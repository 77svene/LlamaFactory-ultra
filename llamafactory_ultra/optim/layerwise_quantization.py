"""
Layerwise Mixed-Precision Quantization for LlamaFactory-ultra
Automatic Mixed-Precision Strategy Search with Bayesian Optimization
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import bitsandbytes as bnb
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class PrecisionType(Enum):
    """Supported precision types for quantization."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"


@dataclass
class LayerConfig:
    """Configuration for a single layer's precision."""
    layer_name: str
    precision: PrecisionType
    sensitivity_score: float = 0.0
    memory_bytes: int = 0
    compute_flops: int = 0


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision quantization."""
    layer_configs: Dict[str, LayerConfig] = field(default_factory=dict)
    total_memory_bytes: int = 0
    total_compute_flops: int = 0
    accuracy_loss: float = 0.0
    optimization_score: float = 0.0


class SensitivityAnalyzer:
    """Analyzes layer sensitivity using gradient-based metrics."""
    
    def __init__(self, model: PreTrainedModel, calibration_data: List[Tensor]):
        self.model = model
        self.calibration_data = calibration_data
        self.hooks = []
        self.gradient_norms = {}
        self.activation_stats = {}
        
    def _register_hooks(self):
        """Register forward and backward hooks for gradient collection."""
        def forward_hook(module, input, output):
            if isinstance(output, Tensor):
                self.activation_stats[module] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'max': output.abs().max().item()
                }
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad_norm = grad_output[0].norm().item()
                self.gradient_norms[module] = grad_norm
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_sensitivity_scores(self) -> Dict[str, float]:
        """
        Compute sensitivity scores for each layer using gradient norms and activation statistics.
        
        Returns:
            Dictionary mapping layer names to sensitivity scores (higher = more sensitive)
        """
        self._register_hooks()
        
        # Run forward and backward passes on calibration data
        self.model.train()
        for data in self.calibration_data:
            if isinstance(data, dict):
                outputs = self.model(**data)
            else:
                outputs = self.model(data)
            
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = outputs.mean()
            
            loss.backward()
        
        # Compute sensitivity scores
        sensitivity_scores = {}
        for name, module in self.model.named_modules():
            if module in self.gradient_norms:
                grad_norm = self.gradient_norms[module]
                act_stats = self.activation_stats.get(module, {})
                
                # Combined sensitivity metric: gradient norm * activation range
                sensitivity = grad_norm * (act_stats.get('max', 1.0) + 1e-8)
                sensitivity_scores[name] = sensitivity
        
        self._remove_hooks()
        self.model.zero_grad()
        
        return sensitivity_scores


class BayesianPrecisionOptimizer:
    """Bayesian optimization for finding optimal mixed-precision configuration."""
    
    def __init__(self, 
                 sensitivity_scores: Dict[str, float],
                 layer_shapes: Dict[str, Tuple[int, ...]],
                 target_memory_reduction: float = 0.5,
                 n_iter: int = 50,
                 init_points: int = 10):
        self.sensitivity_scores = sensitivity_scores
        self.layer_shapes = layer_shapes
        self.target_memory_reduction = target_memory_reduction
        self.n_iter = n_iter
        self.init_points = init_points
        self.layer_names = list(sensitivity_scores.keys())
        self.n_layers = len(self.layer_names)
        
        # Define precision options and their memory/compute characteristics
        self.precision_options = [
            PrecisionType.FP32,
            PrecisionType.FP16,
            PrecisionType.BF16,
            PrecisionType.INT8,
            PrecisionType.INT4,
            PrecisionType.NF4
        ]
        
        # Memory bytes per element for each precision
        self.precision_memory = {
            PrecisionType.FP32: 4,
            PrecisionType.FP16: 2,
            PrecisionType.BF16: 2,
            PrecisionType.INT8: 1,
            PrecisionType.INT4: 0.5,
            PrecisionType.NF4: 0.5
        }
        
        # Compute efficiency factor (relative to FP32)
        self.precision_compute = {
            PrecisionType.FP32: 1.0,
            PrecisionType.FP16: 2.0,
            PrecisionType.BF16: 2.0,
            PrecisionType.INT8: 4.0,
            PrecisionType.INT4: 8.0,
            PrecisionType.NF4: 8.0
        }
    
    def _compute_layer_memory(self, layer_name: str, precision: PrecisionType) -> int:
        """Compute memory usage for a layer in given precision."""
        shape = self.layer_shapes[layer_name]
        elements = np.prod(shape)
        return int(elements * self.precision_memory[precision])
    
    def _compute_layer_flops(self, layer_name: str, precision: PrecisionType) -> int:
        """Compute FLOPs for a layer in given precision."""
        shape = self.layer_shapes[layer_name]
        # Simplified FLOP calculation: 2 * M * N for matrix multiplication
        if len(shape) == 2:  # Linear layer
            flops = 2 * shape[0] * shape[1]
        else:  # Conv or other
            flops = 2 * np.prod(shape)
        
        return int(flops / self.precision_compute[precision])
    
    def _configuration_to_vector(self, config: Dict[str, PrecisionType]) -> np.ndarray:
        """Convert configuration dictionary to optimization vector."""
        vector = np.zeros(self.n_layers)
        for i, layer_name in enumerate(self.layer_names):
            precision = config[layer_name]
            vector[i] = self.precision_options.index(precision)
        return vector
    
    def _vector_to_configuration(self, vector: np.ndarray) -> Dict[str, PrecisionType]:
        """Convert optimization vector to configuration dictionary."""
        config = {}
        for i, layer_name in enumerate(self.layer_names):
            idx = int(np.clip(vector[i], 0, len(self.precision_options) - 1))
            config[layer_name] = self.precision_options[idx]
        return config
    
    def _evaluate_configuration(self, vector: np.ndarray) -> float:
        """
        Evaluate a configuration vector.
        
        Returns:
            Negative optimization score (for minimization)
        """
        config = self._vector_to_configuration(vector)
        
        total_memory = 0
        total_flops = 0
        total_sensitivity_loss = 0
        
        for layer_name, precision in config.items():
            sensitivity = self.sensitivity_scores[layer_name]
            memory = self._compute_layer_memory(layer_name, precision)
            flops = self._compute_layer_flops(layer_name, precision)
            
            total_memory += memory
            total_flops += flops
            
            # Penalize high-precision for sensitive layers (they should stay high precision)
            if sensitivity > np.percentile(list(self.sensitivity_scores.values()), 75):
                if precision in [PrecisionType.INT4, PrecisionType.NF4]:
                    total_sensitivity_loss += sensitivity * 2.0
                elif precision == PrecisionType.INT8:
                    total_sensitivity_loss += sensitivity * 1.0
        
        # Memory reduction ratio
        fp32_memory = sum(self._compute_layer_memory(name, PrecisionType.FP32) 
                         for name in self.layer_names)
        memory_reduction = 1.0 - (total_memory / fp32_memory)
        
        # Optimization score: maximize memory reduction while minimizing sensitivity loss
        memory_score = -abs(memory_reduction - self.target_memory_reduction) * 100
        sensitivity_penalty = total_sensitivity_loss / len(self.layer_names)
        
        # Combined score (higher is better, but we minimize negative)
        score = memory_score - sensitivity_penalty
        
        return -score  # Return negative for minimization
    
    def optimize(self) -> MixedPrecisionConfig:
        """
        Run Bayesian optimization to find optimal mixed-precision configuration.
        
        Returns:
            Optimized MixedPrecisionConfig
        """
        logger.info(f"Starting Bayesian optimization for {self.n_layers} layers")
        
        # Define bounds for each layer (indices into precision_options)
        bounds = [(0, len(self.precision_options) - 1) for _ in range(self.n_layers)]
        
        # Initialize Gaussian Process
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        # Initial random samples
        X_init = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.init_points, self.n_layers)
        )
        y_init = np.array([self._evaluate_configuration(x) for x in X_init])
        
        # Bayesian optimization loop
        X = X_init
        y = y_init
        
        for i in range(self.n_iter):
            # Fit GP
            gp.fit(X, y)
            
            # Find next point to evaluate using Expected Improvement
            x_next = self._propose_location(gp, X, y, bounds)
            y_next = self._evaluate_configuration(x_next)
            
            # Update dataset
            X = np.vstack([X, x_next.reshape(1, -1)])
            y = np.append(y, y_next)
            
            if (i + 1) % 10 == 0:
                best_idx = np.argmin(y)
                logger.info(f"Iteration {i+1}/{self.n_iter}, best score: {-y[best_idx]:.4f}")
        
        # Get best configuration
        best_idx = np.argmin(y)
        best_vector = X[best_idx]
        best_config = self._vector_to_configuration(best_vector)
        
        # Create MixedPrecisionConfig
        layer_configs = {}
        total_memory = 0
        total_flops = 0
        
        for layer_name, precision in best_config.items():
            sensitivity = self.sensitivity_scores[layer_name]
            memory = self._compute_layer_memory(layer_name, precision)
            flops = self._compute_layer_flops(layer_name, precision)
            
            layer_configs[layer_name] = LayerConfig(
                layer_name=layer_name,
                precision=precision,
                sensitivity_score=sensitivity,
                memory_bytes=memory,
                compute_flops=flops
            )
            
            total_memory += memory
            total_flops += flops
        
        # Calculate memory reduction
        fp32_memory = sum(self._compute_layer_memory(name, PrecisionType.FP32) 
                         for name in self.layer_names)
        memory_reduction = 1.0 - (total_memory / fp32_memory)
        
        config = MixedPrecisionConfig(
            layer_configs=layer_configs,
            total_memory_bytes=total_memory,
            total_compute_flops=total_flops,
            optimization_score=-y[best_idx]
        )
        
        logger.info(f"Optimization complete. Memory reduction: {memory_reduction:.2%}")
        return config
    
    def _propose_location(self, gp, X, y, bounds, n_restarts=25):
        """Propose next location using Expected Improvement."""
        best_y = np.min(y)
        
        def expected_improvement(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = max(sigma[0], 1e-9)
            
            z = (best_y - mu) / sigma
            ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei[0]  # Negative for minimization
        
        # Random restart optimization
        best_x = None
        best_ei = float('inf')
        
        for _ in range(n_restarts):
            x0 = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds]
            )
            
            # Simple local search
            for _ in range(10):
                # Perturb each dimension
                for i in range(len(x0)):
                    delta = np.random.uniform(-0.5, 0.5)
                    x_candidate = x0.copy()
                    x_candidate[i] = np.clip(x_candidate[i] + delta, bounds[i][0], bounds[i][1])
                    
                    ei = expected_improvement(x_candidate)
                    if ei < best_ei:
                        best_ei = ei
                        best_x = x_candidate
                        x0 = x_candidate
        
        return best_x if best_x is not None else X[-1]


class DynamicQuantizer:
    """Applies dynamic quantization based on mixed-precision configuration."""
    
    def __init__(self, model: PreTrainedModel, config: MixedPrecisionConfig):
        self.model = model
        self.config = config
        self.original_modules = {}
        
    def _get_quantization_config(self, precision: PrecisionType) -> Dict[str, Any]:
        """Get bitsandbytes quantization config for given precision."""
        if precision == PrecisionType.INT8:
            return {
                'load_in_8bit': True,
                'llm_int8_threshold': 6.0,
                'llm_int8_has_fp16_weight': False
            }
        elif precision == PrecisionType.INT4:
            return {
                'load_in_4bit': True,
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_quant_type': 'fp4',
                'bnb_4bit_use_double_quant': True
            }
        elif precision == PrecisionType.NF4:
            return {
                'load_in_4bit': True,
                'bnb_4bit_compute_dtype': torch.float16,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_use_double_quant': True
            }
        else:
            return {}
    
    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module):
        """Replace a module in the parent module."""
        setattr(parent, child_name, new_module)
    
    def apply_quantization(self):
        """Apply mixed-precision quantization to the model."""
        logger.info("Applying mixed-precision quantization...")
        
        # Store original modules for potential restoration
        for name, module in self.model.named_modules():
            self.original_modules[name] = module
        
        # Apply quantization layer by layer
        for layer_name, layer_config in self.config.layer_configs.items():
            precision = layer_config.precision
            
            # Skip if already in target precision or FP32/FP16/BF16 (no quantization needed)
            if precision in [PrecisionType.FP32, PrecisionType.FP16, PrecisionType.BF16]:
                continue
            
            # Get the module to quantize
            parts = layer_name.split('.')
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            module = getattr(parent, parts[-1])
            
            # Only quantize linear layers
            if not isinstance(module, nn.Linear):
                continue
            
            # Get quantization config
            quant_config = self._get_quantization_config(precision)
            
            if precision == PrecisionType.INT8:
                # Replace with 8-bit linear layer
                new_module = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0
                )
                # Copy weights
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                    
            elif precision in [PrecisionType.INT4, PrecisionType.NF4]:
                # Replace with 4-bit linear layer
                new_module = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                    compress_statistics=True,
                    quant_type='nf4' if precision == PrecisionType.NF4 else 'fp4'
                )
                # Copy weights (will be quantized automatically)
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
            
            # Replace the module
            self._replace_module(parent, parts[-1], new_module)
            logger.debug(f"Quantized {layer_name} to {precision.value}")
        
        logger.info("Mixed-precision quantization applied successfully")
    
    def restore_original(self):
        """Restore original modules (remove quantization)."""
        logger.info("Restoring original model...")
        
        for name, module in self.model.named_modules():
            if name in self.original_modules:
                parts = name.split('.')
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                self._replace_module(parent, parts[-1], self.original_modules[name])
        
        logger.info("Original model restored")


class LayerwiseQuantizationManager:
    """Main manager for layerwise mixed-precision quantization."""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 calibration_data: List[Tensor],
                 target_memory_reduction: float = 0.5,
                 optimization_iterations: int = 50):
        self.model = model
        self.calibration_data = calibration_data
        self.target_memory_reduction = target_memory_reduction
        self.optimization_iterations = optimization_iterations
        self.config: Optional[MixedPrecisionConfig] = None
        self.quantizer: Optional[DynamicQuantizer] = None
        
    def _get_layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shapes of all linear layers in the model."""
        shapes = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                shapes[name] = (module.out_features, module.in_features)
        return shapes
    
    def analyze_and_optimize(self) -> MixedPrecisionConfig:
        """
        Analyze model sensitivity and optimize mixed-precision configuration.
        
        Returns:
            Optimized MixedPrecisionConfig
        """
        logger.info("Starting layerwise quantization analysis...")
        
        # Step 1: Compute sensitivity scores
        analyzer = SensitivityAnalyzer(self.model, self.calibration_data)
        sensitivity_scores = analyzer.compute_sensitivity_scores()
        
        # Step 2: Get layer shapes
        layer_shapes = self._get_layer_shapes()
        
        # Step 3: Run Bayesian optimization
        optimizer = BayesianPrecisionOptimizer(
            sensitivity_scores=sensitivity_scores,
            layer_shapes=layer_shapes,
            target_memory_reduction=self.target_memory_reduction,
            n_iter=self.optimization_iterations
        )
        
        self.config = optimizer.optimize()
        
        # Log configuration summary
        self._log_configuration_summary()
        
        return self.config
    
    def apply_quantization(self):
        """Apply the optimized mixed-precision quantization."""
        if self.config is None:
            raise ValueError("Configuration not computed. Run analyze_and_optimize() first.")
        
        self.quantizer = DynamicQuantizer(self.model, self.config)
        self.quantizer.apply_quantization()
    
    def restore_original(self):
        """Restore original model (remove quantization)."""
        if self.quantizer is not None:
            self.quantizer.restore_original()
    
    def _log_configuration_summary(self):
        """Log summary of the optimized configuration."""
        if self.config is None:
            return
        
        precision_counts = {}
        for layer_config in self.config.layer_configs.values():
            precision = layer_config.precision.value
            precision_counts[precision] = precision_counts.get(precision, 0) + 1
        
        logger.info("Mixed-Precision Configuration Summary:")
        for precision, count in precision_counts.items():
            logger.info(f"  {precision}: {count} layers")
        
        # Calculate memory reduction
        total_fp32 = sum(
            layer_config.memory_bytes * (4 / self.quantizer.precision_memory[layer_config.precision])
            for layer_config in self.config.layer_configs.values()
        )
        memory_reduction = 1.0 - (self.config.total_memory_bytes / total_fp32)
        
        logger.info(f"Total memory: {self.config.total_memory_bytes / 1024**2:.2f} MB")
        logger.info(f"Memory reduction: {memory_reduction:.2%}")
        logger.info(f"Optimization score: {self.config.optimization_score:.4f}")


# Integration with Hugging Face Trainer
class LayerwiseQuantizationCallback:
    """Callback for Hugging Face Trainer to apply layerwise quantization."""
    
    def __init__(self, 
                 calibration_dataloader,
                 target_memory_reduction: float = 0.5,
                 optimization_iterations: int = 50):
        self.calibration_dataloader = calibration_dataloader
        self.target_memory_reduction = target_memory_reduction
        self.optimization_iterations = optimization_iterations
        self.manager = None
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Apply quantization at the beginning of training."""
        if model is None:
            return
        
        # Collect calibration data
        calibration_data = []
        for batch in self.calibration_dataloader:
            if isinstance(batch, dict):
                # Move to same device as model
                batch = {k: v.to(model.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
            calibration_data.append(batch)
        
        # Create and run quantization manager
        self.manager = LayerwiseQuantizationManager(
            model=model,
            calibration_data=calibration_data,
            target_memory_reduction=self.target_memory_reduction,
            optimization_iterations=self.optimization_iterations
        )
        
        # Analyze and apply quantization
        self.manager.analyze_and_optimize()
        self.manager.apply_quantization()
        
        logger.info("Layerwise quantization applied successfully")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Optionally restore original model at the end of training."""
        # Uncomment if you want to restore original model after training
        # if self.manager is not None:
        #     self.manager.restore_original()
        pass


# Utility functions for integration
def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def estimate_memory_savings(original_model: nn.Module, 
                          quantized_model: nn.Module) -> Dict[str, float]:
    """Estimate memory savings from quantization."""
    original_size = get_model_size_mb(original_model)
    quantized_size = get_model_size_mb(quantized_model)
    
    savings = original_size - quantized_size
    reduction_pct = (savings / original_size) * 100 if original_size > 0 else 0
    
    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'savings_mb': savings,
        'reduction_percent': reduction_pct
    }