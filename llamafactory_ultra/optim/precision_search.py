import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import copy
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math
import time
from collections import defaultdict

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt, Linear4bit, Params4bit
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


class PrecisionType(Enum):
    """Supported precision types for mixed-precision training."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class LayerPrecisionConfig:
    """Configuration for a single layer's precision settings."""
    layer_name: str
    precision: PrecisionType
    sensitivity_score: float = 0.0
    memory_saving: float = 0.0
    accuracy_impact: float = 0.0


@dataclass
class PrecisionSearchConfig:
    """Configuration for the precision search algorithm."""
    num_sensitivity_steps: int = 10
    num_optimization_trials: int = 50
    target_memory_reduction: float = 0.5  # 50% memory reduction target
    accuracy_tolerance: float = 0.01  # 1% accuracy drop tolerance
    min_precision: PrecisionType = PrecisionType.INT4
    max_precision: PrecisionType = PrecisionType.FP32
    quantization_granularity: str = "layer"  # "layer", "block", or "module"
    enable_bayesian_optimization: bool = True
    gradient_accumulation_steps: int = 1
    sensitivity_metric: str = "gradient_norm"  # "gradient_norm", "hessian_trace", "fisher_information"


class GradientSensitivityAnalyzer:
    """
    Analyzes layer sensitivity based on gradient statistics during training.
    Higher gradient norms indicate layers more sensitive to quantization.
    """
    
    def __init__(self, model: nn.Module, config: PrecisionSearchConfig):
        self.model = model
        self.config = config
        self.gradient_norms = defaultdict(list)
        self.activation_norms = defaultdict(list)
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks to capture gradients and activations."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Backward hook for gradients
                hook = module.register_backward_hook(
                    lambda module, grad_input, grad_output, name=name: 
                    self._backward_hook(name, grad_output[0])
                )
                self.hooks.append(hook)
                
                # Forward hook for activations
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name:
                    self._forward_hook(name, output)
                )
                self.hooks.append(hook)
    
    def _backward_hook(self, name: str, grad_output: torch.Tensor):
        """Capture gradient statistics during backward pass."""
        if grad_output is not None:
            grad_norm = grad_output.norm(p=2).item()
            self.gradient_norms[name].append(grad_norm)
    
    def _forward_hook(self, name: str, output: torch.Tensor):
        """Capture activation statistics during forward pass."""
        if output is not None:
            act_norm = output.norm(p=2).item()
            self.activation_norms[name].append(act_norm)
    
    def compute_sensitivity_scores(self) -> Dict[str, float]:
        """
        Compute sensitivity scores for each layer based on gradient statistics.
        
        Returns:
            Dictionary mapping layer names to sensitivity scores (higher = more sensitive)
        """
        sensitivity_scores = {}
        
        for name in self.gradient_norms:
            if len(self.gradient_norms[name]) > 0:
                # Use coefficient of variation (std/mean) as sensitivity metric
                grads = np.array(self.gradient_norms[name])
                if np.mean(grads) > 0:
                    sensitivity = np.std(grads) / np.mean(grads)
                else:
                    sensitivity = 0.0
                
                # Combine with activation statistics
                if name in self.activation_norms and len(self.activation_norms[name]) > 0:
                    acts = np.array(self.activation_norms[name])
                    act_sensitivity = np.std(acts) / np.mean(acts) if np.mean(acts) > 0 else 0.0
                    sensitivity = 0.7 * sensitivity + 0.3 * act_sensitivity
                
                sensitivity_scores[name] = sensitivity
        
        return sensitivity_scores
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class BayesianPrecisionOptimizer:
    """
    Bayesian optimization for finding optimal mixed-precision configuration.
    Uses Gaussian Process to model the accuracy-memory trade-off.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 config: PrecisionSearchConfig):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.best_config = None
        self.best_score = float('-inf')
        
        if not HAS_OPTUNA:
            logger.warning("Optuna not installed. Using random search instead.")
    
    def _create_search_space(self, 
                            sensitivity_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Create search space based on sensitivity scores.
        More sensitive layers get narrower search spaces (higher precision options).
        """
        search_space = {}
        
        # Sort layers by sensitivity (most sensitive first)
        sorted_layers = sorted(sensitivity_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        for layer_name, sensitivity in sorted_layers:
            if sensitivity > 0.8:  # Highly sensitive
                # Only allow FP32/FP16/BF16 for highly sensitive layers
                search_space[layer_name] = ["fp32", "fp16", "bf16"]
            elif sensitivity > 0.5:  # Moderately sensitive
                search_space[layer_name] = ["fp16", "bf16", "int8"]
            elif sensitivity > 0.2:  # Low sensitivity
                search_space[layer_name] = ["bf16", "int8", "int4"]
            else:  # Very low sensitivity
                search_space[layer_name] = ["int8", "int4"]
        
        return search_space
    
    def _evaluate_configuration(self, 
                               config: Dict[str, str],
                               sensitivity_scores: Dict[str, float]) -> float:
        """
        Evaluate a mixed-precision configuration.
        
        Returns:
            Combined score (higher is better) balancing memory and accuracy
        """
        # Create a copy of the model with the specified precision
        model_copy = copy.deepcopy(self.model)
        self._apply_precision_config(model_copy, config)
        
        # Evaluate accuracy
        accuracy = self._evaluate_accuracy(model_copy)
        
        # Calculate memory savings
        memory_saving = self._calculate_memory_saving(config)
        
        # Calculate sensitivity-weighted accuracy impact
        accuracy_impact = 0.0
        for layer_name, precision in config.items():
            if layer_name in sensitivity_scores:
                sensitivity = sensitivity_scores[layer_name]
                precision_value = self._precision_to_value(precision)
                # Higher sensitivity and lower precision = higher impact
                impact = sensitivity * (1 - precision_value)
                accuracy_impact += impact
        
        # Normalize accuracy impact
        if len(config) > 0:
            accuracy_impact /= len(config)
        
        # Combined score: maximize memory saving while minimizing accuracy impact
        score = memory_saving - self.config.accuracy_tolerance * accuracy_impact
        
        # Penalize if accuracy drops below tolerance
        if accuracy < (1 - self.config.accuracy_tolerance):
            score -= 10.0  # Large penalty for accuracy drop
        
        return score
    
    def _apply_precision_config(self, model: nn.Module, config: Dict[str, str]):
        """Apply precision configuration to model."""
        for name, module in model.named_modules():
            if name in config:
                precision = config[name]
                self._set_module_precision(module, precision)
    
    def _set_module_precision(self, module: nn.Module, precision: str):
        """Set precision for a specific module."""
        if precision == "fp32":
            module.float()
        elif precision == "fp16":
            module.half()
        elif precision == "bf16":
            if hasattr(module, 'to_bf16'):
                module.to_bf16()
            else:
                module.to(torch.bfloat16)
        elif precision == "int8" and HAS_BITSANDBYTES:
            self._quantize_to_int8(module)
        elif precision == "int4" and HAS_BITSANDBYTES:
            self._quantize_to_int4(module)
    
    def _quantize_to_int8(self, module: nn.Module):
        """Quantize module to INT8 using bitsandbytes."""
        if not HAS_BITSANDBYTES:
            logger.warning("bitsandbytes not available. Skipping INT8 quantization.")
            return
        
        if isinstance(module, nn.Linear):
            # Replace with 8-bit linear layer
            quantized = Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False
            )
            # Copy weights
            quantized.weight = module.weight
            if module.bias is not None:
                quantized.bias = module.bias
            return quantized
    
    def _quantize_to_int4(self, module: nn.Module):
        """Quantize module to INT4 using bitsandbytes."""
        if not HAS_BITSANDBYTES:
            logger.warning("bitsandbytes not available. Skipping INT4 quantization.")
            return
        
        if isinstance(module, nn.Linear):
            # Replace with 4-bit linear layer
            quantized = Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float16,
                compress_statistics=True,
                quant_type="nf4"
            )
            # Copy weights
            quantized.weight = module.weight
            if module.bias is not None:
                quantized.bias = module.bias
            return quantized
    
    def _precision_to_value(self, precision: str) -> float:
        """Convert precision string to numerical value (higher = more precise)."""
        mapping = {
            "fp32": 1.0,
            "bf16": 0.8,
            "fp16": 0.7,
            "int8": 0.5,
            "int4": 0.25
        }
        return mapping.get(precision, 0.0)
    
    def _evaluate_accuracy(self, model: nn.Module) -> float:
        """Evaluate model accuracy on validation set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                inputs, labels = batch
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_memory_saving(self, config: Dict[str, str]) -> float:
        """Calculate memory savings from precision configuration."""
        total_params = sum(p.numel() for p in self.model.parameters())
        memory_saving = 0.0
        
        for layer_name, precision in config.items():
            # Get layer parameters
            layer = dict(self.model.named_modules())[layer_name]
            layer_params = sum(p.numel() for p in layer.parameters())
            
            # Calculate memory reduction for this layer
            if precision == "fp32":
                reduction = 0.0
            elif precision in ["fp16", "bf16"]:
                reduction = 0.5  # 50% reduction from FP32
            elif precision == "int8":
                reduction = 0.75  # 75% reduction from FP32
            elif precision == "int4":
                reduction = 0.875  # 87.5% reduction from FP32
            else:
                reduction = 0.0
            
            memory_saving += (layer_params / total_params) * reduction
        
        return memory_saving
    
    def optimize(self, 
                sensitivity_scores: Dict[str, float]) -> Dict[str, str]:
        """
        Run Bayesian optimization to find optimal precision configuration.
        
        Returns:
            Optimal configuration dictionary
        """
        search_space = self._create_search_space(sensitivity_scores)
        layer_names = list(search_space.keys())
        
        if HAS_OPTUNA and self.config.enable_bayesian_optimization:
            return self._optimize_with_optuna(search_space, layer_names, sensitivity_scores)
        else:
            return self._optimize_with_random_search(search_space, layer_names, sensitivity_scores)
    
    def _optimize_with_optuna(self,
                             search_space: Dict[str, List[str]],
                             layer_names: List[str],
                             sensitivity_scores: Dict[str, float]) -> Dict[str, str]:
        """Optimize using Optuna (Bayesian optimization)."""
        def objective(trial):
            config = {}
            for layer_name in layer_names:
                precision = trial.suggest_categorical(
                    layer_name, 
                    search_space[layer_name]
                )
                config[layer_name] = precision
            
            score = self._evaluate_configuration(config, sensitivity_scores)
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
            
            return score
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.num_optimization_trials)
        
        return self.best_config if self.best_config else study.best_params
    
    def _optimize_with_random_search(self,
                                    search_space: Dict[str, List[str]],
                                    layer_names: List[str],
                                    sensitivity_scores: Dict[str, float]) -> Dict[str, str]:
        """Fallback random search if Optuna is not available."""
        best_config = None
        best_score = float('-inf')
        
        for _ in range(self.config.num_optimization_trials):
            config = {}
            for layer_name in layer_names:
                config[layer_name] = np.random.choice(search_space[layer_name])
            
            score = self._evaluate_configuration(config, sensitivity_scores)
            
            if score > best_score:
                best_score = score
                best_config = config
        
        return best_config


class MixedPrecisionSearcher:
    """
    Main class for automatic mixed-precision strategy search.
    Dynamically determines optimal precision per layer during training.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 config: Optional[PrecisionSearchConfig] = None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or PrecisionSearchConfig()
        
        self.sensitivity_analyzer = GradientSensitivityAnalyzer(model, self.config)
        self.optimizer = BayesianPrecisionOptimizer(
            model, train_dataloader, eval_dataloader, self.config
        )
        
        self.precision_config: Dict[str, PrecisionType] = {}
        self.sensitivity_scores: Dict[str, float] = {}
        self.is_searched = False
        
        logger.info(f"Initialized MixedPrecisionSearcher with config: {self.config}")
    
    def run_sensitivity_analysis(self, 
                                model: Optional[nn.Module] = None,
                                dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Run sensitivity analysis on the model.
        
        Returns:
            Dictionary of sensitivity scores per layer
        """
        if model is None:
            model = self.model
        if dataloader is None:
            dataloader = self.train_dataloader
        
        logger.info("Running sensitivity analysis...")
        
        # Set model to training mode
        model.train()
        
        # Create optimizer for sensitivity analysis
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Run training steps to collect gradient statistics
        for step, batch in enumerate(dataloader):
            if step >= self.config.num_sensitivity_steps:
                break
            
            optimizer.zero_grad()
            
            # Forward pass
            inputs, labels = batch
            outputs = model(inputs)
            
            # Compute loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            if step % 5 == 0:
                logger.info(f"Sensitivity analysis step {step}/{self.config.num_sensitivity_steps}")
        
        # Compute sensitivity scores
        self.sensitivity_scores = self.sensitivity_analyzer.compute_sensitivity_scores()
        
        # Clear hooks
        self.sensitivity_analyzer.clear_hooks()
        
        logger.info(f"Sensitivity analysis complete. Analyzed {len(self.sensitivity_scores)} layers.")
        
        return self.sensitivity_scores
    
    def search_optimal_precision(self) -> Dict[str, PrecisionType]:
        """
        Search for optimal mixed-precision configuration.
        
        Returns:
            Dictionary mapping layer names to optimal precision types
        """
        if not self.sensitivity_scores:
            self.run_sensitivity_analysis()
        
        logger.info("Searching for optimal precision configuration...")
        
        # Run Bayesian optimization
        optimal_config_str = self.optimizer.optimize(self.sensitivity_scores)
        
        # Convert string config to PrecisionType enum
        self.precision_config = {
            layer_name: PrecisionType(precision_str)
            for layer_name, precision_str in optimal_config_str.items()
        }
        
        self.is_searched = True
        
        # Log results
        self._log_search_results()
        
        return self.precision_config
    
    def apply_precision_config(self, 
                              model: Optional[nn.Module] = None,
                              config: Optional[Dict[str, PrecisionType]] = None):
        """
        Apply the optimal precision configuration to the model.
        
        Args:
            model: Model to apply configuration to (uses self.model if None)
            config: Configuration to apply (uses self.precision_config if None)
        """
        if model is None:
            model = self.model
        if config is None:
            if not self.is_searched:
                raise ValueError("No precision configuration available. Run search_optimal_precision() first.")
            config = self.precision_config
        
        logger.info("Applying precision configuration to model...")
        
        # Track statistics
        precision_counts = defaultdict(int)
        total_layers = 0
        
        for name, module in model.named_modules():
            if name in config:
                precision = config[name]
                self._set_module_precision(module, precision)
                precision_counts[precision] += 1
                total_layers += 1
        
        # Log statistics
        logger.info(f"Applied precision configuration to {total_layers} layers:")
        for precision, count in precision_counts.items():
            logger.info(f"  {precision.value}: {count} layers ({count/total_layers*100:.1f}%)")
    
    def _set_module_precision(self, module: nn.Module, precision: PrecisionType):
        """Set precision for a specific module."""
        if precision == PrecisionType.FP32:
            module.float()
        elif precision == PrecisionType.FP16:
            module.half()
        elif precision == PrecisionType.BF16:
            if hasattr(module, 'to_bf16'):
                module.to_bf16()
            else:
                module.to(torch.bfloat16)
        elif precision == PrecisionType.INT8 and HAS_BITSANDBYTES:
            self._quantize_module(module, "int8")
        elif precision == PrecisionType.INT4 and HAS_BITSANDBYTES:
            self._quantize_module(module, "int4")
    
    def _quantize_module(self, module: nn.Module, precision: str):
        """Quantize a module to specified precision."""
        if not HAS_BITSANDBYTES:
            logger.warning(f"bitsandbytes not available. Skipping {precision} quantization.")
            return
        
        if isinstance(module, nn.Linear):
            if precision == "int8":
                quantized = Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False
                )
            elif precision == "int4":
                quantized = Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                    compress_statistics=True,
                    quant_type="nf4"
                )
            else:
                return
            
            # Copy weights and bias
            quantized.weight = module.weight
            if module.bias is not None:
                quantized.bias = module.bias
            
            return quantized
    
    def _log_search_results(self):
        """Log detailed search results."""
        if not self.precision_config:
            return
        
        logger.info("=" * 60)
        logger.info("MIXED-PRECISION SEARCH RESULTS")
        logger.info("=" * 60)
        
        # Group by precision type
        precision_groups = defaultdict(list)
        for layer_name, precision in self.precision_config.items():
            precision_groups[precision].append(layer_name)
        
        # Calculate memory savings
        total_params = sum(p.numel() for p in self.model.parameters())
        memory_saving = 0.0
        
        for layer_name, precision in self.precision_config.items():
            layer = dict(self.model.named_modules())[layer_name]
            layer_params = sum(p.numel() for p in layer.parameters())
            
            if precision == PrecisionType.FP32:
                reduction = 0.0
            elif precision in [PrecisionType.FP16, PrecisionType.BF16]:
                reduction = 0.5
            elif precision == PrecisionType.INT8:
                reduction = 0.75
            elif precision == PrecisionType.INT4:
                reduction = 0.875
            else:
                reduction = 0.0
            
            memory_saving += (layer_params / total_params) * reduction
        
        logger.info(f"Estimated memory saving: {memory_saving*100:.1f}%")
        logger.info(f"Target memory reduction: {self.config.target_memory_reduction*100:.1f}%")
        
        for precision, layers in precision_groups.items():
            logger.info(f"\n{precision.value.upper()} ({len(layers)} layers):")
            for layer in layers[:5]:  # Show first 5 layers
                logger.info(f"  - {layer}")
            if len(layers) > 5:
                logger.info(f"  ... and {len(layers) - 5} more layers")
        
        logger.info("=" * 60)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of the precision configuration."""
        if not self.is_searched:
            return {"status": "not_searched"}
        
        precision_counts = defaultdict(int)
        for precision in self.precision_config.values():
            precision_counts[precision.value] += 1
        
        return {
            "status": "searched",
            "total_layers": len(self.precision_config),
            "precision_distribution": dict(precision_counts),
            "sensitivity_scores": self.sensitivity_scores,
            "config": {k: v.value for k, v in self.precision_config.items()}
        }
    
    def save_config(self, filepath: str):
        """Save precision configuration to file."""
        import json
        
        config_data = {
            "precision_config": {k: v.value for k, v in self.precision_config.items()},
            "sensitivity_scores": self.sensitivity_scores,
            "search_config": {
                "num_sensitivity_steps": self.config.num_sensitivity_steps,
                "num_optimization_trials": self.config.num_optimization_trials,
                "target_memory_reduction": self.config.target_memory_reduction,
                "accuracy_tolerance": self.config.accuracy_tolerance,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Precision configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """Load precision configuration from file."""
        import json
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.precision_config = {
            k: PrecisionType(v) for k, v in config_data["precision_config"].items()
        }
        self.sensitivity_scores = config_data.get("sensitivity_scores", {})
        self.is_searched = True
        
        logger.info(f"Precision configuration loaded from {filepath}")


def create_precision_searcher(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    **kwargs
) -> MixedPrecisionSearcher:
    """
    Factory function to create a MixedPrecisionSearcher instance.
    
    Args:
        model: The model to optimize
        train_dataloader: DataLoader for training data (used for sensitivity analysis)
        eval_dataloader: DataLoader for evaluation data
        **kwargs: Additional configuration parameters
    
    Returns:
        MixedPrecisionSearcher instance
    """
    config = PrecisionSearchConfig(**kwargs)
    return MixedPrecisionSearcher(model, train_dataloader, eval_dataloader, config)


def apply_dynamic_quantization(
    model: nn.Module,
    precision_config: Dict[str, PrecisionType],
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Apply dynamic quantization based on precision configuration.
    
    Args:
        model: Model to quantize
        precision_config: Layer-wise precision configuration
        device: Target device
    
    Returns:
        Quantized model
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create a copy to avoid modifying original
    quantized_model = copy.deepcopy(model)
    
    for name, module in quantized_model.named_modules():
        if name in precision_config:
            precision = precision_config[name]
            
            if precision == PrecisionType.INT8 and HAS_BITSANDBYTES:
                if isinstance(module, nn.Linear):
                    # Replace with quantized layer
                    quantized = Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False
                    )
                    quantized.weight = module.weight
                    if module.bias is not None:
                        quantized.bias = module.bias
                    
                    # Replace in parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent = quantized_model.get_submodule(parent_name)
                        setattr(parent, name.split('.')[-1], quantized)
                    else:
                        setattr(quantized_model, name, quantized)
    
    return quantized_model.to(device)


# Example usage and integration with LlamaFactory-ultra training loop
def integrate_with_training(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int = 3,
    precision_search_interval: int = 1000  # Re-search every 1000 steps
):
    """
    Example integration of precision search with training loop.
    
    Args:
        model: Model to train
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        optimizer: Optimizer
        num_epochs: Number of training epochs
        precision_search_interval: Steps between precision re-search
    """
    # Create precision searcher
    searcher = create_precision_searcher(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_sensitivity_steps=10,
        num_optimization_trials=30,
        target_memory_reduction=0.5,
        accuracy_tolerance=0.02
    )
    
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Regular training step
            optimizer.zero_grad()
            inputs, labels = batch
            outputs = model(inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else nn.CrossEntropyLoss()(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # Periodically re-search precision configuration
            if global_step % precision_search_interval == 0:
                logger.info(f"Re-searching precision configuration at step {global_step}")
                
                # Run sensitivity analysis on recent data
                searcher.run_sensitivity_analysis()
                
                # Search for optimal configuration
                optimal_config = searcher.search_optimal_precision()
                
                # Apply new configuration
                searcher.apply_precision_config()
                
                # Log memory savings
                summary = searcher.get_config_summary()
                logger.info(f"Memory savings: {summary.get('memory_saving', 0)*100:.1f}%")
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
    
    # Final precision search
    logger.info("Running final precision search...")
    optimal_config = searcher.search_optimal_precision()
    searcher.apply_precision_config()
    
    # Save configuration
    searcher.save_config("precision_config.json")
    
    return searcher


# Utility functions for memory estimation
def estimate_model_memory(model: nn.Module, precision_config: Dict[str, PrecisionType]) -> Dict[str, float]:
    """
    Estimate memory usage of model with given precision configuration.
    
    Returns:
        Dictionary with memory estimates in MB
    """
    total_memory = 0
    layer_memories = {}
    
    for name, param in model.named_parameters():
        # Find which layer this parameter belongs to
        layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else name
        
        if layer_name in precision_config:
            precision = precision_config[layer_name]
            
            if precision == PrecisionType.FP32:
                bytes_per_param = 4
            elif precision in [PrecisionType.FP16, PrecisionType.BF16]:
                bytes_per_param = 2
            elif precision == PrecisionType.INT8:
                bytes_per_param = 1
            elif precision == PrecisionType.INT4:
                bytes_per_param = 0.5
            else:
                bytes_per_param = 4
        else:
            bytes_per_param = 4  # Default FP32
        
        param_memory = param.numel() * bytes_per_param
        total_memory += param_memory
        
        if layer_name not in layer_memories:
            layer_memories[layer_name] = 0
        layer_memories[layer_name] += param_memory
    
    # Convert to MB
    total_memory_mb = total_memory / (1024 ** 2)
    layer_memories_mb = {k: v / (1024 ** 2) for k, v in layer_memories.items()}
    
    return {
        "total_memory_mb": total_memory_mb,
        "layer_memories_mb": layer_memories_mb,
        "total_params": sum(p.numel() for p in model.parameters())
    }


def compare_precision_strategies(
    model: nn.Module,
    strategies: List[Dict[str, PrecisionType]]
) -> List[Dict[str, Any]]:
    """
    Compare multiple precision strategies.
    
    Args:
        model: Model to analyze
        strategies: List of precision configurations to compare
    
    Returns:
        List of comparison results
    """
    results = []
    
    for i, strategy in enumerate(strategies):
        memory_info = estimate_model_memory(model, strategy)
        
        # Count precision types
        precision_counts = defaultdict(int)
        for precision in strategy.values():
            precision_counts[precision.value] += 1
        
        results.append({
            "strategy_id": i,
            "total_memory_mb": memory_info["total_memory_mb"],
            "precision_distribution": dict(precision_counts),
            "num_layers": len(strategy),
            "config": {k: v.value for k, v in strategy.items()}
        })
    
    # Sort by memory usage
    results.sort(key=lambda x: x["total_memory_mb"])
    
    return results


# Export main classes and functions
__all__ = [
    "PrecisionType",
    "LayerPrecisionConfig",
    "PrecisionSearchConfig",
    "GradientSensitivityAnalyzer",
    "BayesianPrecisionOptimizer",
    "MixedPrecisionSearcher",
    "create_precision_searcher",
    "apply_dynamic_quantization",
    "integrate_with_training",
    "estimate_model_memory",
    "compare_precision_strategies"
]