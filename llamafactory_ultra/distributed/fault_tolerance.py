"""
Distributed Training Fault Tolerance with Live Migration
Zero-downtime training recovery with live checkpoint migration across heterogeneous clusters.
"""

import os
import time
import json
import pickle
import hashlib
import threading
import queue
import signal
import atexit
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import socket
import psutil
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Node health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MIGRATING = "migrating"
    RECOVERING = "recovering"


class CheckpointType(Enum):
    """Types of checkpoints"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DELTA = "delta"
    LIVE = "live"


@dataclass
class NodeInfo:
    """Information about a node in the cluster"""
    node_id: str
    hostname: str
    ip_address: str
    rank: int
    world_size: int
    gpu_count: int
    gpu_memory: List[int]
    cpu_cores: int
    total_memory: int
    state: NodeState = NodeState.HEALTHY
    last_heartbeat: float = 0.0
    checkpoint_version: int = 0
    workload_shard: Optional[int] = None
    migration_target: Optional[str] = None
    failure_count: int = 0
    recovery_time: float = 0.0


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    timestamp: float
    epoch: int
    global_step: int
    loss: float
    learning_rate: float
    model_hash: str
    optimizer_hash: str
    checkpoint_type: CheckpointType
    parent_checkpoint: Optional[str] = None
    delta_size: int = 0
    total_size: int = 0
    node_states: Dict[str, NodeState] = field(default_factory=dict)
    shard_distribution: Dict[int, List[int]] = field(default_factory=dict)


@dataclass
class MigrationPlan:
    """Plan for migrating workload from failed nodes"""
    source_node: str
    target_nodes: List[str]
    workload_shard: int
    migration_type: str  # "full", "partial", "incremental"
    estimated_time: float
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)


class CheckpointManager:
    """Manages asynchronous checkpointing with incremental state transfer"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        checkpoint_interval: int = 1000,
        incremental_ratio: float = 0.3,
        compression_level: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.incremental_ratio = incremental_ratio
        self.compression_level = compression_level
        
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.current_checkpoint: Optional[str] = None
        self.checkpoint_queue = queue.Queue()
        self.checkpoint_thread = None
        self.running = False
        
        # State tracking for incremental checkpoints
        self.model_state_hashes: Dict[str, str] = {}
        self.optimizer_state_hashes: Dict[str, str] = {}
        self.last_full_checkpoint: Optional[str] = None
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
        
        # Start checkpoint worker
        self.start()
    
    def _load_existing_checkpoints(self):
        """Load metadata from existing checkpoints"""
        metadata_file = self.checkpoint_dir / "checkpoints.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                for cp_id, cp_data in data.items():
                    self.checkpoints[cp_id] = CheckpointMetadata(**cp_data)
    
    def _save_checkpoint_metadata(self):
        """Save checkpoint metadata to disk"""
        metadata_file = self.checkpoint_dir / "checkpoints.json"
        data = {cp_id: asdict(metadata) for cp_id, metadata in self.checkpoints.items()}
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_state_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Compute hash of state dict for change detection"""
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            if tensor.is_cpu:
                hasher.update(tensor.numpy().tobytes())
            else:
                hasher.update(tensor.cpu().numpy().tobytes())
        return hasher.hexdigest()
    
    def _detect_state_changes(
        self,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Detect changed parameters for incremental checkpointing"""
        current_model_hash = self._compute_state_hash(model_state)
        current_optimizer_hash = self._compute_state_hash(optimizer_state)
        
        # If no previous checkpoint or significant changes, do full checkpoint
        if (self.last_full_checkpoint is None or 
            self.model_state_hashes.get(self.last_full_checkpoint) != current_model_hash):
            return model_state, optimizer_state
        
        # Compute delta for incremental checkpoint
        changed_model = {}
        changed_optimizer = {}
        
        # Detect model changes
        for key, tensor in model_state.items():
            if key not in self.model_state_hashes:
                changed_model[key] = tensor
            else:
                # Simple change detection - in production, use more sophisticated diff
                changed_model[key] = tensor
        
        # Detect optimizer changes
        for key, tensor in optimizer_state.items():
            if key not in self.optimizer_state_hashes:
                changed_optimizer[key] = tensor
            else:
                changed_optimizer[key] = tensor
        
        # If changes are small, return delta
        total_params = sum(p.numel() for p in model_state.values())
        changed_params = sum(p.numel() for p in changed_model.values())
        
        if changed_params / total_params < self.incremental_ratio:
            return changed_model, changed_optimizer
        else:
            return model_state, optimizer_state
    
    def _save_checkpoint_async(
        self,
        model: Union[DDP, FSDP, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        loss: float,
        learning_rate: float,
        checkpoint_type: CheckpointType = CheckpointType.FULL,
    ):
        """Save checkpoint asynchronously"""
        checkpoint_id = f"checkpoint_{global_step}_{int(time.time())}"
        
        try:
            # Get state dicts
            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model_state = model.state_dict()
            elif isinstance(model, DDP):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            
            optimizer_state = optimizer.state_dict()
            
            # Detect changes for incremental checkpointing
            if checkpoint_type == CheckpointType.INCREMENTAL:
                model_state, optimizer_state = self._detect_state_changes(
                    model_state, optimizer_state
                )
            
            # Compute hashes
            model_hash = self._compute_state_hash(model_state)
            optimizer_hash = self._compute_state_hash(optimizer_state)
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model state
            model_path = checkpoint_path / "model.pt"
            torch.save(model_state, model_path)
            
            # Save optimizer state
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(optimizer_state, optimizer_path)
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=time.time(),
                epoch=epoch,
                global_step=global_step,
                loss=loss,
                learning_rate=learning_rate,
                model_hash=model_hash,
                optimizer_hash=optimizer_hash,
                checkpoint_type=checkpoint_type,
                parent_checkpoint=self.current_checkpoint,
                delta_size=model_path.stat().st_size + optimizer_path.stat().st_size,
                total_size=self._get_checkpoint_size(checkpoint_path),
            )
            
            # Update tracking
            self.checkpoints[checkpoint_id] = metadata
            self.current_checkpoint = checkpoint_id
            self.model_state_hashes[checkpoint_id] = model_hash
            self.optimizer_state_hashes[checkpoint_id] = optimizer_hash
            
            if checkpoint_type == CheckpointType.FULL:
                self.last_full_checkpoint = checkpoint_id
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save metadata
            self._save_checkpoint_metadata()
            
            logger.info(f"Saved checkpoint {checkpoint_id} (type: {checkpoint_type.value})")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _get_checkpoint_size(self, checkpoint_path: Path) -> int:
        """Calculate total size of checkpoint"""
        total_size = 0
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort checkpoints by timestamp
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        # Keep only the most recent ones
        checkpoints_to_keep = sorted_checkpoints[:self.max_checkpoints]
        checkpoints_to_remove = sorted_checkpoints[self.max_checkpoints:]
        
        for checkpoint_id, _ in checkpoints_to_remove:
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            if checkpoint_path.exists():
                import shutil
                shutil.rmtree(checkpoint_path)
            del self.checkpoints[checkpoint_id]
            logger.info(f"Removed old checkpoint {checkpoint_id}")
    
    def checkpoint_worker(self):
        """Worker thread for asynchronous checkpointing"""
        while self.running:
            try:
                task = self.checkpoint_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break
                
                model, optimizer, epoch, global_step, loss, lr, cp_type = task
                self._save_checkpoint_async(
                    model, optimizer, epoch, global_step, loss, lr, cp_type
                )
                self.checkpoint_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Checkpoint worker error: {e}")
    
    def start(self):
        """Start checkpoint worker thread"""
        if self.checkpoint_thread is None or not self.checkpoint_thread.is_alive():
            self.running = True
            self.checkpoint_thread = threading.Thread(
                target=self.checkpoint_worker,
                daemon=True,
                name="CheckpointWorker"
            )
            self.checkpoint_thread.start()
            logger.info("Checkpoint worker started")
    
    def stop(self):
        """Stop checkpoint worker thread"""
        self.running = False
        if self.checkpoint_thread and self.checkpoint_thread.is_alive():
            self.checkpoint_queue.put(None)  # Poison pill
            self.checkpoint_thread.join(timeout=5.0)
            logger.info("Checkpoint worker stopped")
    
    def request_checkpoint(
        self,
        model: Union[DDP, FSDP, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        loss: float,
        learning_rate: float,
        checkpoint_type: CheckpointType = CheckpointType.FULL,
        priority: bool = False,
    ):
        """Request a checkpoint to be saved"""
        task = (model, optimizer, epoch, global_step, loss, learning_rate, checkpoint_type)
        
        if priority:
            # Clear queue and add high priority task
            while not self.checkpoint_queue.empty():
                try:
                    self.checkpoint_queue.get_nowait()
                except queue.Empty:
                    break
        
        self.checkpoint_queue.put(task)
    
    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        model: Optional[Union[DDP, FSDP, torch.nn.Module]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[int, int, float]:
        """Load a checkpoint"""
        if checkpoint_id is None:
            checkpoint_id = self.current_checkpoint
        
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        # Load model state
        model_path = checkpoint_path / "model.pt"
        if model_path.exists() and model is not None:
            model_state = torch.load(model_path, map_location="cpu")
            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model.load_state_dict(model_state)
            elif isinstance(model, DDP):
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)
        
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists() and optimizer is not None:
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            optimizer.load_state_dict(optimizer_state)
        
        metadata = self.checkpoints[checkpoint_id]
        logger.info(f"Loaded checkpoint {checkpoint_id}")
        
        return metadata.epoch, metadata.global_step, metadata.loss


class NodeHealthMonitor:
    """Monitors node health and detects failures"""
    
    def __init__(
        self,
        node_info: NodeInfo,
        heartbeat_interval: float = 5.0,
        failure_threshold: int = 3,
        check_interval: float = 2.0,
    ):
        self.node_info = node_info
        self.heartbeat_interval = heartbeat_interval
        self.failure_threshold = failure_threshold
        self.check_interval = check_interval
        
        self.nodes: Dict[str, NodeInfo] = {}
        self.heartbeat_queue = queue.Queue()
        self.failure_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        self.running = False
        self.monitor_thread = None
        self.heartbeat_thread = None
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self.stop)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down health monitor")
        self.stop()
    
    def add_node(self, node_info: NodeInfo):
        """Add a node to monitor"""
        self.nodes[node_info.node_id] = node_info
        logger.info(f"Added node {node_info.node_id} to health monitor")
    
    def remove_node(self, node_id: str):
        """Remove a node from monitoring"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed node {node_id} from health monitor")
    
    def register_failure_callback(self, callback: Callable):
        """Register callback for node failures"""
        self.failure_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable):
        """Register callback for node recovery"""
        self.recovery_callbacks.append(callback)
    
    def send_heartbeat(self):
        """Send heartbeat for this node"""
        heartbeat = {
            "node_id": self.node_info.node_id,
            "timestamp": time.time(),
            "state": self.node_info.state.value,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_utilization": self._get_gpu_utilization(),
        }
        self.heartbeat_queue.put(heartbeat)
    
    def _get_gpu_utilization(self) -> List[float]:
        """Get GPU utilization percentages"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            utilizations = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilizations.append(util.gpu)
            return utilizations
        except:
            return []
    
    def heartbeat_worker(self):
        """Worker thread for sending heartbeats"""
        while self.running:
            try:
                self.send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat worker error: {e}")
                time.sleep(1.0)
    
    def monitor_worker(self):
        """Worker thread for monitoring node health"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check each node's last heartbeat
                for node_id, node_info in list(self.nodes.items()):
                    if node_id == self.node_info.node_id:
                        continue  # Skip self
                    
                    time_since_heartbeat = current_time - node_info.last_heartbeat
                    
                    if time_since_heartbeat > self.heartbeat_interval * self.failure_threshold:
                        if node_info.state != NodeState.FAILED:
                            node_info.state = NodeState.FAILED
                            node_info.failure_count += 1
                            logger.warning(f"Node {node_id} marked as FAILED")
                            
                            # Trigger failure callbacks
                            for callback in self.failure_callbacks:
                                try:
                                    callback(node_id, node_info)
                                except Exception as e:
                                    logger.error(f"Failure callback error: {e}")
                    
                    elif time_since_heartbeat > self.heartbeat_interval * 2:
                        if node_info.state == NodeState.HEALTHY:
                            node_info.state = NodeState.DEGRADED
                            logger.warning(f"Node {node_id} marked as DEGRADED")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitor worker error: {e}")
                time.sleep(1.0)
    
    def process_heartbeats(self):
        """Process incoming heartbeats"""
        while self.running:
            try:
                heartbeat = self.heartbeat_queue.get(timeout=1.0)
                node_id = heartbeat["node_id"]
                
                if node_id in self.nodes:
                    node_info = self.nodes[node_id]
                    node_info.last_heartbeat = heartbeat["timestamp"]
                    
                    # Check for recovery
                    if node_info.state in [NodeState.FAILED, NodeState.DEGRADED]:
                        node_info.state = NodeState.HEALTHY
                        logger.info(f"Node {node_id} recovered")
                        
                        # Trigger recovery callbacks
                        for callback in self.recovery_callbacks:
                            try:
                                callback(node_id, node_info)
                            except Exception as e:
                                logger.error(f"Recovery callback error: {e}")
                
                self.heartbeat_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Heartbeat processing error: {e}")
    
    def start(self):
        """Start health monitoring"""
        if not self.running:
            self.running = True
            
            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(
                target=self.heartbeat_worker,
                daemon=True,
                name="HeartbeatWorker"
            )
            self.heartbeat_thread.start()
            
            # Start monitor thread
            self.monitor_thread = threading.Thread(
                target=self.monitor_worker,
                daemon=True,
                name="MonitorWorker"
            )
            self.monitor_thread.start()
            
            # Start heartbeat processing thread
            self.processing_thread = threading.Thread(
                target=self.process_heartbeats,
                daemon=True,
                name="HeartbeatProcessor"
            )
            self.processing_thread.start()
            
            logger.info("Health monitor started")
    
    def stop(self):
        """Stop health monitoring"""
        self.running = False
        
        threads = [
            self.heartbeat_thread,
            self.monitor_thread,
            self.processing_thread,
        ]
        
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        logger.info("Health monitor stopped")


class WorkloadRebalancer:
    """Rebalances workload when nodes fail or join"""
    
    def __init__(
        self,
        total_shards: int,
        node_capabilities: Dict[str, Dict[str, Any]],
        rebalance_strategy: str = "capacity_aware",
    ):
        self.total_shards = total_shards
        self.node_capabilities = node_capabilities
        self.rebalance_strategy = rebalance_strategy
        
        self.shard_distribution: Dict[int, List[str]] = {}
        self.node_workload: Dict[str, List[int]] = {}
        self.migration_history: List[MigrationPlan] = []
        
        self._initialize_distribution()
    
    def _initialize_distribution(self):
        """Initialize shard distribution across nodes"""
        nodes = list(self.node_capabilities.keys())
        if not nodes:
            return
        
        # Simple round-robin distribution
        for shard_id in range(self.total_shards):
            node_id = nodes[shard_id % len(nodes)]
            if shard_id not in self.shard_distribution:
                self.shard_distribution[shard_id] = []
            self.shard_distribution[shard_id].append(node_id)
            
            if node_id not in self.node_workload:
                self.node_workload[node_id] = []
            self.node_workload[node_id].append(shard_id)
    
    def _calculate_node_capacity(self, node_id: str) -> float:
        """Calculate node capacity score"""
        if node_id not in self.node_capabilities:
            return 0.0
        
        caps = self.node_capabilities[node_id]
        
        # Weighted capacity calculation
        gpu_score = caps.get("gpu_count", 0) * 10
        memory_score = caps.get("total_memory", 0) / (1024**3)  # GB
        cpu_score = caps.get("cpu_cores", 0)
        
        return gpu_score * 0.5 + memory_score * 0.3 + cpu_score * 0.2
    
    def _get_available_nodes(self, exclude_nodes: List[str] = None) -> List[str]:
        """Get list of available nodes"""
        exclude_nodes = exclude_nodes or []
        return [
            node_id for node_id in self.node_capabilities.keys()
            if node_id not in exclude_nodes
        ]
    
    def create_migration_plan(
        self,
        failed_nodes: List[str],
        new_nodes: List[str] = None,
    ) -> List[MigrationPlan]:
        """Create migration plan for failed nodes"""
        migration_plans = []
        new_nodes = new_nodes or []
        
        for failed_node in failed_nodes:
            if failed_node not in self.node_workload:
                continue
            
            # Get shards assigned to failed node
            shards_to_migrate = self.node_workload[failed_node].copy()
            
            if not shards_to_migrate:
                continue
            
            # Find target nodes
            available_nodes = self._get_available_nodes(
                exclude_nodes=[failed_node] + new_nodes
            )
            
            if not available_nodes:
                logger.error(f"No available nodes to migrate workload from {failed_node}")
                continue
            
            # Sort nodes by capacity
            available_nodes.sort(
                key=lambda x: self._calculate_node_capacity(x),
                reverse=True
            )
            
            # Distribute shards to target nodes
            target_nodes = []
            for i, shard_id in enumerate(shards_to_migrate):
                target_node = available_nodes[i % len(available_nodes)]
                target_nodes.append(target_node)
                
                # Create migration plan
                plan = MigrationPlan(
                    source_node=failed_node,
                    target_nodes=[target_node],
                    workload_shard=shard_id,
                    migration_type="full",
                    estimated_time=self._estimate_migration_time(shard_id),
                    priority=1,
                )
                migration_plans.append(plan)
            
            logger.info(
                f"Created migration plan for {len(shards_to_migrate)} shards "
                f"from {failed_node} to {len(set(target_nodes))} nodes"
            )
        
        return migration_plans
    
    def _estimate_migration_time(self, shard_id: int) -> float:
        """Estimate migration time for a shard"""
        # Simple estimation based on shard size and network bandwidth
        # In production, use historical data and network monitoring
        return 10.0  # seconds
    
    def execute_migration_plan(
        self,
        migration_plans: List[MigrationPlan],
        checkpoint_manager: CheckpointManager,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        """Execute migration plan"""
        try:
            for plan in sorted(migration_plans, key=lambda x: x.priority, reverse=True):
                logger.info(
                    f"Executing migration: shard {plan.workload_shard} "
                    f"from {plan.source_node} to {plan.target_nodes}"
                )
                
                # Update shard distribution
                for target_node in plan.target_nodes:
                    if plan.workload_shard not in self.shard_distribution:
                        self.shard_distribution[plan.workload_shard] = []
                    
                    if target_node not in self.shard_distribution[plan.workload_shard]:
                        self.shard_distribution[plan.workload_shard].append(target_node)
                    
                    if target_node not in self.node_workload:
                        self.node_workload[target_node] = []
                    
                    if plan.workload_shard not in self.node_workload[target_node]:
                        self.node_workload[target_node].append(plan.workload_shard)
                
                # Remove from source node
                if plan.source_node in self.node_workload:
                    if plan.workload_shard in self.node_workload[plan.source_node]:
                        self.node_workload[plan.source_node].remove(plan.workload_shard)
                
                # Save checkpoint for migration
                checkpoint_manager.request_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=0,  # Will be updated
                    global_step=0,  # Will be updated
                    loss=0.0,  # Will be updated
                    learning_rate=0.0,  # Will be updated
                    checkpoint_type=CheckpointType.LIVE,
                    priority=True,
                )
                
                self.migration_history.append(plan)
                logger.info(f"Migration completed for shard {plan.workload_shard}")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration execution failed: {e}")
            return False
    
    def rebalance_workload(self, active_nodes: List[str]) -> Dict[int, List[str]]:
        """Rebalance workload across active nodes"""
        if not active_nodes:
            return {}
        
        # Calculate total capacity
        total_capacity = sum(
            self._calculate_node_capacity(node_id) for node_id in active_nodes
        )
        
        if total_capacity == 0:
            # Fallback to simple distribution
            return self._simple_rebalance(active_nodes)
        
        # Capacity-aware rebalancing
        new_distribution = {}
        node_loads = {node_id: 0.0 for node_id in active_nodes}
        
        for shard_id in range(self.total_shards):
            # Find node with least load relative to its capacity
            best_node = None
            best_score = float('inf')
            
            for node_id in active_nodes:
                capacity = self._calculate_node_capacity(node_id)
                if capacity == 0:
                    continue
                
                load_ratio = node_loads[node_id] / capacity
                if load_ratio < best_score:
                    best_score = load_ratio
                    best_node = node_id
            
            if best_node is None:
                best_node = active_nodes[0]
            
            if shard_id not in new_distribution:
                new_distribution[shard_id] = []
            new_distribution[shard_id].append(best_node)
            node_loads[best_node] += 1.0
        
        return new_distribution
    
    def _simple_rebalance(self, active_nodes: List[str]) -> Dict[int, List[str]]:
        """Simple round-robin rebalancing"""
        new_distribution = {}
        
        for shard_id in range(self.total_shards):
            node_id = active_nodes[shard_id % len(active_nodes)]
            if shard_id not in new_distribution:
                new_distribution[shard_id] = []
            new_distribution[shard_id].append(node_id)
        
        return new_distribution
    
    def get_node_shards(self, node_id: str) -> List[int]:
        """Get shards assigned to a node"""
        return self.node_workload.get(node_id, [])
    
    def get_shard_nodes(self, shard_id: int) -> List[str]:
        """Get nodes assigned to a shard"""
        return self.shard_distribution.get(shard_id, [])


class FaultTolerantTrainer:
    """Main fault-tolerant trainer with live migration support"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        checkpoint_dir: str,
        node_id: Optional[str] = None,
        world_size: int = 1,
        rank: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.config = config or {}
        
        # Initialize node info
        self.node_id = node_id or self._generate_node_id()
        self.world_size = world_size
        self.rank = rank
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=self.config.get("max_checkpoints", 5),
            checkpoint_interval=self.config.get("checkpoint_interval", 1000),
        )
        
        self.node_info = self._create_node_info()
        self.health_monitor = NodeHealthMonitor(
            node_info=self.node_info,
            heartbeat_interval=self.config.get("heartbeat_interval", 5.0),
            failure_threshold=self.config.get("failure_threshold", 3),
        )
        
        self.workload_rebalancer = WorkloadRebalancer(
            total_shards=len(train_dataloader.dataset),
            node_capabilities=self._gather_node_capabilities(),
            rebalance_strategy=self.config.get("rebalance_strategy", "capacity_aware"),
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_start_time = None
        
        # Recovery state
        self.recovery_mode = False
        self.recovery_checkpoint = None
        self.migration_in_progress = False
        
        # Register callbacks
        self.health_monitor.register_failure_callback(self._on_node_failure)
        self.health_monitor.register_recovery_callback(self._on_node_recovery)
        
        # Register signal handlers
        signal.signal(signal.SIGUSR1, self._handle_migration_signal)
        signal.signal(signal.SIGUSR2, self._handle_recovery_signal)
        
        # Start monitoring
        self.health_monitor.start()
        
        logger.info(f"FaultTolerantTrainer initialized for node {self.node_id}")
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        hostname = socket.gethostname()
        pid = os.getpid()
        timestamp = int(time.time())
        return f"{hostname}_{pid}_{timestamp}"
    
    def _create_node_info(self) -> NodeInfo:
        """Create node information"""
        import GPUtil
        
        # Get GPU information
        try:
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            gpu_memory = [gpu.memoryTotal for gpu in gpus]
        except:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            gpu_memory = []
            for i in range(gpu_count):
                gpu_memory.append(torch.cuda.get_device_properties(i).total_memory)
        
        return NodeInfo(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=socket.gethostbyname(socket.gethostname()),
            rank=self.rank,
            world_size=self.world_size,
            gpu_count=gpu_count,
            gpu_memory=gpu_memory,
            cpu_cores=psutil.cpu_count(),
            total_memory=psutil.virtual_memory().total,
            last_heartbeat=time.time(),
        )
    
    def _gather_node_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Gather capabilities of all nodes"""
        # In production, this would gather info from all nodes via distributed communication
        # For now, return local node capabilities
        capabilities = {}
        
        capabilities[self.node_id] = {
            "gpu_count": self.node_info.gpu_count,
            "gpu_memory": self.node_info.gpu_memory,
            "cpu_cores": self.node_info.cpu_cores,
            "total_memory": self.node_info.total_memory,
            "rank": self.rank,
        }
        
        return capabilities
    
    def _on_node_failure(self, failed_node_id: str, node_info: NodeInfo):
        """Handle node failure"""
        logger.warning(f"Node failure detected: {failed_node_id}")
        
        if self.migration_in_progress:
            logger.info("Migration already in progress, skipping")
            return
        
        self.migration_in_progress = True
        
        try:
            # Create migration plan
            migration_plans = self.workload_rebalancer.create_migration_plan(
                failed_nodes=[failed_node_id]
            )
            
            if not migration_plans:
                logger.error("No migration plans created")
                return
            
            # Execute migration
            success = self.workload_rebalancer.execute_migration_plan(
                migration_plans=migration_plans,
                checkpoint_manager=self.checkpoint_manager,
                model=self.model,
                optimizer=self.optimizer,
            )
            
            if success:
                logger.info(f"Successfully migrated workload from {failed_node_id}")
            else:
                logger.error(f"Failed to migrate workload from {failed_node_id}")
        
        except Exception as e:
            logger.error(f"Error handling node failure: {e}")
        
        finally:
            self.migration_in_progress = False
    
    def _on_node_recovery(self, recovered_node_id: str, node_info: NodeInfo):
        """Handle node recovery"""
        logger.info(f"Node recovered: {recovered_node_id}")
        
        # Rebalance workload to include recovered node
        active_nodes = [
            nid for nid, ninfo in self.health_monitor.nodes.items()
            if ninfo.state != NodeState.FAILED
        ]
        
        new_distribution = self.workload_rebalancer.rebalance_workload(active_nodes)
        logger.info(f"Workload rebalanced after recovery of {recovered_node_id}")
    
    def _handle_migration_signal(self, signum, frame):
        """Handle migration signal"""
        logger.info("Received migration signal")
        # Trigger graceful migration
        self._graceful_migration()
    
    def _handle_recovery_signal(self, signum, frame):
        """Handle recovery signal"""
        logger.info("Received recovery signal")
        # Trigger recovery process
        self._recover_from_failure()
    
    def _graceful_migration(self):
        """Perform graceful migration"""
        logger.info("Starting graceful migration")
        
        # Save current state
        self.checkpoint_manager.request_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            global_step=self.global_step,
            loss=self.best_loss,
            learning_rate=self._get_current_lr(),
            checkpoint_type=CheckpointType.LIVE,
            priority=True,
        )
        
        # Wait for checkpoint to complete
        time.sleep(2.0)
        
        logger.info("Graceful migration checkpoint saved")
    
    def _recover_from_failure(self):
        """Recover from failure"""
        logger.info("Starting recovery from failure")
        
        self.recovery_mode = True
        
        try:
            # Load latest checkpoint
            epoch, global_step, loss = self.checkpoint_manager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
            )
            
            self.current_epoch = epoch
            self.global_step = global_step
            self.best_loss = loss
            
            logger.info(f"Recovered to epoch {epoch}, step {global_step}, loss {loss}")
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            # Try to load from backup or start fresh
            self._emergency_recovery()
        
        finally:
            self.recovery_mode = False
    
    def _emergency_recovery(self):
        """Emergency recovery when normal recovery fails"""
        logger.warning("Starting emergency recovery")
        
        # Try to find any available checkpoint
        checkpoint_dir = Path(self.checkpoint_manager.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_*"))
        
        if checkpoints:
            # Load the most recent checkpoint
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            checkpoint_id = latest_checkpoint.name
            
            try:
                epoch, global_step, loss = self.checkpoint_manager.load_checkpoint(
                    checkpoint_id=checkpoint_id,
                    model=self.model,
                    optimizer=self.optimizer,
                )
                
                self.current_epoch = epoch
                self.global_step = global_step
                self.best_loss = loss
                
                logger.info(f"Emergency recovery successful from {checkpoint_id}")
                
            except Exception as e:
                logger.error(f"Emergency recovery failed: {e}")
                # Reset to initial state
                self.current_epoch = 0
                self.global_step = 0
                self.best_loss = float('inf')
        else:
            logger.error("No checkpoints available for recovery")
    
    def _get_current_lr(self) -> float:
        """Get current learning rate"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with fault tolerance"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.training_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Check for recovery mode
                if self.recovery_mode:
                    logger.info("In recovery mode, pausing training")
                    time.sleep(1.0)
                    continue
                
                # Training step
                loss = self._training_step(batch)
                total_loss += loss
                num_batches += 1
                self.global_step += 1
                
                # Periodic checkpointing
                if self.global_step % self.checkpoint_manager.checkpoint_interval == 0:
                    self.checkpoint_manager.request_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        global_step=self.global_step,
                        loss=loss,
                        learning_rate=self._get_current_lr(),
                        checkpoint_type=CheckpointType.INCREMENTAL,
                    )
                
                # Log progress
                if batch_idx % 100 == 0:
                    avg_loss = total_loss / num_batches
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}, "
                        f"Step: {self.global_step}"
                    )
                
                # Update best loss
                if loss < self.best_loss:
                    self.best_loss = loss
                
            except Exception as e:
                logger.error(f"Training step failed: {e}")
                # Attempt recovery
                self._recover_from_failure()
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # End of epoch checkpoint
        self.checkpoint_manager.request_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            loss=avg_loss,
            learning_rate=self._get_current_lr(),
            checkpoint_type=CheckpointType.FULL,
        )
        
        return avg_loss
    
    def _training_step(self, batch: Any) -> float:
        """Perform a single training step"""
        # This should be implemented based on the specific model and task
        # For now, return a dummy loss
        return 0.0
    
    def train(self, num_epochs: int):
        """Main training loop with fault tolerance"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            try:
                self.current_epoch = epoch
                avg_loss = self.train_epoch(epoch)
                
                logger.info(
                    f"Epoch {epoch} completed. "
                    f"Average loss: {avg_loss:.4f}, "
                    f"Best loss: {self.best_loss:.4f}"
                )
                
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            
            except Exception as e:
                logger.error(f"Epoch {epoch} failed: {e}")
                self._recover_from_failure()
                continue
        
        logger.info("Training completed")
    
    def shutdown(self):
        """Shutdown the trainer"""
        logger.info("Shutting down FaultTolerantTrainer")
        
        # Stop health monitor
        self.health_monitor.stop()
        
        # Stop checkpoint manager
        self.checkpoint_manager.stop()
        
        # Save final checkpoint
        self.checkpoint_manager.request_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            global_step=self.global_step,
            loss=self.best_loss,
            learning_rate=self._get_current_lr(),
            checkpoint_type=CheckpointType.FULL,
            priority=True,
        )
        
        logger.info("Shutdown complete")


# Utility functions for integration with existing codebase
def create_fault_tolerant_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    checkpoint_dir: str,
    **kwargs,
) -> FaultTolerantTrainer:
    """Factory function to create FaultTolerantTrainer"""
    return FaultTolerantTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )


def setup_fault_tolerance(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    checkpoint_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> FaultTolerantTrainer:
    """Setup fault tolerance for training"""
    config = config or {}
    
    # Get distributed info if available
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    
    trainer = FaultTolerantTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        checkpoint_dir=checkpoint_dir,
        world_size=world_size,
        rank=rank,
        config=config,
    )
    
    return trainer


# Example usage
if __name__ == "__main__":
    # This would be integrated with the existing training script
    pass