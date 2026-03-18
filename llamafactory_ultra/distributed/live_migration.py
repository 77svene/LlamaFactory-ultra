import logging
import os
import time
import multiprocessing
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.distributed import ProcessGroup
from torch.distributed._functional_options import _get_process_group_backend
import torch.multiprocessing as mp

from llamafactory_ultra.config import DataConfig, ModelConfig, TrainArg, TrainStrategy

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status enumeration for distributed training nodes."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MIGRATING = "migrating"
    RECOVERED = "recovered"


@dataclass
class MigrationConfig:
    """Configuration for live migration and fault tolerance."""
    enable_live_migration: bool = True
    checkpoint_interval: int = 1000  # steps
    max_recovery_time: float = 60.0  # seconds
    health_check_interval: float = 10.0
    migration_timeout: float = 300.0
    retry_count: int = 3
    backup_path: str = "/tmp/llamafactory_ultra/checkpoints/"
    rpc_backend: str = "gloo"
    rpc_timeout: float = 600.0
    rebalance_strategy: str = "load_aware"
    allow_cross_node_migration: bool = True
    sync_optimizer_state: bool = True
    sync_gradient_state: bool = True
    sync_rng_state: bool = True


@dataclass
class NodeHealth:
    """Represents the health status of a single training node."""
    node_id: str
    rank: int
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: float = 0.0
    load: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0


class HealthMonitor:
    """
    Monitors the health of nodes in the distributed cluster.
    Detects failures and triggers migration workflows.
    """

    def __init__(self, health_check_interval: float, node_id: str):
        self.interval = health_check_interval
        self.node_id = node_id
        self.nodes: Dict[str, NodeHealth] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[NodeStatus], None]] = []
        self._running = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)

    def register_callback(self, callback: Callable[[NodeStatus], None]) -> None:
        """Register a callback to be invoked on status change."""
        self._callbacks.append(callback)

    def update_node_status(self, node_id: str, status: NodeStatus, load: float = 0.0) -> None:
        """Update the status of a specific node."""
        with self._lock:
            if node_id not in self.nodes:
                self.nodes[node_id] = NodeHealth(node_id=node_id, rank=-1)
            
            node = self.nodes[node_id]
            node.status = status
            node.load = load
            node.last_heartbeat = time.time()

            if node.status == NodeStatus.FAILED:
                for cb in self._callbacks:
                    cb(NodeStatus.FAILED)

    def check_health(self) -> None:
        """Perform health check on all tracked nodes."""
        with self._lock:
            for node_id, node in self.nodes.items():
                if time.time() - node.last_heartbeat > self.interval * 3:
                    node.status = NodeStatus.FAILED
                    logger.warning(f"Node {node_id} failed to heartbeat. Marked as FAILED.")
                    # Notify rebalancer
                    if hasattr(self, 'rebalancer'):
                        self.rebalancer.on_node_failure(node_id)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            self.check_health()
            time.sleep(self.interval)

    def start(self) -> None:
        """Start the monitoring thread."""
        self._running = True
        self._thread.start()
        logger.info(f"Health monitor for {self.node_id} started.")

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._running = False
        self._thread.join()

    def get_failed_nodes(self) -> List[str]:
        """Get list of failed node IDs."""
        with self._lock:
            return [nid for nid, n in self.nodes.items() if n.status == NodeStatus.FAILED]


class CheckpointMigrator:
    """
    Handles asynchronous checkpoint migration across heterogeneous clusters.
    Optimized for <1 minute recovery time.
    """

    def __init__(
        self,
        backup_path: str,
        sync_optimizer_state: bool = True,
        sync_gradient_state: bool = True,
        sync_rng_state: bool = True,
        rpc_backend: str = "gloo",
    ):
        self.backup_path = backup_path
        self.sync_optimizer = sync_optimizer_state
        self.sync_gradient = sync_gradient_state
        self.sync_rng = sync_rng_state
        self.backend = rpc_backend
        self._lock = threading.Lock()
        self._migration_queue: List[Tuple[str, Dict[str, Any]]] = []

    def _serialize_state(self, state_dict: Dict[str, Any]) -> bytes:
        """Serialize state dict to bytes for efficient transfer."""
        # Use torch.save with custom compression if available
        # For production, ensure we don't hold large tensors in memory unnecessarily
        import pickle
        return pickle.dumps(state_dict)

    def migrate_state_dict(
        self,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Optional[Dict[str, Any]] = None,
        target_node: str = None,
    ) -> bool:
        """
        Migrate state dict to a target node or backup location.
        Returns True if migration succeeded.
        """
        try:
            # In a real scenario, this would use torch.distributed.rpc or a custom RPC
            # For this implementation, we assume a logical migration path
            with self._lock:
                # Simulate incremental transfer
                logger.info(f"Migrating state to {target_node or 'backup_path'}")
                
                # Serialize model
                model_bytes = torch.save(model_state, self.backup_path + "model.pt")
                
                # Serialize optimizer if requested
                if self.sync_optimizer and optimizer_state:
                    optimizer_bytes = torch.save(optimizer_state, self.backup_path + "optimizer.pt")
                else:
                    optimizer_bytes = None

                # Serialize gradients if requested (requires gradient accumulation buffer)
                if self.sync_gradient:
                    grad_bytes = torch.save(model_state, self.backup_path + "gradients.pt")
                else:
                    grad_bytes = None

                # Simulate transfer time (keep it fast for <1 min recovery)
                time.sleep(1)  # Placeholder for actual network transfer
                
                return True
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def load_state_dict(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load state dict from a specific node's backup."""
        path = os.path.join(self.backup_path, f"{node_id}.pt")
        if os.path.exists(path):
            return torch.load(path)
        return None


class WorkloadRebalancer:
    """
    Automatically redistributes workload when nodes fail.
    Supports preemptible instances with dynamic reassignment.
    """

    def __init__(
        self,
        rebalance_strategy: str = "load_aware",
        allow_cross_node_migration: bool = True,
    ):
        self.strategy = rebalance_strategy
        self.allow_cross_node = allow_cross_node_migration
        self._available_nodes: Set[str] = set()
        self._failed_nodes: Set[str] = set()

    def on_node_failure(self, failed_node: str) -> None:
        """Handle the logic when a node fails."""
        self._failed_nodes.add(failed_node)
        self._available_nodes.discard(failed_node)
        logger.warning(f"Node {failed_node} failed. Initiating rebalancing.")

    def rebalance_workload(self, available_nodes: List[str], failed_node: str) -> Dict[str, int]:
        """
        Calculate new rank assignments for surviving nodes.
        Returns a mapping of node -> new rank count.
        """
        total