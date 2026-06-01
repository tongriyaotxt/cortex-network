"""
M5-1: Branch Isolation and Allocation

Provides:
- BranchMask: marks which DCU branches are active for a given task.
- BranchAllocator: manages branch assignment across tasks.
- ElasticPlasticity: dynamic per-branch learning rate modulation.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import BranchMask, UsageStats


class BranchAllocator:
    """
    Allocates subsets of dendritic branches to different tasks.
    
    Strategy:
    - Track usage frequency of each branch
    - When a new task arrives, allocate the least-used branches
    - Soft release: don't freeze, just lower learning rate
    """

    def __init__(self, n_branches: int, n_tasks_max: int = 100):
        self.n_branches = n_branches
        self.n_tasks_max = n_tasks_max
        self.masks: Dict[str, BranchMask] = {}
        self._usage_counts = torch.zeros(n_branches)
        self._task_count = 0

    def allocate(
        self,
        task_id: str,
        required_branches: int = 2,
    ) -> BranchMask:
        """
        Allocate branches for a new task.
        Returns a BranchMask with `required_branches` ones.
        """
        if task_id in self.masks:
            return self.masks[task_id]

        # Select least-used branches
        sorted_indices = torch.argsort(self._usage_counts)
        selected = sorted_indices[:required_branches]

        mask = torch.zeros(self.n_branches)
        mask[selected] = 1.0

        # Mark these branches as used
        self._usage_counts[selected] += 1.0

        branch_mask = BranchMask(
            task_id=task_id,
            mask=mask,
            usage_count=0,
        )
        self.masks[task_id] = branch_mask
        self._task_count += 1
        return branch_mask

    def release(self, task_id: str):
        """Soft-release: track but don't delete."""
        if task_id in self.masks:
            mask = self.masks[task_id].mask
            self._usage_counts -= mask
            self._usage_counts.clamp_(min=0)

    def update_usage(self, task_id: str, performance: float):
        """Record task performance."""
        if task_id in self.masks:
            self.masks[task_id].performance_history.append(performance)
            self.masks[task_id].usage_count += 1

    def get_mask(self, task_id: str) -> Optional[BranchMask]:
        return self.masks.get(task_id)

    def get_all_masks(self) -> List[BranchMask]:
        return list(self.masks.values())


class ElasticPlasticity(nn.Module):
    """
    Dynamically adjusts per-branch plasticity (effective learning rate).
    
    Plasticity rules:
    - High-usage branch -> low plasticity (protect old knowledge)
    - Low-usage branch -> high plasticity (learn new knowledge)
    - Performance drop -> temporarily raise plasticity (repair)
    - Consolidation signal -> permanently lower plasticity (lock in)
    """

    def __init__(self, n_branches: int):
        super().__init__()
        self.n_branches = n_branches

        # Plasticity coefficients [0, 1]
        self.register_buffer('plasticity', torch.ones(n_branches))
        self.register_buffer('usage_counts', torch.zeros(n_branches))
        self.register_buffer('performance_history', torch.zeros(n_branches, 10))

        # Plasticity predictor
        self.predictor = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def compute_plasticity(
        self,
        branch_id: int,
        usage_stats: UsageStats,
    ) -> float:
        """
        Compute plasticity for a specific branch.
        Returns value in [0, 1].
        """
        features = torch.tensor([
            1.0 - usage_stats.activation_frequency,  # less used = more plastic
            1.0 / (1.0 + usage_stats.recency * 0.01),  # recent use = less plastic
            -usage_stats.performance_trend,  # declining performance = more plastic
        ], dtype=torch.float32)

        plasticity = self.predictor(features).item()
        return plasticity

    def update(self, branch_id: int, delta: float):
        """
        Update plasticity directly (e.g., from consolidation signal).
        delta > 0: increase plasticity
        delta < 0: decrease plasticity
        """
        if 0 <= branch_id < self.n_branches:
            self.plasticity[branch_id] = torch.clamp(
                self.plasticity[branch_id] + delta,
                0.01, 1.0
            )

    def consolidate(self, branch_ids: List[int]):
        """Lower plasticity for specified branches (memory consolidation)."""
        for bid in branch_ids:
            self.update(bid, -0.3)

    def get_mask(self) -> torch.Tensor:
        """Return current plasticity mask."""
        return self.plasticity

    def apply_to_gradient(self, grad: torch.Tensor, branch_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply plasticity mask to gradient.
        Args:
            grad: (n_branches, ...) gradient tensor
            branch_mask: (n_branches,) which branches to update
        Returns:
            masked gradient
        """
        plasticity = self.plasticity * branch_mask
        # Expand plasticity to match grad shape
        while plasticity.dim() < grad.dim():
            plasticity = plasticity.unsqueeze(-1)
        return grad * plasticity
