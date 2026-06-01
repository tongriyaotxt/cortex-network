"""
M5-3: Offline Consolidation (Sleep Phase)

Provides:
- OfflineConsolidation: replays important memories during "sleep"
  to strengthen them and reduce branch plasticity.
"""

from typing import List
import torch
import torch.nn as nn

from .memory_cabinet import MemoryCabinet
from .branch_isolation import ElasticPlasticity


class OfflineConsolidation:
    """
    Simulates sleep-phase memory consolidation.

    Process:
    1. Close external input
    2. Sample high-importance memories from MemoryCabinet
    3. Replay them through the model (forward pass, no external labels)
    4. Strengthen corresponding slow timescale states
    5. Lower plasticity of active branches
    """

    def __init__(
        self,
        model,
        memory_cabinet: MemoryCabinet,
        plasticity: ElasticPlasticity,
    ):
        self.model = model
        self.memory_cabinet = memory_cabinet
        self.plasticity = plasticity
        self.consolidation_history = []

    def sleep_phase(
        self,
        duration_steps: int = 100,
        replay_temperature: float = 1.0,
    ):
        """
        Run consolidation for a number of steps.

        Args:
            duration_steps: how many replay steps to run
            replay_temperature: controls randomness in memory sampling
        """
        # Collect memories to replay
        memories = self.memory_cabinet.replay(
            task_id="consolidation",
            n_samples=duration_steps,
        )

        if not memories:
            return

        for mem in memories:
            # Replay through model
            mem_batch = mem.unsqueeze(0)
            with torch.no_grad():
                _ = self.model.forward_for_consolidation(mem_batch)

            # Identify active branches
            active_branches = self._get_active_branches()

            # Lower plasticity for active branches (consolidate)
            self.plasticity.consolidate(active_branches)

            self.consolidation_history.append({
                'memory_id': len(self.consolidation_history),
                'active_branches': active_branches,
            })

    def wake_phase(self):
        """Restore normal operation."""
        pass  # Currently a no-op; future: restore input gates

    def _get_active_branches(self) -> List[int]:
        """
        Identify which branches were most active during replay.
        Simplified: return all branches that have plasticity > 0.5.
        """
        active = []
        for i, p in enumerate(self.plasticity.plasticity):
            if p > 0.5:
                active.append(i)
        return active
