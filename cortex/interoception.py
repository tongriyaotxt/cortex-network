"""
M3-3: Interoception Channel

Converts internal system state into interoceptive signals (hunger, pain,
confusion, fatigue) that feed into the SelfModule and GNW.
"""

import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import SystemState


class InteroceptionChannel(nn.Module):
    """
    Encodes system state into an interoceptive vector.

    Interoceptive signals:
    - prediction_error_stress: cumulative prediction error -> "confusion / anxiety"
    - energy_fatigue: computation cost -> "fatigue"
    - reward_pleasure: external reward -> "pleasure / pain"
    - goal_progress: progress toward goal -> "satisfaction / frustration"
    - memory_pressure: memory usage -> "mental clutter"
    """

    def __init__(self, d_intero: int = 64):
        super().__init__()
        self.d_intero = d_intero

        # Scalar -> intero vector mapping
        self.encoder = nn.Sequential(
            nn.Linear(5, d_intero // 2),
            nn.SiLU(),
            nn.Linear(d_intero // 2, d_intero),
        )

        # Learnable baseline (homeostatic setpoint)
        self.baseline = nn.Parameter(torch.zeros(d_intero))

    def compute_signals(self, system_state: SystemState) -> torch.Tensor:
        """
        Args:
            system_state: see agi_protocol.SystemState
        Returns:
            intero: (d_intero,) or (batch, d_intero)
        """
        # Aggregate scalar signals
        pred_error = sum(system_state.layer_errors.values()) / max(len(system_state.layer_errors), 1)
        spike_load = sum(system_state.spike_rates.values()) / max(len(system_state.spike_rates), 1)

        scalars = torch.tensor([
            pred_error,
            spike_load,
            system_state.reward,
            system_state.energy_consumption,
            float(system_state.memory_retrieval_hits),
        ], dtype=torch.float32)

        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        intero = self.encoder(scalars) + self.baseline.unsqueeze(0)
        return intero

    def forward(self, system_state: SystemState) -> torch.Tensor:
        """Alias for compute_signals."""
        return self.compute_signals(system_state)
