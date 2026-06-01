"""
M3-1: Action LPM and Action Head

Provides:
- ActionLPM: a Local Processing Module that translates GNW intention
  into concrete action distributions.
- ActionHead: the output head attached to CORTEXModel for action generation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import ActionDistribution, ActionSpace


class ActionLPM(nn.Module):
    """
    An LPM that receives the broadcasted intention from GNW and produces
    an ActionDistribution (discrete logits + continuous params + saliency).

    This module competes in GNW like any other LPM: when the model decides
    to "act", this LPM wins the competition and emits actions.
    """

    def __init__(self, d_model: int, action_space: ActionSpace):
        super().__init__()
        self.d_model = d_model
        self.action_space = action_space

        # Intention processor
        self.intention_processor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Discrete action head
        if action_space.n_actions > 0:
            self.discrete_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, action_space.n_actions),
            )
        else:
            self.discrete_head = None

        # Continuous parameter head
        if action_space.n_continuous > 0:
            self.continuous_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, action_space.n_continuous),
            )
        else:
            self.continuous_head = None

        # Saliency: how urgent is this action?
        self.saliency_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, intention: torch.Tensor) -> Tuple[ActionDistribution, torch.Tensor]:
        """
        Args:
            intention: (batch, seq_len, d_model) or (batch, d_model)
        Returns:
            action_dist: ActionDistribution
            saliency:    (batch,) or scalar
        """
        orig_shape = intention.shape
        if intention.dim() == 3:
            batch, seq_len, d = intention.shape
            x = intention.mean(dim=1)  # pool over sequence
        else:
            batch, d = intention.shape
            x = intention

        h = self.intention_processor(x)

        # Discrete logits
        discrete_logits = None
        if self.discrete_head is not None:
            discrete_logits = self.discrete_head(h)

        # Continuous params
        continuous_params = None
        if self.continuous_head is not None:
            continuous_params = self.continuous_head(h)

        # Saliency
        saliency = self.saliency_head(h).squeeze(-1)

        action_dist = ActionDistribution(
            discrete_logits=discrete_logits,
            continuous_params=continuous_params,
            saliency=saliency.mean().item(),
        )

        return action_dist, saliency


class ActionHead(nn.Module):
    """
    The top-level action head attached to CORTEXModel.
    Similar to output_head but for actions.
    """

    def __init__(self, d_model: int, action_space: ActionSpace, dropout: float = 0.1):
        super().__init__()
        self.action_space = action_space

        # Share the same structure as ActionLPM but as a standalone head
        self.action_lpm = ActionLPM(d_model, action_space)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> ActionDistribution:
        """
        Args:
            h: (batch, seq_len, d_model) final hidden state
        Returns:
            ActionDistribution
        """
        h = self.dropout(h)
        action_dist, _ = self.action_lpm(h)
        return action_dist
