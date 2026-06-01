"""
M4-3: Subroutine LPM

Provides:
- SubroutineLPM: a reusable skill module that can be called like a function.
  Each subroutine has its own local workspace and cost estimator.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import WorkspaceContext


class SubroutineLPM(nn.Module):
    """
    A callable skill module.

    Example subroutines:
    - Sort: sort a list of embeddings
    - Search: find matching items
    - Compare: evaluate similarity
    - Summarize: compress information

    Each subroutine owns:
    - A skill embedding (identifies what it does)
    - A processing network
    - A cost estimator
    """

    def __init__(
        self,
        d_model: int,
        subroutine_name: str,
        skill_embedding: Optional[torch.Tensor] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.subroutine_name = subroutine_name
        self.hidden_dim = hidden_dim or d_model

        # Skill embedding (learnable or fixed)
        if skill_embedding is not None:
            self.register_buffer('_skill_emb', skill_embedding)
        else:
            self.skill_embedding = nn.Parameter(torch.randn(d_model) * 0.02)

        # Processing network
        self.processor = nn.Sequential(
            nn.Linear(d_model * 2, self.hidden_dim),  # input + skill
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, d_model),
        )

        # Cost estimator
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def skill_emb(self) -> torch.Tensor:
        if hasattr(self, '_skill_emb'):
            return self._skill_emb
        return self.skill_embedding

    def forward(
        self,
        input_state: torch.Tensor,
        context: Optional[WorkspaceContext] = None,
    ) -> torch.Tensor:
        """
        Execute the subroutine.
        Args:
            input_state: (batch, d_model) or (d_model,)
            context: optional workspace context
        Returns:
            result: same shape as input_state
        """
        if input_state.dim() == 1:
            input_state = input_state.unsqueeze(0)

        batch = input_state.size(0)
        skill = self.skill_emb.unsqueeze(0).expand(batch, -1)

        combined = torch.cat([input_state, skill], dim=-1)
        result = self.processor(combined)

        return result

    def estimate_cost(self, input_state: torch.Tensor) -> float:
        """
        Estimate the computational cost of executing this subroutine.
        Used by the workspace for cost-benefit competition.
        """
        if input_state.dim() == 1:
            input_state = input_state.unsqueeze(0)

        cost = self.cost_head(input_state).mean().item()
        return cost

    def get_saliency(self, input_state: torch.Tensor) -> torch.Tensor:
        """
        How relevant is this subroutine for the current input?
        Computed as cosine similarity between input and skill embedding.
        """
        if input_state.dim() == 1:
            input_state = input_state.unsqueeze(0)

        # Cosine similarity
        skill = self.skill_emb.unsqueeze(0).expand(input_state.size(0), -1)
        sim = F.cosine_similarity(input_state, skill, dim=-1)
        return sim
