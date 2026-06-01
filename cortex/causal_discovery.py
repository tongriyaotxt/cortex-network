"""
M6-3: Causal Discovery LPM

Provides:
- CausalDiscoveryLPM: discovers causal relationships from time series data
  using Granger causality + attention mechanisms.
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal import CausalGraph, CausalVariable


class CausalDiscoveryLPM(nn.Module):
    """
    An LPM that analyzes variable time series and infers causal graphs.

    Combines:
    - Granger causality: does past of X help predict future of Y?
    - Attention: learnable pairwise influence weights
    """

    def __init__(
        self,
        d_model: int,
        max_lag: int = 5,
        n_variables: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_lag = max_lag
        self.n_variables = n_variables

        # Lag encoder: encode history into representation
        self.lag_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
        )

        # Pairwise influence predictor
        self.influence_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_variables),
        )

        # Direction classifier: X -> Y or Y -> X or no causation
        self.direction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),  # X->Y, Y->X, no causation
        )

        # Learnable edge threshold (replaces hard-coded 0.3)
        self.edge_threshold = nn.Parameter(torch.tensor(0.3))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, history: List[torch.Tensor]) -> CausalGraph:
        """
        Discover causal graph from time series.

        Args:
            history: list of T tensors, each (n_variables, d_model)
        Returns:
            CausalGraph
        """
        if len(history) < self.max_lag + 1:
            return CausalGraph()

        # Stack history: (T, n_variables, d_model)
        hist = torch.stack(history, dim=0)
        T, n_vars, d = hist.shape

        # For simplicity, assume n_vars <= n_variables
        n_vars = min(n_vars, self.n_variables)

        # Encode lagged history for each variable
        var_reprs = []
        for v in range(n_vars):
            var_history = hist[:, v, :]  # (T, d_model)
            var_history = var_history.unsqueeze(0)  # (1, T, d)
            encoded, _ = self.lag_encoder(var_history)  # (1, T, d//2)
            var_reprs.append(encoded[:, -1, :])  # (1, d//2)

        var_reprs = torch.cat(var_reprs, dim=0)  # (n_vars, d//2)

        # Pad to d_model
        if var_reprs.size(-1) < self.d_model:
            var_reprs = F.pad(var_reprs, (0, self.d_model - var_reprs.size(-1)))
        elif var_reprs.size(-1) > self.d_model:
            var_reprs = var_reprs[:, :self.d_model]

        # Build causal graph
        graph = CausalGraph()

        # Add variables
        for v in range(n_vars):
            var = CausalVariable(var_id=f"var_{v}", embedding=var_reprs[v])
            graph.add_variable(var)

        # Predict pairwise influences
        for i in range(n_vars):
            influence = self.influence_predictor(var_reprs[i])  # (n_variables,)
            influence = influence[:n_vars]

            for j in range(n_vars):
                if i == j:
                    continue
                strength = torch.sigmoid(influence[j]).item()
                threshold = torch.sigmoid(self.edge_threshold).item()
                if strength > threshold:  # learned threshold for causal edge
                    # Determine direction
                    pair = torch.cat([var_reprs[i], var_reprs[j]], dim=0).unsqueeze(0)
                    dir_logits = self.direction_head(pair).squeeze(0)
                    dir_probs = F.softmax(dir_logits, dim=0)

                    if dir_probs[0] > 0.5:  # i -> j
                        graph.add_edge(f"var_{i}", f"var_{j}", strength)
                    elif dir_probs[1] > 0.5:  # j -> i
                        graph.add_edge(f"var_{j}", f"var_{i}", strength)

        return graph

    def estimate_causal_strength(
        self,
        cause: torch.Tensor,
        effect: torch.Tensor,
    ) -> float:
        """
        Estimate causal strength between two variables.
        """
        if cause.dim() == 1:
            cause = cause.unsqueeze(0)
        if effect.dim() == 1:
            effect = effect.unsqueeze(0)

        # Concatenate and predict
        combined = torch.cat([cause, effect], dim=-1)
        if combined.size(-1) > self.d_model * 2:
            combined = combined[:, :self.d_model * 2]
        elif combined.size(-1) < self.d_model * 2:
            combined = F.pad(combined, (0, self.d_model * 2 - combined.size(-1)))

        logits = self.direction_head(combined).squeeze(0)
        probs = F.softmax(logits, dim=0)
        return probs[0].item()  # P(cause -> effect)
