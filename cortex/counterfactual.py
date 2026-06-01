"""
M6-2: Counterfactual Workspace

Provides:
- CounterfactualWorkspace: runs multiple workspace copies in parallel,
  each with different intervention settings.
- Compares factual vs counterfactual outcomes to estimate causal effects.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .workspace import GlobalWorkspaceLayer
from .agi_protocol import SymbolicToken, CausalEffect
from .causal import CausalGraph


class CounterfactualWorkspace:
    """
    Runs multiple parallel workspace instances for counterfactual reasoning.

    Architecture:
    - Base workspace: the "real" world (no interventions)
    - Counterfactual workspaces: copies with do-operator applied
    - All share base LPMs but receive different input conditions
    """

    def __init__(
        self,
        base_workspace: GlobalWorkspaceLayer,
        n_counterfactuals: int = 2,
    ):
        self.base_workspace = base_workspace
        self.n_counterfactuals = n_counterfactuals

        # Create counterfactual copies
        self.counterfactuals: List[GlobalWorkspaceLayer] = []
        for _ in range(n_counterfactuals):
            # Create a copy with same architecture
            cf = GlobalWorkspaceLayer(
                d_model=base_workspace.d_model,
                n_modules=base_workspace.n_modules,
                workspace_dim=base_workspace.workspace_dim,
            )
            # Share parameters with base (they are the same model)
            cf.load_state_dict(base_workspace.state_dict())
            self.counterfactuals.append(cf)

    def run_counterfactual(
        self,
        intervention: Dict[str, SymbolicToken],
        base_state: torch.Tensor,
        cf_index: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run a single counterfactual workspace with intervention.

        Args:
            intervention: {var_id: value} dict
            base_state: input state
            cf_index: which counterfactual copy to use
        Returns:
            (output, info) same as GlobalWorkspaceLayer
        """
        if cf_index >= len(self.counterfactuals):
            cf_index = 0

        cf_workspace = self.counterfactuals[cf_index]

        # Apply intervention to input state
        intervened_state = base_state.clone()
        
        # Real intervention: modulate state with intervention token embeddings
        for var_id, token in intervention.items():
            if token.embedding is not None:
                emb = token.embedding
                # Ensure embedding can be broadcast to state dimensions
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0).unsqueeze(0)  # (1, 1, d_embed)
                elif emb.dim() == 2:
                    emb = emb.unsqueeze(1)  # (batch, 1, d_embed)
                
                d_state = intervened_state.size(-1)
                d_emb = emb.size(-1)
                
                if d_emb < d_state:
                    emb = F.pad(emb, (0, d_state - d_emb))
                elif d_emb > d_state:
                    emb = emb[..., :d_state]
                
                # Broadcast batch dimension if needed
                if emb.size(0) < intervened_state.size(0):
                    emb = emb.expand(intervened_state.size(0), -1, -1)
                
                # Add intervention as a residual modulation
                # This simulates "setting the variable to this value"
                # by biasing the state in the direction of the token embedding
                intervened_state = intervened_state + 0.3 * emb

        # Run counterfactual workspace
        out, info = cf_workspace(intervened_state, return_consciousness=False)
        info['intervention'] = intervention
        info['counterfactual_index'] = cf_index

        return out, info

    def compare(
        self,
        factual_result: torch.Tensor,
        counterfactual_results: List[torch.Tensor],
        outcome_var: str = "",
    ) -> CausalEffect:
        """
        Compare factual vs counterfactual outcomes.

        Args:
            factual_result: output from base workspace
            counterfactual_results: list of outputs from CF workspaces
            outcome_var: name of outcome variable
        Returns:
            CausalEffect
        """
        # Compute mean difference
        factual_mean = factual_result.mean().item()
        cf_means = [r.mean().item() for r in counterfactual_results]

        if len(cf_means) >= 2:
            effect = cf_means[0] - cf_means[1]
        elif len(cf_means) == 1:
            effect = cf_means[0] - factual_mean
        else:
            effect = 0.0

        # Uncertainty as std across counterfactuals
        if len(cf_means) > 1:
            uncertainty = torch.tensor(cf_means).std().item()
        else:
            uncertainty = abs(effect) * 0.5

        return CausalEffect(
            effect_on_var=outcome_var,
            mean_delta=effect,
            uncertainty=uncertainty,
            n_samples=len(counterfactual_results),
        )

    def run_all(
        self,
        base_state: torch.Tensor,
        interventions: List[Dict[str, SymbolicToken]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, dict]], CausalEffect]:
        """
        Run base + all counterfactuals and return comparison.

        Args:
            base_state: input
            interventions: list of intervention dicts
        Returns:
            factual_result, list of (cf_result, cf_info), CausalEffect
        """
        # Factual
        factual, factual_info = self.base_workspace(base_state)

        # Counterfactuals
        cf_results = []
        for i, interv in enumerate(interventions):
            cf_out, cf_info = self.run_counterfactual(interv, base_state, cf_index=i)
            cf_results.append((cf_out, cf_info))

        # Compare
        cf_tensors = [r for r, _ in cf_results]
        effect = self.compare(factual, cf_tensors)

        return factual, cf_results, effect
