"""
HebbianDCU: One-shot dendritic plasticity via Hebbian learning.

Core insight: Biological learning often happens in ONE interaction.
When Chappie watches a human fight once, he learns the moves immediately.
This is because dendritic spines can grow and strengthen within seconds
via Hebbian mechanisms:
    - Pre-before-post timing  → Long-Term Potentiation (LTP)
    - Post-before-pre timing  → Long-Term Depression (LTD)
    - Synchronous firing      → Synaptic strengthening

We implement this as a DCU wrapper that maintains a "plasticity trace"
per dendritic branch. After each forward pass, the trace is updated
based on pre- and post-synaptic activity. No backward pass needed.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import math


class HebbianTrace:
    """
    Maintain a running trace of pre- and post-synaptic activity
    for Hebbian updates.
    """
    def __init__(self, shape: Tuple[int, ...], device: str = "cpu"):
        self.shape = shape
        self.device = device
        # Eligibility trace: decaying average of co-activity
        self.eligibility = torch.zeros(shape, device=device)
        # Pre-synaptic running mean
        self.pre_mean = torch.zeros(shape[1] if len(shape) > 1 else shape[0], device=device)
        # Post-synaptic running mean
        self.post_mean = torch.zeros(shape[0], device=device)
        self.timestep = 0
        
    def update(
        self,
        pre: torch.Tensor,      # (in_dim,) or (batch, in_dim)
        post: torch.Tensor,     # (out_dim,) or (batch, out_dim)
        decay: float = 0.9,
        learning_rate: float = 0.01
    ):
        """
        Update the eligibility trace based on pre/post activity.
        
        Implements a simplified STDP (Spike-Timing Dependent Plasticity):
        - If pre and post fire together, strengthen connection
        - If only pre fires, weaken (predictive coding: no surprise)
        - If only post fires, weaken (false alarm)
        """
        # Batch average if needed
        if pre.dim() > 1:
            pre = pre.mean(dim=0)
        if post.dim() > 1:
            post = post.mean(dim=0)
        
        self.timestep += 1
        
        # Update running means
        self.pre_mean = decay * self.pre_mean + (1 - decay) * pre
        self.post_mean = decay * self.post_mean + (1 - decay) * post
        
        # Centered activities
        pre_centered = pre - self.pre_mean
        post_centered = post - self.post_mean
        
        # Outer product = co-activity matrix
        # This is the core Hebbian update: post_i * pre_j
        coactivity = torch.outer(post_centered, pre_centered)
        
        # Update eligibility trace (decaying average)
        self.eligibility = decay * self.eligibility + (1 - decay) * coactivity
        
        return self.eligibility * learning_rate


class HebbianRule:
    """
    Configurable Hebbian learning rules.
    """
    @staticmethod
    def plain_hebbian(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Classic Hebb: Δw = η * post ⊗ pre"""
        if pre.dim() > 1:
            pre = pre.mean(dim=0)
        if post.dim() > 1:
            post = post.mean(dim=0)
        return torch.outer(post, pre)
    
    @staticmethod
    def oja(pre: torch.Tensor, post: torch.Tensor, weight: torch.Tensor, lr: float = 0.01) -> torch.Tensor:
        """
        Oja's rule: Δw = η * (post ⊗ pre - post² * w)
        Prevents unbounded growth, normalizes weights naturally.
        """
        if pre.dim() > 1:
            pre = pre.mean(dim=0)
        if post.dim() > 1:
            post = post.mean(dim=0)
        hebb = torch.outer(post, pre)
        decay = torch.outer(post ** 2, weight.sum(dim=1))
        return lr * (hebb - decay)
    
    @staticmethod
    def bcm(pre: torch.Tensor, post: torch.Tensor, weight: torch.Tensor, 
            post_mean: torch.Tensor, lr: float = 0.01, theta: float = 0.5) -> torch.Tensor:
        """
        BCM rule: Δw = η * post * (post - θ) ⊗ pre
        Only strengthens when post is ABOVE a threshold (sliding threshold).
        This creates selectivity — neurons become tuned to specific patterns.
        """
        if pre.dim() > 1:
            pre = pre.mean(dim=0)
        if post.dim() > 1:
            post = post.mean(dim=0)
        
        # Sliding threshold based on recent mean activity
        selective_post = post * (post - theta * post_mean.clamp(min=0.1))
        return lr * torch.outer(selective_post, pre)
    
    @staticmethod
    def stdp(
        pre_times: torch.Tensor,
        post_times: torch.Tensor,
        weight: torch.Tensor,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.01
    ) -> torch.Tensor:
        """
        Spike-Timing Dependent Plasticity.
        
        Args:
            pre_times: spike times of pre-synaptic neurons (ms)
            post_times: spike times of post-synaptic neurons (ms)
            
        If pre fires BEFORE post (pre_time < post_time): LTP (strengthen)
        If pre fires AFTER post (pre_time > post_time): LTD (weaken)
        """
        # Time difference matrix: dt[i,j] = post_time[i] - pre_time[j]
        dt = post_times.unsqueeze(1) - pre_times.unsqueeze(0)  # (out, in)
        
        # LTP for pre-before-post (dt > 0)
        # LTD for post-before-pre (dt < 0)
        dw = torch.zeros_like(weight)
        
        # Positive dt: LTP
        pos_mask = dt > 0
        dw[pos_mask] = A_plus * torch.exp(-dt[pos_mask] / tau_plus)
        
        # Negative dt: LTD
        neg_mask = dt < 0
        dw[neg_mask] = -A_minus * torch.exp(dt[neg_mask] / tau_minus)
        
        return dw


class HebbianDCU(nn.Module):
    """
    A DCU wrapper that adds one-shot Hebbian plasticity.
    
    After each forward pass, dendritic branch weights are updated
    via local Hebbian rules. No global backward pass needed.
    
    This enables:
        - One-shot learning: show an example once, it's learned
        - Continual adaptation: weights adapt during inference
        - Self-organization: branches spontaneously specialize
    """
    def __init__(
        self,
        base_dcu: nn.Module,
        rule: Literal["plain", "oja", "bcm", "stdp"] = "oja",
        lr: float = 0.01,
        plasticity_decay: float = 0.95,
        max_weight_change: float = 0.1,
        freeze_after: Optional[int] = None  # Stop plasticity after N steps
    ):
        super().__init__()
        self.base_dcu = base_dcu
        self.rule_name = rule
        self.lr = lr
        self.plasticity_decay = plasticity_decay
        self.max_weight_change = max_weight_change
        self.freeze_after = freeze_after
        self.step_count = 0
        
        # Initialize traces for each branch
        self.traces: list[HebbianTrace] = []
        
        # For BCM rule: running mean of post-synaptic activity per branch
        self.post_means: list[torch.Tensor] = []
        
        self._init_traces()
        
    def _init_traces(self):
        """Initialize plasticity traces for each dendritic branch."""
        if hasattr(self.base_dcu, 'branches') and isinstance(self.base_dcu.branches, nn.ModuleList):
            for branch in self.base_dcu.branches:
                if hasattr(branch, 'weight'):
                    w = branch.weight.data
                    self.traces.append(HebbianTrace(w.shape, device=w.device))
                    self.post_means.append(torch.zeros(w.shape[0], device=w.device))
        elif hasattr(self.base_dcu, 'branch_weights'):
            for w in self.base_dcu.branch_weights:
                self.traces.append(HebbianTrace(w.shape, device=w.device))
                self.post_means.append(torch.zeros(w.shape[0], device=w.device))
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass + Hebbian update.
        
        1. Run normal DCU forward
        2. For each branch, compute pre- and post-synaptic activity
        3. Apply Hebbian rule to update branch weights
        """
        # Store pre-synaptic input for each branch
        # x shape: (batch, seq_len, d) or (batch, d)
        
        # Forward through base DCU
        output = self.base_dcu(x, **kwargs)
        
        # Apply Hebbian plasticity (if not frozen)
        if self.freeze_after is None or self.step_count < self.freeze_after:
            self._hebbian_update(x, output)
        
        self.step_count += 1
        return output
    
    def _hebbian_update(self, pre_input: torch.Tensor, post_output: torch.Tensor):
        """Apply the configured Hebbian rule to all branches."""
        branches = []
        weights = []
        
        if hasattr(self.base_dcu, 'branches') and isinstance(self.base_dcu.branches, nn.ModuleList):
            branches = list(self.base_dcu.branches)
            weights = [b.weight for b in branches if hasattr(b, 'weight')]
        elif hasattr(self.base_dcu, 'branch_weights'):
            weights = self.base_dcu.branch_weights
        
        for idx, (w, trace) in enumerate(zip(weights, self.traces)):
            if idx >= len(self.traces):
                break
            
            # Get branch-specific pre-input
            # For simplicity, we use the same pre_input but in practice
            # each branch might receive a projected version
            branch_pre = pre_input
            if branch_pre.dim() > 2:
                branch_pre = branch_pre.mean(dim=1)  # Average over sequence
            
            # Get branch-specific post-output
            branch_post = post_output
            if branch_post.dim() > 2:
                branch_post = branch_post.mean(dim=1)
            
            # Compute weight change
            if self.rule_name == "plain":
                dw = HebbianRule.plain_hebbian(branch_pre, branch_post) * self.lr
            elif self.rule_name == "oja":
                dw = HebbianRule.oja(branch_pre, branch_post, w.data, self.lr)
            elif self.rule_name == "bcm":
                dw = HebbianRule.bcm(
                    branch_pre, branch_post, w.data,
                    self.post_means[idx], self.lr
                )
                # Update running post mean
                self.post_means[idx] = 0.9 * self.post_means[idx] + 0.1 * branch_post.mean(dim=0)
            else:
                dw = HebbianRule.plain_hebbian(branch_pre, branch_post) * self.lr
            
            # Clip to prevent runaway changes
            dw = torch.clamp(dw, -self.max_weight_change, self.max_weight_change)
            
            # Apply update
            with torch.no_grad():
                w.data += dw.to(w.device)
    
    def consolidate(self, consolidation_factor: float = 0.5):
        """
        Consolidate recent plasticity into stable weights.
        
        Like sleep in biological systems: move fast Hebbian changes
        into slower, more permanent weight structures.
        
        After consolidation, the fast plasticity traces are reset,
        ready for new one-shot learning.
        """
        for trace in self.traces:
            trace.eligibility *= (1 - consolidation_factor)
        self.step_count = 0
    
    def reset_plasticity(self):
        """Reset all plasticity traces (forget recent one-shot learning)."""
        for trace in self.traces:
            trace.eligibility.zero_()
            trace.pre_mean.zero_()
            trace.post_mean.zero_()
            trace.timestep = 0
        self.step_count = 0
    
    def get_branch_specialization(self) -> list[dict]:
        """
        Analyze what each branch has learned.
        
        Returns a list of metrics showing how selective each branch is.
        High selectivity = the branch has learned a specific pattern.
        """
        specs = []
        for idx, trace in enumerate(self.traces):
            elig = trace.eligibility
            # Measure how "structured" the eligibility trace is
            # High variance = selective; low variance = general
            flat = elig.flatten()
            variance = flat.var().item()
            sparsity = (flat.abs() < 0.01).float().mean().item()
            specs.append({
                "branch": idx,
                "selectivity": variance,
                "sparsity": sparsity,
                "timestep": trace.timestep,
            })
        return specs
