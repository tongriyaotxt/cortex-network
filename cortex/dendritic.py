"""
Dendritic Computation Unit (DCU)

Biologically-inspired computation units that model dendritic branches
with separate nonlinearities, inspired by:
- Spruston, N. (2008). Pyramidal neurons: dendritic structure and synaptic integration.
- Major, G., Larkum, M. E., & Schiller, J. (2013). Active properties of neocortical pyramidal neuron dendrites.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriticBranch(nn.Module):
    """
    A single dendritic branch with its own nonlinear dynamics.
    
    Biologically, dendritic branches can perform local nonlinear
    computations (NMDA spikes, Ca2+ spikes) independent of the soma.
    """
    def __init__(self, d_in, d_branch, branch_type='excitatory', dropout=0.0):
        super().__init__()
        self.d_in = d_in
        self.d_branch = d_branch
        self.branch_type = branch_type
        
        self.linear = nn.Linear(d_in, d_branch, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Branch-specific nonlinearities
        if branch_type == 'excitatory':
            self.nonlin = nn.SiLU()  # Smooth ReLU-like
        elif branch_type == 'inhibitory':
            self.nonlin = lambda x: -F.softplus(x)  # Negative softplus
        elif branch_type == 'rectifying':
            # Models NMDA-like voltage-gated behavior
            self.nonlin = lambda x: F.relu(x) ** 2
        elif branch_type == 'modulatory':
            # Gating/modulatory branch (like neuromodulators)
            self.nonlin = nn.Sigmoid()
        else:
            self.nonlin = nn.Tanh()
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_in) or (batch, d_in)
        Returns:
            h: (batch, seq_len, d_branch) or (batch, d_branch)
        """
        h = self.linear(x)
        h = self.nonlin(h)
        h = self.dropout(h)
        return h


class SpikeGenerator(nn.Module):
    """
    Differentiable spike generator using surrogate gradients.
    
    Forward: Heaviside threshold function
    Backward: Triangular surrogate gradient
    
    Reference: Zenke, F., & Vogels, T. P. (2021). The remarkable robustness of surrogate gradient learning.
    """
    def __init__(self, threshold=1.0, surrogate_width=1.0):
        super().__init__()
        self.threshold = threshold
        self.surrogate_width = surrogate_width
    
    def forward(self, membrane_potential):
        """
        Args:
            membrane_potential: continuous membrane potential
        Returns:
            spikes: binary spikes (0 or 1)
        """
        # Forward: threshold
        spikes = (membrane_potential >= self.threshold).float()
        
        # Backward: surrogate gradient
        if self.training:
            # Detach for forward, attach gradient for backward
            spikes = spikes + self.surrogate(membrane_potential) - self.surrogate(membrane_potential).detach()
        
        return spikes
    
    def surrogate(self, v):
        """Triangular surrogate gradient."""
        diff = v - self.threshold
        return torch.clamp(1.0 - torch.abs(diff) / self.surrogate_width, min=0.0)


class DendriticComputationUnit(nn.Module):
    """
    Complete dendritic computation unit with multiple branches,
    somatic integration, and spike output.
    
    Architecture:
        Input → [Branch 1, Branch 2, ..., Branch B] → Somatic Integration → Spike Output
                (each with different nonlinearity)
    """
    def __init__(
        self,
        d_in,
        d_out,
        n_branches=4,
        d_branch=None,
        branch_types=None,
        branch_ratios=None,
        threshold=1.0,
        max_gating=True,
        dropout=0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_branches = n_branches
        self.max_gating = max_gating
        
        if d_branch is None:
            d_branch = d_out
        
        # Default branch types if not specified
        if branch_types is None:
            branch_types = ['excitatory', 'excitatory', 'inhibitory', 'modulatory']
        
        if branch_ratios is None:
            branch_ratios = [0.4, 0.3, 0.2, 0.1]
        
        # Ensure branch dimensions sum to d_branch
        branch_dims = [max(1, int(d_branch * r)) for r in branch_ratios]
        # Adjust last one to match exactly
        branch_dims[-1] = d_branch - sum(branch_dims[:-1])
        
        # Create branches
        self.branches = nn.ModuleList()
        self.branch_dims = []
        for i, (btype, bdim) in enumerate(zip(branch_types, branch_dims)):
            self.branches.append(DendriticBranch(d_in, bdim, btype, dropout))
            self.branch_dims.append(bdim)
        
        # Somatic integration: combines branch outputs
        total_branch_dim = sum(self.branch_dims)
        self.somatic_linear = nn.Linear(total_branch_dim, d_out, bias=True)
        
        # Max-gating weight (learnable gate on max branch activation)
        if max_gating:
            self.max_gate_weight = nn.Parameter(torch.ones(d_out) * 0.5)
        
        # Spike generator for output
        self.spike_gen = SpikeGenerator(threshold=threshold)
        
        # Branch attention weights (learnable combination)
        self.branch_weights = nn.Parameter(torch.ones(len(self.branches)) / len(self.branches))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize with biological-inspired scales
        for branch in self.branches:
            nn.init.xavier_uniform_(branch.linear.weight, gain=0.8)
            if branch.linear.bias is not None:
                nn.init.zeros_(branch.linear.bias)
        
        nn.init.xavier_uniform_(self.somatic_linear.weight, gain=0.5)
        if self.somatic_linear.bias is not None:
            nn.init.zeros_(self.somatic_linear.bias)
    
    def forward(self, x, return_spike=True, return_continuous=False):
        """
        Args:
            x: (batch, seq_len, d_in) or (batch, d_in)
            return_spike: whether to return spike output
            return_continuous: whether to return continuous somatic potential
        Returns:
            Depending on flags, returns spike and/or continuous output
        """
        # Compute all branch activations
        branch_activations = []
        for branch in self.branches:
            branch_activations.append(branch(x))
        
        # Softmax-weighted branch combination
        branch_weights = F.softmax(self.branch_weights, dim=0)
        
        # Weighted concatenation
        weighted_branches = []
        for w, act in zip(branch_weights, branch_activations):
            weighted_branches.append(w * act)
        
        combined = torch.cat(weighted_branches, dim=-1)
        
        # Somatic integration
        soma = self.somatic_linear(combined)
        
        # Max-gating: strong max branch activation boosts somatic response
        if self.max_gating:
            # Max over branches for each output dimension
            max_vals = []
            for act in branch_activations:
                # Pad or project to d_out
                if act.shape[-1] != self.d_out:
                    pad = self.d_out - act.shape[-1]
                    act_padded = F.pad(act, (0, pad))
                else:
                    act_padded = act
                max_vals.append(act_padded)
            max_branch = torch.stack(max_vals, dim=-1).max(dim=-1)[0]
            soma = soma + torch.sigmoid(self.max_gate_weight) * max_branch
        
        # Somatic nonlinearity (sigmoid-like, but not saturating too quickly)
        soma = F.silu(soma)
        
        outputs = []
        
        if return_continuous or not return_spike:
            outputs.append(soma)
        
        if return_spike:
            spikes = self.spike_gen(soma)
            outputs.append(spikes)
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def get_spike_rate(self, x):
        """Compute average spike rate for regularization."""
        spikes = self.forward(x, return_spike=True, return_continuous=False)
        return spikes.mean()


class DendriticFFN(nn.Module):
    """
    Feed-forward network using dendritic computation units.
    Replaces standard MLP with biologically-inspired dendritic layers.
    """
    def __init__(self, d_model, d_ff=None, n_branches=4, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.dcu1 = DendriticComputationUnit(
            d_model, d_ff, n_branches=n_branches, dropout=dropout
        )
        self.dcu2 = DendriticComputationUnit(
            d_ff, d_model, n_branches=max(2, n_branches//2), dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # First DCU: continuous output
        h = self.dcu1(x, return_spike=False, return_continuous=True)
        h = self.dropout(h)
        
        # Second DCU: spike output for sparse activation
        out = self.dcu2(h, return_spike=True, return_continuous=False)
        return out
