"""
CORTEX Building Block

Combines all core components into a single transformer-style block:
1. Multi-timescale dendritic processing
2. Global workspace integration
3. Predictive coding
4. Spike gating
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dendritic import DendriticComputationUnit
from .workspace import GlobalWorkspaceLayer
from .predictive_coding import PredictiveCodingLayer
from .multiscale_state import MultiTimescaleStateLayer
from .spike_encoding import HybridSpikeEncoder


class CORTEXBlock(nn.Module):
    """
    A single CORTEX processing block.
    
    Processing flow:
    1. Multi-timescale state update
    2. Dendritic computation (with spike gating)
    3. Global workspace broadcast
    4. Predictive coding update
    5. Feed-forward dendritic network
    """
    def __init__(
        self,
        d_model,
        n_modules=4,
        workspace_dim=None,
        n_branches=4,
        n_timescales=3,
        timescales=None,
        spike_dim_ratio=0.3,
        dropout=0.1,
        use_workspace=True,
        use_predictive=True,
        use_multiscale=True,
        use_spike=True,
        causal=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_workspace = use_workspace
        self.use_predictive = use_predictive
        self.use_multiscale = use_multiscale
        self.use_spike = use_spike
        self.causal = causal
        
        # 1. Multi-timescale processing
        if use_multiscale:
            self.multiscale = MultiTimescaleStateLayer(
                d_model,
                timescales=timescales,
                n_timescales=n_timescales,
            )
        else:
            self.multiscale = None
        
        # 2. Spike encoding
        if use_spike:
            self.spike_encoder = HybridSpikeEncoder(
                d_model,
                spike_dim_ratio=spike_dim_ratio,
            )
        else:
            self.spike_encoder = None
        
        # 3. Dendritic self-attention replacement
        # Instead of standard Q,K,V attention, use dendritic computation
        self.dendritic_attn = DendriticComputationUnit(
            d_model, d_model,
            n_branches=n_branches,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # 4. Global workspace
        if use_workspace:
            self.workspace = GlobalWorkspaceLayer(
                d_model,
                n_modules=n_modules,
                workspace_dim=workspace_dim,
            )
        else:
            self.workspace = None
        
        # 5. Predictive coding
        if use_predictive:
            self.predictive = PredictiveCodingLayer(d_model)
        else:
            self.predictive = None
        
        # 6. Feed-forward dendritic network
        d_ff = 4 * d_model
        self.dendritic_ff = nn.Sequential(
            DendriticComputationUnit(d_model, d_ff, n_branches=n_branches, dropout=dropout),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, multiscale_states=None, return_info=False, causal=None, branch_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            multiscale_states: list of states for multiscale layer
            return_info: whether to return intermediate info
            causal: whether to enforce causality (default from init)
            branch_mask: (n_branches,) optional mask for DCU branches (M5)
        Returns:
            out: (batch, seq_len, d_model)
            new_states: multiscale states
            info: dict with intermediate values
        """
        causal = self.causal if causal is None else causal
        info = {}
        # CORTEX architecture is inherently causal:
        # - MultiTimescaleStateLayer processes sequentially (RNN-style)
        # - DCU and Workspace operate position-wise independently
        # The causal flag is passed for explicit documentation and future extensions.
        
        # Step 1: Multi-timescale processing
        if self.use_multiscale and self.multiscale is not None:
            h, new_states = self.multiscale(x, states=multiscale_states)
        else:
            h = x
            new_states = None
        
        # Step 2: Spike encoding
        spike_info = None
        if self.use_spike and self.spike_encoder is not None:
            h, spike_info = self.spike_encoder(h)
            if spike_info is not None:
                info['spike_rate'] = spike_info['spike_rate']
        
        # Step 3: Dendritic attention (with residual)
        h_attn = self.dendritic_attn(h, return_spike=False, return_continuous=True, branch_mask=branch_mask)
        h = self.norm1(h + self.dropout(h_attn))
        
        # Step 4: Global workspace
        workspace_info = None
        if self.use_workspace and self.workspace is not None:
            h, workspace_info = self.workspace(h)
            if isinstance(workspace_info, tuple):
                h, workspace_info = workspace_info[0], workspace_info[1] if len(workspace_info) > 1 else {}
            info['workspace'] = workspace_info
        
        # Step 5: Predictive coding
        pred_info = None
        if self.use_predictive and self.predictive is not None:
            # Predict next layer (treat current as prediction of next)
            h, pred_info = self.predictive(h, h_target=None, return_error=False)
            info['prediction'] = pred_info
        
        # Step 6: Feed-forward (with residual)
        h_ff = self.dendritic_ff(h)
        out = self.norm2(h + self.dropout(h_ff))
        
        if return_info:
            return out, new_states, info
        return out, new_states


class CORTEXLayer(nn.Module):
    """
    A stack of CORTEX blocks with shared workspace consciousness.
    
    Consciousness state is maintained across all blocks in the layer.
    """
    def __init__(
        self,
        d_model,
        n_blocks=4,
        n_modules=4,
        workspace_dim=None,
        n_branches=4,
        n_timescales=3,
        dropout=0.1,
        share_workspace=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.share_workspace = share_workspace
        
        if share_workspace:
            # Shared workspace across all blocks
            self.workspace = GlobalWorkspaceLayer(
                d_model, n_modules=n_modules, workspace_dim=workspace_dim
            )
            
            self.blocks = nn.ModuleList([
                CORTEXBlock(
                    d_model,
                    n_modules=0,  # No per-block workspace
                    workspace_dim=workspace_dim,
                    n_branches=n_branches,
                    n_timescales=n_timescales,
                    dropout=dropout,
                    use_workspace=False,  # Use shared instead
                )
                for _ in range(n_blocks)
            ])
        else:
            self.workspace = None
            self.blocks = nn.ModuleList([
                CORTEXBlock(
                    d_model,
                    n_modules=n_modules,
                    workspace_dim=workspace_dim,
                    n_branches=n_branches,
                    n_timescales=n_timescales,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ])
    
    def forward(self, x, multiscale_states=None, return_info=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            multiscale_states: list of lists of states
        Returns:
            out: processed output
            info: dict with layer-wide info
        """
        h = x
        all_states = multiscale_states or [None] * self.n_blocks
        all_info = []
        
        # Shared workspace path
        if self.share_workspace and self.workspace is not None:
            for i, block in enumerate(self.blocks):
                h, state, block_info = block(h, multiscale_states=all_states[i], return_info=True)
                all_states[i] = state
                all_info.append(block_info)
            
            # Apply shared workspace once at the end
            h, workspace_info = self.workspace(h)
            all_info.append({'shared_workspace': workspace_info})
        else:
            for i, block in enumerate(self.blocks):
                h, state, block_info = block(h, multiscale_states=all_states[i], return_info=True)
                all_states[i] = state
                all_info.append(block_info)
        
        if return_info:
            return h, all_states, all_info
        return h, all_states
