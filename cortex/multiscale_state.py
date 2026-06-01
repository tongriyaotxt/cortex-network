"""
Multi-Timescale Heterogeneous State Processing

Models the brain's multiple temporal scales of processing:
- Fast (γ ~ 40Hz): sensory processing, attention
- Medium (α ~ 10Hz): working memory integration
- Slow (δ ~ 1-3Hz): long-range prediction, context

Reference:
- Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks.
- Stern, M., Istrate, N., & Mazzucato, L. (2023). A reservoir of timescales emerges in recurrent circuits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimescaleUnit(nn.Module):
    """
    A recurrent unit operating at a specific timescale.
    
    dh/dt = (-h + f(x, h_other)) / τ
    
    Discrete: h_{t+1} = (1 - dt/τ) * h_t + (dt/τ) * f(x_t, h_other)
    """
    def __init__(self, d_model, timescale_ms=100.0, dt_ms=10.0):
        super().__init__()
        self.d_model = d_model
        self.timescale_ms = timescale_ms
        self.dt_ms = dt_ms
        
        # Decay factor
        self.register_buffer('decay', torch.tensor(1.0 - dt_ms / timescale_ms))
        
        # State update function
        self.update_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # input + other timescales
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, h_prev, h_other=None):
        """
        Args:
            x: (batch, seq_len, d_model) input
            h_prev: (batch, d_model) previous state
            h_other: (batch, d_model) or None, coupled timescale state
        Returns:
            h: (batch, seq_len, d_model) updated states over sequence
        """
        batch, seq_len, d = x.shape
        device = x.device
        
        h = h_prev
        h_trace = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Combine input with other timescale
            if h_other is not None:
                combined = torch.cat([x_t, h_other], dim=-1)
            else:
                combined = torch.cat([x_t, h], dim=-1)
            
            # Compute update
            update = self.update_fn(combined)
            
            # Leaky integration
            h = self.decay * h + (1.0 - self.decay) * update
            
            h_trace.append(h)
        
        h_trace = torch.stack(h_trace, dim=1)  # (batch, seq_len, d)
        return h_trace


class MultiTimescaleStateLayer(nn.Module):
    """
    Layer that maintains and integrates multiple timescale states.
    
    Different subpopulations of neurons operate at different speeds,
    and they are coupled through cross-timescale connections.
    """
    def __init__(
        self,
        d_model,
        timescales=None,  # in milliseconds
        n_timescales=None,
        dt_ms=10.0,
        coupling_strength=0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.dt_ms = dt_ms
        self.coupling_strength = coupling_strength
        
        # Default timescales (brain-inspired)
        if timescales is None:
            if n_timescales is None:
                n_timescales = 3
            # Generate logarithmically spaced timescales
            timescales = [25.0 * (4.0 ** i) for i in range(n_timescales)]
        
        self.n_timescales = len(timescales)
        self.timescales = timescales
        
        # Dimension per timescale
        self.d_per_scale = d_model // self.n_timescales
        assert self.d_per_scale * self.n_timescales == d_model, \
            "d_model must be divisible by n_timescales"
        
        # Create timescale units
        self.units = nn.ModuleList([
            TimescaleUnit(self.d_per_scale, tau, dt_ms)
            for tau in timescales
        ])
        
        # Cross-timescale coupling
        # Each timescale receives input from others
        self.cross_coupling = nn.ModuleList([
            nn.Linear(self.d_per_scale, self.d_per_scale)
            for _ in range(self.n_timescales)
        ])
        
        # Input projection to split into timescales
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Output fusion
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for coupling in self.cross_coupling:
            nn.init.xavier_uniform_(coupling.weight, gain=0.3)
            nn.init.zeros_(coupling.bias)
        
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
    
    def forward(self, x, states=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            states: list of (batch, d_per_scale) or None
        Returns:
            out: (batch, seq_len, d_model) fused multi-timescale output
            new_states: list of final states for each timescale
        """
        batch, seq_len, d = x.shape
        device = x.device
        
        # Initialize states if needed
        if states is None:
            states = [
                torch.zeros(batch, self.d_per_scale, device=device)
                for _ in range(self.n_timescales)
            ]
        
        # Project input
        x_proj = self.input_proj(x)
        
        # Split into timescales
        x_scales = torch.chunk(x_proj, self.n_timescales, dim=-1)
        
        # Process each timescale with cross-coupling
        scale_outputs = []
        new_states = []
        
        for i, (unit, x_s, h_prev) in enumerate(zip(self.units, x_scales, states)):
            # Compute coupling from other timescales
            h_other = torch.zeros(batch, self.d_per_scale, device=device)
            for j, h_j in enumerate(states):
                if i != j:
                    h_other = h_other + self.coupling_strength * self.cross_coupling[i](h_j)
            
            # Update this timescale
            h_trace = unit(x_s, h_prev, h_other)
            scale_outputs.append(h_trace)
            new_states.append(h_trace[:, -1, :])  # Final state (keep gradient for training)
        
        # Concatenate all timescales
        combined = torch.cat(scale_outputs, dim=-1)  # (batch, seq_len, d_model)
        
        # Fusion
        out = self.output_fusion(combined)
        out = self.norm(out + x)  # Residual + norm
        
        return out, new_states
    
    def get_timescale_features(self, x, states=None):
        """Return features from each timescale separately."""
        batch, seq_len, d = x.shape
        device = x.device
        
        if states is None:
            states = [
                torch.zeros(batch, self.d_per_scale, device=device)
                for _ in range(self.n_timescales)
            ]
        
        x_proj = self.input_proj(x)
        x_scales = torch.chunk(x_proj, self.n_timescales, dim=-1)
        
        scale_outputs = []
        new_states = []
        
        for i, (unit, x_s, h_prev) in enumerate(zip(self.units, x_scales, states)):
            h_other = torch.zeros(batch, self.d_per_scale, device=device)
            for j, h_j in enumerate(states):
                if i != j:
                    h_other = h_other + self.coupling_strength * self.cross_coupling[i](h_j)
            
            h_trace = unit(x_s, h_prev, h_other)
            scale_outputs.append(h_trace)
            new_states.append(h_trace[:, -1, :])
        
        return scale_outputs, new_states
