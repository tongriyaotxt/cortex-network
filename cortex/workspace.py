"""
Global Neuronal Workspace (GNW) Layer

Implements the Global Workspace Theory of consciousness by Dehaene & Changeux:
- Information from local modules competes for access
- Winning information triggers "ignition" and global broadcast
- Produces an explicit "consciousness state vector"

References:
- Dehaene, S., & Changeux, J. P. (2011). Experimental and theoretical approaches to conscious processing.
- Dehaene, S., et al. (1998). A neuronal network model linking subjective reports and objective physiological data.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalProcessingModule(nn.Module):
    """
    A local processing module that computes representations
    and estimates its own saliency for workspace competition.
    """
    def __init__(self, d_model, module_id=0):
        super().__init__()
        self.d_model = d_model
        self.module_id = module_id
        
        # Representation processing
        self.processor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Saliency estimation (how important is this module's output?)
        self.saliency_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )
        
        # Learnable gain (some modules naturally have higher saliency)
        self.gain = nn.Parameter(torch.ones(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            z: processed representation
            saliency: scalar saliency score per position
        """
        z = self.processor(x)
        
        # Compute saliency
        saliency_raw = self.saliency_estimator(z).squeeze(-1)  # (batch, seq_len)
        saliency = saliency_raw * torch.abs(self.gain)  # Apply learnable gain
        
        return z, saliency


class IgnitionMechanism(nn.Module):
    """
    Models the "ignition" phenomenon in GNW:
    When salient information reaches threshold, it triggers
    a rapid, self-sustaining broadcast across the workspace.
    """
    def __init__(self, workspace_dim, n_modules, ignition_threshold=0.5):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.n_modules = n_modules
        self.ignition_threshold = ignition_threshold
        
        # Ignition classifier
        self.ignition_gate = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 2),
            nn.SiLU(),
            nn.Linear(workspace_dim // 2, 1),
        )
        
        # Recurrent ignition state (simulates self-sustaining activity)
        self.recurrent = nn.Linear(workspace_dim, workspace_dim)
        
        # Ignition threshold is learnable
        self.threshold_param = nn.Parameter(torch.tensor(ignition_threshold))
    
    def forward(self, workspace_state, prev_consciousness=None):
        """
        Args:
            workspace_state: (batch, seq_len, workspace_dim)
            prev_consciousness: previous consciousness state for recurrence
        Returns:
            ignition_prob: (batch, seq_len) probability of ignition
            new_consciousness: updated consciousness state
        """
        batch, seq_len, d = workspace_state.shape
        
        # Add recurrent component if available
        if prev_consciousness is not None:
            recurrent_input = self.recurrent(prev_consciousness)
            workspace_state = workspace_state + 0.1 * recurrent_input
        
        # Compute ignition probability
        ignition_logits = self.ignition_gate(workspace_state).squeeze(-1)
        ignition_prob = torch.sigmoid(ignition_logits)
        
        # Apply adaptive threshold
        threshold = torch.sigmoid(self.threshold_param)
        
        # Hard ignition decision (with straight-through for differentiability)
        ignition = (ignition_prob > threshold).float()
        ignition = ignition + ignition_prob - ignition_prob.detach()
        
        # Update consciousness state only when ignition occurs
        if prev_consciousness is None:
            new_consciousness = workspace_state * ignition.unsqueeze(-1)
        else:
            # Momentum update: consciousness persists and updates
            new_consciousness = 0.7 * prev_consciousness + 0.3 * workspace_state * ignition.unsqueeze(-1)
        
        return ignition_prob, ignition, new_consciousness


class GlobalWorkspaceLayer(nn.Module):
    """
    Global Neuronal Workspace Layer.
    
    Coordinates multiple local processing modules through:
    1. Competition: modules compete based on saliency
    2. Ignition: winning content triggers global broadcast
    3. Consciousness: maintains explicit state vector
    """
    def __init__(
        self,
        d_model,
        n_modules=8,
        workspace_dim=None,
        competition_temp=0.5,
        ignition_threshold=0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_modules = n_modules
        self.workspace_dim = workspace_dim or d_model // 2
        self.competition_temp = competition_temp
        
        # Local processing modules (use lpms to avoid conflict with nn.Module.modules)
        self.lpms = nn.ModuleList([
            LocalProcessingModule(d_model, module_id=i)
            for i in range(n_modules)
        ])
        
        # Project each module's output to workspace dimension
        self.module_projections = nn.ModuleList([
            nn.Linear(d_model, self.workspace_dim)
            for _ in range(n_modules)
        ])
        
        # Ignition mechanism
        self.ignition = IgnitionMechanism(
            self.workspace_dim, n_modules, ignition_threshold
        )
        
        # Broadcast back to full dimension
        self.broadcast = nn.Linear(self.workspace_dim, d_model)
        
        # Consciousness state (initialized as zeros)
        self.register_buffer('consciousness_init', torch.zeros(1, 1, self.workspace_dim))
        
        # Feedback to modules
        self.feedback = nn.Linear(self.workspace_dim, d_model)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for proj in self.module_projections:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        nn.init.xavier_uniform_(self.broadcast.weight, gain=0.5)
        nn.init.xavier_uniform_(self.feedback.weight, gain=0.3)
    
    def forward(self, x, return_consciousness=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            return_consciousness: whether to return consciousness state
        Returns:
            out: (batch, seq_len, d_model) broadcast workspace content
            consciousness_state: (batch, seq_len, workspace_dim) if requested
            info: dict with workspace statistics
        """
        batch, seq_len, d = x.shape
        device = x.device
        
        # Step 1: Local processing and saliency
        module_outputs = []
        module_saliencies = []
        
        for module, proj in zip(self.lpms, self.module_projections):
            z, saliency = module(x)
            z_proj = proj(z)
            module_outputs.append(z_proj)
            module_saliencies.append(saliency)
        
        # Stack: (n_modules, batch, seq_len, workspace_dim)
        module_outputs = torch.stack(module_outputs, dim=0)
        module_saliencies = torch.stack(module_saliencies, dim=0)  # (n_modules, batch, seq_len)
        
        # Step 2: Competition (softmax over modules)
        # Normalize saliencies
        saliencies_norm = module_saliencies / (math.sqrt(self.d_model) + 1e-6)
        
        # Temperature-scaled softmax competition
        competition_weights = F.softmax(saliencies_norm / self.competition_temp, dim=0)
        # (n_modules, batch, seq_len)
        
        # Step 3: Weighted aggregation into workspace
        workspace = torch.zeros(batch, seq_len, self.workspace_dim, device=device)
        for m in range(self.n_modules):
            w = competition_weights[m].unsqueeze(-1)  # (batch, seq_len, 1)
            workspace = workspace + w * module_outputs[m]
        
        # Step 4: Ignition and consciousness
        prev_consciousness = self.consciousness_init.expand(batch, seq_len, -1).to(device)
        
        ignition_prob, ignition, consciousness = self.ignition(workspace, prev_consciousness)
        
        # Step 5: Global broadcast
        broadcast_signal = self.broadcast(consciousness)
        
        # Step 6: Add feedback to original input
        feedback_signal = self.feedback(consciousness)
        out = x + broadcast_signal + 0.1 * feedback_signal
        
        # Step 7: Layer normalization
        out = F.layer_norm(out, out.shape[-1:])
        
        info = {
            'workspace': workspace,
            'ignition_prob': ignition_prob,
            'ignition': ignition,
            'competition_weights': competition_weights,
            'module_saliencies': module_saliencies,
        }
        
        if return_consciousness:
            return out, consciousness, info
        return out, info
    
    def get_consciousness_content(self, x):
        """
        Extract the current consciousness content for analysis.
        """
        _, consciousness, info = self.forward(x, return_consciousness=True)
        return consciousness, info
