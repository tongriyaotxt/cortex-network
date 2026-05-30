"""
CORTEX Complete Model

Full sequence model combining all CORTEX components.
Can be used for language modeling, sequence classification,
or any other sequence task.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cortex_block import CORTEXBlock, CORTEXLayer
from .workspace import GlobalWorkspaceLayer


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - used in modern LLMs.
    Provides relative position encoding through rotation.
    """
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotation angles
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Return half-dimensional embeddings (will be applied to x1, x2)
        return freqs.cos(), freqs.sin()
    
    def apply_rotary(self, x, cos, sin):
        """
        Apply rotary embeddings to input.
        x: (batch, seq_len, dim)
        """
        x1, x2 = x[..., ::2], x[..., 1::2]
        # cos and sin have shape (seq_len, dim/2)
        # Need to broadcast: (1, seq_len, dim/2)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack([y1, y2], dim=-1).flatten(-2)


class CORTEXModel(nn.Module):
    """
    Complete CORTEX sequence model.
    
    Architecture:
    - Token embeddings + positional encoding
    - Stack of CORTEX layers
    - Global workspace across all layers
    - Task-specific output head
    
    The model maintains an explicit "consciousness state" that
    can be monitored, analyzed, or used for interpretability.
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=12,
        n_modules=8,
        workspace_dim=256,
        n_branches=4,
        n_timescales=3,
        timescales=None,
        spike_dim_ratio=0.3,
        max_seq_len=4096,
        dropout=0.1,
        tie_weights=False,
        num_classes=None,  # For classification tasks
        consciousness_output=False,  # Return consciousness state
        causal=True,  # True for GPT-style, False for BERT-style
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.consciousness_output = consciousness_output
        self.causal = causal
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = RotaryPositionalEmbedding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Main CORTEX layers
        self.layers = nn.ModuleList([
            CORTEXBlock(
                d_model=d_model,
                n_modules=max(2, n_modules // 2),
                workspace_dim=workspace_dim,
                n_branches=n_branches,
                n_timescales=n_timescales,
                timescales=timescales,
                spike_dim_ratio=spike_dim_ratio,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(n_layers)
        ])
        
        # Global workspace across all layers (higher-level integration)
        self.global_workspace = GlobalWorkspaceLayer(
            d_model=d_model,
            n_modules=n_modules,
            workspace_dim=workspace_dim,
        )
        
        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output head
        if num_classes is not None:
            # Classification head
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, num_classes),
            )
        else:
            # Language modeling head
            self.output_head = nn.Linear(d_model, vocab_size, bias=False)
            if tie_weights:
                self.output_head.weight = self.token_embedding.weight
        
        # Consciousness projection (optional output)
        if consciousness_output:
            self.consciousness_proj = nn.Sequential(
                nn.Linear(workspace_dim, workspace_dim // 2),
                nn.SiLU(),
                nn.Linear(workspace_dim // 2, workspace_dim // 4),
            )
        
        self.num_classes = num_classes
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        return_consciousness=False,
        return_all_info=False,
        causal=None,
    ):
        causal = self.causal if causal is None else causal
        """
        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: optional mask
            labels: optional labels for loss computation
            return_consciousness: whether to return consciousness state
            return_all_info: whether to return all intermediate info
        Returns:
            outputs: logits or classification scores
            consciousness: optional consciousness state
            info: optional dict with all intermediate info
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings + positional encoding
        x = self.token_embedding(input_ids)
        
        # Apply rotary positional encoding
        cos, sin = self.pos_encoding(seq_len, device)
        x = self.pos_encoding.apply_rotary(x, cos, sin)
        x = self.dropout(x)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) with 1 for valid, 0 for masked
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask
        
        # Note: CORTEX is inherently causal due to:
        # - MultiTimescaleStateLayer: RNN-style sequential processing
        # - DendriticComputationUnit: position-wise independent computation
        # - GlobalWorkspaceLayer: per-position competition and aggregation
        # The `causal` flag serves as explicit documentation and future extension point
        # for hybrid architectures that may incorporate standard attention heads.
        
        # Pass through CORTEX layers
        h = x
        multiscale_states = [None] * self.n_layers
        all_layer_info = []
        
        for i, layer in enumerate(self.layers):
            h, new_state, layer_info = layer(
                h,
                multiscale_states=multiscale_states[i],
                return_info=True,
            )
            multiscale_states[i] = new_state
            all_layer_info.append(layer_info)
        
        # Global workspace integration
        h, global_workspace_info = self.global_workspace(h)
        
        # Extract consciousness state
        if isinstance(global_workspace_info, tuple):
            consciousness = global_workspace_info[1] if len(global_workspace_info) > 1 else None
        else:
            # Try to get from the info dict
            if hasattr(global_workspace_info, 'get'):
                consciousness = global_workspace_info.get('workspace', None)
            else:
                consciousness = None
        
        # Final normalization
        h = self.final_norm(h)
        
        # Output projection
        logits = self.output_head(h)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.num_classes is not None:
                # Classification loss
                loss = F.cross_entropy(
                    logits.mean(dim=1),  # Pool over sequence
                    labels
                )
            else:
                # Language modeling loss
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )
        
        outputs = {'logits': logits}
        if loss is not None:
            outputs['loss'] = loss
        
        if return_consciousness or self.consciousness_output:
            if consciousness is not None:
                consciousness_vec = self.consciousness_proj(consciousness.mean(dim=1))
            else:
                consciousness_vec = None
            outputs['consciousness'] = consciousness_vec
        
        if return_all_info:
            outputs['layer_info'] = all_layer_info
            outputs['global_workspace'] = global_workspace_info
        
        return outputs
    
    def get_consciousness_state(self, input_ids, attention_mask=None):
        """
        Extract the consciousness state for a given input.
        Useful for interpretability and analysis.
        """
        outputs = self.forward(
            input_ids,
            attention_mask=attention_mask,
            return_consciousness=True,
            return_all_info=True,
        )
        return outputs.get('consciousness'), outputs.get('global_workspace')
    
    def get_spike_statistics(self, input_ids):
        """
        Get spike rate statistics across all layers.
        """
        outputs = self.forward(
            input_ids,
            return_all_info=True,
        )
        
        spike_rates = []
        for layer_info in outputs.get('layer_info', []):
            if 'spike_rate' in layer_info:
                spike_rates.append(layer_info['spike_rate'])
        
        return {
            'mean_spike_rate': sum(spike_rates) / len(spike_rates) if spike_rates else 0.0,
            'spike_rates': spike_rates,
        }
    
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
    ):
        """
        Auto-regressive generation.
        
        Note: For full generation support, the model should be trained
        with causal masking (not yet implemented in this basic version).
        """
        self.eval()
        device = input_ids.device
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            # Forward pass (autoregressive: only see past tokens)
            outputs = self.forward(input_ids, causal=True)
            logits = outputs['logits']
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
