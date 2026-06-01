"""
M1: Symbolic Reasoning Module

Provides:
- SymbolicBranch: a DCU branch that discretizes continuous representations
  via vector quantization, producing explicit symbolic tokens.
- SymbolicWorkspace: a discrete buffer inside GNW that operates on symbolic
  tokens, supports composition, rewriting, and broadcast back to continuous.
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import SymbolicToken, SymbolicExpression, RewriteRule


# =============================================================================
# SymbolicBranch: VQ-based dendritic branch
# =============================================================================

class SymbolicBranch(nn.Module):
    """
    A DCU branch that maps continuous input to discrete symbolic tokens
    via a learned codebook (VQ-VAE style).

    Forward:
        Input x -> linear projection -> nearest neighbor lookup in codebook
        -> hard token (forward), straight-through gradient (backward)

    This branch can be used alongside excitatory/inhibitory/modulatory
    branches inside a DendriticComputationUnit.
    """

    def __init__(self, d_in: int, d_branch: int, codebook_size: int = 512):
        super().__init__()
        self.d_in = d_in
        self.d_branch = d_branch
        self.codebook_size = codebook_size

        # Project input to codebook dimension
        self.project = nn.Linear(d_in, d_branch, bias=True)

        # Learnable codebook: each row is a codeword
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_branch))
        nn.init.uniform_(self.codebook, -1.0 / codebook_size, 1.0 / codebook_size)

        # Token confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_branch, d_branch // 4),
            nn.SiLU(),
            nn.Linear(d_branch // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[SymbolicToken]]:
        """
        Args:
            x: (batch, seq_len, d_in) or (batch, d_in)
        Returns:
            quantized: (batch, seq_len, d_branch) continuous embedding
            tokens:    list of SymbolicToken per (batch, seq) position
        """
        orig_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, d = x.shape
            x_flat = x.reshape(-1, d)
        else:
            batch, d = x.shape
            seq_len = 1
            x_flat = x

        # Project to codebook space
        z = self.project(x_flat)  # (B*S, d_branch)

        # Find nearest codeword
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.pow(2).sum(dim=1, keepdim=True).t()
            - 2 * z @ self.codebook.t()
        )
        indices = distances.argmin(dim=1)  # (B*S,)

        # Hard quantization (forward), straight-through (backward)
        z_q = self.codebook[indices]  # (B*S, d_branch)
        quantized = z + (z_q - z).detach()

        # Confidence
        confidence = self.confidence_head(quantized).squeeze(-1)  # (B*S,)

        # Build tokens
        tokens = []
        for i in range(quantized.size(0)):
            tokens.append(SymbolicToken(
                token_id=int(indices[i].item()),
                embedding=quantized[i],
                confidence=float(confidence[i].item()),
            ))

        # Reshape
        if len(orig_shape) == 3:
            quantized = quantized.reshape(batch, seq_len, self.d_branch)
        else:
            quantized = quantized.reshape(batch, self.d_branch)

        return quantized, tokens

    def embed_token(self, token_id: int) -> torch.Tensor:
        """Look up the embedding for a token id."""
        return self.codebook[token_id]


# =============================================================================
# SymbolicWorkspace: discrete logic inside GNW
# =============================================================================

class SymbolicWorkspace(nn.Module):
    """
    A symbolic buffer that runs inside the GlobalWorkspaceLayer.

    Processing pipeline:
      1. quantize:   continuous -> SymbolicToken list
      2. compose:    tokens -> SymbolicExpression tree
      3. apply_rule: rewrite expressions via rules
      4. broadcast:  symbolic result -> continuous residual

    The continuous residual is added back to the GNW workspace state.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 512,
        d_embed: int = 64,
        n_rules: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_embed = d_embed

        # Project continuous to embed space before quantization
        self.input_proj = nn.Linear(d_model, d_embed)

        # Symbol embedding table for broadcast
        self.symbol_embed = nn.Embedding(vocab_size, d_embed)

        # Broadcast network: symbol -> continuous residual
        self.broadcast_net = nn.Sequential(
            nn.Linear(d_embed, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Composition attention: combine multiple tokens into expression
        self.compose_query = nn.Linear(d_embed, d_embed)
        self.compose_key = nn.Linear(d_embed, d_embed)
        self.compose_value = nn.Linear(d_embed, d_embed)

        # Built-in rewrite rules (learnable pattern matching)
        self.rule_patterns = nn.Parameter(torch.randn(n_rules, d_embed * 2))
        self.rule_replacements = nn.Parameter(torch.randn(n_rules, d_embed))
        self.rule_gate = nn.Linear(d_embed * 2, n_rules)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.symbol_embed.weight)
        for m in self.broadcast_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def quantize(self, continuous: torch.Tensor) -> List[SymbolicToken]:
        """
        Convert continuous vectors to symbolic tokens by nearest-neighbor
        lookup in the symbol embedding table.

        Args:
            continuous: (batch, seq_len, d_model)
        Returns:
            tokens: flat list of SymbolicToken
        """
        if continuous.dim() == 3:
            batch, seq_len, d = continuous.shape
            x = continuous.reshape(-1, d)
        else:
            x = continuous.reshape(1, -1)

        # Project to embed space
        x_embed = self.input_proj(x)  # (N, d_embed)
        # Compute similarity with codebook
        z = x_embed @ self.symbol_embed.weight.t()  # (N, vocab_size)
        indices = z.argmax(dim=1)
        confidences = z.softmax(dim=1).max(dim=1).values

        tokens = []
        for i in range(indices.size(0)):
            tid = int(indices[i].item())
            tokens.append(SymbolicToken(
                token_id=tid,
                embedding=self.symbol_embed(torch.tensor(tid, device=indices.device)),
                confidence=float(confidences[i].item()),
            ))
        return tokens

    def compose(self, tokens: List[SymbolicToken]) -> SymbolicExpression:
        """
        Compose a list of tokens into a structured expression using
        self-attention over token embeddings.

        Current implementation: simple aggregation into a flat expression.
        Future: full tree-structured parsing.
        """
        if not tokens:
            return SymbolicExpression(head="Empty", args=[])

        emb = torch.stack([t.embedding for t in tokens], dim=0)  # (n, d_embed)

        # Self-attention composition
        q = self.compose_query(emb).unsqueeze(0)    # (1, n, d)
        k = self.compose_key(emb).unsqueeze(0)      # (1, n, d)
        v = self.compose_value(emb).unsqueeze(0)    # (1, n, d)

        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(emb.size(-1)), dim=-1)
        composed = (attn @ v).squeeze(0).mean(dim=0)  # (d_embed,)

        # Build expression: treat the most confident token as head
        best_token = max(tokens, key=lambda t: t.confidence)
        args = [t for t in tokens if t.token_id != best_token.token_id]

        return SymbolicExpression(
            head=f"Token_{best_token.token_id}",
            args=[t.token_id for t in args],
            confidence=best_token.confidence,
        )

    def apply_rule(
        self,
        expr: SymbolicExpression,
    ) -> SymbolicExpression:
        """
        Apply learnable rewrite rules to a symbolic expression.

        For now, we implement a soft matching: compute similarity between
        expr embedding and rule patterns, then gate a replacement.
        """
        # Encode expr into a single vector
        expr_emb = self._encode_expression(expr)  # (d_embed,)

        # Compare with all rules
        # (n_rules, d_embed*2)
        rule_input = torch.cat([expr_emb, expr_emb], dim=0).unsqueeze(0)
        gate_logits = self.rule_gate(rule_input).squeeze(0)  # (n_rules,)
        rule_weights = torch.softmax(gate_logits, dim=0)

        # Weighted combination of replacements
        replacement = rule_weights @ self.rule_replacements  # (d_embed,)

        # If a rule fires strongly, return modified expression
        max_weight = rule_weights.max().item()
        if max_weight > 0.5:
            return SymbolicExpression(
                head=f"Rewritten_{expr.head}",
                args=expr.args + ["rule_applied"],
                confidence=max_weight,
            )
        return expr

    def broadcast(self, expr: SymbolicExpression) -> torch.Tensor:
        """
        Convert a symbolic expression back to a continuous residual vector.
        Args:
            expr: SymbolicExpression
        Returns:
            residual: (d_embed,) or (1, d_model) after broadcast_net
        """
        expr_emb = self._encode_expression(expr)  # (d_embed,)
        residual = self.broadcast_net(expr_emb)   # (d_model,)
        return residual

    def _encode_expression(self, expr: SymbolicExpression) -> torch.Tensor:
        """Encode a symbolic expression into a single embedding vector."""
        device = self.symbol_embed.weight.device
        # Head embedding (hash-based lookup)
        head_id = hash(expr.head) % self.vocab_size
        head_emb = self.symbol_embed(torch.tensor(head_id, device=device))

        # Arg embeddings
        arg_embs = []
        for arg in expr.args:
            if isinstance(arg, int):
                arg_embs.append(self.symbol_embed(torch.tensor(arg % self.vocab_size, device=device)))
            else:
                arg_embs.append(self.symbol_embed(torch.tensor(hash(str(arg)) % self.vocab_size, device=device)))

        if arg_embs:
            args_emb = torch.stack(arg_embs).mean(dim=0)
        else:
            args_emb = torch.zeros_like(head_emb)

        return (head_emb + args_emb) / 2.0

    def forward(
        self,
        continuous: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[SymbolicToken], SymbolicExpression]:
        """
        End-to-end symbolic processing pipeline.

        Args:
            continuous: (batch, seq_len, d_model)
        Returns:
            residual:   (batch, seq_len, d_model) broadcast back
            tokens:     list of SymbolicToken
            expr:       final SymbolicExpression
        """
        batch, seq_len, d = continuous.shape

        # 1. Quantize
        tokens = self.quantize(continuous)

        # 2. Compose
        expr = self.compose(tokens)

        # 3. Apply rules
        expr = self.apply_rule(expr)

        # 4. Broadcast
        residual = self.broadcast(expr)  # (d_model,)
        residual = residual.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)

        return residual, tokens, expr
