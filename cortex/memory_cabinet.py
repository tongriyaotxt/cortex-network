"""
M5-2: Memory Cabinet (External Long-Term Storage)

Provides:
- MemoryCabinet: explicit archive and retrieval of slow timescale states.
- Supports importance-weighted storage and similarity-based retrieval.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import MemoryKey, MemoryRecord


class MemoryCabinet(nn.Module):
    """
    An external memory bank for long-term storage of slow timescale states.
    
    Architecture:
    - Keys: encoded from slow_state + context
    - Values: the slow_state itself
    - Retrieval: cosine similarity + context filtering
    - Replay: used during offline consolidation
    """

    def __init__(
        self,
        d_memory: int = 256,
        capacity: int = 10000,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_memory = d_memory
        self.capacity = capacity
        self.temperature = temperature

        # Memory bank
        self.register_buffer('keys', torch.zeros(capacity, d_memory))
        self.register_buffer('values', torch.zeros(capacity, d_memory))
        self.register_buffer('masks', torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer('importance', torch.zeros(capacity))
        self.register_buffer('timestamps', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('context_tags', torch.zeros(capacity, dtype=torch.long))

        self.write_ptr = 0

        # Key encoder
        self.key_encoder = nn.Sequential(
            nn.Linear(d_memory, d_memory),
            nn.SiLU(),
            nn.Linear(d_memory, d_memory),
        )

    def archive(
        self,
        slow_state: torch.Tensor,
        context_tag: str = "",
        importance: float = 0.5,
        timestamp: int = 0,
    ) -> MemoryKey:
        """
        Archive a slow timescale state.
        Returns a MemoryKey for later retrieval.
        """
        if slow_state.dim() == 1:
            slow_state = slow_state.unsqueeze(0)

        # Encode key
        key = self.key_encoder(slow_state).squeeze(0).detach()

        # Store
        idx = self.write_ptr % self.capacity
        self.keys[idx] = key
        self.values[idx] = slow_state.squeeze(0).detach()
        self.masks[idx] = True
        self.importance[idx] = importance
        self.timestamps[idx] = timestamp
        self.context_tags[idx] = hash(context_tag) % (2**31)

        self.write_ptr += 1

        return MemoryKey(
            key_id=f"mem_{idx}_{timestamp}",
            context_tag=context_tag,
            importance=importance,
        )

    def retrieve(
        self,
        query: torch.Tensor,
        context_hint: Optional[str] = None,
        k: int = 3,
    ) -> List[MemoryRecord]:
        """
        Retrieve k most similar memories.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)

        q = self.key_encoder(query).squeeze(0)

        valid = self.masks
        if not valid.any():
            return []

        valid_keys = self.keys[valid]
        valid_values = self.values[valid]
        valid_importance = self.importance[valid]

        # Cosine similarity
        sim = F.cosine_similarity(q.unsqueeze(0), valid_keys, dim=-1)

        # Weight by importance
        weighted_sim = sim * valid_importance.clamp(min=0.1)

        # Context filtering
        if context_hint is not None:
            hint_hash = hash(context_hint) % (2**31)
            context_match = (self.context_tags[valid] == hint_hash).float()
            weighted_sim = weighted_sim * (1.0 + context_match)

        top_k = min(k, weighted_sim.size(0))
        scores, indices = torch.topk(weighted_sim, top_k)

        results = []
        for idx_in_valid, score in zip(indices.tolist(), scores.tolist()):
            key_id = f"mem_{idx_in_valid}"
            results.append(MemoryRecord(
                key=MemoryKey(key_id=key_id, importance=valid_importance[idx_in_valid].item()),
                slow_state=valid_values[idx_in_valid],
                similarity_score=score,
            ))
        return results

    def replay(
        self,
        task_id: str,
        n_samples: int = 10,
    ) -> List[torch.Tensor]:
        """
        Retrieve historical samples for a specific task.
        Used during offline consolidation to prevent forgetting.
        """
        # Randomly sample from memory bank
        valid_indices = torch.where(self.masks)[0]
        if len(valid_indices) == 0:
            return []

        n = min(n_samples, len(valid_indices))
        sampled = valid_indices[torch.randperm(len(valid_indices))[:n]]
        return [self.values[i] for i in sampled.tolist()]

    def forward(self, query: torch.Tensor, k: int = 3):
        """Convenience forward."""
        return self.retrieve(query, k=k)
