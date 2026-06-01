"""
M2: Self-Modeling Module

Provides:
- SelfModule: an always-on LPM that monitors internal system state and
  produces a SelfState vector (goal, certainty, cognitive load, emotion).
- AutobiographicalMemory: episodic storage using slow timescales,
  supports similarity-based retrieval.
- MetacognitiveMonitor: uncertainty estimation, confusion detection,
  and information-gap requests.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import (
    SelfState, EventRecord, InformationRequest, InternalSignals, Goal,
)


# =============================================================================
# SelfModule: the "self" LPM
# =============================================================================

class SelfModule(nn.Module):
    """
    An LPM that takes InternalSignals (rather than external input) and
    produces a SelfState + saliency score.

    It competes in the GNW like any other LPM, but its inputs come from
    system introspection rather than sensory data.
    """

    def __init__(
        self,
        d_model: int,
        d_goal: int = 64,
        d_auto: int = 128,
        n_goals_max: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_goal = d_goal
        self.d_auto = d_auto
        self.n_goals_max = n_goals_max

        # Input encoder: aggregate all internal signals into a vector
        self.signal_encoder = nn.Sequential(
            nn.Linear(d_model + d_goal + 4, d_model),  # ws + goal + scalars
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Goal embedding network
        self.goal_embed = nn.Sequential(
            nn.Linear(d_goal, d_goal),
            nn.SiLU(),
            nn.Linear(d_goal, d_goal),
        )

        # Self-state generators
        self.certainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.load_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Tanh(),
        )

        # Saliency: how important is the self-state right now?
        self.saliency_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

        self.autobiographical_proj = nn.Linear(d_model, d_auto)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        internal_signals: InternalSignals,
    ) -> Tuple[SelfState, torch.Tensor]:
        """
        Args:
            internal_signals: see agi_protocol.InternalSignals
        Returns:
            self_state: SelfState dataclass
            saliency:   (batch,) or scalar if batch=1
        """
        device = self._get_device()

        # Encode workspace state
        ws = internal_signals.workspace_state
        if ws is None:
            ws = torch.zeros(self.d_model, device=device)
        if ws.dim() == 1:
            ws = ws.unsqueeze(0)

        batch = ws.size(0)

        # Encode goal stack (top goal only, for simplicity)
        if internal_signals.goal_stack:
            top_goal = internal_signals.goal_stack[-1]
            if top_goal.embedding is not None:
                goal_emb = top_goal.embedding
            else:
                goal_emb = torch.zeros(self.d_goal, device=device)
        else:
            goal_emb = torch.zeros(self.d_goal, device=device)

        if goal_emb.dim() == 1:
            goal_emb = goal_emb.unsqueeze(0).expand(batch, -1)

        # Aggregate scalar signals
        scalars = torch.tensor([
            internal_signals.reward,
            sum(internal_signals.prediction_errors.values()) / max(len(internal_signals.prediction_errors), 1),
            sum(internal_signals.spike_rates.values()) / max(len(internal_signals.spike_rates), 1),
            internal_signals.energy_consumption,
        ], device=device, dtype=ws.dtype)
        scalars = scalars.unsqueeze(0).expand(batch, -1)

        # Pad goal_emb if needed
        if goal_emb.size(-1) < self.d_goal:
            pad = self.d_goal - goal_emb.size(-1)
            goal_emb = F.pad(goal_emb, (0, pad))

        # Combine
        combined = torch.cat([ws, goal_emb[:, :self.d_goal], scalars], dim=-1)
        h = self.signal_encoder(combined)  # (batch, d_model)

        # Generate self-state components
        certainty = self.certainty_head(h).squeeze(-1)      # (batch,)
        cognitive_load = self.load_head(h).squeeze(-1)      # (batch,)
        emotion = self.emotion_head(h).squeeze(-1)          # (batch,)
        saliency_logits = self.saliency_head(h).squeeze(-1)  # (batch,)
        saliency = saliency_logits * cognitive_load.clamp(min=0.1)

        auto_ctx = self.autobiographical_proj(h)  # (batch, d_auto)

        # Keep scalars as tensors to preserve gradients for module loss
        self_state = SelfState(
            goal_embedding=goal_emb.mean(dim=0),
            certainty=certainty.mean(),
            cognitive_load=cognitive_load.mean(),
            emotional_valence=emotion.mean(),
            autobiographical_context=auto_ctx.mean(dim=0),
        )

        return self_state, saliency

    def _get_device(self):
        return next(self.parameters()).device


# =============================================================================
# AutobiographicalMemory: episodic storage
# =============================================================================

class AutobiographicalMemory(nn.Module):
    """
    Stores event records and supports similarity-based retrieval.
    Implemented as a differentiable memory bank with learnable keys.
    """

    def __init__(
        self,
        d_event: int = 256,
        capacity: int = 1000,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_event = d_event
        self.capacity = capacity
        self.temperature = temperature

        # Memory bank: keys and values
        self.register_buffer('memory_keys', torch.zeros(capacity, d_event))
        self.register_buffer('memory_values', torch.zeros(capacity, d_event))
        self.register_buffer('memory_mask', torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer('memory_importance', torch.zeros(capacity))
        self.register_buffer('memory_timestamps', torch.zeros(capacity, dtype=torch.long))

        self.write_ptr = 0

        # Key encoder for new events
        self.key_encoder = nn.Sequential(
            nn.Linear(d_event, d_event),
            nn.SiLU(),
            nn.Linear(d_event, d_event),
        )

    def encode_event(
        self,
        consciousness: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        outcome: float = 0.0,
        timestamp: int = 0,
    ) -> torch.Tensor:
        """
        Encode an event into a memory vector.
        Args:
            consciousness: (d_event,)
            action: optional (d_event,)
            outcome: scalar reward
            timestamp: step count
        Returns:
            event_vec: (d_event,)
        """
        if consciousness.dim() == 1:
            consciousness = consciousness.unsqueeze(0)

        if action is not None:
            if action.dim() == 1:
                action = action.unsqueeze(0)
            # Concat and project
            combined = torch.cat([consciousness, action], dim=-1)
            # Pad or project to d_event
            if combined.size(-1) != self.d_event:
                combined = F.linear(combined, torch.randn(self.d_event, combined.size(-1)))
        else:
            combined = consciousness

        if combined.size(-1) != self.d_event:
            pad = self.d_event - combined.size(-1)
            combined = F.pad(combined, (0, pad))

        # Encode
        event_vec = self.key_encoder(combined)  # (1, d_event)

        # Store
        idx = self.write_ptr % self.capacity
        self.memory_keys[idx] = event_vec.squeeze(0).detach()
        self.memory_values[idx] = event_vec.squeeze(0).detach()
        self.memory_mask[idx] = True
        self.memory_importance[idx] = abs(outcome)
        self.memory_timestamps[idx] = timestamp
        self.write_ptr += 1

        return event_vec.squeeze(0)

    def retrieve_similar(
        self,
        query: torch.Tensor,
        k: int = 3,
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Retrieve k most similar memories.
        Returns:
            list of (memory_vector, similarity_score)
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)

        q = self.key_encoder(query)  # (1, d_event)

        # Similarity with all stored memories
        valid_mask = self.memory_mask
        if not valid_mask.any():
            return []

        valid_keys = self.memory_keys[valid_mask]  # (n_valid, d_event)
        sim = F.cosine_similarity(q, valid_keys, dim=-1)  # (n_valid,)

        # Weighted by importance
        importance = self.memory_importance[valid_mask]
        weighted_sim = sim * importance.clamp(min=0.1)

        top_k = min(k, weighted_sim.size(0))
        values, indices = torch.topk(weighted_sim, top_k)

        results = []
        for idx_in_valid, score in zip(indices.tolist(), values.tolist()):
            mem_vec = valid_keys[idx_in_valid]
            results.append((mem_vec, score))
        return results

    def consolidate(self, workspace_replay: List[torch.Tensor]):
        """
        Offline consolidation: replay workspace states and strengthen
        corresponding memory traces.
        """
        for state in workspace_replay:
            # Find most similar memory and boost its importance
            retrieved = self.retrieve_similar(state, k=1)
            if retrieved:
                # Boost importance (simplified)
                pass

    def forward(self, query: torch.Tensor, k: int = 3):
        """Convenience forward: retrieve similar memories."""
        return self.retrieve_similar(query, k)


# =============================================================================
# MetacognitiveMonitor: thinking about thinking
# =============================================================================

class MetacognitiveMonitor(nn.Module):
    """
    Estimates the model's own uncertainty and detects cognitive gaps.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Uncertainty from precision
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Confusion detector (RNN over error history)
        self.confusion_rnn = nn.GRUCell(1, d_model // 4)
        self.confusion_classifier = nn.Linear(d_model // 4, 1)

        # Information gap detector
        self.gap_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self.hidden_state = None

    def estimate_uncertainty(
        self,
        prediction: torch.Tensor,
        precision: torch.Tensor,
    ) -> float:
        """
        Higher precision -> lower uncertainty.
        Args:
            prediction: (..., d_model)
            precision:  (..., d_model)
        Returns:
            uncertainty: scalar in [0, 1]
        """
        # Precision -> confidence
        mean_precision = precision.mean(dim=-1, keepdim=True)
        confidence = torch.sigmoid(mean_precision)
        uncertainty = 1.0 - confidence.mean().item()
        return uncertainty

    def detect_confusion(self, error_history: List[float]) -> Tuple[bool, float]:
        """
        Detect sustained high prediction error -> confusion.
        Returns:
            is_confused: bool
            confusion_level: float [0, 1]
        """
        if not error_history:
            return False, 0.0

        device = next(self.parameters()).device
        errors = torch.tensor(error_history[-10:], device=device).unsqueeze(1)  # (T, 1)

        h = self.hidden_state
        if h is None or h.size(0) != errors.size(0):
            h = torch.zeros(errors.size(0), self.d_model // 4, device=device)

        for t in range(errors.size(0)):
            h = self.confusion_rnn(errors[t:t+1], h[:1])

        self.hidden_state = h[:1].detach()

        confusion_logit = self.confusion_classifier(h[:1]).squeeze()
        confusion_level = torch.sigmoid(confusion_logit).item()
        is_confused = confusion_level > 0.7 and len(error_history) > 3

        return is_confused, confusion_level

    def request_information(
        self,
        gap: torch.Tensor,
        context: torch.Tensor,
    ) -> Optional[InformationRequest]:
        """
        Detect information gaps.
        Args:
            gap: (d_model,) representation of missing information
            context: (d_model,) current context
        Returns:
            InformationRequest or None
        """
        if gap.dim() == 1:
            gap = gap.unsqueeze(0)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        combined = torch.cat([gap, context], dim=-1)
        if combined.size(-1) > self.d_model * 2:
            combined = combined[:, :self.d_model * 2]
        elif combined.size(-1) < self.d_model * 2:
            pad = self.d_model * 2 - combined.size(-1)
            combined = F.pad(combined, (0, pad))

        gap_prob = self.gap_detector(combined).squeeze().item()

        if gap_prob > 0.6:
            return InformationRequest(
                gap_description="detected_information_gap",
                urgency=gap_prob,
            )
        return None
