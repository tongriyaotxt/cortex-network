"""
AGI-CORTEX Integrated Model

Extends CORTEXModel with all six AGI modules (M1-M6):
- M1: Symbolic Reasoning
- M2: Self-Modeling
- M3: Embodied Interaction
- M4: Hierarchical Planning
- M5: Continual Learning
- M6: Causal Inference

This model is backward-compatible with CORTEXModel: all new features are
opt-in via constructor flags.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cortex_model import CORTEXModel, RotaryPositionalEmbedding
from .workspace import GlobalWorkspaceLayer
from .cortex_block import CORTEXBlock
from .symbolic import SymbolicWorkspace
from .self_modeling import SelfModule, AutobiographicalMemory, MetacognitiveMonitor
from .action import ActionHead, ActionDistribution
from .forward_model import ForwardModel
from .interoception import InteroceptionChannel
from .hierarchical_workspace import HierarchicalWorkspace
from .goal import GoalStack
from .subroutine import SubroutineLPM
from .branch_isolation import BranchAllocator, ElasticPlasticity
from .memory_cabinet import MemoryCabinet
from .consolidation import OfflineConsolidation
from .causal import CausalGraph
from .counterfactual import CounterfactualWorkspace
from .causal_discovery import CausalDiscoveryLPM
from .agi_protocol import (
    WorkspacePacket, InternalSignals, SelfState, Action, ActionSpace,
    Goal, SystemState,
)


class AGICORTEXModel(nn.Module):
    """
    AGI-CORTEX: Fully integrated model with all six extension modules.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_modules: int = 8,
        workspace_dim: int = 256,
        n_branches: int = 4,
        n_timescales: int = 3,
        timescales=None,
        spike_dim_ratio: float = 0.3,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        tie_weights: bool = False,
        num_classes: Optional[int] = None,
        consciousness_output: bool = True,
        causal: bool = True,
        # AGI module switches
        use_symbolic: bool = False,
        use_self_modeling: bool = False,
        use_embodied: bool = False,
        use_hierarchical: bool = False,
        use_continual: bool = False,
        use_causal: bool = False,
        # AGI module configs
        symbolic_vocab_size: int = 512,
        n_subroutines: int = 4,
        action_space: Optional[ActionSpace] = None,
        memory_capacity: int = 10000,
        max_goal_depth: int = 5,
        n_counterfactuals: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.consciousness_output = consciousness_output
        self.causal = causal

        # Module flags
        self.use_symbolic = use_symbolic
        self.use_self_modeling = use_self_modeling
        self.use_embodied = use_embodied
        self.use_hierarchical = use_hierarchical
        self.use_continual = use_continual
        self.use_causal = use_causal

        # ===== Core CORTEX (same as baseline) =====
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = RotaryPositionalEmbedding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

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

        # ===== M1: Symbolic Reasoning =====
        if use_symbolic:
            self.symbolic_workspace = SymbolicWorkspace(
                d_model=d_model,
                vocab_size=symbolic_vocab_size,
            )
        else:
            self.symbolic_workspace = None

        # ===== M2: Self-Modeling =====
        if use_self_modeling:
            self.self_module = SelfModule(d_model=d_model)
            self.autobiographical_memory = AutobiographicalMemory(d_event=d_model)
            self.metacognitive_monitor = MetacognitiveMonitor(d_model=d_model)
            self.self_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, 64),
            )
        else:
            self.self_module = None
            self.autobiographical_memory = None
            self.metacognitive_monitor = None

        # ===== M3: Embodied Interaction =====
        if use_embodied:
            if action_space is None:
                action_space = ActionSpace(n_actions=10, n_continuous=0)
            self.action_space = action_space
            self.action_head = ActionHead(d_model, action_space, dropout)
            self.forward_model = ForwardModel(d_state=d_model, d_action=action_space.n_actions + action_space.n_continuous)
            self.interoception = InteroceptionChannel(d_intero=64)
        else:
            self.action_space = None
            self.action_head = None
            self.forward_model = None
            self.interoception = None

        # ===== M4: Hierarchical Planning =====
        if use_hierarchical:
            self.hierarchical_workspace = HierarchicalWorkspace(
                d_model=d_model,
                n_modules=n_modules,
                workspace_dim=workspace_dim,
                max_depth=max_goal_depth,
            )
            self.subroutines = nn.ModuleList([
                SubroutineLPM(d_model, f"subroutine_{i}")
                for i in range(n_subroutines)
            ])
        else:
            self.hierarchical_workspace = None
            self.subroutines = None

        # ===== M5: Continual Learning =====
        if use_continual:
            self.branch_allocator = BranchAllocator(n_branches=n_branches)
            self.elastic_plasticity = ElasticPlasticity(n_branches=n_branches)
            self.memory_cabinet = MemoryCabinet(d_memory=d_model, capacity=memory_capacity)
        else:
            self.branch_allocator = None
            self.elastic_plasticity = None
            self.memory_cabinet = None

        # ===== M6: Causal Inference =====
        if use_causal:
            self.causal_discovery = CausalDiscoveryLPM(d_model=d_model)
            self.counterfactual_workspace = None  # Initialized lazily
            self.n_counterfactuals = n_counterfactuals
        else:
            self.causal_discovery = None

        # ===== Workspace (use hierarchical if enabled) =====
        if use_hierarchical and self.hierarchical_workspace is not None:
            self.global_workspace = self.hierarchical_workspace
        else:
            self.global_workspace = GlobalWorkspaceLayer(
                d_model=d_model,
                n_modules=n_modules,
                workspace_dim=workspace_dim,
            )

        # ===== Output heads =====
        self.final_norm = nn.LayerNorm(d_model)

        if num_classes is not None:
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, num_classes),
            )
        else:
            self.output_head = nn.Linear(d_model, vocab_size, bias=False)
            if tie_weights:
                self.output_head.weight = self.token_embedding.weight

        if consciousness_output:
            self.consciousness_proj = nn.Sequential(
                nn.Linear(workspace_dim, workspace_dim // 2),
                nn.SiLU(),
                nn.Linear(workspace_dim // 2, workspace_dim // 4),
            )
        else:
            self.consciousness_proj = None

        self.num_classes = num_classes

        # State tracking
        self._internal_signals = InternalSignals()
        self._goal_stack = GoalStack(max_depth=max_goal_depth) if use_hierarchical else None
        self._step_counter = 0
        self._prev_self_state = None  # For self-consistency loss

        # Loss weights for AGI modules
        self.lambda_symbolic = 0.01
        self.lambda_self = 0.01
        self.lambda_action = 0.01
        self.lambda_plasticity = 0.001
        self.lambda_causal = 0.001

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
        return_self_state=False,
        return_action=False,
        return_symbolic=False,
        causal=None,
    ):
        causal = self.causal if causal is None else causal
        batch, seq_len = input_ids.shape
        device = input_ids.device
        self._step_counter += 1

        # Embeddings
        x = self.token_embedding(input_ids)
        cos, sin = self.pos_encoding(seq_len, device)
        x = self.pos_encoding.apply_rotary(x, cos, sin)
        x = self.dropout(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask

        # Pass through layers
        h = x
        multiscale_states = [None] * self.n_layers
        all_layer_info = []
        all_spike_rates = []
        all_errors = {}

        # M5: Get branch mask for continual learning
        branch_mask = None
        if self.use_continual and self.branch_allocator is not None:
            # Use a default task or current task's mask
            current_task = getattr(self, '_current_task', 'default')
            branch_mask_obj = self.branch_allocator.get_mask(current_task)
            if branch_mask_obj is not None:
                branch_mask = branch_mask_obj.mask.to(device)

        for i, layer in enumerate(self.layers):
            h, new_state, layer_info = layer(
                h,
                multiscale_states=multiscale_states[i],
                return_info=True,
                branch_mask=branch_mask,
            )
            multiscale_states[i] = new_state
            all_layer_info.append(layer_info)

            if 'spike_rate' in layer_info:
                all_spike_rates.append(layer_info['spike_rate'])
            if 'prediction' in layer_info and isinstance(layer_info['prediction'], dict):
                if 'error_magnitude' in layer_info['prediction']:
                    all_errors[f'layer_{i}'] = layer_info['prediction']['error_magnitude']

        # Global workspace
        if self.use_hierarchical and self.hierarchical_workspace is not None:
            h, global_workspace_info = self.hierarchical_workspace(h, return_info=True)
        else:
            h, global_workspace_info = self.global_workspace(h)

        # Extract consciousness
        consciousness = None
        if isinstance(global_workspace_info, dict):
            consciousness = global_workspace_info.get('workspace', None)
        elif isinstance(global_workspace_info, tuple):
            consciousness = global_workspace_info[1] if len(global_workspace_info) > 1 else None

        # M1: Symbolic reasoning
        symbolic_tokens = None
        symbolic_expr = None
        if self.use_symbolic and self.symbolic_workspace is not None:
            sym_residual, symbolic_tokens, symbolic_expr = self.symbolic_workspace(h)
            h = h + 0.1 * sym_residual

        # M2: Self-modeling
        self_state = None
        if self.use_self_modeling and self.self_module is not None:
            # Build internal signals
            self._internal_signals.workspace_state = h.mean(dim=1) if h.dim() == 3 else h
            self._internal_signals.spike_rates = {f'layer_{i}': r for i, r in enumerate(all_spike_rates)}
            self._internal_signals.prediction_errors = all_errors
            if self._goal_stack is not None:
                self._internal_signals.goal_stack = self._goal_stack.stack

            self_state, self_saliency = self.self_module(self._internal_signals)

            # Store event
            if self.autobiographical_memory is not None:
                self.autobiographical_memory.encode_event(
                    consciousness=self_state.autobiographical_context if self_state.autobiographical_context is not None else torch.zeros(self.d_model, device=device),
                    timestamp=self._step_counter,
                )

        # Final norm
        h = self.final_norm(h)

        # Output heads
        logits = self.output_head(h)

        # M3: Action output
        action_dist = None
        if self.use_embodied and self.action_head is not None:
            action_dist = self.action_head(h)

        # Compute loss
        loss = None
        if labels is not None:
            if self.num_classes is not None:
                loss = F.cross_entropy(logits.mean(dim=1), labels)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )

        # Build outputs
        outputs = {'logits': logits}
        if loss is not None:
            outputs['loss'] = loss

        # === Compute AGI module losses (only in training mode) ===
        if self.training:
            total_module_loss = torch.tensor(0.0, device=device)
            module_losses = {}

            # M1: Symbolic consistency loss
            if self.use_symbolic and symbolic_expr is not None:
                sym_loss = self._compute_symbolic_loss(symbolic_expr, device)
                if sym_loss is not None:
                    module_losses['symbolic'] = sym_loss.item()
                    total_module_loss = total_module_loss + self.lambda_symbolic * sym_loss

            # M2: Self consistency loss
            if self.use_self_modeling and self_state is not None:
                self_loss = self._compute_self_loss(self_state, device)
                if self_loss is not None:
                    module_losses['self'] = self_loss.item()
                    total_module_loss = total_module_loss + self.lambda_self * self_loss

            # M3: Action sparsity regularization
            if self.use_embodied and action_dist is not None:
                action_loss = self._compute_action_loss(action_dist)
                if action_loss is not None:
                    module_losses['action'] = action_loss.item()
                    total_module_loss = total_module_loss + self.lambda_action * action_loss

            # M5: Plasticity regularization
            if self.use_continual and self.elastic_plasticity is not None:
                plasticity_loss = self._compute_plasticity_loss()
                if plasticity_loss is not None:
                    module_losses['plasticity'] = plasticity_loss.item()
                    total_module_loss = total_module_loss + self.lambda_plasticity * plasticity_loss

            if loss is not None:
                outputs['loss'] = loss + total_module_loss
            else:
                outputs['loss'] = total_module_loss

            outputs['module_losses'] = module_losses
            outputs['base_loss'] = loss

        if return_consciousness or self.consciousness_output:
            if consciousness is not None and self.consciousness_proj is not None:
                outputs['consciousness'] = self.consciousness_proj(consciousness.mean(dim=1))
            else:
                outputs['consciousness'] = None

        if return_self_state and self_state is not None:
            outputs['self_state'] = self_state

        if return_action and action_dist is not None:
            outputs['action_distribution'] = action_dist

        if return_symbolic and symbolic_tokens is not None:
            outputs['symbolic_tokens'] = symbolic_tokens
            outputs['symbolic_expression'] = symbolic_expr

        if return_all_info:
            outputs['layer_info'] = all_layer_info
            outputs['global_workspace'] = global_workspace_info
            outputs['multiscale_states'] = multiscale_states

        return outputs

    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
        use_agi_modules=True,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            outputs = self.forward(
                input_ids,
                causal=True,
                return_self_state=use_agi_modules and self.use_self_modeling,
                return_action=use_agi_modules and self.use_embodied,
                return_symbolic=use_agi_modules and self.use_symbolic,
            )
            logits = outputs['logits']
            next_logits = logits[:, -1, :] / temperature

            # === M2: Self-modeling influences generation ===
            if use_agi_modules and 'self_state' in outputs and outputs['self_state'] is not None:
                ss = outputs['self_state']
                if isinstance(ss, list):
                    ss = ss[0]
                def _scalar(val):
                    return val.item() if isinstance(val, torch.Tensor) else float(val)
                # Low certainty -> sharpen (more conservative)
                if hasattr(ss, 'certainty') and _scalar(ss.certainty) < 0.3:
                    next_logits = next_logits * 1.3
                # High cognitive load -> flatten (more exploratory)
                if hasattr(ss, 'cognitive_load') and _scalar(ss.cognitive_load) > 0.7:
                    next_logits = next_logits * 0.7
                # Negative emotion -> avoid extreme probabilities
                if hasattr(ss, 'emotional_valence') and _scalar(ss.emotional_valence) < -0.5:
                    next_logits = torch.tanh(next_logits * 0.5) * 2.0

            # === M4: Goal stack influences generation ===
            if use_agi_modules and self._goal_stack is not None and not self._goal_stack.is_empty():
                top_goal = self._goal_stack.peek()
                if top_goal is not None and top_goal.embedding is not None:
                    goal_emb = top_goal.embedding
                    if goal_emb.dim() == 1:
                        goal_emb = goal_emb.unsqueeze(0)
                    # Project goal to vocab logits space
                    if goal_emb.size(-1) != next_logits.size(-1):
                        if not hasattr(self, '_goal_to_vocab'):
                            self._goal_to_vocab = nn.Linear(
                                goal_emb.size(-1), next_logits.size(-1), bias=False
                            ).to(next_logits.device)
                        goal_vocab = self._goal_to_vocab(goal_emb)
                    else:
                        goal_vocab = goal_emb
                    # Add as bias (only if shapes match after batch handling)
                    if goal_vocab.dim() == 2 and goal_vocab.size(0) == next_logits.size(0):
                        next_logits = next_logits + 0.15 * goal_vocab

            # === M3: Action head can override token generation ===
            if use_agi_modules and 'action_distribution' in outputs and outputs['action_distribution'] is not None:
                ad = outputs['action_distribution']
                if ad.saliency > 0.8 and hasattr(self.action_space, 'action_types'):
                    # High-saliency action: if we have a special "ACT" token space,
                    # this would output an action token. For now, we modulate logits
                    # to favor action-related tokens (simplified as flattening).
                    next_logits = next_logits * 0.5

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def get_self_state(self, input_ids, attention_mask=None):
        outputs = self.forward(
            input_ids,
            attention_mask=attention_mask,
            return_self_state=True,
            return_all_info=True,
        )
        return outputs.get('self_state')

    def push_goal(self, description: str, priority: float = 0.5) -> bool:
        if self._goal_stack is not None:
            goal = Goal(goal_id=f"g_{self._step_counter}", description=description, priority=priority)
            return self._goal_stack.push(goal)
        return False

    def forward_for_consolidation(self, x):
        """Simplified forward for offline consolidation."""
        for layer in self.layers:
            x, _, _ = layer(x, return_info=False)
        return x

    # =============================================================================
    # AGI Module Loss Functions
    # =============================================================================

    def _compute_symbolic_loss(self, symbolic_expr, device):
        """
        M1: Encourage symbolic expressions to have high confidence
        and discourage trivial (Empty) expressions.
        """
        if symbolic_expr is None:
            return None
        loss = torch.tensor(0.0, device=device)
        # Confidence should be high
        if hasattr(symbolic_expr, 'confidence'):
            loss = loss + (1.0 - symbolic_expr.confidence) ** 2
        # Non-empty expressions preferred
        if hasattr(symbolic_expr, 'head') and symbolic_expr.head == "Empty":
            loss = loss + 1.0
        return loss

    def _compute_self_loss(self, self_state, device):
        """
        M2: Self-consistency and narrative coherence.
        - Certainty should correlate with low prediction error
        - Self-state should not change too abruptly
        """
        if self_state is None:
            return None
        loss = torch.tensor(0.0, device=device)

        # Helper to get tensor value (works for both float and Tensor)
        def _t(val):
            if isinstance(val, torch.Tensor):
                return val
            return torch.tensor(val, device=device)

        # Temporal consistency: self-state should not drift too fast
        if self._prev_self_state is not None:
            if (hasattr(self_state, 'goal_embedding') and hasattr(self._prev_self_state, 'goal_embedding')
                and self_state.goal_embedding is not None and self._prev_self_state.goal_embedding is not None):
                drift = (self_state.goal_embedding - self._prev_self_state.goal_embedding).pow(2).mean()
                loss = loss + drift
            if hasattr(self_state, 'certainty') and hasattr(self._prev_self_state, 'certainty'):
                c1 = _t(self_state.certainty)
                c0 = _t(self._prev_self_state.certainty)
                loss = loss + (c1 - c0).pow(2)
            if hasattr(self_state, 'cognitive_load') and hasattr(self._prev_self_state, 'cognitive_load'):
                l1 = _t(self_state.cognitive_load)
                l0 = _t(self._prev_self_state.cognitive_load)
                loss = loss + (l1 - l0).pow(2)
            if hasattr(self_state, 'emotional_valence') and hasattr(self._prev_self_state, 'emotional_valence'):
                e1 = _t(self_state.emotional_valence)
                e0 = _t(self._prev_self_state.emotional_valence)
                loss = loss + (e1 - e0).pow(2)

        # Detach to avoid backward-through-graph in next batch
        from .agi_protocol import SelfState
        detached_state = SelfState(
            goal_embedding=self_state.goal_embedding.detach() if self_state.goal_embedding is not None else None,
            certainty=self_state.certainty.detach() if isinstance(self_state.certainty, torch.Tensor) else self_state.certainty,
            cognitive_load=self_state.cognitive_load.detach() if isinstance(self_state.cognitive_load, torch.Tensor) else self_state.cognitive_load,
            emotional_valence=self_state.emotional_valence.detach() if isinstance(self_state.emotional_valence, torch.Tensor) else self_state.emotional_valence,
            autobiographical_context=self_state.autobiographical_context.detach() if self_state.autobiographical_context is not None else None,
        )
        self._prev_self_state = detached_state

        # Cognitive load should stay in reasonable range [0.1, 0.9]
        if hasattr(self_state, 'cognitive_load'):
            load = _t(self_state.cognitive_load)
            loss = loss + (load - 0.5).pow(2) * 0.1  # encourage mid-range
            loss = loss + F.relu(0.1 - load).pow(2) * 0.5
            loss = loss + F.relu(load - 0.9).pow(2) * 0.5

        # Emotional valence should not be extreme
        if hasattr(self_state, 'emotional_valence'):
            emotion = _t(self_state.emotional_valence)
            loss = loss + emotion.pow(2) * 0.1

        # Certainty calibration: encourage moderate certainty
        if hasattr(self_state, 'certainty'):
            cert = _t(self_state.certainty)
            loss = loss + (cert - 0.7).pow(2) * 0.05

        return loss

    def _compute_action_loss(self, action_dist):
        """
        M3: Action regularization.
        - Saliency should not always be high (sparse actions)
        - Discrete logits should not be too peaked unless necessary
        """
        if action_dist is None:
            return None
        loss = torch.tensor(0.0)
        if action_dist.discrete_logits is not None:
            # Entropy regularization: encourage exploration
            probs = F.softmax(action_dist.discrete_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = loss - 0.1 * entropy  # Maximize entropy
        # Saliency sparsity: encourage low average saliency
        loss = loss + (action_dist.saliency ** 2) * 0.5
        return loss

    def _compute_plasticity_loss(self):
        """
        M5: Plasticity regularization.
        - Plasticity should stay in [0.1, 1.0]
        - Very low plasticity is penalized (don't freeze everything)
        """
        if self.elastic_plasticity is None:
            return None
        plasticity = self.elastic_plasticity.plasticity
        # Penalize extreme values
        too_low = (0.1 - plasticity).clamp(min=0.0)
        loss = too_low.pow(2).sum()
        return loss
