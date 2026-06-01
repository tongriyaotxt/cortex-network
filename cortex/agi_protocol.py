"""
AGI-CORTEX Unified Data Protocol

Defines the common data structures and interfaces used by all AGI extension
modules (M1-M6). These types serve as the communication fabric between
symbolic reasoning, self-modeling, embodied interaction, hierarchical planning,
continual learning, and causal inference components.

All new modules should operate on these types to ensure interoperability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
import torch
from torch import Tensor


# =============================================================================
# M1: Symbolic Reasoning Primitives
# =============================================================================

@dataclass
class SymbolicToken:
    """
    A discrete symbol atom that can travel through the Global Workspace
    alongside continuous embeddings.
    """
    token_id: int
    embedding: Tensor  # (d_embed,)
    confidence: float = 1.0  # [0, 1]
    source_module: int = -1  # Which LPM produced this token


@dataclass
class SymbolicExpression:
    """
    A composed symbolic structure, e.g.  Color(Object=APPLE, Value=RED).
    Implemented as a tree for recursive composition.
    """
    head: str  # e.g. "Color", "And", "Not"
    args: List[Any] = field(default_factory=list)
    # args may contain SymbolicToken, SymbolicExpression, or raw values
    confidence: float = 1.0


@dataclass
class RewriteRule:
    """Pattern-directed rewrite rule for symbolic computation."""
    pattern: SymbolicExpression
    replacement: SymbolicExpression
    condition: Optional[callable] = None  # Optional guard function


# =============================================================================
# M2: Self-Modeling Primitives
# =============================================================================

@dataclass
class SelfState:
    """
    The model's explicit representation of "who I am / what I am doing".
    """
    goal_embedding: Tensor          # (d_goal,)
    certainty: float = 0.5          # [0, 1]
    cognitive_load: float = 0.0     # [0, 1]
    emotional_valence: float = 0.0  # [-1, 1]
    autobiographical_context: Optional[Tensor] = None  # (d_auto,)


@dataclass
class EventRecord:
    """A single autobiographical event."""
    timestamp: int
    consciousness: Tensor           # (d_model,) snapshot of consciousness
    action: Optional[Tensor] = None
    outcome: float = 0.0            # [-1, 1]  reward / result quality
    importance: float = 0.5         # [0, 1]  consolidation priority
    tags: List[str] = field(default_factory=list)


@dataclass
class InformationRequest:
    """Emitted when the model detects an information gap."""
    gap_description: str
    target_variable: Optional[str] = None
    urgency: float = 0.5            # [0, 1]


# =============================================================================
# M3: Embodied Interaction Primitives
# =============================================================================

@dataclass
class ActionSpace:
    """Definition of all actions the model can perform."""
    n_actions: int = 0
    n_continuous: int = 0
    action_types: List[str] = field(default_factory=list)


@dataclass
class ActionDistribution:
    """Stochastic action output."""
    discrete_logits: Optional[Tensor] = None      # (n_actions,)
    continuous_params: Optional[Tensor] = None    # (n_continuous,)
    saliency: float = 0.0                         # urgency of the action


@dataclass
class Action:
    """Unified action representation (discrete + continuous)."""
    discrete_action: Optional[int] = None
    continuous_params: Optional[Tensor] = None
    action_type: str = "noop"
    confidence: float = 1.0
    expected_outcome: Optional[Tensor] = None
    causal_effect_estimate: Optional[float] = None


@dataclass
class SystemState:
    """Snapshot of the whole system for interoception."""
    workspace_state: Optional[Tensor] = None
    layer_errors: Dict[str, float] = field(default_factory=dict)
    spike_rates: Dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    energy_consumption: float = 0.0
    memory_retrieval_hits: int = 0
    branch_usage: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# M4: Hierarchical Planning Primitives
# =============================================================================

@dataclass
class Goal:
    """A node in the hierarchical goal tree."""
    goal_id: str
    description: str = ""
    embedding: Optional[Tensor] = None
    parent: Optional[str] = None      # parent goal_id
    children: List[str] = field(default_factory=list)
    status: Literal['pending', 'active', 'completed', 'failed'] = 'pending'
    priority: float = 0.5
    deadline: Optional[int] = None


@dataclass
class DecompositionStrategy:
    """Strategy for breaking a goal into sub-goals."""
    strategy_type: str = "sequential"  # sequential / parallel / conditional
    max_children: int = 5
    # Additional heuristics can be added here


@dataclass
class WorkspaceContext:
    """Context passed to a subroutine LPM."""
    caller_goal: Optional[str] = None
    parent_workspace_id: Optional[str] = None
    available_memory_keys: List[str] = field(default_factory=list)


# =============================================================================
# M5: Continual Learning Primitives
# =============================================================================

@dataclass
class BranchMask:
    """Which dendritic branches are active for a given task."""
    task_id: str
    mask: Tensor                    # (n_branches,) bool or float [0,1]
    usage_count: int = 0
    performance_history: List[float] = field(default_factory=list)


@dataclass
class UsageStats:
    """Statistics for branch usage, driving plasticity."""
    activation_frequency: float = 0.0
    recency: int = 0                # steps since last use
    performance_trend: float = 0.0  # slope of recent performance


@dataclass
class MemoryKey:
    """Reference key for a stored memory."""
    key_id: str
    context_tag: str = ""
    importance: float = 0.5


@dataclass
class MemoryRecord:
    """Retrieved memory content."""
    key: MemoryKey
    slow_state: Tensor
    similarity_score: float = 0.0


# =============================================================================
# M6: Causal Inference Primitives
# =============================================================================

@dataclass
class CausalVariable:
    """A node in the causal graph."""
    var_id: str
    embedding: Optional[Tensor] = None
    possible_values: List[SymbolicToken] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


@dataclass
class CausalEffect:
    """Estimated causal effect of an intervention."""
    effect_on_var: str
    mean_delta: float = 0.0
    uncertainty: float = 0.0
    n_samples: int = 0


# =============================================================================
# Unified Workspace Packet (cross-module communication)
# =============================================================================

@dataclass
class WorkspacePacket:
    """
    The canonical data structure that flows through the extended Global
    Workspace. Every module can attach its payload; receivers inspect only
    the fields they care about.
    """

    # --- Core continuous content (always present) ---
    continuous: Tensor                      # (batch, seq_len, d_model)

    # --- M1: Symbolic reasoning ---
    symbolic: Optional[List[SymbolicToken]] = None
    symbolic_expression: Optional[SymbolicExpression] = None

    # --- M2: Self-modeling ---
    self_state: Optional[SelfState] = None
    autobiographical_hint: Optional[Tensor] = None

    # --- M3: Embodied interaction ---
    action_distribution: Optional[ActionDistribution] = None
    imagined_outcome: Optional[Tensor] = None

    # --- M4: Hierarchical planning ---
    goal_context: Optional[Goal] = None
    subroutine_call: Optional[str] = None

    # --- M5: Continual learning ---
    memory_key: Optional[MemoryKey] = None
    consolidation_request: bool = False

    # --- M6: Causal inference ---
    causal_graph: Optional[Any] = None      # Lazy: actual type in causal.py
    intervention: Optional[Dict[str, SymbolicToken]] = None
    counterfactual_tag: Optional[str] = None

    # --- Metadata ---
    timestamp: int = 0
    source_module: str = "unknown"
    saliency: float = 0.0
    ignition: bool = False

    def has_symbolic(self) -> bool:
        return self.symbolic is not None or self.symbolic_expression is not None

    def has_self(self) -> bool:
        return self.self_state is not None

    def has_action(self) -> bool:
        return self.action_distribution is not None

    def has_goal(self) -> bool:
        return self.goal_context is not None

    def has_memory(self) -> bool:
        return self.memory_key is not None or self.consolidation_request

    def has_causal(self) -> bool:
        return self.causal_graph is not None or self.intervention is not None


# =============================================================================
# Internal Signals (fed to SelfModule)
# =============================================================================

@dataclass
class InternalSignals:
    """Comprehensive snapshot of internal system state."""
    workspace_state: Optional[Tensor] = None
    goal_stack: List[Goal] = field(default_factory=list)
    prediction_errors: Dict[str, float] = field(default_factory=dict)
    spike_rates: Dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    energy_consumption: float = 0.0
    memory_retrieval_hits: int = 0
    branch_usage: Dict[str, int] = field(default_factory=dict)
