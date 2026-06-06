"""
CHAPPIE: Conscious Heuristic Adaptive Programming via Predictive Instant Encoding

A non-training paradigm for CORTEX that replaces gradient descent with:
    1. Knowledge Compilation (symbolic → neural weights)
    2. Hebbian One-Shot Plasticity (single interaction learning)
    3. Predictive Bootstrapping (self-supervised imagination)
    4. Metacognitive Self-Modification (code-as-consciousness)
    5. Multi-Timescale Instant Consolidation (working → long-term memory)

Inspired by the idea that consciousness and intelligence can emerge from
structured knowledge injection and self-organizing plasticity rather than
massive data training.
"""

from .knowledge_compiler import KnowledgeCompiler
from .hebbian_dcu import HebbianDCU, HebbianRule
from .workspace_bootstrapper import WorkspaceBootstrapper
from .self_modifying import SelfModifyingInterface, WeightEditor
from .oneshot_consolidator import OneShotConsolidator
from .chappie_model import ChappieCORTEX

__all__ = [
    "KnowledgeCompiler",
    "HebbianDCU",
    "HebbianRule",
    "WorkspaceBootstrapper",
    "SelfModifyingInterface",
    "WeightEditor",
    "OneShotConsolidator",
    "ChappieCORTEX",
]
