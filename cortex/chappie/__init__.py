"""
CHAPPIE: Conscious Heuristic Adaptive Programming via Predictive Instant Encoding

A non-training paradigm for CORTEX that replaces gradient descent with:
    1. Knowledge Compilation (symbolic → neural weights)
    2. Hebbian One-Shot Plasticity (single interaction learning)
    3. Predictive Bootstrapping (self-supervised imagination)
    4. Metacognitive Self-Modification (code-as-consciousness)
    5. Multi-Timescale Instant Consolidation (working → long-term memory)

Phase 1: Personality Engine — Emotion, Morality, Self-Narrative
Phase 2: Affective Computing — Face/Voice/Text emotion perception
Phase 3: Embodied Perception + Motor Control — Vision, Audio, Touch, Movement
Phase 4: Social Intelligence — Trust, Deception, Groups, Norms
Phase 5: Creative System — Art, Stories, Humor
Phase 6: Consciousness Transfer — Mind upload/download/backup
Phase 7: Self-Rewriting + Hardware Introspection — Code modification, body awareness
"""

from .knowledge_compiler import KnowledgeCompiler
from .hebbian_dcu import HebbianDCU, HebbianRule
from .workspace_bootstrapper import WorkspaceBootstrapper
from .self_modifying import SelfModifyingInterface, WeightEditor
from .oneshot_consolidator import OneShotConsolidator
from .chappie_model import ChappieCORTEX
from .personality_engine import EmotionalState, EmotionSystem, MoralRule, MoralDevelopmentModule, LifeEvent, SelfNarrativeGenerator
from .affective_computing import AffectivePerception, EmotionalExpressionGenerator, EmpathyModel
from .perception import VisionEncoder, AuditionEncoder, TactileEncoder, ProprioceptionEncoder, MultimodalFusion
from .motor_cortex import MotorPrimitive, MotorPrimitiveLibrary, MotorSequenceComposer, LowLevelController, FacialExpressionController, VocalController
from .social_intelligence import TrustModel, DeceptionDetector, GroupDynamicsModel, SocialNormLearner
from .creative_system import ArtGenerator, Storyteller, HumorGenerator
from .consciousness_transfer import MindStateSerializer, ConsciousnessBackupManager, BodyAdapter
from .self_rewriting import CodeIntrospection, SafeCodeModifier, NeuralArchitectureSearch
from .hardware_introspection import BodySchema, DamageDetector, ResourceMonitor, SelfRepairPlanner

__all__ = [
    # Core non-training
    "KnowledgeCompiler",
    "HebbianDCU",
    "HebbianRule",
    "WorkspaceBootstrapper",
    "SelfModifyingInterface",
    "WeightEditor",
    "OneShotConsolidator",
    "ChappieCORTEX",
    # Phase 1: Personality
    "EmotionalState",
    "EmotionSystem",
    "MoralRule",
    "MoralDevelopmentModule",
    "LifeEvent",
    "SelfNarrativeGenerator",
    # Phase 2: Affective
    "AffectivePerception",
    "EmotionalExpressionGenerator",
    "EmpathyModel",
    # Phase 3: Perception + Motor
    "VisionEncoder",
    "AuditionEncoder",
    "TactileEncoder",
    "ProprioceptionEncoder",
    "MultimodalFusion",
    "MotorPrimitive",
    "MotorPrimitiveLibrary",
    "MotorSequenceComposer",
    "LowLevelController",
    "FacialExpressionController",
    "VocalController",
    # Phase 4: Social
    "TrustModel",
    "DeceptionDetector",
    "GroupDynamicsModel",
    "SocialNormLearner",
    # Phase 5: Creative
    "ArtGenerator",
    "Storyteller",
    "HumorGenerator",
    # Phase 6: Transfer
    "MindStateSerializer",
    "ConsciousnessBackupManager",
    "BodyAdapter",
    # Phase 7: Self-rewriting + Hardware
    "CodeIntrospection",
    "SafeCodeModifier",
    "NeuralArchitectureSearch",
    "BodySchema",
    "DamageDetector",
    "ResourceMonitor",
    "SelfRepairPlanner",
]
