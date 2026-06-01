"""
CORTEX: Conscious Orchestrated Recurrent Transformer with Excitatory Integration

A biologically-motivated neural architecture integrating:
- Global Workspace Theory (consciousness modeling)
- Dendritic computation
- Hybrid spike-continuous encoding
- Predictive coding
- Multi-timescale heterogeneous processing

AGI Extensions (v1.0):
- M1: Symbolic Reasoning
- M2: Self-Modeling
- M3: Embodied Interaction
- M4: Hierarchical Planning
- M5: Continual Learning
- M6: Causal Inference
"""

# Core components
from .cortex_model import CORTEXModel
from .dendritic import DendriticComputationUnit
from .workspace import GlobalWorkspaceLayer
from .spike_encoding import HybridSpikeEncoder
from .predictive_coding import PredictiveCodingLayer
from .multiscale_state import MultiTimescaleStateLayer

# AGI Extension Modules
from .agi_protocol import (
    WorkspacePacket, InternalSignals, Action, SelfState,
    SymbolicToken, SymbolicExpression, Goal, CausalVariable,
)
from .symbolic import SymbolicBranch, SymbolicWorkspace
from .self_modeling import SelfModule, AutobiographicalMemory, MetacognitiveMonitor
from .action import ActionLPM, ActionHead
from .forward_model import ForwardModel
from .interoception import InteroceptionChannel
from .env_wrapper import EnvironmentWrapper, CORTEXEnvWrapper, MockEnvironment
from .goal import GoalStack
from .hierarchical_workspace import HierarchicalWorkspace
from .subroutine import SubroutineLPM
from .branch_isolation import BranchAllocator, ElasticPlasticity
from .memory_cabinet import MemoryCabinet
from .consolidation import OfflineConsolidation
from .causal import CausalGraph
from .counterfactual import CounterfactualWorkspace
from .causal_discovery import CausalDiscoveryLPM

# Integrated AGI Model
from .agi_cortex_model import AGICORTEXModel

__all__ = [
    # Core
    'CORTEXModel',
    'DendriticComputationUnit',
    'GlobalWorkspaceLayer',
    'HybridSpikeEncoder',
    'PredictiveCodingLayer',
    'MultiTimescaleStateLayer',
    # AGI Protocol
    'WorkspacePacket',
    'InternalSignals',
    'Action',
    'SelfState',
    'SymbolicToken',
    'SymbolicExpression',
    'Goal',
    'CausalVariable',
    # M1 Symbolic
    'SymbolicBranch',
    'SymbolicWorkspace',
    # M2 Self-Modeling
    'SelfModule',
    'AutobiographicalMemory',
    'MetacognitiveMonitor',
    # M3 Embodied
    'ActionLPM',
    'ActionHead',
    'ForwardModel',
    'InteroceptionChannel',
    'EnvironmentWrapper',
    'CORTEXEnvWrapper',
    'MockEnvironment',
    # M4 Hierarchical
    'GoalStack',
    'HierarchicalWorkspace',
    'SubroutineLPM',
    # M5 Continual
    'BranchAllocator',
    'ElasticPlasticity',
    'MemoryCabinet',
    'OfflineConsolidation',
    # M6 Causal
    'CausalGraph',
    'CounterfactualWorkspace',
    'CausalDiscoveryLPM',
    # Integrated
    'AGICORTEXModel',
]

__version__ = '1.0.0'
