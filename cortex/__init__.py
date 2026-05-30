"""
CORTEX: Conscious Orchestrated Recurrent Transformer with Excitatory Integration

A biologically-motivated neural architecture integrating:
- Global Workspace Theory (consciousness modeling)
- Dendritic computation
- Hybrid spike-continuous encoding
- Predictive coding
- Multi-timescale heterogeneous processing
"""

from .cortex_model import CORTEXModel
from .dendritic import DendriticComputationUnit
from .workspace import GlobalWorkspaceLayer
from .spike_encoding import HybridSpikeEncoder
from .predictive_coding import PredictiveCodingLayer
from .multiscale_state import MultiTimescaleStateLayer

__all__ = [
    'CORTEXModel',
    'DendriticComputationUnit',
    'GlobalWorkspaceLayer',
    'HybridSpikeEncoder',
    'PredictiveCodingLayer',
    'MultiTimescaleStateLayer',
]

__version__ = '0.1.0'
