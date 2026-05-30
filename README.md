# CORTEX: Conscious Orchestrated Recurrent Transformer with Excitatory Integration

**Author:** tongriyao

---

## Overview

CORTEX is a fundamentally new neural network architecture that bridges three historically disjoint fields: **deep learning engineering**, **computational neuroscience**, and **cognitive science**. It is the first architecture to systematically integrate:

- **State Space Models** (e.g., Mamba/Mamba-2) for linear-time sequence modeling
- **Spiking Neural Networks (SNN)** for event-driven sparse computation
- **Global Workspace Theory** of consciousness from cognitive neuroscience
- **Biologically-plausible computation**: dendritic branching, predictive coding, and multi-timescale dynamics

Unlike conventional models that treat neural networks as pure function approximators, CORTEX is designed as a **dynamic system** that maintains and updates an explicit internal world model, much closer to how biological brains process information.

---

## Core Innovations

### 1. Global Neuronal Workspace (GNW) Layer
Inspired by Dehaene & Changeux's Global Neuronal Workspace theory of consciousness, this layer implements **competitive selection**, **threshold ignition**, and **global broadcasting**. Information undergoes a phase transition from local unconscious processing to globally available conscious content.

### 2. Dendritic Computation Unit (DCU)
Motivated by the electrophysiology of pyramidal neurons, each DCU contains multiple dendritic branches with **independent nonlinearities** (excitatory, inhibitory, rectifying, modulatory). This enables nonlinear spatial aggregation that is strictly more expressive than standard MLPs.

### 3. Hybrid Spike-Continuous Encoding
A novel training-inference decoupled design:
- **Training**: Uses differentiable soft spikes (probabilities) to maintain gradient flow
- **Inference**: Can be deployed as true sparse spikes on neuromorphic hardware (e.g., Intel Loihi)
- Combines the energy efficiency of SNNs with the trainability of ANNs

### 4. Integrated Predictive Coding
Every layer implements **prediction-error decomposition** with precision weighting, inspired by Friston's Free Energy Principle. The model learns not just to map inputs to outputs, but to predict the state of the layer above it, using prediction errors to drive learning.

### 5. Multi-Timescale Heterogeneous Processing
Different subpopulations operate at different timescales (fast ~25ms, medium ~100ms, slow ~500ms), coupled through cross-timescale connections. This naturally unifies short-term working memory and long-term contextual memory without external memory mechanisms.

### 6. Consciousness-Gated Attention
Attention weights are modulated by the global workspace consciousness state `C_t`, implementing **top-down cognitive control**. What the model is "thinking about" influences what it "pays attention to".

---

## Installation

```bash
pip install torch numpy
```

---

## Quick Start

```python
import torch
from cortex import CORTEXModel

# Create a CORTEX model
model = CORTEXModel(
    d_model=512,
    n_layers=12,
    n_modules=8,            # Number of local processing modules
    workspace_dim=256,      # Workspace dimension
    n_branches=4,           # Dendritic branches per unit
    n_timescales=3,         # Number of timescales
    vocab_size=50000,
    max_seq_len=4096,
    consciousness_output=True
)

# Forward pass
x = torch.randint(0, 50000, (2, 1024))  # (batch, seq_len)
outputs = model(x, return_consciousness=True)

logits = outputs['logits']
consciousness = outputs['consciousness']  # Extract the model's "conscious state"

# Generate text
generated = model.generate(
    input_ids=x[:, :8],
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

---

## Architecture Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete mathematical formulation, biological motivation, and design principles.

---

## Project Structure

```
cortex/
├── __init__.py
├── dendritic.py              # Dendritic Computation Units
├── spike_encoding.py         # Hybrid spike-continuous encoding
├── workspace.py              # Global Neuronal Workspace layer
├── predictive_coding.py      # Predictive coding layers
├── multiscale_state.py       # Multi-timescale state management
├── cortex_block.py           # CORTEX building blocks
└── cortex_model.py           # Full sequence model

tests/
└── test_cortex.py            # Unit tests

examples/
├── train_language_model.py   # Language modeling example
└── consciousness_analysis.py # Consciousness state analysis
```

---

## Key Design Philosophy

| Traditional Deep Learning | CORTEX |
|---|---|
| Function approximation | Dynamic system maintaining an internal model |
| Self-attention (all-to-all) | Global workspace (competition + broadcast) |
| Single timescale | Heterogeneous multi-timescale coupling |
| Dense computation | Spike-gated sparse computation |
| End-to-end backprop only | Local prediction-error signals + global task loss |

---

## Citation

If you use CORTEX in your research, please cite:

```bibtex
@article{cortex2025,
  title={CORTEX: A Biologically-Motivated Neural Architecture Integrating Global Workspace Theory, 
         Dendritic Computation, and Hybrid Spike-Continuous Dynamics},
  author={tongriyao},
  year={2025}
}
```

---

## Acknowledgments

This architecture synthesizes insights from:
- **Transformer/SSM research** (Vaswani et al., Gu & Dao, Dao & Gu)
- **Global Workspace Theory** (Baars, Dehaene & Changeux)
- **Predictive Processing** (Friston, Rao & Ballard)
- **Neuromorphic computing** (Maass, Indiveri, Zenke)
- **Dendritic computation** (Spruston, Larkum, Major)

---

*Designed and implemented by tongriyao.*
