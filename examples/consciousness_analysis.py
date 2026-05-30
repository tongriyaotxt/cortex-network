"""
Example: Analyzing consciousness states in CORTEX.

This demonstrates how to extract and visualize the model's
"consciousness state" for interpretability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from cortex.cortex_model import CORTEXModel


def analyze_consciousness():
    """Analyze consciousness states for different inputs."""
    print("Consciousness State Analysis")
    print("=" * 60)
    
    # Create model
    model = CORTEXModel(
        vocab_size=100,
        d_model=126,
        n_layers=4,
        n_modules=4,
        workspace_dim=63,
        max_seq_len=32,
        consciousness_output=True,
    )
    model.eval()
    
    # Create different input patterns
    patterns = {
        'random': torch.randint(0, 100, (1, 16)),
        'repeated': torch.tensor([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]),
        'alternating': torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]),
        'structured': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]]),
    }
    
    print("\nConsciousness states for different input patterns:")
    print("-" * 60)
    
    for name, input_ids in patterns.items():
        with torch.no_grad():
            outputs = model(
                input_ids,
                return_consciousness=True,
                return_all_info=True,
            )
        
        consciousness = outputs['consciousness']
        info = outputs.get('global_workspace', {})
        
        print(f"\nPattern: {name}")
        print(f"  Input: {input_ids[0].tolist()}")
        
        if consciousness is not None:
            print(f"  Consciousness shape: {consciousness.shape}")
            print(f"  Consciousness mean: {consciousness.mean().item():.4f}")
            print(f"  Consciousness std: {consciousness.std().item():.4f}")
            print(f"  Consciousness max: {consciousness.max().item():.4f}")
        
        # Analyze workspace info
        if isinstance(info, dict):
            if 'ignition_prob' in info:
                ig = info['ignition_prob']
                print(f"  Ignition prob mean: {ig.mean().item():.4f}")
                print(f"  Ignition prob max: {ig.max().item():.4f}")
            
            if 'competition_weights' in info:
                cw = info['competition_weights']
                print(f"  Competition entropy: {compute_entropy(cw).item():.4f}")
    
    # Spike statistics
    print("\n" + "-" * 60)
    print("Spike Statistics:")
    for name, input_ids in patterns.items():
        stats = model.get_spike_statistics(input_ids)
        print(f"  {name}: mean rate = {stats['mean_spike_rate']:.4f}")
    
    print("\n" + "=" * 60)


def compute_entropy(weights):
    """Compute entropy of competition weights."""
    weights = weights + 1e-8
    return -(weights * weights.log()).sum(dim=0).mean()


def compare_attention_patterns():
    """Compare how CORTEX processes different sequences."""
    print("\nAttention Pattern Comparison")
    print("=" * 60)
    
    model = CORTEXModel(
        vocab_size=50,
        d_model=66,
        n_layers=2,
        n_modules=2,
        workspace_dim=33,
        max_seq_len=16,
    )
    model.eval()
    
    # Compare two similar inputs
    input1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    input2 = torch.tensor([[1, 2, 3, 4, 10, 11, 12, 13]])  # Different second half
    
    with torch.no_grad():
        out1 = model(input1, return_all_info=True)
        out2 = model(input2, return_all_info=True)
    
    # Compare layer-wise spike rates
    print("\nLayer-wise spike rates:")
    for i, (info1, info2) in enumerate(zip(out1['layer_info'], out2['layer_info'])):
        r1 = info1.get('spike_rate', 0)
        r2 = info2.get('spike_rate', 0)
        print(f"  Layer {i}: input1={r1:.4f}, input2={r2:.4f}")
    
    print("=" * 60)


if __name__ == '__main__':
    analyze_consciousness()
    compare_attention_patterns()
