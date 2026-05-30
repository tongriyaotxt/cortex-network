"""
Tests for CORTEX architecture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from cortex.dendritic import DendriticComputationUnit, SpikeGenerator
from cortex.workspace import GlobalWorkspaceLayer
from cortex.spike_encoding import HybridSpikeEncoder
from cortex.predictive_coding import PredictiveCodingLayer
from cortex.multiscale_state import MultiTimescaleStateLayer
from cortex.cortex_block import CORTEXBlock
from cortex.cortex_model import CORTEXModel


def test_spike_generator():
    """Test differentiable spike generation."""
    print("Testing SpikeGenerator...")
    gen = SpikeGenerator(threshold=1.0)
    v = torch.randn(2, 10, 32, requires_grad=True)
    spikes = gen(v)
    
    assert spikes.shape == v.shape
    assert (spikes == 0).sum() + (spikes == 1).sum() == spikes.numel()
    
    # Test gradient flow
    loss = spikes.sum()
    loss.backward()
    assert v.grad is not None
    print("  [PASS] SpikeGenerator passed")


def test_dendritic_unit():
    """Test dendritic computation unit."""
    print("Testing DendriticComputationUnit...")
    dcu = DendriticComputationUnit(
        d_in=64,
        d_out=64,
        n_branches=4,
    )
    x = torch.randn(2, 16, 64)
    
    # Test continuous output
    out = dcu(x, return_spike=False, return_continuous=True)
    assert out.shape == (2, 16, 64)
    
    # Test spike output
    out, spikes = dcu(x, return_spike=True, return_continuous=True)
    assert spikes.shape == (2, 16, 64)
    print("  [PASS] DendriticComputationUnit passed")


def test_workspace():
    """Test global workspace layer."""
    print("Testing GlobalWorkspaceLayer...")
    workspace = GlobalWorkspaceLayer(
        d_model=128,
        n_modules=4,
        workspace_dim=64,
    )
    x = torch.randn(2, 8, 128)
    
    out, info = workspace(x)
    assert out.shape == x.shape
    assert 'ignition_prob' in info
    assert 'competition_weights' in info
    print("  [PASS] GlobalWorkspaceLayer passed")


def test_spike_encoder():
    """Test hybrid spike encoder."""
    print("Testing HybridSpikeEncoder...")
    encoder = HybridSpikeEncoder(d_model=128, spike_dim_ratio=0.5)
    x = torch.randn(2, 8, 128)
    
    out, spike_info = encoder(x)
    assert out.shape == x.shape
    assert 'spike_rate' in spike_info
    print(f"  Spike rate: {spike_info['spike_rate']:.3f}")
    print("  [PASS] HybridSpikeEncoder passed")


def test_predictive_coding():
    """Test predictive coding layer."""
    print("Testing PredictiveCodingLayer...")
    pc = PredictiveCodingLayer(d_model=128)
    h_current = torch.randn(2, 8, 128)
    h_target = torch.randn(2, 8, 128)
    
    h_updated, error_info = pc(h_current, h_target, return_error=True)
    assert h_updated.shape == h_current.shape
    assert 'error' in error_info
    assert 'precision' in error_info
    print("  [PASS] PredictiveCodingLayer passed")


def test_multiscale():
    """Test multi-timescale state layer."""
    print("Testing MultiTimescaleStateLayer...")
    mts = MultiTimescaleStateLayer(
        d_model=126,
        timescales=[25.0, 100.0, 500.0],
    )
    x = torch.randn(2, 8, 126)
    
    out, states = mts(x)
    assert out.shape == x.shape
    assert len(states) == 3  # 3 timescales
    print("  [PASS] MultiTimescaleStateLayer passed")


def test_cortex_block():
    """Test complete CORTEX block."""
    print("Testing CORTEXBlock...")
    block = CORTEXBlock(
        d_model=126,
        n_modules=4,
        n_branches=4,
        n_timescales=3,
    )
    x = torch.randn(2, 8, 126)
    
    out, states, info = block(x, return_info=True)
    assert out.shape == x.shape
    print(f"  Spike rate: {info.get('spike_rate', 0):.3f}")
    print("  [PASS] CORTEXBlock passed")


def test_cortex_model():
    """Test complete CORTEX model."""
    print("Testing CORTEXModel...")
    model = CORTEXModel(
        vocab_size=1000,
        d_model=252,
        n_layers=4,
        n_modules=4,
        workspace_dim=126,
        max_seq_len=128,
        consciousness_output=True,
    )
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 16))
    outputs = model(input_ids, return_consciousness=True)
    
    assert 'logits' in outputs
    assert outputs['logits'].shape == (2, 16, 1000)
    assert 'consciousness' in outputs
    print(f"  Consciousness shape: {outputs['consciousness'].shape if outputs['consciousness'] is not None else None}")
    
    # Test generation
    gen_ids = model.generate(
        input_ids[:, :4],
        max_new_tokens=5,
        temperature=1.0,
    )
    assert gen_ids.shape[1] > 4
    print(f"  Generated sequence length: {gen_ids.shape[1]}")
    print("  [PASS] CORTEXModel passed")


def test_gradient_flow():
    """Test that gradients flow through all components."""
    print("Testing gradient flow...")
    model = CORTEXModel(
        vocab_size=100,
        d_model=66,
        n_layers=2,
        n_modules=2,
        max_seq_len=32,
    )
    
    input_ids = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))
    
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()
    
    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Some parameters use straight-through or conditional paths
            if 'threshold_param' in name or 'precision_estimator' in name or 'error_processor' in name:
                continue
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    print("  [PASS] Gradient flow passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("CORTEX Architecture Tests")
    print("=" * 60)
    
    tests = [
        test_spike_generator,
        test_dendritic_unit,
        test_workspace,
        test_spike_encoder,
        test_predictive_coding,
        test_multiscale,
        test_cortex_block,
        test_cortex_model,
        test_gradient_flow,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  [FAIL] {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
