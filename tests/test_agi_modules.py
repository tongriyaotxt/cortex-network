"""
Tests for AGI-CORTEX extension modules (M1-M6).

Run with:
    python tests/test_agi_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from cortex import AGICORTEXModel
from cortex.agi_protocol import (
    SymbolicToken, SymbolicExpression, Goal, ActionSpace, InternalSignals,
)
from cortex.symbolic import SymbolicBranch, SymbolicWorkspace
from cortex.self_modeling import SelfModule, AutobiographicalMemory, MetacognitiveMonitor
from cortex.action import ActionLPM, ActionHead
from cortex.forward_model import ForwardModel
from cortex.goal import GoalStack
from cortex.hierarchical_workspace import HierarchicalWorkspace
from cortex.branch_isolation import BranchAllocator, ElasticPlasticity
from cortex.memory_cabinet import MemoryCabinet
from cortex.causal import CausalGraph
from cortex.counterfactual import CounterfactualWorkspace
from cortex.causal_discovery import CausalDiscoveryLPM


def test_symbolic_branch():
    """Test M1: SymbolicBranch VQ quantization."""
    print("Testing SymbolicBranch...")
    branch = SymbolicBranch(d_in=64, d_branch=32, codebook_size=128)
    x = torch.randn(2, 8, 64)
    quantized, tokens = branch(x)
    assert quantized.shape == (2, 8, 32)
    assert len(tokens) == 2 * 8
    assert all(isinstance(t, SymbolicToken) for t in tokens)
    print(f"  Quantized shape: {quantized.shape}, Tokens: {len(tokens)}")
    print("  [PASS] SymbolicBranch passed")


def test_symbolic_workspace():
    """Test M1: SymbolicWorkspace end-to-end."""
    print("Testing SymbolicWorkspace...")
    ws = SymbolicWorkspace(d_model=128, vocab_size=256)
    x = torch.randn(2, 8, 128)
    residual, tokens, expr = ws(x)
    assert residual.shape == (2, 8, 128)
    assert len(tokens) == 2 * 8
    assert isinstance(expr, SymbolicExpression)
    print(f"  Residual shape: {residual.shape}, Expression head: {expr.head}")
    print("  [PASS] SymbolicWorkspace passed")


def test_self_module():
    """Test M2: SelfModule produces SelfState."""
    print("Testing SelfModule...")
    module = SelfModule(d_model=64)
    signals = InternalSignals()
    signals.workspace_state = torch.randn(2, 64)
    signals.spike_rates = {'layer_0': 0.1, 'layer_1': 0.2}
    signals.prediction_errors = {'layer_0': 0.05}
    
    self_state, saliency = module(signals)
    assert hasattr(self_state, 'certainty')
    assert hasattr(self_state, 'cognitive_load')
    assert hasattr(self_state, 'emotional_valence')
    assert 0 <= self_state.certainty <= 1
    assert saliency.numel() == 2
    print(f"  Certainty: {self_state.certainty:.3f}, Load: {self_state.cognitive_load:.3f}")
    print("  [PASS] SelfModule passed")


def test_autobiographical_memory():
    """Test M2: AutobiographicalMemory store and retrieve."""
    print("Testing AutobiographicalMemory...")
    mem = AutobiographicalMemory(d_event=64, capacity=100)
    
    # Store events
    for i in range(10):
        mem.encode_event(
            consciousness=torch.randn(64),
            action=torch.randn(64),
            outcome=0.5,
            timestamp=i,
        )
    
    # Retrieve
    query = torch.randn(64)
    results = mem.retrieve_similar(query, k=3)
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    print(f"  Stored: 10, Retrieved: {len(results)}")
    print("  [PASS] AutobiographicalMemory passed")


def test_action_lpm():
    """Test M3: ActionLPM outputs ActionDistribution."""
    print("Testing ActionLPM...")
    action_space = ActionSpace(n_actions=5, n_continuous=2)
    lpm = ActionLPM(d_model=64, action_space=action_space)
    intention = torch.randn(2, 8, 64)
    action_dist, saliency = lpm(intention)
    assert action_dist.discrete_logits is not None
    assert action_dist.discrete_logits.shape == (2, 5)
    assert action_dist.continuous_params is not None
    assert action_dist.continuous_params.shape == (2, 2)
    print(f"  Discrete: {action_dist.discrete_logits.shape}, Saliency: {saliency.shape}")
    print("  [PASS] ActionLPM passed")


def test_forward_model():
    """Test M3: ForwardModel predicts next state."""
    print("Testing ForwardModel...")
    fm = ForwardModel(d_state=64, d_action=5)
    state = torch.randn(2, 64)
    action = torch.randn(2, 5)
    next_state, uncertainty = fm.predict_next_state(state, action)
    assert next_state.shape == (2, 64)
    assert uncertainty.shape == (2, 64)
    
    # Test imagination
    actions = [torch.randn(5) for _ in range(3)]
    traj = fm.imagine_trajectory(state[0], actions)
    assert len(traj) == 3
    print(f"  Next state: {next_state.shape}, Trajectory length: {len(traj)}")
    print("  [PASS] ForwardModel passed")


def test_goal_stack():
    """Test M4: GoalStack push/pop/decompose."""
    print("Testing GoalStack...")
    stack = GoalStack(max_depth=3)
    goal = Goal(goal_id="g1", description="write_paper", priority=0.8)
    assert stack.push(goal) == True
    assert stack.depth() == 1
    
    # Decompose
    from cortex.agi_protocol import DecompositionStrategy
    strategy = DecompositionStrategy(max_children=3)
    sub_goals = stack.decompose(goal, strategy)
    assert len(sub_goals) > 0
    assert all(sg.parent == "g1" for sg in sub_goals)
    
    # Pop
    popped = stack.pop()
    assert popped.goal_id == "g1"
    assert stack.is_empty()
    print(f"  Sub-goals: {len(sub_goals)}, Stack empty: {stack.is_empty()}")
    print("  [PASS] GoalStack passed")


def test_hierarchical_workspace():
    """Test M4: HierarchicalWorkspace with child workspaces."""
    print("Testing HierarchicalWorkspace...")
    hw = HierarchicalWorkspace(d_model=64, n_modules=4, max_depth=3)
    x = torch.randn(2, 8, 64)
    
    # Without goals: should work like normal workspace
    out, info = hw(x, return_info=True)
    assert out.shape == x.shape
    
    # With goals: should create child workspaces
    goal = Goal(goal_id="g_test", description="test_goal")
    goal.embedding = torch.randn(64)
    hw.push_goal(goal)
    out2, info2 = hw(x, return_info=True)
    assert out2.shape == x.shape
    assert info2['goal_stack_depth'] == 1
    assert 'g_test' in info2['active_goals']
    print(f"  Output shape: {out2.shape}, Active goals: {info2['active_goals']}")
    print("  [PASS] HierarchicalWorkspace passed")


def test_branch_allocator():
    """Test M5: BranchAllocator assigns branches."""
    print("Testing BranchAllocator...")
    allocator = BranchAllocator(n_branches=4)
    mask = allocator.allocate("task_A", required_branches=2)
    assert mask.mask.sum() == 2
    
    mask2 = allocator.allocate("task_B", required_branches=2)
    # Should get different branches
    overlap = (mask.mask * mask2.mask).sum()
    print(f"  Task A branches: {mask.mask.tolist()}, Task B: {mask2.mask.tolist()}, Overlap: {overlap}")
    print("  [PASS] BranchAllocator passed")


def test_elastic_plasticity():
    """Test M5: ElasticPlasticity modulates learning rates."""
    print("Testing ElasticPlasticity...")
    ep = ElasticPlasticity(n_branches=4)
    from cortex.agi_protocol import UsageStats
    stats = UsageStats(activation_frequency=0.1, recency=10, performance_trend=-0.5)
    p = ep.compute_plasticity(0, stats)
    assert 0 <= p <= 1
    ep.consolidate([0, 1])
    assert ep.plasticity[0] < 1.0
    print(f"  Plasticity: {ep.plasticity.tolist()}")
    print("  [PASS] ElasticPlasticity passed")


def test_memory_cabinet():
    """Test M5: MemoryCabinet archive and retrieve."""
    print("Testing MemoryCabinet...")
    cabinet = MemoryCabinet(d_memory=64, capacity=100)
    
    for i in range(20):
        cabinet.archive(
            slow_state=torch.randn(64),
            context_tag=f"task_{i % 3}",
            importance=0.5 + 0.5 * (i % 2),
            timestamp=i,
        )
    
    query = torch.randn(64)
    results = cabinet.retrieve(query, context_hint="task_0", k=5)
    assert len(results) > 0
    print(f"  Archived: 20, Retrieved: {len(results)}")
    print("  [PASS] MemoryCabinet passed")


def test_causal_graph():
    """Test M6: CausalGraph with do-operator."""
    print("Testing CausalGraph...")
    from cortex.agi_protocol import CausalVariable, SymbolicToken
    
    graph = CausalGraph()
    v1 = CausalVariable(var_id="rain", embedding=torch.randn(16))
    v2 = CausalVariable(var_id="wet", embedding=torch.randn(16))
    graph.add_variable(v1)
    graph.add_variable(v2)
    graph.add_edge("rain", "wet", strength=0.9)
    
    # Do-operator
    token = SymbolicToken(token_id=0, embedding=torch.randn(16))
    intervened = graph.do("rain", token)
    assert ("rain", "wet") in intervened.edges
    assert intervened.nodes["rain"].possible_values == [token]
    
    # Query
    effect = graph.query({"rain": token}, outcome_var="wet")
    assert effect.effect_on_var == "wet"
    print(f"  Effect on 'wet': {effect.mean_delta:.4f} ± {effect.uncertainty:.4f}")
    print("  [PASS] CausalGraph passed")


def test_counterfactual_workspace():
    """Test M6: CounterfactualWorkspace runs parallel worlds."""
    print("Testing CounterfactualWorkspace...")
    from cortex.workspace import GlobalWorkspaceLayer
    from cortex.agi_protocol import SymbolicToken
    
    base_ws = GlobalWorkspaceLayer(d_model=64, n_modules=4)
    cf_ws = CounterfactualWorkspace(base_ws, n_counterfactuals=2)
    
    state = torch.randn(2, 8, 64)
    intervention = {"var_0": SymbolicToken(token_id=0, embedding=torch.randn(64))}
    
    out, info = cf_ws.run_counterfactual(intervention, state, cf_index=0)
    assert out.shape == state.shape
    print(f"  Counterfactual output shape: {out.shape}")
    print("  [PASS] CounterfactualWorkspace passed")


def test_causal_discovery():
    """Test M6: CausalDiscoveryLPM infers graphs."""
    print("Testing CausalDiscoveryLPM...")
    cd = CausalDiscoveryLPM(d_model=64, n_variables=3)
    
    # Create synthetic time series: var_0 causes var_1
    history = []
    for t in range(10):
        v0 = torch.randn(3, 64)
        v0[1] = v0[0] * 0.8 + torch.randn(64) * 0.2  # var_1 depends on var_0
        history.append(v0)
    
    graph = cd(history)
    assert len(graph.nodes) == 3
    print(f"  Discovered nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")
    print("  [PASS] CausalDiscoveryLPM passed")


def test_agi_cortex_model_forward():
    """Test integrated AGICORTEXModel with all modules."""
    print("Testing AGICORTEXModel (all modules ON)...")
    model = AGICORTEXModel(
        vocab_size=100,
        d_model=66,
        n_layers=2,
        n_modules=4,
        max_seq_len=32,
        use_symbolic=True,
        use_self_modeling=True,
        use_embodied=True,
        use_hierarchical=True,
        use_continual=True,
        use_causal=True,
    )
    
    x = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))
    
    # Training forward
    model.train()
    outputs = model(x, labels=labels)
    assert 'loss' in outputs
    assert 'logits' in outputs
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    # Backward
    outputs['loss'].backward()
    
    # Check that core components have gradients
    # (Not all AGI module params get grads every step because some outputs
    #  are stored as Python floats in dataclasses, not tensors)
    # Check that the main output path has gradients
    # Some parameters may legitimately have zero grad due to:
    # - Sparse spikes (dendritic_ff.2 gets zero input if no spikes fire)
    # - Detached memory operations (autobiographical memory stores with .detach())
    core_paths = [
        'token_embedding',
        'output_head',
        'layers.0.dendritic_attn',
        'layers.1.dendritic_attn',
        'symbolic_workspace.broadcast_net',
        'action_head.action_lpm.discrete_head',
    ]
    failed = []
    for path in core_paths:
        found = False
        for name, p in model.named_parameters():
            if path in name and p.requires_grad:
                found = True
                if p.grad is None or p.grad.abs().sum() == 0:
                    failed.append(name)
        if not found:
            failed.append(f"Path not found: {path}")
    
    assert not failed, f"Core parameters without gradient: {failed}"
    
    # Eval forward with all returns
    model.eval()
    with torch.no_grad():
        outputs = model(
            x,
            return_self_state=True,
            return_action=True,
            return_symbolic=True,
            return_all_info=True,
        )
    assert 'self_state' in outputs
    assert 'action_distribution' in outputs
    assert 'symbolic_tokens' in outputs
    print(f"  Self certainty: {outputs['self_state'].certainty:.3f}")
    print(f"  Action saliency: {outputs['action_distribution'].saliency:.3f}")
    
    # Generation with AGI modules
    with torch.no_grad():
        gen = model.generate(x[:, :4], max_new_tokens=3, use_agi_modules=True)
    assert gen.shape[1] > 4
    print(f"  Generated length: {gen.shape[1]}")
    print("  [PASS] AGICORTEXModel passed")


def test_agi_cortex_model_modules_off():
    """Test backward compatibility: all modules OFF."""
    print("Testing AGICORTEXModel (all modules OFF)...")
    model = AGICORTEXModel(
        vocab_size=100,
        d_model=66,
        n_layers=2,
        max_seq_len=32,
        use_symbolic=False,
        use_self_modeling=False,
        use_embodied=False,
        use_hierarchical=False,
        use_continual=False,
        use_causal=False,
    )
    
    x = torch.randint(0, 100, (2, 8))
    outputs = model(x)
    assert outputs['logits'].shape == (2, 8, 100)
    print("  [PASS] AGICORTEXModel (modules off) passed")


def run_all_tests():
    print("=" * 70)
    print("AGI-CORTEX Extension Module Tests")
    print("=" * 70)
    
    tests = [
        test_symbolic_branch,
        test_symbolic_workspace,
        test_self_module,
        test_autobiographical_memory,
        test_action_lpm,
        test_forward_model,
        test_goal_stack,
        test_hierarchical_workspace,
        test_branch_allocator,
        test_elastic_plasticity,
        test_memory_cabinet,
        test_causal_graph,
        test_counterfactual_workspace,
        test_causal_discovery,
        test_agi_cortex_model_forward,
        test_agi_cortex_model_modules_off,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
