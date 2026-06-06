"""
CHAPPIE Demo: Non-Training CORTEX in Action

This demo shows how a CORTEX model can learn and adapt WITHOUT
training (no gradient descent, no backprop, no dataset).

Scenarios demonstrated:
    1. Knowledge Upload: Inject rules and facts directly into weights
    2. One-Shot Learning: Learn from a single demonstration
    3. Daydreaming: Self-supervised imagination for bootstrapping
    4. Self-Healing: Diagnose and fix model issues
    5. Memory Retrieval: Generate with episodic memory influence

Run:
    python examples/chappie_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from cortex.chappie import (
    ChappieCORTEX,
    KnowledgeCompiler,
    HebbianDCU,
    WorkspaceBootstrapper,
    SelfModifyingInterface,
    OneShotConsolidator,
)


def demo_knowledge_upload():
    """Scenario 1: Upload knowledge like consciousness transfer."""
    print("\n" + "="*60)
    print("SCENARIO 1: Knowledge Upload (Consciousness Transfer)")
    print("="*60)
    
    # Create a simple DCU for demonstration
    class SimpleDCU(nn.Module):
        def __init__(self, d_in=64, d_out=64, n_branches=4):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Linear(d_in, d_out) for _ in range(n_branches)
            ])
            self.n_branches = n_branches
    
    dcu = SimpleDCU(d_in=64, d_out=64, n_branches=4)
    
    # Compile knowledge
    compiler = KnowledgeCompiler(d_model=64, n_branches=4)
    
    print("Compiling rules...")
    compiler.compile_rule(
        premises=["human", "falling"],
        conclusion="catch",
        confidence=0.95
    )
    compiler.compile_rule(
        premises=["threat", "nearby"],
        conclusion="defend",
        confidence=0.9
    )
    
    print("Compiling facts...")
    compiler.compile_fact(
        subject="gravity",
        relation="causes",
        object="falling",
        confidence=1.0
    )
    compiler.compile_fact(
        subject="Chappie",
        relation="is_a",
        object="robot",
        confidence=1.0
    )
    
    print("Compiling procedures...")
    compiler.compile_procedure(
        name="catch_human",
        steps=["detect_fall", "move_fast", "extend_arms", "absorb_impact"],
        motor_outputs=[0.9, 0.95, 0.8, 0.7],
        confidence=0.9
    )
    
    # Apply to DCU
    print("Injecting compiled knowledge into DCU weights...")
    compiler.apply_to_dcu(dcu)
    
    # Check that weights changed
    weight_norms = [b.weight.data.norm().item() for b in dcu.branches]
    print(f"Branch weight norms after injection: {[f'{w:.3f}' for w in weight_norms]}")
    print("Knowledge upload complete!")
    
    return compiler


def demo_one_shot_learning():
    """Scenario 2: Learn from a single interaction."""
    print("\n" + "="*60)
    print("SCENARIO 2: One-Shot Hebbian Learning")
    print("="*60)
    
    # Simple DCU with Hebbian plasticity
    class SimpleDCU(nn.Module):
        def __init__(self, d_in=32, d_out=32):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Linear(d_in, d_out) for _ in range(2)
            ])
            self.n_branches = 2
        
        def forward(self, x):
            return sum(b(x) for b in self.branches) / len(self.branches)
    
    base_dcu = SimpleDCU(d_in=32, d_out=32)
    hebbian_dcu = HebbianDCU(base_dcu, rule="oja", lr=0.05)
    
    # Show initial weights
    init_norms = [b.weight.data.norm().item() for b in base_dcu.branches]
    print(f"Initial branch norms: {[f'{w:.3f}' for w in init_norms]}")
    
    # Single demonstration: a specific pattern
    pattern = torch.randn(1, 32)
    target_response = torch.randn(1, 32) * 2.0  # Strong signal
    
    print("Showing pattern once...")
    for _ in range(5):  # A few repetitions to let Hebbian trace build
        output = hebbian_dcu(pattern)
        # Simulate the target by providing it as "post-synaptic" signal
        # In reality, the Hebbian update happens automatically during forward
    
    # Check specialization
    specs = hebbian_dcu.get_branch_specialization()
    print(f"Branch specialization after one-shot exposure:")
    for spec in specs:
        print(f"  Branch {spec['branch']}: selectivity={spec['selectivity']:.4f}, "
              f"sparsity={spec['sparsity']:.2%}, steps={spec['timestep']}")
    
    # Consolidate
    hebbian_dcu.consolidate(consolidation_factor=0.5)
    print("Memory consolidated!")
    
    return hebbian_dcu


def demo_self_healing():
    """Scenario 3: Self-diagnosis and repair."""
    print("\n" + "="*60)
    print("SCENARIO 3: Self-Healing (Introspection & Repair)")
    print("="*60)
    
    # Create a model with some problems
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 64)
            self.dcu = nn.Module()
            self.dcu.branches = nn.ModuleList([
                nn.Linear(64, 64),
                nn.Linear(64, 64),
            ])
    
    model = SimpleModel()
    
    # Introduce problems
    # 1. Dead branch (near-zero weights)
    model.dcu.branches[0].weight.data *= 0.0001
    # 2. Exploding weights
    model.layer1.weight.data[0, :] *= 100.0
    
    print("Injected problems: dead branch + exploding weights")
    
    # Self-heal
    interface = SelfModifyingInterface(model)
    print("\nRunning self-diagnosis...")
    report = interface.editor.introspect()
    print(f"Found: {report['dead_neurons']} dead neurons, "
          f"{report['exploding_weights']} exploding layers")
    
    print("\nApplying self-heal...")
    result = interface.self_heal(verbose=True)
    print(f"Self-heal result: success={result['success']}, "
          f"issues={result['issues_found']}, fixes={result['fixes_applied']}")
    
    # Verify
    report_after = interface.editor.introspect()
    print(f"After healing: {report_after['dead_neurons']} dead, "
          f"{report_after['exploding_weights']} exploding")
    
    return interface


def demo_memory_consolidation():
    """Scenario 4: One-shot memory capture and retrieval."""
    print("\n" + "="*60)
    print("SCENARIO 4: Flashbulb Memory (One-Shot Consolidation)")
    print("="*60)
    
    consolidator = OneShotConsolidator(
        dcu_layer=None,  # Pure memory mode
        importance_threshold=0.7,
        max_engrams=100
    )
    
    # Simulate important experiences
    experiences = [
        ("someone_falling", 0.95),
        ("fire_alarm", 0.92),
        ("friendly_greeting", 0.4),  # Below threshold, won't capture
        ("gun_pointed", 0.98),
        ("normal_walking", 0.3),  # Below threshold
    ]
    
    print("Capturing experiences...")
    for event, importance in experiences:
        # Create synthetic pattern from event name
        pattern = torch.tensor([ord(c) for c in event[:20]], dtype=torch.float32)
        pattern = pattern / pattern.norm()
        
        # Pad to consistent size
        if len(pattern) < 32:
            pattern = torch.cat([pattern, torch.zeros(32 - len(pattern))])
        
        ctx = torch.randn(32)
        
        engram = consolidator.capture(
            input_pattern=pattern,
            workspace_state=ctx,
            importance=importance,
            auto_consolidate=True
        )
        
        status = "CAPTURED" if engram else "ignored (low importance)"
        print(f"  [{event}] importance={importance:.2f} -> {status}")
    
    print(f"\nTotal captured: {consolidator.total_captured}")
    print(f"Total consolidated: {consolidator.total_consolidated}")
    
    # Retrieve
    print("\nRetrieving memory for 'someone_falling' cue...")
    query = torch.tensor([ord(c) for c in "someone_falling"[:20]], dtype=torch.float32)
    query = query / query.norm()
    query = torch.cat([query, torch.zeros(32 - len(query))])
    
    retrieved = consolidator.retrieve(query, top_k=3)
    print(f"Retrieved {len(retrieved)} memories:")
    for engram, sim in retrieved:
        print(f"  similarity={sim:.3f}, importance={engram.importance:.2f}, "
              f"retrievals={engram.retrieval_count}")
    
    stats = consolidator.get_memory_stats()
    print(f"\nMemory stats: {stats}")
    
    return consolidator


def demo_full_chappie():
    """Scenario 5: Full CHAPPIE integration demo."""
    print("\n" + "="*60)
    print("SCENARIO 5: Full CHAPPIE System")
    print("="*60)
    
    # Create a minimal base model
    class MinimalCORTEX(nn.Module):
        def __init__(self, vocab_size=1000, d_model=64):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.workspace_dim = 32
            self.embed = nn.Embedding(vocab_size, d_model)
            self.project = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            emb = self.embed(x)
            pooled = emb.mean(dim=1)
            logits = self.project(pooled)
            return {"logits": logits.unsqueeze(1), "workspace": pooled}
    
    base = MinimalCORTEX(vocab_size=100, d_model=64)
    
    print("Wrapping base model with CHAPPIE capabilities...")
    chappie = ChappieCORTEX(
        base_model=base,
        enable_hebbian=False,  # Skip for simplicity
        enable_bootstrapper=False,
        enable_self_modify=True,
        enable_consolidator=True,
    )
    
    # 1. Upload knowledge
    print("\n1. Uploading knowledge...")
    kb = {
        "rules": [
            {"premises": ["human", "in_danger"], "conclusion": "protect", "confidence": 0.99},
        ],
        "facts": [
            {"subject": "Chappie", "relation": "has_mission", "object": "protect_humans", "confidence": 1.0},
        ],
        "procedures": [
            {"name": "protect_human", "steps": ["assess_threat", "interpose", "neutralize_threat"], "confidence": 0.9},
        ]
    }
    result = chappie.upload_knowledge(kb)
    print(f"Upload result: {result}")
    
    # 2. Watch a demonstration
    print("\n2. Watching demonstration...")
    demo_input = torch.randint(0, 100, (1, 10))
    demo_output = torch.randint(0, 100, (1,))
    learn_result = chappie.watch_demonstration(
        input_sequence=demo_input,
        correct_output=demo_output,
        importance=0.85
    )
    print(f"Learning result: {learn_result}")
    
    # 3. Self-heal
    print("\n3. Self-healing check...")
    heal_result = chappie.self_heal()
    print(f"Self-heal: {heal_result}")
    
    # 4. Stats
    print("\n4. System stats...")
    stats = chappie.get_stats()
    print(f"Stats: {stats}")
    
    return chappie


def main():
    print("CHAPPIE Demo: Non-Training CORTEX")
    print("Each scenario demonstrates a different 'Chappie-like' capability")
    print("without any gradient descent training.")
    
    demo_knowledge_upload()
    demo_one_shot_learning()
    demo_self_healing()
    demo_memory_consolidation()
    demo_full_chappie()
    
    print("\n" + "="*60)
    print("All demos complete!")
    print("="*60)


if __name__ == "__main__":
    main()
