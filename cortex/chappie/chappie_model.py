"""
ChappieCORTEX: Integrated non-training CORTEX system.

This is the main entry point that wires together all non-training modules:
    - KnowledgeCompiler: inject symbolic knowledge directly into weights
    - HebbianDCU: one-shot learning from single interactions
    - WorkspaceBootstrapper: self-supervised imagination and curiosity
    - SelfModifyingInterface: introspection and self-repair
    - OneShotConsolidator: permanent memory from single experiences

Usage:
    # Create a base CORTEX model
    base_model = AGICORTEXModel(...)
    
    # Wrap with CHAPPIE capabilities
    chappie = ChappieCORTEX(base_model)
    
    # 1. Inject knowledge (like uploading consciousness)
    chappie.upload_knowledge({
        "rules": [{"premises": ["human", "falling"], "conclusion": "catch"}],
        "facts": [{"subject": "gravity", "relation": "causes", "object": "falling"}],
        "procedures": [{"name": "catch_human", "steps": ["see_fall", "move", "extend_arms", "absorb_impact"]}]
    })
    
    # 2. One-shot learn from a demonstration
    chappie.watch_demonstration(
        input_sequence=demo_input,
        correct_output=demo_output,
        importance=0.95  # High importance = immediate consolidation
    )
    
    # 3. Daydream to bootstrap internal models
    chappie.daydream(steps=100, cycles=5)
    
    # 4. Self-diagnose and heal
    chappie.self_heal()
    
    # 5. Generate with memory retrieval
    output = chappie.generate_with_memory(prompt, max_new_tokens=50)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math

from .knowledge_compiler import KnowledgeCompiler
from .hebbian_dcu import HebbianDCU
from .workspace_bootstrapper import WorkspaceBootstrapper
from .self_modifying import SelfModifyingInterface, WeightEditor
from .oneshot_consolidator import OneShotConsolidator


class ChappieCORTEX(nn.Module):
    """
    A CORTEX model augmented with non-training cognitive capabilities.
    
    Unlike standard CORTEX which requires gradient descent training,
    ChappieCORTEX learns via:
        - Knowledge injection (compile rules/facts into weights)
        - One-shot Hebbian plasticity (learn from single demonstrations)
        - Predictive bootstrapping (self-supervised imagination)
        - Self-modification (diagnose and repair its own weights)
        - Memory consolidation (turn experiences into permanent structure)
    """
    def __init__(
        self,
        base_model: nn.Module,
        enable_knowledge_compiler: bool = True,
        enable_hebbian: bool = True,
        enable_bootstrapper: bool = True,
        enable_self_modify: bool = True,
        enable_consolidator: bool = True,
        hebbian_lr: float = 0.01,
        hebbian_rule: str = "oja",
        importance_threshold: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.base_model = base_model
        self.device = device
        
        # Extract dimensions from base model
        self.d_model = self._get_d_model()
        self.vocab_size = self._get_vocab_size()
        
        # Move base model to device
        self.base_model = base_model.to(device)
        
        # Initialize non-training modules
        self.knowledge_compiler: Optional[KnowledgeCompiler] = None
        self.hebbian_layers: List[HebbianDCU] = []
        self.bootstrapper: Optional[WorkspaceBootstrapper] = None
        self.self_modifier: Optional[SelfModifyingInterface] = None
        self.consolidator: Optional[OneShotConsolidator] = None
        
        if enable_knowledge_compiler:
            n_branches = self._get_n_branches()
            self.knowledge_compiler = KnowledgeCompiler(
                d_model=self.d_model,
                n_branches=n_branches or 4,
                device=device
            )
        
        if enable_hebbian:
            self._wrap_hebbian(rule=hebbian_rule, lr=hebbian_lr)
        
        if enable_bootstrapper:
            workspace_dim = self._get_workspace_dim()
            self.bootstrapper = WorkspaceBootstrapper(
                workspace_dim=workspace_dim or 256,
                vocab_size=self.vocab_size or 50000,
                device=device
            )
        
        if enable_self_modify:
            self.self_modifier = SelfModifyingInterface(base_model)
        
        if enable_consolidator:
            # Attach to first Hebbian layer or None
            target_dcu = self.hebbian_layers[0].base_dcu if self.hebbian_layers else None
            self.consolidator = OneShotConsolidator(
                dcu_layer=target_dcu,
                importance_threshold=importance_threshold,
                device=device
            )
        
        # Interaction history
        self.interaction_count = 0
        self.total_surprise = 0.0
        
    def _get_d_model(self) -> int:
        """Infer d_model from base model."""
        if hasattr(self.base_model, 'd_model'):
            return self.base_model.d_model
        # Try to infer from embedding
        for name, param in self.base_model.named_parameters():
            if 'embed' in name or 'token' in name:
                return param.shape[-1]
        return 512
    
    def _get_vocab_size(self) -> int:
        """Infer vocab_size from base model."""
        if hasattr(self.base_model, 'vocab_size'):
            return self.base_model.vocab_size
        for name, param in self.base_model.named_parameters():
            if 'embed' in name or 'token' in name:
                return param.shape[0]
        return 50000
    
    def _get_n_branches(self) -> int:
        """Infer n_branches from base model DCUs."""
        for _, module in self.base_model.named_modules():
            if hasattr(module, 'n_branches'):
                return module.n_branches
            if hasattr(module, 'branches') and isinstance(module.branches, nn.ModuleList):
                return len(module.branches)
        return 4
    
    def _get_workspace_dim(self) -> int:
        """Infer workspace_dim from base model."""
        if hasattr(self.base_model, 'workspace_dim'):
            return self.base_model.workspace_dim
        return self.d_model // 2
    
    def _wrap_hebbian(self, rule: str = "oja", lr: float = 0.01):
        """Wrap DCU layers in the base model with Hebbian plasticity."""
        for name, module in self.base_model.named_modules():
            # Heuristic: identify DCU layers
            is_dcu = (
                'dcu' in name.lower() or
                'dendritic' in name.lower() or
                (hasattr(module, 'branches') and isinstance(module.branches, nn.ModuleList))
            )
            
            if is_dcu and not isinstance(module, HebbianDCU):
                hebbian_wrapper = HebbianDCU(
                    base_dcu=module,
                    rule=rule,
                    lr=lr,
                    freeze_after=None  # Always plastic
                )
                # Replace in parent — this is tricky, so we just keep a list
                self.hebbian_layers.append(hebbian_wrapper)
    
    def upload_knowledge(self, knowledge: Dict[str, List[Dict]]) -> Dict:
        """
        Upload structured knowledge into the model's weights.
        
        Like Chappie's consciousness upload: rules, facts, and skills
        are compiled directly into neural structure.
        
        Args:
            knowledge: Dict with keys "rules", "facts", "procedures"
                Each containing a list of knowledge items.
        
        Returns:
            Stats about what was compiled.
        """
        if self.knowledge_compiler is None:
            return {"error": "Knowledge compiler not enabled"}
        
        # Compile rules
        for rule in knowledge.get("rules", []):
            self.knowledge_compiler.compile_rule(
                premises=rule["premises"],
                conclusion=rule["conclusion"],
                confidence=rule.get("confidence", 1.0)
            )
        
        # Compile facts
        for fact in knowledge.get("facts", []):
            self.knowledge_compiler.compile_fact(
                subject=fact["subject"],
                relation=fact["relation"],
                object=fact["object"],
                confidence=fact.get("confidence", 1.0)
            )
        
        # Compile procedures
        for proc in knowledge.get("procedures", []):
            self.knowledge_compiler.compile_procedure(
                name=proc["name"],
                steps=proc["steps"],
                motor_outputs=proc.get("motor_outputs"),
                confidence=proc.get("confidence", 1.0)
            )
        
        # Apply to DCU layers
        applied = 0
        for hebb_layer in self.hebbian_layers:
            self.knowledge_compiler.apply_to_dcu(hebb_layer.base_dcu)
            applied += 1
        
        return {
            "rules_compiled": len(knowledge.get("rules", [])),
            "facts_compiled": len(knowledge.get("facts", [])),
            "procedures_compiled": len(knowledge.get("procedures", [])),
            "dcu_layers_modified": applied,
        }
    
    def watch_demonstration(
        self,
        input_sequence: torch.Tensor,
        correct_output: Optional[torch.Tensor] = None,
        importance: float = 0.8,
        context: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Learn from a single demonstration (one-shot learning).
        
        Like Chappie watching a human demonstrate something once:
        the model immediately updates its weights via Hebbian plasticity
        and consolidates the memory if important enough.
        
        Args:
            input_sequence: The input tokens/states shown in the demo
            correct_output: The expected output (for supervised one-shot)
            importance: How important this demonstration is (0-1)
            context: Additional context (e.g., workspace state)
        """
        input_seq = input_sequence.to(self.device)
        
        # Forward pass through base model
        with torch.no_grad():
            if hasattr(self.base_model, 'forward'):
                output = self.base_model(input_seq)
            else:
                output = input_seq  # Fallback
        
        # Compute surprise (prediction error)
        surprise = 0.5  # Default
        if correct_output is not None:
            if torch.is_tensor(output) and torch.is_tensor(correct_output):
                error = (output - correct_output.to(self.device)).abs().mean()
                surprise = min(1.0, error.item())
            elif isinstance(output, dict) and 'logits' in output:
                pred = output['logits'].argmax(dim=-1)
                target = correct_output.to(self.device)
                mismatch = (pred != target).float().mean()
                surprise = mismatch.item()
        
        # Use provided importance or computed surprise
        effective_importance = max(importance, surprise)
        
        # Trigger Hebbian updates (the forward pass already did this
        # if Hebbian layers are active, but we can force extra updates)
        for hebb_layer in self.hebbian_layers:
            # The forward pass already updated traces; now consolidate
            if effective_importance > 0.8:
                hebb_layer.consolidate(consolidation_factor=0.5)
        
        # Capture in one-shot consolidator
        if self.consolidator is not None:
            ws_state = context if context is not None else input_seq.float().mean(dim=1)
            self.consolidator.capture(
                input_pattern=input_seq.float().mean(dim=1),
                workspace_state=ws_state,
                importance=effective_importance,
                auto_consolidate=True
            )
        
        self.interaction_count += 1
        self.total_surprise += surprise
        
        return {
            "surprise": surprise,
            "effective_importance": effective_importance,
            "hebbian_layers_active": len(self.hebbian_layers),
            "memory_captured": effective_importance > self.consolidator.importance_threshold if self.consolidator else False,
        }
    
    def daydream(self, steps: int = 100, cycles: int = 3) -> Dict:
        """
        Run imagination cycles to bootstrap internal models.
        
        Like Chappie visualizing scenarios in his head:
        the model generates imagined sequences and learns from
        its own prediction errors.
        """
        if self.bootstrapper is None:
            return {"error": "Bootstrapper not enabled"}
        
        # Use a random seed from memory if available
        seed = torch.randint(0, self.vocab_size, (1, 10), device=self.device)
        if self.consolidator and len(self.consolidator.consolidated_engrams) > 0:
            # Seed from a random memory
            mem = self.consolidator.consolidated_engrams[
                torch.randint(0, len(self.consolidator.consolidated_engrams), (1,)).item()
            ]
            seed = mem.pattern[:10].unsqueeze(0).long().clamp(0, self.vocab_size - 1)
        
        stats = self.bootstrapper.sleep_cycle(
            model=self.base_model,
            hebbian_layers=self.hebbian_layers,
            cycles=cycles
        )
        
        return {
            "cycles_completed": stats.get("cycles", 0),
            "mean_surprise": stats.get("mean_surprise", 0.0),
            "total_learning_steps": stats.get("total_learning_steps", 0),
        }
    
    def self_heal(self, verbose: bool = False) -> Dict:
        """
        Diagnose and fix model issues automatically.
        
        Like Chappie noticing his arm is broken and rewiring it:
        introspect, identify problems, repair, validate.
        """
        if self.self_modifier is None:
            return {"error": "Self-modification not enabled"}
        
        return self.self_modifier.self_heal(verbose=verbose)
    
    def evolve(self, objective: str = "efficiency") -> Dict:
        """
        Guided self-evolution toward an objective.
        
        Objectives:
            - "efficiency": prune dead branches, normalize weights
            - "capacity": add new dendritic branches
            - "stability": balance activations, prevent explosion
        """
        if self.self_modifier is None:
            return {"error": "Self-modification not enabled"}
        
        return self.self_modifier.evolve(objective=objective)
    
    def generate_with_memory(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        memory_strength: float = 0.3
    ) -> torch.Tensor:
        """
        Generate tokens with episodic memory retrieval.
        
        Retrieved memories are injected into the workspace state,
        influencing generation without changing the underlying model.
        """
        generated = prompt.clone().to(self.device)
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                if hasattr(self.base_model, 'forward'):
                    out = self.base_model(generated)
                else:
                    break
            
            # Get logits
            if isinstance(out, dict) and 'logits' in out:
                logits = out['logits'][:, -1, :]  # Last position
            elif torch.is_tensor(out):
                logits = out[:, -1, :]
            else:
                break
            
            # Retrieve memory and bias logits
            if self.consolidator is not None and memory_strength > 0:
                query = generated.float().mean(dim=1)
                retrieved = self.consolidator.retrieve(query, top_k=1)
                if len(retrieved) > 0:
                    engram, sim = retrieved[0]
                    if sim > 0.6:
                        # Bias logits toward tokens from the memory pattern
                        mem_tokens = engram.pattern.long().clamp(0, self.vocab_size - 1)
                        for t in mem_tokens[:5]:  # Boost top 5 tokens from memory
                            logits[:, t] += memory_strength * sim * 2.0
            
            # Sample
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_k > 0:
                top_probs, top_indices = torch.topk(probs, top_k)
                top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
                next_token = top_indices.gather(-1, torch.multinomial(top_probs, 1))
            else:
                next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def forward(self, x: torch.Tensor, **kwargs) -> Any:
        """Standard forward pass through base model."""
        return self.base_model(x, **kwargs)
    
    def get_stats(self) -> Dict:
        """Get comprehensive stats about the CHAPPIE system."""
        stats = {
            "interactions": self.interaction_count,
            "avg_surprise": self.total_surprise / max(1, self.interaction_count),
            "modules": {
                "knowledge_compiler": self.knowledge_compiler is not None,
                "hebbian_layers": len(self.hebbian_layers),
                "bootstrapper": self.bootstrapper is not None,
                "self_modifier": self.self_modifier is not None,
                "consolidator": self.consolidator is not None,
            }
        }
        
        if self.consolidator:
            stats["memory"] = self.consolidator.get_memory_stats()
        
        if self.self_modifier and self.self_modifier.editor:
            stats["self_edits"] = len(self.self_modifier.editor.edit_history)
        
        return stats
