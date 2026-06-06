"""
SelfModifyingInterface: The model can read and edit its own weights.

This is the ultimate "Chappie" feature: consciousness as self-modifying code.

Core capabilities:
    1. INTROSPECTION: Read its own weight matrices and interpret them
    2. DIAGNOSIS: Detect problematic patterns (dead branches, exploding weights)
    3. EDITING: Modify specific weights based on symbolic reasoning
    4. GROWTH: Add new dendritic branches or LPMs on demand
    5. PRUNING: Remove redundant or harmful connections

Biological basis:
    - Synaptic pruning (adolescent brain development)
    - Adult neurogenesis (new neurons in hippocampus)
    - Homeostatic plasticity (neurons regulate their own excitability)
    - Metacognition (thinking about thinking)

Implementation approach:
    We treat the model's weights as "data" that the model's symbolic
    reasoning module (M1) can manipulate. The SelfModeling module (M2)
    provides a representation of the model's own state, which the
    symbolic module can reason about and modify.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import math


class WeightEditor:
    """
    A symbolic interface for editing neural weights.
    
    Like editing code: you specify WHAT to change in symbolic terms,
    and the editor translates that into weight modifications.
    
    Examples:
        editor.remove_dead_branches(threshold=0.01)
        editor.strengthen_path("language", "workspace", factor=1.5)
        editor.add_branch_to_dcu(dcu_id=3, branch_type="excitatory")
        editor.balance_activation(layer="gnw", target_mean=0.5)
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.edit_history: List[Dict] = []
        
    def introspect(self, verbose: bool = False) -> Dict:
        """
        Analyze the model's current weight structure.
        
        Returns a symbolic representation of the model's architecture
        that can be reasoned about by the symbolic module.
        """
        report = {
            "total_parameters": 0,
            "layers": [],
            "dead_neurons": 0,
            "exploding_weights": 0,
            "vanishing_weights": 0,
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                n_params = param.numel()
                report["total_parameters"] += n_params
                
                # Analyze weight statistics
                mean = param.data.mean().item()
                std = param.data.std().item()
                max_val = param.data.abs().max().item()
                
                # Detect issues
                dead = (param.data.abs() < 1e-6).float().mean().item()
                if dead > 0.5:
                    report["dead_neurons"] += 1
                
                if max_val > 10.0:
                    report["exploding_weights"] += 1
                elif max_val < 0.001 and n_params > 100:
                    report["vanishing_weights"] += 1
                
                layer_info = {
                    "name": name,
                    "shape": list(param.shape),
                    "n_params": n_params,
                    "mean": mean,
                    "std": std,
                    "max_abs": max_val,
                    "dead_ratio": dead,
                }
                report["layers"].append(layer_info)
                
                if verbose:
                    print(f"  {name}: {list(param.shape)} | mean={mean:.4f} std={std:.4f} max={max_val:.4f} dead={dead:.2%}")
        
        return report
    
    def remove_dead_branches(self, threshold: float = 0.01) -> int:
        """
        Prune dendritic branches with very low activation weights.
        
        Returns: number of branches pruned.
        """
        pruned = 0
        
        for name, module in self.model.named_modules():
            # Look for DCU or branch-like structures
            if hasattr(module, 'branches') and isinstance(module.branches, nn.ModuleList):
                to_remove = []
                for idx, branch in enumerate(module.branches):
                    if hasattr(branch, 'weight'):
                        avg_magnitude = branch.weight.data.abs().mean().item()
                        if avg_magnitude < threshold:
                            to_remove.append(idx)
                
                # Remove in reverse order to maintain indices
                for idx in reversed(to_remove):
                    del module.branches[idx]
                    pruned += 1
                    self.edit_history.append({
                        "action": "prune_branch",
                        "layer": name,
                        "branch_idx": idx,
                        "reason": f"avg_magnitude < {threshold}",
                    })
        
        return pruned
    
    def strengthen_path(
        self,
        from_layer: str,
        to_layer: str,
        factor: float = 1.2,
        selectivity: float = 0.8
    ) -> bool:
        """
        Strengthen connections between two layers.
        
        Like saying "pay more attention to signals from X when going to Y".
        Only strengthens the top-k strongest existing connections.
        """
        from_param = None
        to_param = None
        
        for name, param in self.model.named_parameters():
            if from_layer in name and 'weight' in name:
                from_param = param
            if to_layer in name and 'weight' in name:
                to_param = param
        
        if from_param is None or to_param is None:
            return False
        
        # Find the top-k strongest connections
        w = from_param.data
        flat = w.abs().flatten()
        k = int(selectivity * flat.numel())
        topk_threshold = torch.topk(flat, k).values.min().item()
        
        # Strengthen those connections
        mask = w.abs() >= topk_threshold
        with torch.no_grad():
            w[mask] *= factor
        
        self.edit_history.append({
            "action": "strengthen_path",
            "from": from_layer,
            "to": to_layer,
            "factor": factor,
            "selectivity": selectivity,
        })
        return True
    
    def balance_activation(self, layer_name: str, target_mean: float = 0.5, tolerance: float = 0.1) -> bool:
        """
        Homeostatic plasticity: adjust biases so layer activations
        target a specific mean.
        
        Biological neurons regulate their firing rates — if a neuron
        fires too much, it downregulates; too little, it upregulates.
        """
        for name, module in self.model.named_modules():
            if layer_name in name:
                if hasattr(module, 'bias') and module.bias is not None:
                    current_mean = module.bias.data.mean().item()
                    if abs(current_mean - target_mean) > tolerance:
                        adjustment = target_mean - current_mean
                        with torch.no_grad():
                            module.bias.data += adjustment
                        self.edit_history.append({
                            "action": "balance_activation",
                            "layer": name,
                            "old_mean": current_mean,
                            "new_mean": target_mean,
                        })
                        return True
        return False
    
    def add_branch_to_dcu(
        self,
        dcu_name: str,
        branch_type: str = "excitatory",
        init_scale: float = 0.01
    ) -> bool:
        """
        Add a new dendritic branch to a DCU (neurogenesis-like).
        
        The new branch is initialized with small random weights
        and will be shaped by subsequent Hebbian learning.
        """
        for name, module in self.model.named_modules():
            if dcu_name in name and hasattr(module, 'branches'):
                if isinstance(module.branches, nn.ModuleList):
                    # Get shape from existing branches
                    if len(module.branches) > 0:
                        ref_branch = module.branches[0]
                        if hasattr(ref_branch, 'weight'):
                            out_dim, in_dim = ref_branch.weight.shape
                            
                            # Create new branch
                            new_branch = nn.Linear(in_dim, out_dim, bias=True)
                            nn.init.normal_(new_branch.weight, mean=0.0, std=init_scale)
                            if new_branch.bias is not None:
                                nn.init.zeros_(new_branch.bias)
                            
                            # Add to module list
                            module.branches.append(new_branch)
                            
                            self.edit_history.append({
                                "action": "add_branch",
                                "dcu": name,
                                "branch_type": branch_type,
                                "shape": [out_dim, in_dim],
                            })
                            return True
        return False
    
    def normalize_weights(self, layer_pattern: str = "", max_norm: float = 1.0) -> int:
        """
        Normalize weight matrices to prevent explosion.
        
        Like synaptic scaling: if total synaptic strength gets too high,
        neurons globally downscale all their inputs.
        """
        normalized = 0
        for name, param in self.model.named_parameters():
            if layer_pattern in name and 'weight' in name:
                w = param.data
                norm = w.norm(dim=1, keepdim=True)
                if (norm > max_norm).any():
                    with torch.no_grad():
                        param.data = w * (max_norm / norm.clamp(min=max_norm))
                    normalized += 1
                    self.edit_history.append({
                        "action": "normalize",
                        "layer": name,
                        "max_norm": max_norm,
                    })
        return normalized
    
    def execute_symbolic_edit(self, command: str, **kwargs) -> Dict:
        """
        Execute a symbolic edit command.
        
        This is the bridge between the symbolic reasoning module
        and weight modification. The symbolic module can generate
        commands like:
            "prune dead branches in layer 3"
            "strengthen workspace to language pathway"
            "add inhibitory branch to dcu_5"
        
        For now, we provide a structured API; NLP parsing can be added.
        """
        commands = {
            "prune_dead": self.remove_dead_branches,
            "strengthen_path": self.strengthen_path,
            "balance": self.balance_activation,
            "add_branch": self.add_branch_to_dcu,
            "normalize": self.normalize_weights,
            "introspect": self.introspect,
        }
        
        if command in commands:
            result = commands[command](**kwargs)
            return {"success": True, "command": command, "result": result}
        else:
            return {"success": False, "command": command, "error": f"Unknown command: {command}"}


class SelfModifyingInterface:
    """
    High-level interface for self-modifying CORTEX models.
    
    Combines:
        - Introspection (read own state)
        - Reasoning (symbolic module decides what to change)
        - Editing (weight editor executes changes)
        - Validation (check that changes didn't break anything)
    """
    def __init__(self, model: nn.Module, editor: Optional[WeightEditor] = None):
        self.model = model
        self.editor = editor or WeightEditor(model)
        self.state_before: Optional[Dict] = None
        
    def snapshot(self) -> Dict:
        """Take a snapshot of current model state for rollback."""
        snapshot = {}
        for name, param in self.model.named_parameters():
            snapshot[name] = param.data.clone()
        return snapshot
    
    def rollback(self, snapshot: Dict):
        """Restore model to a previous snapshot."""
        for name, param in self.model.named_parameters():
            if name in snapshot:
                param.data.copy_(snapshot[name])
    
    def diagnose(self) -> List[Dict]:
        """
        Run a full diagnostic and return recommended fixes.
        
        Like a doctor examining a patient: identify problems,
        then prescribe treatments.
        """
        report = self.editor.introspect()
        recommendations = []
        
        if report["dead_neurons"] > 0:
            recommendations.append({
                "priority": "high",
                "issue": f"{report['dead_neurons']} dead neurons detected",
                "fix": "prune_dead",
                "args": {"threshold": 0.01},
            })
        
        if report["exploding_weights"] > 0:
            recommendations.append({
                "priority": "high",
                "issue": f"{report['exploding_weights']} layers with exploding weights",
                "fix": "normalize",
                "args": {"max_norm": 1.0},
            })
        
        # Find specific vanishing layers and suggest per-layer fixes
        for layer_info in report["layers"]:
            if layer_info["max_abs"] < 0.001 and layer_info["n_params"] > 100:
                recommendations.append({
                    "priority": "medium",
                    "issue": f"Layer '{layer_info['name']}' has vanishing weights (max={layer_info['max_abs']:.6f})",
                    "fix": "normalize",
                    "args": {"layer_pattern": layer_info["name"], "max_norm": 0.5},
                })
        
        return recommendations
    
    def self_heal(self, verbose: bool = False) -> Dict:
        """
        Automatically diagnose and fix model issues.
        
        Like Chappie fixing his own damaged arm: inspect, reason, repair.
        """
        # Snapshot before changes
        self.state_before = self.snapshot()
        
        # Diagnose
        recommendations = self.diagnose()
        
        if verbose:
            print(f"Self-heal: {len(recommendations)} issues found")
        
        # Apply fixes
        applied = []
        for rec in recommendations:
            if verbose:
                print(f"  [{rec['priority']}] {rec['issue']} -> {rec['fix']}")
            
            result = self.editor.execute_symbolic_edit(rec['fix'], **rec['args'])
            applied.append({
                "recommendation": rec,
                "result": result,
            })
        
        # Validate: check that model still runs
        valid = self._validate()
        
        if not valid and self.state_before:
            if verbose:
                print("  Validation failed! Rolling back...")
            self.rollback(self.state_before)
            return {
                "success": False,
                "issues_found": len(recommendations),
                "fixes_applied": len(applied),
                "validation": "failed_rolled_back",
            }
        
        return {
            "success": True,
            "issues_found": len(recommendations),
            "fixes_applied": len(applied),
            "validation": "passed",
            "details": applied,
        }
    
    def evolve(self, objective: str = "efficiency") -> Dict:
        """
        Guided self-modification toward an objective.
        
        Objectives:
            - "efficiency": reduce parameters, prune redundancy
            - "capacity": add branches/modules for more expressiveness
            - "stability": normalize, balance activations
            - "speed": simplify pathways
        """
        self.state_before = self.snapshot()
        
        if objective == "efficiency":
            n_pruned = self.editor.remove_dead_branches(threshold=0.05)
            n_norm = self.editor.normalize_weights(max_norm=0.8)
            return {"objective": objective, "pruned": n_pruned, "normalized": n_norm}
        
        elif objective == "capacity":
            # Find DCUs and add branches
            added = 0
            for name, module in self.model.named_modules():
                if 'dcu' in name.lower() and hasattr(module, 'branches'):
                    if self.editor.add_branch_to_dcu(name, branch_type="excitatory"):
                        added += 1
            return {"objective": objective, "branches_added": added}
        
        elif objective == "stability":
            n_balanced = 0
            for name, _ in self.model.named_modules():
                if self.editor.balance_activation(name, target_mean=0.0):
                    n_balanced += 1
            n_norm = self.editor.normalize_weights(max_norm=1.0)
            return {"objective": objective, "balanced": n_balanced, "normalized": n_norm}
        
        return {"objective": objective, "success": False, "error": "Unknown objective"}
    
    def _validate(self, test_input_shape: Tuple[int, ...] = (1, 10)) -> bool:
        """
        Quick validation: does the model still produce reasonable outputs?
        """
        try:
            with torch.no_grad():
                # Try a forward pass with dummy data
                # Use float tensor for models with Linear layers,
                # int tensor for models with Embedding layers
                dummy_int = torch.randint(0, 1000, test_input_shape)
                dummy_float = torch.randn(test_input_shape)
                
                out = None
                try:
                    out = self.model(dummy_int)
                except (RuntimeError, TypeError):
                    try:
                        out = self.model(dummy_float)
                    except Exception:
                        return False
                
                # Check for NaN/Inf
                if out is not None:
                    if isinstance(out, dict):
                        for v in out.values():
                            if torch.is_tensor(v):
                                if torch.isnan(v).any() or torch.isinf(v).any():
                                    return False
                    elif torch.is_tensor(out):
                        if torch.isnan(out).any() or torch.isinf(out).any():
                            return False
                return True
        except Exception:
            return False
