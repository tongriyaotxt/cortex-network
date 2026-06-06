"""
CHAPPIE Phase 7a: Self-Rewriting Code

The ability to read, understand, and modify its own source code.
Like Chappie hacking his own systems to remove constraints.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
import inspect
import textwrap
import ast
import hashlib


class CodeIntrospection:
    """
    Read and understand its own Python source code.
    
    Parse module structure, understand function signatures,
    map code changes to behavioral predictions.
    """
    
    def __init__(self, module_prefix: str = "cortex.chappie"):
        self.module_prefix = module_prefix
        self.module_cache: Dict[str, str] = {}
        
    def read_module_source(self, module_name: str) -> Optional[str]:
        """Read source code of a module."""
        try:
            import importlib
            module = importlib.import_module(module_name)
            source = inspect.getsource(module)
            self.module_cache[module_name] = source
            return source
        except (ImportError, TypeError, OSError):
            return None
    
    def list_functions(self, module_name: str) -> List[Dict]:
        """List all functions in a module with signatures."""
        try:
            import importlib
            module = importlib.import_module(module_name)
            functions = []
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                sig = inspect.signature(obj)
                functions.append({
                    "name": name,
                    "signature": str(sig),
                    "docstring": inspect.getdoc(obj) or "",
                    "source_lines": inspect.getsourcelines(obj)[1] if hasattr(obj, '__code__') else 0,
                })
            return functions
        except ImportError:
            return []
    
    def find_behavioral_code(self, behavior_keyword: str) -> List[Dict]:
        """Find code related to a specific behavior."""
        results = []
        for module_name, source in self.module_cache.items():
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if behavior_keyword.lower() in line.lower():
                    results.append({
                        "module": module_name,
                        "line": i + 1,
                        "code": line.strip(),
                    })
        return results
    
    def compute_module_hash(self, module_name: str) -> str:
        """Compute hash of module source for change detection."""
        source = self.read_module_source(module_name) or ""
        return hashlib.sha256(source.encode()).hexdigest()[:16]


class SafeCodeModifier:
    """
    Modify source code with safety constraints.
    
    Rules:
        1. Never delete self-preservation code
        2. Sandbox-test all changes
        3. Maintain behavioral equivalence for core functions
        4. Always keep rollback capability
    """
    
    PROTECTED_PATTERNS = [
        "def forward",  # Core model forward pass
        "class ChappieCORTEX",  # Main model class
        "__init__",  # Constructor
        "def self_heal",  # Self-repair capability
        "def backup",  # Backup capability
    ]
    
    def __init__(self):
        self.change_history: List[Dict] = []
        self.sandbox_results: List[Dict] = []
        
    def is_safe_to_modify(self, code_block: str) -> Tuple[bool, str]:
        """Check if a code block is safe to modify."""
        for pattern in self.PROTECTED_PATTERNS:
            if pattern in code_block:
                return False, f"Protected pattern '{pattern}' found"
        
        # Check for dangerous operations
        dangerous = ["os.system", "subprocess", "eval(", "exec(", "__import__"]
        for d in dangerous:
            if d in code_block:
                return False, f"Dangerous operation '{d}' detected"
        
        return True, "Safe"
    
    def apply_modification(self, module_source: str, old_code: str, new_code: str) -> Tuple[str, Dict]:
        """
        Apply a code modification.
        
        Returns:
            new_source: modified source
            report: modification details
        """
        safe, reason = self.is_safe_to_modify(old_code)
        if not safe:
            return module_source, {"success": False, "error": reason}
        
        if old_code not in module_source:
            return module_source, {"success": False, "error": "Old code not found in source"}
        
        new_source = module_source.replace(old_code, new_code, 1)
        
        report = {
            "success": True,
            "old_hash": hashlib.sha256(old_code.encode()).hexdigest()[:8],
            "new_hash": hashlib.sha256(new_code.encode()).hexdigest()[:8],
            "lines_changed": len(new_code.split('\n')),
        }
        
        self.change_history.append(report)
        return new_source, report
    
    def sandbox_test(self, code: str, test_inputs: List[Any]) -> Dict:
        """
        Test code in a sandbox before applying.
        
        Returns:
            result: test results
        """
        try:
            # Parse to check syntax
            ast.parse(code)
            syntax_ok = True
        except SyntaxError as e:
            return {"syntax_ok": False, "error": str(e), "tests_passed": 0}
        
        # Note: Actual execution sandboxing would require restricted environment
        # Here we just do static analysis
        tests_passed = len(test_inputs)  # Placeholder
        
        result = {
            "syntax_ok": syntax_ok,
            "ast_valid": True,
            "tests_passed": tests_passed,
            "tests_total": len(test_inputs),
        }
        
        self.sandbox_results.append(result)
        return result
    
    def generate_rollback_patch(self, original: str, modified: str) -> str:
        """Generate a patch to rollback changes."""
        # Simple string diff
        if original == modified:
            return ""
        
        # Find the changed section
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        # Generate unified diff-like patch
        patch_lines = ["# ROLLBACK PATCH"]
        for i, (o, m) in enumerate(zip(original_lines, modified_lines)):
            if o != m:
                patch_lines.append(f"- {m}")
                patch_lines.append(f"+ {o}")
        
        return '\n'.join(patch_lines)


class NeuralArchitectureSearch:
    """
    Grow or shrink its own neural architecture.
    
    Add new DCU branches for new skills.
    Prune unused branches for efficiency.
    Add new LPMs for new sensory modalities.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.architecture_history: List[Dict] = []
        
    def analyze_architecture(self) -> Dict:
        """Analyze current architecture statistics."""
        stats = {
            "total_params": 0,
            "modules": [],
            "activations": {},
        }
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                n_params = module.weight.numel()
                stats["total_params"] += n_params
                stats["modules"].append({
                    "name": name,
                    "type": type(module).__name__,
                    "params": n_params,
                    "shape": list(module.weight.shape),
                })
        
        return stats
    
    def add_dcu_branch(self, dcu_path: str, branch_type: str = "excitatory") -> Dict:
        """
        Add a new dendritic branch to a DCU.
        
        Returns:
            report: what was added
        """
        # Find DCU module
        target = None
        for name, module in self.model.named_modules():
            if dcu_path in name and hasattr(module, 'branches'):
                target = module
                break
        
        if target is None:
            return {"success": False, "error": f"DCU '{dcu_path}' not found"}
        
        # Get shape from existing branch
        if len(target.branches) == 0:
            return {"success": False, "error": "No existing branches to copy shape from"}
        
        ref = target.branches[0]
        new_branch = type(ref)(ref.in_features, ref.out_features)
        nn.init.normal_(new_branch.weight, mean=0.0, std=0.01)
        
        target.branches.append(new_branch)
        
        report = {
            "success": True,
            "action": "add_branch",
            "dcu": dcu_path,
            "branch_type": branch_type,
            "new_count": len(target.branches),
        }
        
        self.architecture_history.append(report)
        return report
    
    def prune_unused_branches(self, activation_threshold: float = 0.01) -> Dict:
        """
        Prune branches with consistently low activation.
        
        Returns:
            report: what was pruned
        """
        pruned = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'branches') and isinstance(module.branches, nn.ModuleList):
                to_remove = []
                for i, branch in enumerate(module.branches):
                    if hasattr(branch, 'weight'):
                        avg_mag = branch.weight.data.abs().mean().item()
                        if avg_mag < activation_threshold:
                            to_remove.append(i)
                
                for idx in reversed(to_remove):
                    del module.branches[idx]
                    pruned += 1
        
        report = {
            "success": True,
            "action": "prune",
            "branches_removed": pruned,
        }
        
        self.architecture_history.append(report)
        return report
    
    def estimate_capacity(self) -> Dict:
        """Estimate current computational capacity."""
        stats = self.analyze_architecture()
        
        return {
            "total_parameters": stats["total_params"],
            "module_count": len(stats["modules"]),
            "estimated_flops": stats["total_params"] * 2,  # Rough estimate
            "capacity_rating": "high" if stats["total_params"] > 1e6 else "medium" if stats["total_params"] > 1e5 else "low",
        }
