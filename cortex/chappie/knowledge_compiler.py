"""
KnowledgeCompiler: Compile symbolic knowledge directly into neural weights.

Core insight: Instead of learning weights via gradient descent on data,
we directly COMPILE structured knowledge (rules, facts, procedures) into
the dendritic branching patterns of DCUs.

This is inspired by:
    - Chappie's consciousness upload (direct knowledge transfer)
    - Hyperdimensional Computing (Plate, 2003)
    - Neural Program Induction (Graves et al.)
    - The idea that dendrites are ALREADY structured for specific computations

Example:
    compiler = KnowledgeCompiler(d_model=512, n_branches=4)
    
    # Compile a rule: "if A and B then C"
    compiler.compile_rule(
        premises=["A", "B"],
        conclusion="C",
        confidence=0.95
    )
    
    # Compile a fact: "Paris is the capital of France"
    compiler.compile_fact(
        subject="Paris",
        relation="capital_of",
        object="France"
    )
    
    # Compile a procedure (like Chappie learning to fight)
    compiler.compile_procedure(
        name="block_punch",
        steps=["detect_fist", "raise_arm", "deflect"],
        motor_outputs=[0.8, 0.9, 0.7]
    )
    
    # Apply to a CORTEX DCU layer
    dcu = compiler.apply_to_dcu(dcu_layer)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import hashlib


class HypervectorBank:
    """
    Maintain a bank of high-dimensional random vectors for symbols.
    Each symbol gets a stable dense hypervector via hashing.
    """
    def __init__(self, dim: int = 512, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self._cache: Dict[str, torch.Tensor] = {}
        
    def get(self, symbol: str) -> torch.Tensor:
        """Get or create a stable hypervector for a symbol."""
        if symbol not in self._cache:
            # Deterministic but pseudo-random via hashing
            hash_val = int(hashlib.md5(f"{symbol}:{self.seed}".encode()).hexdigest(), 16)
            rng = np.random.RandomState(hash_val % (2**31))
            vec = rng.randn(self.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            self._cache[symbol] = torch.from_numpy(vec)
        return self._cache[symbol]
    
    def bind(self, a: str, b: str) -> torch.Tensor:
        """Binding: circular convolution for associative pairs."""
        va = self.get(a)
        vb = self.get(b)
        # FFT-based circular convolution
        Va = torch.fft.rfft(va)
        Vb = torch.fft.rfft(vb)
        bound = torch.fft.irfft(Va * Vb, n=self.dim)
        return bound / (bound.norm() + 1e-8)
    
    def bundle(self, symbols: List[str]) -> torch.Tensor:
        """Bundling: superposition of multiple symbols."""
        vecs = [self.get(s) for s in symbols]
        bundled = torch.stack(vecs).sum(dim=0)
        return bundled / (bundled.norm() + 1e-8)


class KnowledgeCompiler:
    """
    Compile symbolic knowledge into dendritic weight patterns.
    
    Each DCU branch is treated as a "logic gate" that can be programmed
    to detect specific symbol combinations via its weight pattern.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_branches: int = 4,
        branch_dim: int = 128,
        hyper_dim: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.d_model = d_model
        self.n_branches = n_branches
        self.branch_dim = branch_dim
        self.device = device
        self.hv = HypervectorBank(dim=hyper_dim, seed=42)
        
        # Knowledge registry: tracks what has been compiled
        self.rules: List[Dict] = []
        self.facts: List[Dict] = []
        self.procedures: List[Dict] = []
        
    def compile_rule(
        self,
        premises: List[str],
        conclusion: str,
        confidence: float = 1.0,
        branch_type: str = "excitatory"
    ) -> Dict:
        """
        Compile a logical rule into a dendritic branch pattern.
        
        Mechanism:
            - Premises are bundled into a query hypervector
            - The branch weight is shaped to detect this bundle
            - The conclusion is bound to the output projection
            - Confidence scales the weight magnitude (stronger = more reliable)
        """
        premise_bundle = self.hv.bundle(premises)
        conclusion_vec = self.hv.get(conclusion)
        
        # Create weight pattern: W_branch detects premise_bundle
        # W_out maps to conclusion
        rule_spec = {
            "type": "rule",
            "premises": premises,
            "conclusion": conclusion,
            "confidence": confidence,
            "branch_type": branch_type,
            "premise_hv": premise_bundle,
            "conclusion_hv": conclusion_vec,
        }
        self.rules.append(rule_spec)
        return rule_spec
    
    def compile_fact(
        self,
        subject: str,
        relation: str,
        object: str,
        confidence: float = 1.0
    ) -> Dict:
        """
        Compile a triple fact (S, R, O) into associative memory weights.
        
        Uses binding to create a single vector representing (S-R-O),
        then maps this to a dendritic branch that fires when
        any two of the three are present.
        """
        s_vec = self.hv.get(subject)
        r_vec = self.hv.get(relation)
        o_vec = self.hv.get(object)
        
        # (S ⊗ R) ⊗ O as the full fact representation
        sr_bound = self.hv.bind(subject, relation)
        # We store the binding pattern
        fact_hv = sr_bound + o_vec  # simplified: SR + O
        
        fact_spec = {
            "type": "fact",
            "subject": subject,
            "relation": relation,
            "object": object,
            "confidence": confidence,
            "fact_hv": fact_hv,
        }
        self.facts.append(fact_spec)
        return fact_spec
    
    def compile_procedure(
        self,
        name: str,
        steps: List[str],
        motor_outputs: Optional[List[float]] = None,
        confidence: float = 1.0
    ) -> Dict:
        """
        Compile a procedural skill (sequence of actions) into weights.
        
        Like Chappie watching a fight once and knowing how to block:
        the temporal sequence is encoded as a transition matrix
        in the multi-timescale state layer.
        """
        step_vectors = [self.hv.get(s) for s in steps]
        
        # Create transition weights: step_i → step_{i+1}
        transitions = []
        for i in range(len(steps) - 1):
            transitions.append({
                "from": steps[i],
                "to": steps[i+1],
                "from_hv": step_vectors[i],
                "to_hv": step_vectors[i+1],
            })
        
        proc_spec = {
            "type": "procedure",
            "name": name,
            "steps": steps,
            "motor_outputs": motor_outputs or [1.0] * len(steps),
            "confidence": confidence,
            "transitions": transitions,
            "step_vectors": step_vectors,
        }
        self.procedures.append(proc_spec)
        return proc_spec
    
    def apply_to_dcu(
        self,
        dcu_layer: nn.Module,
        branch_assignment: Optional[str] = None
    ) -> nn.Module:
        """
        Apply all compiled knowledge to a DCU layer's weights.
        
        This directly MODIFIES the DCU weights to encode knowledge.
        No training, no gradients — just structured weight injection.
        
        Args:
            dcu_layer: A DendriticComputationUnit or compatible layer
            branch_assignment: How to assign knowledge to branches
                - "separate": each rule/fact/procedure gets its own branch
                - "interleaved": mix across branches (default, more robust)
        """
        branch_assignment = branch_assignment or "interleaved"
        
        # Collect all knowledge hypervectors
        all_knowledge = []
        for rule in self.rules:
            all_knowledge.append({
                "hv": rule["premise_hv"],
                "out": rule["conclusion_hv"],
                "conf": rule["confidence"],
                "type": "rule"
            })
        for fact in self.facts:
            all_knowledge.append({
                "hv": fact["fact_hv"],
                "out": fact["fact_hv"],  # auto-associative
                "conf": fact["confidence"],
                "type": "fact"
            })
        for proc in self.procedures:
            for t in proc["transitions"]:
                all_knowledge.append({
                    "hv": t["from_hv"],
                    "out": t["to_hv"],
                    "conf": proc["confidence"],
                    "type": "procedure"
                })
        
        if len(all_knowledge) == 0:
            return dcu_layer
        
        # Project hypervectors into the DCU's weight space
        # We create a projection matrix from hyper_dim to branch_dim
        proj_in = self._create_projection(self.hv.dim, self.branch_dim)
        proj_out = self._create_projection(self.hv.dim, self.d_model)
        
        # Assign to branches
        n_items = len(all_knowledge)
        items_per_branch = max(1, n_items // self.n_branches)
        
        with torch.no_grad():
            for b_idx in range(self.n_branches):
                start = b_idx * items_per_branch
                end = start + items_per_branch if b_idx < self.n_branches - 1 else n_items
                branch_items = all_knowledge[start:end]
                
                if len(branch_items) == 0:
                    continue
                
                # Average the knowledge for this branch
                hvs = torch.stack([item["hv"] for item in branch_items])
                outs = torch.stack([item["out"] for item in branch_items])
                confs = torch.tensor([item["conf"] for item in branch_items])
                
                # Weighted average by confidence
                weights = confs / confs.sum()
                branch_hv = (hvs * weights.unsqueeze(1)).sum(dim=0)
                branch_out = (outs * weights.unsqueeze(1)).sum(dim=0)
                
                # Project into DCU weight space
                w_branch = torch.outer(proj_in(branch_hv), proj_out(branch_out))
                
                # Inject into DCU
                self._inject_branch_weight(dcu_layer, b_idx, w_branch)
        
        return dcu_layer
    
    def _create_projection(self, in_dim: int, out_dim: int) -> Callable:
        """Create a fixed random projection (like in random features)."""
        proj = torch.randn(in_dim, out_dim, device=self.device) / np.sqrt(in_dim)
        def project(x: torch.Tensor) -> torch.Tensor:
            x = x.to(self.device)
            return x @ proj
        return project
    
    def _inject_branch_weight(self, dcu_layer: nn.Module, branch_idx: int, weight_matrix: torch.Tensor):
        """
        Inject a compiled weight pattern into a specific DCU branch.
        
        This modifies the DCU's branch weight to detect the compiled pattern.
        The exact method depends on the DCU implementation.
        """
        # Try common attribute names
        if hasattr(dcu_layer, 'branch_weights'):
            # List of weight tensors per branch
            if branch_idx < len(dcu_layer.branch_weights):
                old_w = dcu_layer.branch_weights[branch_idx]
                target = weight_matrix[:old_w.shape[0], :old_w.shape[1]].to(old_w.device)
                # Blend: 70% compiled knowledge, 30% existing (if any)
                if old_w.abs().sum() > 0:
                    new_w = 0.7 * target + 0.3 * old_w
                else:
                    new_w = target
                dcu_layer.branch_weights[branch_idx].copy_(new_w)
        elif hasattr(dcu_layer, 'branches') and isinstance(dcu_layer.branches, nn.ModuleList):
            branch = dcu_layer.branches[branch_idx]
            if hasattr(branch, 'weight'):
                old_w = branch.weight.data
                target = weight_matrix[:old_w.shape[0], :old_w.shape[1]].to(old_w.device)
                if old_w.abs().sum() > 0:
                    branch.weight.data = 0.7 * target + 0.3 * old_w
                else:
                    branch.weight.data = target
    
    def save_knowledge_base(self, path: str):
        """Save the compiled knowledge base to disk."""
        kb = {
            "rules": [{k: v for k, v in r.items() if k not in ("premise_hv", "conclusion_hv")} for r in self.rules],
            "facts": [{k: v for k, v in f.items() if k not in ("fact_hv",)} for f in self.facts],
            "procedures": [{k: v for k, v in p.items() if k not in ("step_vectors",)} for p in self.procedures],
        }
        torch.save(kb, path)
    
    def load_knowledge_base(self, path: str):
        """Load and re-compile a knowledge base."""
        kb = torch.load(path, map_location="cpu")
        for r in kb.get("rules", []):
            self.compile_rule(r["premises"], r["conclusion"], r.get("confidence", 1.0))
        for f in kb.get("facts", []):
            self.compile_fact(f["subject"], f["relation"], f["object"], f.get("confidence", 1.0))
        for p in kb.get("procedures", []):
            self.compile_procedure(p["name"], p["steps"], p.get("motor_outputs"), p.get("confidence", 1.0))
