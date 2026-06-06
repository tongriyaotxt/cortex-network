"""
OneShotConsolidator: Turn a single experience into long-term memory.

In the movie, Chappie watches a human get thrown once and immediately
knows how to catch someone. Biological brains do this too:
    - Flashbulb memories (9/11, personal traumas)
    - One-trial taste aversion
    - Episodic memory formation in single exposures

The mechanism in CORTEX:
    - Fast timescale (τ=25ms): immediate working memory
    - Medium timescale (τ=100ms): episodic buffer
    - Slow timescale (τ=500ms): long-term structure
    
    When an event is marked as "important" (high surprise or emotional tag),
    the OneShotConsolidator creates a PERMANENT structural change:
        1. A new dendritic branch is grown (fast)
        2. It's tuned to the exact pattern via Hebbian LTP (medium)
        3. The branch is integrated into the DCU's stable weights (slow)
        4. A "memory index" is stored in the Memory Cabinet for retrieval

This is NOT training — it's structural plasticity, like growing a new synapse.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


class MemoryEngram:
    """
    A single-shot memory trace.
    
    Contains:
        - The pattern that triggered it (input representation)
        - The context (workspace state at the time)
        - The emotional/importance tag (precision weight)
        - The structural change that was made (which branch, which weights)
    """
    def __init__(
        self,
        pattern: torch.Tensor,
        context: torch.Tensor,
        importance: float,
        branch_id: int,
        weight_delta: torch.Tensor,
        timestamp: int = 0
    ):
        self.pattern = pattern
        self.context = context
        self.importance = importance
        self.branch_id = branch_id
        self.weight_delta = weight_delta
        self.timestamp = timestamp
        self.retrieval_count = 0
        
    def similarity(self, query: torch.Tensor, context_query: Optional[torch.Tensor] = None) -> float:
        """Measure similarity to a query pattern for retrieval."""
        pattern_sim = torch.cosine_similarity(self.pattern, query, dim=-1).item()
        
        if context_query is not None:
            ctx_sim = torch.cosine_similarity(self.context, context_query, dim=-1).item()
            return 0.7 * pattern_sim + 0.3 * ctx_sim
        return pattern_sim
    
    def strengthen(self, factor: float = 1.1):
        """Strengthen this memory with each successful retrieval (rehearsal)."""
        self.importance = min(1.0, self.importance * factor)
        self.retrieval_count += 1


class OneShotConsolidator:
    """
    Consolidate single experiences into permanent structural memory.
    
    Usage:
        consolidator = OneShotConsolidator(dcu_layer, importance_threshold=0.7)
        
        # During normal operation, mark important events
        if surprise > threshold:
            consolidator.capture(
                input_pattern=x,
                workspace_state=workspace,
                importance=surprise,
                hebbian_trace=dcu.traces[0].eligibility
            )
        
        # Periodically consolidate captured memories
        stats = consolidator.consolidate_all()
    """
    def __init__(
        self,
        dcu_layer: Optional[nn.Module] = None,
        importance_threshold: float = 0.7,
        max_engrams: int = 1000,
        consolidation_rate: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.dcu_layer = dcu_layer
        self.importance_threshold = importance_threshold
        self.max_engrams = max_engrams
        self.consolidation_rate = consolidation_rate
        self.device = device
        
        # Captured but not yet consolidated
        self.pending_engrams: List[MemoryEngram] = []
        
        # Fully consolidated (permanent)
        self.consolidated_engrams: List[MemoryEngram] = []
        
        # Fast cache for retrieval (most recently used)
        self.retrieval_cache: List[MemoryEngram] = []
        self.cache_size = 100
        
        self.total_captured = 0
        self.total_consolidated = 0
        
    def capture(
        self,
        input_pattern: torch.Tensor,
        workspace_state: torch.Tensor,
        importance: float,
        hebbian_trace: Optional[torch.Tensor] = None,
        auto_consolidate: bool = True
    ) -> Optional[MemoryEngram]:
        """
        Capture an important experience for potential consolidation.
        
        Args:
            input_pattern: The input that caused the important response
            workspace_state: Global workspace state at that moment
            importance: 0-1 score of how important this experience is
            hebbian_trace: The eligibility trace from Hebbian learning
            auto_consolidate: If True, immediately consolidate if important enough
        
        Returns:
            The captured MemoryEngram, or None if not important enough
        """
        if importance < self.importance_threshold:
            return None
        
        self.total_captured += 1
        
        # Flatten to consistent shape
        pattern = input_pattern.flatten().to(self.device)
        context = workspace_state.flatten().to(self.device)
        
        # Determine which branch to assign this memory to
        branch_id = self._select_branch(pattern)
        
        # Use Hebbian trace as the weight delta, or compute from pattern
        if hebbian_trace is not None:
            weight_delta = hebbian_trace.clone()
        else:
            # Create a rank-1 update from outer product
            weight_delta = torch.outer(context[:pattern.shape[0]], pattern)
        
        engram = MemoryEngram(
            pattern=pattern,
            context=context,
            importance=importance,
            branch_id=branch_id,
            weight_delta=weight_delta,
            timestamp=self.total_captured
        )
        
        self.pending_engrams.append(engram)
        
        # Keep pending list bounded
        if len(self.pending_engrams) > self.max_engrams // 2:
            # Remove lowest importance pending engram
            self.pending_engrams.sort(key=lambda e: e.importance)
            self.pending_engrams = self.pending_engrams[-(self.max_engrams // 2):]
        
        if auto_consolidate and importance > 0.9:
            # High importance = immediate consolidation (flashbulb memory)
            self._consolidate_engram(engram)
        
        return engram
    
    def consolidate_all(self, batch_size: int = 10) -> Dict:
        """
        Consolidate all pending engrams into permanent memory.
        
        This is like sleep: replay and strengthen important experiences.
        Returns statistics about what was consolidated.
        """
        if len(self.pending_engrams) == 0:
            return {"consolidated": 0, "pending": 0, "total": len(self.consolidated_engrams)}
        
        # Sort by importance (most important first)
        self.pending_engrams.sort(key=lambda e: e.importance, reverse=True)
        
        consolidated_now = 0
        failed = 0
        
        for engram in self.pending_engrams:
            success = self._consolidate_engram(engram)
            if success:
                consolidated_now += 1
            else:
                failed += 1
        
        self.pending_engrams = []
        
        return {
            "consolidated": consolidated_now,
            "failed": failed,
            "total_consolidated": len(self.consolidated_engrams),
            "total_pending": len(self.pending_engrams),
        }
    
    def _consolidate_engram(self, engram: MemoryEngram) -> bool:
        """
        Make a single engram permanent by modifying DCU weights.
        
        This is the core structural plasticity:
            weight_new = weight_old + consolidation_rate * importance * weight_delta
        """
        if self.dcu_layer is None:
            # No DCU to modify — just store the engram
            self.consolidated_engrams.append(engram)
            self.total_consolidated += 1
            return True
        
        try:
            # Find the target branch
            branch = self._get_branch(engram.branch_id)
            if branch is None:
                return False
            
            # Compute scaled update
            scale = self.consolidation_rate * engram.importance
            delta = engram.weight_delta
            
            # Match dimensions
            if hasattr(branch, 'weight'):
                w = branch.weight.data
                if delta.shape != w.shape:
                    # Project delta to match weight shape
                    delta = self._project_delta(delta, w.shape)
                
                # Apply permanent change
                with torch.no_grad():
                    w += scale * delta.to(w.device)
            
            # Mark as consolidated
            engram.timestamp = self.total_consolidated
            self.consolidated_engrams.append(engram)
            self.total_consolidated += 1
            
            # Add to retrieval cache
            self.retrieval_cache.insert(0, engram)
            if len(self.retrieval_cache) > self.cache_size:
                self.retrieval_cache = self.retrieval_cache[:self.cache_size]
            
            return True
            
        except Exception:
            return False
    
    def retrieve(
        self,
        query_pattern: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        top_k: int = 3
    ) -> List[Tuple[MemoryEngram, float]]:
        """
        Retrieve the most relevant consolidated memories.
        
        Like episodic memory recall: given a cue, find similar past experiences.
        """
        query = query_pattern.flatten().to(self.device)
        ctx = context.flatten().to(self.device) if context is not None else None
        
        # Search consolidated engrams
        scored = []
        for engram in self.consolidated_engrams:
            sim = engram.similarity(query, ctx)
            scored.append((engram, sim))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        
        top = scored[:top_k]
        
        # Strengthen retrieved memories (rehearsal effect)
        for engram, _ in top:
            engram.strengthen(factor=1.05)
        
        return top
    
    def retrieve_and_inject(
        self,
        query_pattern: torch.Tensor,
        workspace_state: torch.Tensor,
        injection_strength: float = 0.3
    ) -> torch.Tensor:
        """
        Retrieve relevant memories and inject them into the workspace.
        
        This is like "remembering" influencing "current thinking".
        The retrieved memory pattern is blended into the workspace state.
        """
        retrieved = self.retrieve(query_pattern, workspace_state, top_k=1)
        
        if len(retrieved) == 0:
            return workspace_state
        
        engram, sim = retrieved[0]
        
        if sim < 0.5:
            # Not similar enough — don't inject
            return workspace_state
        
        # Inject the memory pattern into workspace
        # Reshape to match workspace
        mem_pattern = engram.pattern
        ws_flat = workspace_state.flatten()
        
        if mem_pattern.shape[0] != ws_flat.shape[0]:
            # Project memory to workspace dimension
            mem_projected = self._project_to_dim(mem_pattern, ws_flat.shape[0])
        else:
            mem_projected = mem_pattern
        
        # Blend: workspace + strength * similarity * memory
        blended = ws_flat + injection_strength * sim * mem_projected.to(ws_flat.device)
        
        return blended.reshape(workspace_state.shape)
    
    def _select_branch(self, pattern: torch.Tensor) -> int:
        """Select which DCU branch should store this memory."""
        # Simple hash-based selection for determinism
        hash_val = int(pattern.sum().item() * 1000) % max(1, self.n_branches())
        return hash_val
    
    def _get_branch(self, branch_id: int) -> Optional[nn.Module]:
        """Get a specific branch from the DCU layer."""
        if self.dcu_layer is None:
            return None
        
        if hasattr(self.dcu_layer, 'branches') and isinstance(self.dcu_layer.branches, nn.ModuleList):
            if branch_id < len(self.dcu_layer.branches):
                return self.dcu_layer.branches[branch_id]
        
        if hasattr(self.dcu_layer, 'branch_weights'):
            if branch_id < len(self.dcu_layer.branch_weights):
                # Return a mock module with weight attribute
                class MockBranch:
                    def __init__(self, weight):
                        self.weight = weight
                return MockBranch(self.dcu_layer.branch_weights[branch_id])
        
        return None
    
    def n_branches(self) -> int:
        """Count branches in the attached DCU layer."""
        if self.dcu_layer is None:
            return 4  # Default
        if hasattr(self.dcu_layer, 'branches'):
            return len(self.dcu_layer.branches)
        if hasattr(self.dcu_layer, 'branch_weights'):
            return len(self.dcu_layer.branch_weights)
        return 4
    
    def _project_delta(self, delta: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Project a weight delta to match target dimensions."""
        flat = delta.flatten()
        target_size = math.prod(target_shape)
        
        if flat.numel() >= target_size:
            # Truncate and reshape
            return flat[:target_size].reshape(target_shape)
        else:
            # Pad and reshape
            padded = torch.zeros(target_size, device=flat.device)
            padded[:flat.numel()] = flat
            return padded.reshape(target_shape)
    
    def _project_to_dim(self, vec: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Project a vector to a target dimension."""
        if vec.numel() == target_dim:
            return vec
        
        # Use random projection
        proj = torch.randn(vec.numel(), target_dim, device=vec.device) / math.sqrt(vec.numel())
        return vec @ proj
    
    def get_memory_stats(self) -> Dict:
        """Return statistics about the memory system."""
        if len(self.consolidated_engrams) == 0:
            return {
                "total_captured": self.total_captured,
                "total_consolidated": 0,
                "pending": len(self.pending_engrams),
                "avg_importance": 0.0,
                "retrieval_distribution": {},
            }
        
        importances = [e.importance for e in self.consolidated_engrams]
        retrievals = [e.retrieval_count for e in self.consolidated_engrams]
        
        return {
            "total_captured": self.total_captured,
            "total_consolidated": len(self.consolidated_engrams),
            "pending": len(self.pending_engrams),
            "avg_importance": sum(importances) / len(importances),
            "max_importance": max(importances),
            "avg_retrievals": sum(retrievals) / len(retrievals),
            "retrieval_distribution": {
                "never": sum(1 for r in retrievals if r == 0),
                "1-5": sum(1 for r in retrievals if 1 <= r <= 5),
                "6+": sum(1 for r in retrievals if r >= 6),
            }
        }
