"""
M6-1: Causal Graph and do-operator

Provides:
- CausalGraph: explicit DAG representation of causal relationships.
- do-operator (Pearl): intervention that cuts parent links.
- Causal queries: P(Y | do(X=x))
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import CausalVariable, CausalEffect, SymbolicToken


class CausalGraph:
    """
    Explicit causal DAG.

    Nodes: CausalVariable
    Edges: directed with strength weight
    """

    def __init__(self):
        self.nodes: Dict[str, CausalVariable] = {}
        self.edges: Dict[Tuple[str, str], float] = {}
        self._adjacency: Dict[str, List[str]] = {}  # parent -> children

    def add_variable(self, var: CausalVariable):
        self.nodes[var.var_id] = var
        if var.var_id not in self._adjacency:
            self._adjacency[var.var_id] = []

    def add_edge(self, cause: str, effect: str, strength: float = 1.0):
        if cause in self.nodes and effect in self.nodes:
            self.edges[(cause, effect)] = strength
            self._adjacency[cause].append(effect)
            # Update node parent/child lists
            if cause not in self.nodes[effect].parents:
                self.nodes[effect].parents.append(cause)
            if effect not in self.nodes[cause].children:
                self.nodes[cause].children.append(effect)

    def remove_edge(self, cause: str, effect: str):
        key = (cause, effect)
        if key in self.edges:
            del self.edges[key]
            self._adjacency[cause].remove(effect)
            self.nodes[effect].parents.remove(cause)
            self.nodes[cause].children.remove(effect)

    def do(self, var_id: str, value: SymbolicToken) -> 'CausalGraph':
        """
        Pearl's do-operator.
        Returns a new graph where var_id's parents are cut and
        the variable is set to the given value.
        """
        new_graph = CausalGraph()
        # Copy all nodes
        for vid, var in self.nodes.items():
            new_var = CausalVariable(
                var_id=var.var_id,
                embedding=var.embedding,
                possible_values=var.possible_values,
                parents=list(var.parents),
                children=list(var.children),
            )
            new_graph.add_variable(new_var)

        # Copy edges except those into var_id
        for (c, e), strength in self.edges.items():
            if e == var_id:
                continue  # cut parent links
            new_graph.add_edge(c, e, strength)

        # Set the intervened variable
        if var_id in new_graph.nodes:
            # Override possible values to just the intervention
            new_graph.nodes[var_id].possible_values = [value]

        return new_graph

    def topological_sort(self) -> List[str]:
        """Return variables in topological order."""
        visited = set()
        order = []

        def visit(vid):
            if vid in visited:
                return
            visited.add(vid)
            for child in self._adjacency.get(vid, []):
                visit(child)
            order.append(vid)

        for vid in self.nodes:
            visit(vid)

        return list(reversed(order))

    def query(
        self,
        intervention: Dict[str, SymbolicToken],
        outcome_var: str,
        n_samples: int = 10,
    ) -> CausalEffect:
        """
        Estimate P(outcome | do(intervention)).

        Simplified implementation: perform do-intervention, then
        simulate forward propagation.
        """
        graph = self
        for var_id, value in intervention.items():
            graph = graph.do(var_id, value)

        # Simulate forward (simplified: just propagate embeddings)
        order = graph.topological_sort()
        state = {}
        for vid in order:
            var = graph.nodes[vid]
            if vid in intervention:
                state[vid] = intervention[vid].embedding
            else:
                # Aggregate parent influences
                parent_vals = []
                for p in var.parents:
                    if p in state:
                        parent_vals.append(state[p])
                if parent_vals:
                    state[vid] = torch.stack(parent_vals).mean(dim=0)
                elif var.embedding is not None:
                    state[vid] = var.embedding
                else:
                    state[vid] = torch.zeros(1)

        if outcome_var in state:
            outcome = state[outcome_var]
            return CausalEffect(
                effect_on_var=outcome_var,
                mean_delta=outcome.mean().item(),
                uncertainty=outcome.std().item(),
                n_samples=n_samples,
            )
        return CausalEffect(effect_on_var=outcome_var)

    def to_tensor(self) -> torch.Tensor:
        """Serialize graph adjacency to tensor for neural processing."""
        n = len(self.nodes)
        adj = torch.zeros(n, n)
        id_to_idx = {vid: i for i, vid in enumerate(self.nodes)}
        for (c, e), strength in self.edges.items():
            adj[id_to_idx[c], id_to_idx[e]] = strength
        return adj
