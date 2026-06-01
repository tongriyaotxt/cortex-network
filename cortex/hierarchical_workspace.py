"""
M4-2: Hierarchical Workspace

Provides:
- HierarchicalWorkspace: nested workspaces where each child workspace
  handles a sub-goal. Parent workspaces can create/close children.
- WorkspaceNode: a single node in the workspace tree.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .workspace import GlobalWorkspaceLayer
from .agi_protocol import Goal, WorkspaceContext, WorkspacePacket
from .goal import GoalStack


class WorkspaceNode(nn.Module):
    """
    A node in the hierarchical workspace tree.
    Each node owns a GlobalWorkspaceLayer and optionally child nodes.
    """

    def __init__(
        self,
        node_id: str,
        d_model: int,
        n_modules: int = 4,
        workspace_dim: Optional[int] = None,
        parent: Optional['WorkspaceNode'] = None,
    ):
        super().__init__()
        self.node_id = node_id
        self.parent = parent
        self._children: Dict[str, 'WorkspaceNode'] = {}
        self.goal: Optional[Goal] = None

        # The actual workspace layer
        self.workspace = GlobalWorkspaceLayer(
            d_model=d_model,
            n_modules=n_modules,
            workspace_dim=workspace_dim,
        )

        # Result accumulator (closed child results)
        self.child_results: Dict[str, torch.Tensor] = {}

    def create_child(
        self,
        child_id: str,
        goal: Goal,
        d_model: int,
        n_modules: int = 4,
        workspace_dim: Optional[int] = None,
    ) -> 'WorkspaceNode':
        """Create a child workspace for a sub-goal."""
        child = WorkspaceNode(
            node_id=child_id,
            d_model=d_model,
            n_modules=n_modules,
            workspace_dim=workspace_dim,
            parent=self,
        )
        child.goal = goal
        self._children[child_id] = child
        return child

    def close_child(self, child_id: str) -> Optional[torch.Tensor]:
        """
        Close a child workspace and return its result summary.
        The result is stored in child_results.
        """
        if child_id not in self._children:
            return None
        child = self._children[child_id]
        # Summarize child workspace state as result
        result = child.summarize()
        self.child_results[child_id] = result
        del self._children[child_id]
        return result

    def summarize(self) -> torch.Tensor:
        """
        Summarize this workspace's state into a single vector.
        Used when a child workspace returns its result to parent.
        """
        # Use the workspace's consciousness state as summary
        # In practice, this would be a learned summary network
        return torch.zeros(1)

    def escalate(self, problem: str):
        """Escalate a problem to parent workspace."""
        if self.parent is not None:
            self.parent.child_results[self.node_id] = torch.tensor([1.0])  # problem flag


class HierarchicalWorkspace(nn.Module):
    """
    Manages a tree of workspaces.

    Processing flow:
      1. Root workspace processes input
      2. If a goal needs decomposition, create child workspace
      3. Child workspace runs independently
      4. Child returns result, parent integrates
    """

    def __init__(
        self,
        d_model: int,
        n_modules: int = 8,
        workspace_dim: Optional[int] = None,
        max_depth: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_modules = n_modules
        self.workspace_dim = workspace_dim or d_model // 2
        self.max_depth = max_depth

        # Root workspace
        self.root = WorkspaceNode(
            node_id="root",
            d_model=d_model,
            n_modules=n_modules,
            workspace_dim=workspace_dim,
        )

        # Goal stack attached to the root
        self.goal_stack = GoalStack(max_depth=max_depth)

        # Sub-goal decomposition network
        self.decomposition_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def push_goal(self, goal: Goal) -> bool:
        """Push a new goal onto the stack."""
        return self.goal_stack.push(goal)

    def pop_goal(self) -> Optional[Goal]:
        """Pop completed goal."""
        return self.goal_stack.pop()

    def decompose_current_goal(
        self,
        strategy,
    ) -> List[Goal]:
        """
        Decompose the current top goal into sub-goals.
        For each sub-goal, create a child workspace.
        """
        current = self.goal_stack.peek()
        if current is None:
            return []

        # Generate sub-goal embeddings
        if current.embedding is not None:
            emb = current.embedding
        else:
            emb = torch.zeros(self.d_model)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        sub_embs = self.decomposition_net(emb)  # (1, d_model)
        # Split into multiple sub-goal embeddings
        n_subs = min(strategy.max_children, self.max_depth - self.goal_stack.depth())
        if n_subs <= 0:
            return []

        sub_embs = sub_embs.view(n_subs, self.d_model // n_subs)
        # Pad back to d_model
        sub_embs = F.pad(sub_embs, (0, self.d_model - sub_embs.size(-1)))

        sub_goals = self.goal_stack.decompose(current, strategy, decomposition_net=self.decomposition_net)
        for sg, se in zip(sub_goals, sub_embs):
            sg.embedding = se

        return sub_goals

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the hierarchical workspace.
        
        If the goal stack has active goals, each goal gets its own child
        workspace. The outputs are fused back into the root workspace result.
        """
        batch, seq_len, d = x.shape
        device = x.device
        
        # Step 1: Process through root workspace
        out, info = self.root.workspace(x, return_consciousness=False)
        
        # Step 2: If there are active goals, create/run child workspaces
        active_goals = self.goal_stack.get_all_active()
        if active_goals:
            child_outputs = []
            for goal in active_goals:
                child_id = goal.goal_id
                if child_id not in self.root._children:
                    child = self.root.create_child(
                        child_id=child_id,
                        goal=goal,
                        d_model=self.d_model,
                        n_modules=max(2, self.n_modules // 2),
                        workspace_dim=self.workspace_dim,
                    )
                else:
                    child = self.root._children[child_id]
                
                # Modulate input with goal embedding if available
                x_child = x
                if goal.embedding is not None:
                    goal_emb = goal.embedding.to(device)
                    if goal_emb.dim() == 1:
                        goal_emb = goal_emb.unsqueeze(0).unsqueeze(1)
                    elif goal_emb.dim() == 2:
                        goal_emb = goal_emb.unsqueeze(1)
                    # Broadcast and add as residual
                    if goal_emb.size(-1) < d:
                        goal_emb = F.pad(goal_emb, (0, d - goal_emb.size(-1)))
                    elif goal_emb.size(-1) > d:
                        goal_emb = goal_emb[..., :d]
                    if goal_emb.size(0) < batch:
                        goal_emb = goal_emb.expand(batch, -1, -1)
                    x_child = x + 0.1 * goal_emb
                
                # Run child workspace
                child_out, _ = child.workspace(x_child, return_consciousness=False)
                child_outputs.append(child_out)
            
            # Fuse child outputs into root output (mean residual)
            if child_outputs:
                child_fusion = torch.stack(child_outputs, dim=0).mean(dim=0)
                out = out + 0.1 * child_fusion
        
        info['goal_stack_depth'] = self.goal_stack.depth()
        info['active_goals'] = [g.goal_id for g in active_goals]
        return out, info
