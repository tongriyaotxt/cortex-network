"""
M4-1: Goal and GoalStack

Provides:
- Goal: a node in the hierarchical goal tree.
- GoalStack: a stack-based goal manager with decomposition strategies.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agi_protocol import Goal as GoalProto, DecompositionStrategy


class GoalStack:
    """
    Manages a stack of goals with push/pop/decompose operations.
    
    The stack is stored as a list of Goal objects. The top of the stack
    is the current active goal. When a goal is too complex, it is
    decomposed into sub-goals and pushed onto the stack.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.stack: List[GoalProto] = []
        self.completed: List[GoalProto] = []
        self._goal_counter = 0

    def _new_id(self) -> str:
        self._goal_counter += 1
        return f"goal_{self._goal_counter}"

    def push(self, goal: GoalProto) -> bool:
        """
        Push a new goal onto the stack.
        Returns False if stack would overflow.
        """
        if len(self.stack) >= self.max_depth:
            return False
        goal.status = 'active'
        self.stack.append(goal)
        return True

    def pop(self) -> Optional[GoalProto]:
        """Pop the top goal (mark as completed)."""
        if not self.stack:
            return None
        goal = self.stack.pop()
        goal.status = 'completed'
        self.completed.append(goal)
        return goal

    def peek(self) -> Optional[GoalProto]:
        """View the top goal without removing."""
        return self.stack[-1] if self.stack else None

    def fail_top(self) -> Optional[GoalProto]:
        """Mark the top goal as failed and pop it."""
        if not self.stack:
            return None
        goal = self.stack.pop()
        goal.status = 'failed'
        return goal

    def decompose(
        self,
        goal: GoalProto,
        strategy: DecompositionStrategy,
        sub_goal_embeddings: Optional[List[torch.Tensor]] = None,
        decomposition_net: Optional[nn.Module] = None,
    ) -> List[GoalProto]:
        """
        Decompose a goal into sub-goals.
        
        If decomposition_net is provided, it is called with the goal embedding
        to produce learned sub-goal embeddings and descriptions.
        Otherwise falls back to heuristic splitting.
        """
        n_children = min(strategy.max_children, self.max_depth - len(self.stack))
        if n_children <= 0:
            return []

        # Use learned decomposition network if available
        if decomposition_net is not None and goal.embedding is not None:
            emb = goal.embedding
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            with torch.no_grad() if not decomposition_net.training else torch.enable_grad():
                net_output = decomposition_net(emb)
            # Split network output into sub-goal embeddings
            d = net_output.size(-1)
            sub_embs = net_output.view(n_children, d // n_children) if d >= n_children else net_output.unsqueeze(0).expand(n_children, -1)
            if sub_embs.size(-1) < goal.embedding.size(-1):
                sub_embs = torch.nn.functional.pad(sub_embs, (0, goal.embedding.size(-1) - sub_embs.size(-1)))
            sub_goal_embeddings = [sub_embs[i] for i in range(n_children)]

        sub_goals = []
        for i in range(n_children):
            child_id = self._new_id()
            desc = f"{goal.description}_sub_{i}"
            # If learned embeddings provided, use more descriptive naming
            if decomposition_net is not None:
                desc = f"{goal.description}_learned_sub_{i}"
            child = GoalProto(
                goal_id=child_id,
                description=desc,
                parent=goal.goal_id,
                status='pending',
                priority=goal.priority * 0.9,  # slightly lower priority
            )
            if sub_goal_embeddings is not None and i < len(sub_goal_embeddings):
                child.embedding = sub_goal_embeddings[i]
            sub_goals.append(child)

        goal.children = [sg.goal_id for sg in sub_goals]
        return sub_goals

    def escalate(self, problem_description: str) -> Optional[GoalProto]:
        """
        Escalate a problem to the parent goal.
        Pop current, modify parent, return parent.
        """
        if len(self.stack) < 2:
            return None
        current = self.stack.pop()
        parent = self.stack[-1]
        parent.description += f" [escalated:{problem_description}]"
        parent.status = 'active'
        return parent

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def depth(self) -> int:
        return len(self.stack)

    def get_all_active(self) -> List[GoalProto]:
        return [g for g in self.stack if g.status == 'active']

    def to_tensor(self, d_goal: int, device='cpu') -> torch.Tensor:
        """
        Encode the entire stack as a tensor for neural processing.
        Returns: (depth, d_goal)
        """
        if not self.stack:
            return torch.zeros(1, d_goal, device=device)

        embs = []
        for g in self.stack:
            if g.embedding is not None:
                emb = g.embedding
            else:
                emb = torch.zeros(d_goal, device=device)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            embs.append(emb)

        stack_tensor = torch.cat(embs, dim=0)  # (depth, d_goal)
        if stack_tensor.size(0) < self.max_depth:
            pad = self.max_depth - stack_tensor.size(0)
            stack_tensor = F.pad(stack_tensor, (0, 0, 0, pad))
        return stack_tensor[:self.max_depth]
