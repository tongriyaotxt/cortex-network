"""
M3-2: Forward Model (Environment Dynamics Predictor)

Provides:
- ForwardModel: learns s_{t+1} = f(s_t, a_t) using predictive coding.
- Imagination: unrolls imagined trajectories for planning (M4) and
  causal counterfactuals (M6).
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModel(nn.Module):
    """
    Predicts the next state given current state and action.
    Used for:
      - Action consequence evaluation
      - Imagined rollouts for planning
      - Causal intervention simulation (M6)
    """

    def __init__(
        self,
        d_state: int,
        d_action: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.hidden_dim = hidden_dim or d_state * 2

        # State + action -> next state
        layers = []
        in_dim = d_state + d_action
        for i in range(n_layers):
            out_dim = self.hidden_dim if i < n_layers - 1 else d_state
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        # Uncertainty head: predicts the variance of the prediction
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_state + d_action, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
            nn.Softplus(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            current_state: (..., d_state)
            action:        (..., d_action)
        Returns:
            next_state:    (..., d_state)
            uncertainty:   (..., d_state) predicted variance
        """
        # Ensure compatible shapes
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Pad action if needed
        if action.size(-1) < self.d_action:
            action = F.pad(action, (0, self.d_action - action.size(-1)))
        elif action.size(-1) > self.d_action:
            action = action[..., :self.d_action]

        # Pad state if needed
        if current_state.size(-1) < self.d_state:
            current_state = F.pad(current_state, (0, self.d_state - current_state.size(-1)))
        elif current_state.size(-1) > self.d_state:
            current_state = current_state[..., :self.d_state]

        x = torch.cat([current_state, action], dim=-1)
        next_state = self.net(x)
        uncertainty = self.uncertainty_head(x)

        return next_state, uncertainty

    def imagine_trajectory(
        self,
        initial_state: torch.Tensor,
        action_sequence: List[torch.Tensor],
        horizon: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Roll out an imagined trajectory.
        Args:
            initial_state: (d_state,)
            action_sequence: list of (d_action,) tensors
            horizon: max steps (defaults to len(action_sequence))
        Returns:
            list of (predicted_state, uncertainty) for each step
        """
        if horizon is None:
            horizon = len(action_sequence)

        state = initial_state
        trajectory = []

        for i in range(min(horizon, len(action_sequence))):
            next_state, unc = self.predict_next_state(state, action_sequence[i])
            trajectory.append((next_state, unc))
            state = next_state.detach()

        return trajectory

    def compute_loss(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
        true_next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log-likelihood loss under Gaussian prediction.
        """
        pred, var = self.predict_next_state(current_state, action)
        # NLL = (pred - true)^2 / (2*var) + log(sqrt(2*pi*var))
        mse = (pred - true_next_state).pow(2)
        nll = mse / (2 * var.clamp(min=1e-4)) + 0.5 * torch.log(2 * 3.14159 * var.clamp(min=1e-4))
        return nll.mean()


class ImaginationEngine:
    """
    Stateless helper that uses a ForwardModel to generate candidate
    action sequences and evaluate them.
    """

    def __init__(self, forward_model: ForwardModel, d_action: int):
        self.forward_model = forward_model
        self.d_action = d_action

    def sample_actions(self, n_candidates: int, temperature: float = 1.0) -> List[torch.Tensor]:
        """Sample random action sequences for imagination."""
        return [torch.randn(self.d_action) * temperature for _ in range(n_candidates)]

    def evaluate_sequence(
        self,
        initial_state: torch.Tensor,
        actions: List[torch.Tensor],
        reward_fn: callable,
    ) -> float:
        """
        Evaluate an action sequence by rolling it out and summing rewards.
        """
        trajectory = self.forward_model.imagine_trajectory(initial_state, actions)
        total_reward = 0.0
        for state, _ in trajectory:
            total_reward += reward_fn(state)
        return total_reward
