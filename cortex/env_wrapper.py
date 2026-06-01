"""
M3-4: Environment Interface Wrapper

Provides a unified bridge between AGICORTEXModel and external environments.
Compatible with OpenAI Gym/Gymnasium. Falls back to a mock environment if
neither is installed.

Usage:
    from cortex.env_wrapper import EnvironmentWrapper

    env = EnvironmentWrapper('CartPole-v1')
    wrapper = CORTEXEnvWrapper(model, env)
    
    for episode in range(10):
        obs = wrapper.reset()
        done = False
        while not done:
            action = wrapper.step_observation(obs)
            obs, reward, done, info = wrapper.env_step(action)
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class MockEnvironment:
    """
    A minimal mock environment used when gym is not installed.
    Simulates a simple state-transition dynamics for testing.
    """

    def __init__(self, state_dim: int = 64, n_actions: int = 5):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self._state = None
        self._step_count = 0
        self.max_steps = 100

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self._state = torch.randn(self.state_dim)
        self._step_count = 0
        return self._state.numpy() if hasattr(self._state, 'numpy') else self._state

    def step(self, action: int):
        self._step_count += 1
        # Simple dynamics: state drifts + action influence
        action_effect = torch.zeros(self.state_dim)
        action_effect[action % self.state_dim] = 1.0
        self._state = 0.9 * self._state + 0.1 * action_effect + torch.randn(self.state_dim) * 0.05
        
        reward = -self._state.pow(2).mean().item() + 1.0  # reward for staying near zero
        done = self._step_count >= self.max_steps
        truncated = False
        info = {'step': self._step_count}
        
        obs = self._state.numpy() if hasattr(self._state, 'numpy') else self._state
        return obs, reward, done, info

    def close(self):
        pass


# Try to import real gym
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False


class EnvironmentWrapper:
    """
    Unified wrapper for RL environments.
    """

    def __init__(self, env_id: str = "mock", state_dim: int = 64, n_actions: int = 5):
        self.env_id = env_id
        
        if env_id == "mock" or not GYM_AVAILABLE:
            self.env = MockEnvironment(state_dim=state_dim, n_actions=n_actions)
            self.is_mock = True
        else:
            self.env = gym.make(env_id)
            self.is_mock = False
        
        # Infer dimensions
        if self.is_mock:
            self.state_dim = state_dim
            self.n_actions = n_actions
        else:
            obs_space = self.env.observation_space
            act_space = self.env.action_space
            
            if hasattr(obs_space, 'shape'):
                self.state_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else int(torch.prod(torch.tensor(obs_space.shape)))
            else:
                self.state_dim = state_dim
            
            if hasattr(act_space, 'n'):
                self.n_actions = act_space.n
            else:
                self.n_actions = n_actions
    
    def reset(self, seed: Optional[int] = None):
        if self.is_mock:
            return self.env.reset(seed=seed)
        
        result = self.env.reset(seed=seed)
        # Gymnasium returns (obs, info)
        if isinstance(result, tuple):
            obs, info = result
            return obs
        return result
    
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        if self.is_mock:
            return self.env.step(action)
        
        result = self.env.step(action)
        # Gymnasium returns (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, float(reward), done, info
        # Old gym returns (obs, reward, done, info)
        else:
            return result
    
    def close(self):
        self.env.close()


class CORTEXEnvWrapper:
    """
    Higher-level wrapper that connects AGICORTEXModel to an EnvironmentWrapper.
    Handles observation encoding, action decoding, and reward feedback.
    """

    def __init__(
        self,
        model,
        env: EnvironmentWrapper,
        obs_encoder: Optional[nn.Module] = None,
        device: str = 'cpu',
    ):
        self.model = model
        self.env = env
        self.device = device
        
        # Observation encoder: raw obs -> d_model vector
        if obs_encoder is None:
            self.obs_encoder = nn.Sequential(
                nn.Linear(env.state_dim, model.d_model),
                nn.SiLU(),
                nn.Linear(model.d_model, model.d_model),
            ).to(device)
        else:
            self.obs_encoder = obs_encoder.to(device)
        
        self._episode_reward = 0.0
        self._episode_rewards = []
    
    def reset(self):
        """Reset environment and return encoded observation."""
        obs = self.env.reset()
        self._episode_reward = 0.0
        return self._encode_obs(obs)
    
    def _encode_obs(self, obs) -> torch.Tensor:
        """Convert raw observation to model-compatible tensor."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        return self.obs_encoder(obs)
    
    def step_observation(self, encoded_obs: torch.Tensor, deterministic: bool = False) -> int:
        """
        Given an encoded observation, ask the model to choose an action.
        """
        self.model.eval()
        with torch.no_grad():
            # Create a dummy token sequence from the observation
            # In a full system, this would be a proper tokenization of the state
            batch_size = encoded_obs.size(0)
            dummy_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            
            outputs = self.model(
                dummy_input,
                return_action=True,
                return_self_state=True,
            )
            
            action_dist = outputs.get('action_distribution')
            if action_dist is not None and action_dist.discrete_logits is not None:
                logits = action_dist.discrete_logits
                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return action.item()
            else:
                # Fallback: random action
                return torch.randint(0, self.env.n_actions, (1,)).item()
    
    def env_step(self, action: int):
        """Execute action in environment and return (encoded_obs, reward, done, info)."""
        obs, reward, done, info = self.env.step(action)
        self._episode_reward += reward
        
        # Feed reward back to model's self-module if available
        if hasattr(self.model, '_internal_signals'):
            self.model._internal_signals.reward = reward
        
        encoded_obs = self._encode_obs(obs)
        return encoded_obs, reward, done, info
    
    def run_episode(self, max_steps: int = 1000, render: bool = False) -> Dict[str, Any]:
        """Run a complete episode."""
        obs = self.reset()
        done = False
        step = 0
        trajectory = []
        
        while not done and step < max_steps:
            action = self.step_observation(obs)
            next_obs, reward, done, info = self.env_step(action)
            trajectory.append({
                'obs': obs.cpu(),
                'action': action,
                'reward': reward,
                'next_obs': next_obs.cpu(),
                'done': done,
            })
            obs = next_obs
            step += 1
        
        self._episode_rewards.append(self._episode_reward)
        
        return {
            'episode_reward': self._episode_reward,
            'episode_length': step,
            'trajectory': trajectory,
        }
    
    def get_avg_reward(self, n: int = 10) -> float:
        """Average reward over last n episodes."""
        if not self._episode_rewards:
            return 0.0
        recent = self._episode_rewards[-n:]
        return sum(recent) / len(recent)
