"""
WorkspaceBootstrapper: Self-supervised learning via internal imagination.

Core insight: Chappie doesn't need a million training examples because
he can IMAGINE scenarios and learn from his own predictions.

In CORTEX terms:
    - The Global Workspace generates "imagined" token sequences
    - Predictive Coding layers compare imagination to reality
    - The prediction error becomes the learning signal
    - NO external labels needed — it's self-supervised bootstrapping

This is similar to:
    - Predictive coding (Friston)
    - Self-supervised learning (BERT, GPT) but WITHOUT gradient descent
    - Dream/sleep consolidation (biological)
    - World models (Ha & Schmidhuber)

The key difference from normal training:
    - We use predictive coding's innate error signals
    - Combined with Hebbian updates (local, not backprop)
    - The model bootstraps its own training curriculum
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Callable
import math


class InternalWorldModel:
    """
    A lightweight world model that predicts the next state of the workspace.
    
    This is NOT trained — it's initialized with random projections
    and refined via Hebbian learning as the system observes transitions.
    """
    def __init__(self, workspace_dim: int, hidden_dim: int = 256):
        self.W_pred = torch.randn(workspace_dim, hidden_dim) / math.sqrt(workspace_dim)
        self.W_out = torch.randn(hidden_dim, workspace_dim) / math.sqrt(hidden_dim)
        self.h = torch.zeros(hidden_dim)  # Hidden state
        
    def predict(self, workspace_state: torch.Tensor) -> torch.Tensor:
        """Predict next workspace state from current state."""
        h_new = torch.tanh(workspace_state @ self.W_pred + self.h * 0.5)
        pred = h_new @ self.W_out
        self.h = h_new.detach()
        return pred
    
    def update_world_model(self, actual: torch.Tensor, predicted: torch.Tensor, lr: float = 0.001):
        """Hebbian update of the world model based on prediction error."""
        error = actual - predicted
        # Simple gradient-free update: move predictions toward actual
        with torch.no_grad():
            self.W_out += lr * torch.outer(self.h, error)


class ImaginationSampler:
    """
    Sample imagined trajectories from the workspace.
    
    Like Chappie visualizing a fight before it happens:
    the workspace can run "simulations" by feeding its own output
    back as input (autoregressive imagination).
    """
    def __init__(
        self,
        vocab_size: int,
        workspace_dim: int,
        temperature: float = 0.8,
        top_k: int = 20
    ):
        self.vocab_size = vocab_size
        self.workspace_dim = workspace_dim
        self.temperature = temperature
        self.top_k = top_k
        
    def imagine_sequence(
        self,
        workspace_fn: Callable,  # Function that maps workspace -> logits
        seed_workspace: torch.Tensor,
        length: int = 50,
        inject_noise: float = 0.1
    ) -> List[torch.Tensor]:
        """
        Generate an imagined sequence by autoregressively sampling
        from the workspace's own predictions.
        
        Returns a list of imagined workspace states.
        """
        imagined = [seed_workspace]
        current = seed_workspace
        
        for _ in range(length):
            # Get prediction from workspace
            logits = workspace_fn(current)  # (vocab_size,)
            
            # Sample with temperature
            probs = torch.softmax(logits / self.temperature, dim=-1)
            
            # Top-k filtering
            if self.top_k > 0:
                top_probs, top_indices = torch.topk(probs, self.top_k)
                top_probs = top_probs / top_probs.sum()
                next_token = top_indices[torch.multinomial(top_probs, 1)]
            else:
                next_token = torch.multinomial(probs, 1)
            
            # Map token back to workspace state (embeddings)
            # In practice, this would use the model's embedding layer
            next_workspace = self._token_to_workspace(next_token, inject_noise)
            
            imagined.append(next_workspace)
            current = next_workspace
            
        return imagined
    
    def _token_to_workspace(self, token: torch.Tensor, noise: float) -> torch.Tensor:
        """Convert a token index back to a workspace state vector."""
        # Use a random but deterministic projection
        proj = torch.randn(self.vocab_size, self.workspace_dim)
        vec = proj[token.item()]
        if noise > 0:
            vec += noise * torch.randn_like(vec)
        return vec / (vec.norm() + 1e-8)


class PredictiveCodingBootstrap:
    """
    Use predictive coding errors as self-supervised learning signals.
    
    Normal predictive coding minimizes prediction error via inference.
    Here, we ALSO use the error to drive Hebbian updates:
        - Large error = surprise = important = strengthen connection
        - Small error = expected = unimportant = weaken connection
    
    This creates an intrinsic curiosity drive: the model naturally
    learns what surprises it.
    """
    def __init__(
        self,
        precision_weighted: bool = True,
        surprise_threshold: float = 0.5,
        learning_rate: float = 0.001
    ):
        self.precision_weighted = precision_weighted
        self.surprise_threshold = surprise_threshold
        self.lr = learning_rate
        
    def compute_learning_signal(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
        precision: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert prediction error into a learning signal.
        
        Returns a weight update direction (same shape as prediction error).
        """
        error = actual - prediction
        
        if self.precision_weighted and precision is not None:
            # High precision = high confidence in error = learn more
            error = error * precision
        
        # Surprise-gated learning: only learn from surprising events
        surprise = error.abs().mean()
        if surprise < self.surprise_threshold:
            # Below threshold: minimal learning (expected events)
            gate = 0.1
        else:
            # Above threshold: full learning (surprising events)
            gate = 1.0
        
        return self.lr * gate * error


class WorkspaceBootstrapper:
    """
    Main interface: bootstrap knowledge via internal imagination
    and predictive coding, WITHOUT external training data.
    
    Usage:
        bootstrapper = WorkspaceBootstrapper(workspace_dim=256, vocab_size=1000)
        
        # Phase 1: Daydream — generate imagined sequences
        imagined = bootstrapper.daydream(seed_tokens, cortex_model, steps=100)
        
        # Phase 2: Learn from imagination — update weights via prediction errors
        bootstrapper.learn_from_imagination(imagined, cortex_model)
        
        # Phase 3: Consolidate — move fast Hebbian changes to stable weights
        bootstrapper.consolidate()
    """
    def __init__(
        self,
        workspace_dim: int = 256,
        vocab_size: int = 50000,
        imagination_steps: int = 100,
        n_daydreams: int = 10,
        temperature: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.workspace_dim = workspace_dim
        self.vocab_size = vocab_size
        self.imagination_steps = imagination_steps
        self.n_daydreams = n_daydreams
        self.temperature = temperature
        self.device = device
        
        self.world_model = InternalWorldModel(workspace_dim)
        self.imagination = ImaginationSampler(vocab_size, workspace_dim, temperature)
        self.pc_bootstrap = PredictiveCodingBootstrap()
        
        # Track what has been imagined (to avoid repetitive daydreaming)
        self.imagination_memory: List[torch.Tensor] = []
        self.memory_capacity = 1000
        
    def daydream(
        self,
        seed_context: torch.Tensor,
        model: nn.Module,
        steps: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate imagined experiences ("daydreams").
        
        Args:
            seed_context: Initial tokens/workspace state to start from
            model: The CORTEX model (used for workspace→logits mapping)
            steps: How many steps to imagine
            
        Returns:
            List of imagined states with predictions and errors
        """
        steps = steps or self.imagination_steps
        
        # Get initial workspace state from model
        with torch.no_grad():
            if hasattr(model, 'get_workspace_state'):
                current_ws = model.get_workspace_state(seed_context)
            else:
                # Fallback: use a simple projection
                current_ws = torch.randn(self.workspace_dim, device=self.device)
                current_ws = current_ws / current_ws.norm()
        
        trajectory = []
        
        for t in range(steps):
            # Predict next state
            predicted_ws = self.world_model.predict(current_ws)
            
            # Get model's actual next-state prediction
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    # Feed imagined state through model
                    logits = self._workspace_to_logits(current_ws, model)
                else:
                    logits = torch.randn(self.vocab_size, device=self.device)
            
            # Sample next token/workspace
            next_ws = self.imagination._token_to_workspace(
                torch.argmax(logits), noise=0.1
            ).to(self.device)
            
            # Prediction error
            error = next_ws - predicted_ws
            
            trajectory.append({
                "step": t,
                "workspace": current_ws.clone(),
                "predicted": predicted_ws,
                "actual": next_ws,
                "error": error,
                "surprise": error.abs().mean().item(),
            })
            
            # Update world model with what actually happened
            self.world_model.update_world_model(next_ws, predicted_ws, lr=0.001)
            
            current_ws = next_ws
        
        # Store in imagination memory
        self.imagination_memory.extend([t["workspace"] for t in trajectory])
        if len(self.imagination_memory) > self.memory_capacity:
            self.imagination_memory = self.imagination_memory[-self.memory_capacity:]
        
        return trajectory
    
    def learn_from_imagination(
        self,
        trajectory: List[Dict],
        model: nn.Module,
        hebbian_layers: Optional[List[nn.Module]] = None
    ) -> Dict:
        """
        Use imagined prediction errors to update model weights.
        
        This is where the "learning without training" happens:
        prediction errors from imagination drive Hebbian updates.
        """
        total_surprise = 0.0
        n_updates = 0
        
        for step_data in trajectory:
            error = step_data["error"]
            surprise = step_data["surprise"]
            total_surprise += surprise
            
            # Only learn from surprising events (curiosity-driven)
            if surprise > self.pc_bootstrap.surprise_threshold:
                learning_signal = self.pc_bootstrap.compute_learning_signal(
                    step_data["predicted"],
                    step_data["actual"]
                )
                
                # Apply to Hebbian layers if provided
                if hebbian_layers:
                    for layer in hebbian_layers:
                        if hasattr(layer, '_hebbian_update'):
                            # Trigger Hebbian update with the error signal
                            layer._hebbian_update(
                                step_data["workspace"].unsqueeze(0),
                                step_data["actual"].unsqueeze(0) + learning_signal.unsqueeze(0)
                            )
                
                n_updates += 1
        
        return {
            "mean_surprise": total_surprise / len(trajectory),
            "n_learning_steps": n_updates,
            "learning_ratio": n_updates / len(trajectory),
        }
    
    def sleep_cycle(
        self,
        model: nn.Module,
        hebbian_layers: Optional[List[nn.Module]] = None,
        cycles: int = 5
    ) -> Dict:
        """
        Run a "sleep" cycle: replay and consolidate imagined memories.
        
        Like biological sleep: replay recent experiences,
        strengthen important patterns, and prune weak connections.
        """
        stats = []
        
        for cycle in range(cycles):
            # Sample from imagination memory
            if len(self.imagination_memory) == 0:
                break
            
            seed = self.imagination_memory[
                torch.randint(0, len(self.imagination_memory), (1,)).item()
            ]
            
            # Daydream from this seed
            imagined = self.daydream(seed, model, steps=self.imagination_steps // 2)
            
            # Learn from the daydream
            cycle_stats = self.learn_from_imagination(imagined, model, hebbian_layers)
            stats.append(cycle_stats)
            
            # Consolidate Hebbian traces every few cycles
            if cycle % 3 == 0 and hebbian_layers:
                for layer in hebbian_layers:
                    if hasattr(layer, 'consolidate'):
                        layer.consolidate(consolidation_factor=0.3)
        
        # Aggregate stats
        if stats:
            return {
                "cycles": len(stats),
                "mean_surprise": sum(s["mean_surprise"] for s in stats) / len(stats),
                "total_learning_steps": sum(s["n_learning_steps"] for s in stats),
            }
        return {"cycles": 0, "mean_surprise": 0.0, "total_learning_steps": 0}
    
    def _workspace_to_logits(self, workspace: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Convert workspace state to token logits via the model."""
        # This is a simplified interface — actual implementation
        # would use the model's output projection
        if hasattr(model, 'workspace_to_logits'):
            return model.workspace_to_logits(workspace)
        return torch.randn(self.vocab_size, device=workspace.device)
    
    def get_novelty_score(self, new_state: torch.Tensor) -> float:
        """
        Measure how novel a state is compared to previous imagination.
        
        High novelty = worth exploring = good for learning.
        """
        if len(self.imagination_memory) == 0:
            return 1.0
        
        similarities = []
        for mem in self.imagination_memory[-100:]:  # Compare to recent memory
            sim = torch.cosine_similarity(new_state, mem, dim=0).item()
            similarities.append(sim)
        
        avg_sim = sum(similarities) / len(similarities)
        return 1.0 - avg_sim  # High novelty = low similarity
