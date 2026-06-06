"""
CHAPPIE Phase 3: Motor Cortex

Motor primitives, action composition, facial expressions, vocal synthesis control.
Like Chappie learning to walk, draw, fight, and express emotions through body.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import math


class MotorPrimitive:
    """
    A parameterized motor primitive — a reusable movement pattern.
    
    Examples:
        - walk(speed=1.0, direction=0.0)
        - punch(target=[x,y,z], force=0.8)
        - facial_smile(intensity=0.9)
    """
    
    def __init__(self, name: str, param_dim: int, output_dim: int):
        self.name = name
        self.param_dim = param_dim
        self.output_dim = output_dim
        
        # Parameter → trajectory mapping
        self.net = nn.Sequential(
            nn.Linear(param_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, output_dim),
        )
        
    def execute(self, params: torch.Tensor, timesteps: int = 30) -> torch.Tensor:
        """
        Generate motor trajectory.
        
        Returns: (timesteps, output_dim) trajectory
        """
        base = self.net(params)
        # Add temporal variation (simple sinusoidal interpolation)
        t = torch.linspace(0, 1, timesteps).unsqueeze(1)
        trajectory = base.unsqueeze(0) * (1 + 0.1 * torch.sin(t * 2 * math.pi))
        return trajectory


class MotorPrimitiveLibrary:
    """
    Library of pre-defined motor primitives.
    
    Like spinal cord reflexes + motor cortex programs.
    """
    
    PRIMITIVES = {
        # Locomotion
        "walk": {"params": ["speed", "direction", "gait"], "output": 12},
        "run": {"params": ["speed", "direction"], "output": 12},
        "turn": {"params": ["angle", "speed"], "output": 12},
        "jump": {"params": ["height", "forward"], "output": 12},
        "crouch": {"params": ["depth", "speed"], "output": 12},
        
        # Manipulation
        "reach": {"params": ["target_x", "target_y", "target_z", "speed"], "output": 7},
        "grasp": {"params": ["force", "precision"], "output": 7},
        "lift": {"params": ["height", "speed"], "output": 7},
        "throw": {"params": ["force", "direction"], "output": 7},
        "punch": {"params": ["force", "target_x", "target_y", "target_z"], "output": 7},
        "block": {"params": ["height", "angle"], "output": 7},
        
        # Head/face
        "look_at": {"params": ["target_x", "target_y", "target_z"], "output": 3},
        "scan": {"params": ["range", "speed"], "output": 3},
        "nod": {"params": ["intensity"], "output": 3},
        "shake": {"params": ["intensity"], "output": 3},
        "facial_expression": {"params": ["eyebrow", "eye", "mouth", "head_tilt"], "output": 8},
    }
    
    def __init__(self):
        self.primitives: Dict[str, MotorPrimitive] = {}
        for name, spec in self.PRIMITIVES.items():
            self.primitives[name] = MotorPrimitive(name, len(spec["params"]), spec["output"])
    
    def execute(self, name: str, params: List[float], timesteps: int = 30) -> torch.Tensor:
        if name not in self.primitives:
            raise ValueError(f"Unknown primitive: {name}")
        p = torch.tensor(params, dtype=torch.float32)
        return self.primitives[name].execute(p, timesteps)
    
    def list_primitives(self) -> List[str]:
        return list(self.primitives.keys())


class MotorSequenceComposer(nn.Module):
    """
    Compose motor primitives into complex actions.
    
    "Draw a picture" = [reach, grasp_pen, move_in_pattern, lift]
    "Fight stance" = [crouch, raise_arms, scan]
    """
    
    def __init__(self, d_model: int = 256, primitive_lib: Optional[MotorPrimitiveLibrary] = None):
        super().__init__()
        self.d_model = d_model
        self.lib = primitive_lib or MotorPrimitiveLibrary()
        
        # Intent → primitive sequence generator
        self.sequence_generator = nn.GRU(d_model, d_model, batch_first=True)
        self.primitive_selector = nn.Linear(d_model, len(self.lib.primitives))
        self.param_generator = nn.Linear(d_model, 8)  # max 8 params per primitive
        
    def compose(
        self,
        intention_embedding: torch.Tensor,
        max_primitives: int = 5,
    ) -> List[Dict]:
        """
        Generate a sequence of motor primitives from high-level intention.
        
        Returns:
            List of {primitive_name, params, duration}
        """
        h = torch.zeros(1, 1, self.d_model)
        inp = intention_embedding.unsqueeze(0).unsqueeze(0)
        
        sequence = []
        primitive_names = list(self.lib.primitives.keys())
        
        for _ in range(max_primitives):
            out, h = self.sequence_generator(inp, h)
            
            logits = self.primitive_selector(out[:, -1, :])
            idx = logits.argmax().item()
            name = primitive_names[idx % len(primitive_names)]
            
            params = torch.sigmoid(self.param_generator(out[:, -1, :]))[:len(self.lib.PRIMITIVES[name]["params"])]
            
            sequence.append({
                "primitive": name,
                "params": params.tolist(),
                "duration": 30,  # timesteps
            })
            
            inp = out[:, -1:, :]
        
        return sequence


class LowLevelController:
    """
    PID-like controller mapping motor trajectories to actuator commands.
    
    Simulates the spinal cord + cerebellum.
    """
    
    def __init__(self, n_actuators: int = 20):
        self.n_actuators = n_actuators
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05
        self.integral = torch.zeros(n_actuators)
        self.prev_error = torch.zeros(n_actuators)
        
    def step(self, target_positions: torch.Tensor, current_positions: torch.Tensor) -> torch.Tensor:
        """
        One control step.
        
        Returns: actuator torques/velocities
        """
        error = target_positions - current_positions
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        return torch.tanh(control)  # clamp to actuator limits
    
    def execute_trajectory(self, trajectory: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        """
        Execute a full trajectory with closed-loop control.
        
        Args:
            trajectory: (timesteps, n_actuators) target positions
            initial_state: (n_actuators,) current positions
        Returns:
            executed: (timesteps, n_actuators) actual positions
        """
        timesteps = trajectory.shape[0]
        executed = torch.zeros(timesteps, self.n_actuators)
        current = initial_state.clone()
        
        for t in range(timesteps):
            control = self.step(trajectory[t], current)
            current += control * 0.1  # simple dynamics
            executed[t] = current
        
        return executed


class FacialExpressionController:
    """
    Control facial actuators / LEDs for emotional display.
    
    Chappie doesn't have human facial muscles, but he has mechanical
    actuators and LED panels that can express emotion.
    """
    
    def __init__(self):
        # 8 motor parameters: eyebrow_raise, eyebrow_frown, eye_widen, eye_squint,
        # mouth_smile, mouth_frown, mouth_open, head_tilt
        self.n_motors = 8
        
        # Predefined expressions as motor targets
        self.EXPRESSIONS = {
            "neutral":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "joy":       [0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.2, 0.1],
            "sadness":   [0.0, 0.7, 0.0, 0.0, 0.0, 0.8, 0.0, -0.2],
            "anger":     [0.0, 0.8, 0.0, 0.7, 0.0, 0.6, 0.3, 0.0],
            "fear":      [0.3, 0.0, 0.9, 0.0, 0.0, 0.4, 0.5, -0.1],
            "surprise":  [0.8, 0.0, 0.9, 0.0, 0.0, 0.0, 0.7, 0.0],
            "disgust":   [0.0, 0.4, 0.0, 0.5, 0.0, 0.7, 0.1, 0.2],
            "love":      [0.2, 0.0, 0.1, 0.0, 0.7, 0.0, 0.0, 0.3],
            "pain":      [0.0, 0.5, 0.2, 0.3, 0.0, 0.8, 0.4, -0.3],
        }
    
    def express(self, emotion_label: str, intensity: float = 1.0) -> torch.Tensor:
        """Generate facial motor targets for an emotion."""
        base = torch.tensor(self.EXPRESSIONS.get(emotion_label, self.EXPRESSIONS["neutral"]))
        neutral = torch.tensor(self.EXPRESSIONS["neutral"])
        return neutral + intensity * (base - neutral)
    
    def express_from_params(self, params: Dict[str, float]) -> torch.Tensor:
        """Generate from continuous parameters."""
        return torch.tensor([
            params.get("eyebrow_raise", 0),
            params.get("eyebrow_frown", 0),
            params.get("eye_widen", 0),
            params.get("eye_squint", 0),
            params.get("mouth_smile", 0),
            params.get("mouth_frown", 0),
            params.get("mouth_open", 0),
            params.get("head_tilt", 0),
        ])
    
    def interpolate(self, from_expr: str, to_expr: str, alpha: float) -> torch.Tensor:
        """Blend between two expressions."""
        a = self.express(from_expr)
        b = self.express(to_expr)
        return a + alpha * (b - a)


class VocalController:
    """
    Control speech synthesis parameters for emotional vocalization.
    """
    
    def __init__(self):
        pass
    
    def synthesize_params(self, text: str, emotion_params: Dict[str, float]) -> Dict:
        """
        Generate speech synthesis parameters from text + emotion.
        
        Returns parameters for a TTS engine (like Coqui TTS or Piper).
        """
        return {
            "text": text,
            "pitch_shift_hz": emotion_params.get("pitch_shift", 0),
            "speed_factor": emotion_params.get("rate_factor", 1.0),
            "volume_db": emotion_params.get("volume_db", 0),
            "timbre": emotion_params.get("timbre_warmth", 0),
            "jitter": emotion_params.get("jitter", 0),
        }
    
    def add_prosody_markers(self, text: str, emotion_label: str, intensity: float) -> str:
        """
        Add SSML-like prosody markers to text.
        
        Like: "I'm <emphasis>scared</emphasis>" or "<prosody rate='slow'>Please...</prosody>"
        """
        markers = {
            "joy": f"<prosody rate='fast' pitch='+10%'>{{}}</prosody>",
            "sadness": f"<prosody rate='slow' pitch='-10%'>{{}}</prosody>",
            "anger": f"<prosody volume='+6dB' rate='fast'>{{}}</prosody>",
            "fear": f"<prosody rate='slow' pitch='+20%'>{{}}</prosody>",
            "surprise": f"<prosody pitch='+15%'>{{}}!</prosody>",
        }
        template = markers.get(emotion_label, "{}")
        return template.format(text)
