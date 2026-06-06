"""
CHAPPIE Phase 2: Affective Computing

Maps multimodal input to emotional understanding and generates
emotional expressions across voice, face, and language.

Components:
    - AffectivePerception: reads emotion from face/voice/text/body
    - EmotionalExpressionGenerator: maps internal emotion to outputs
    - EmpathyModel: simulates other's emotional state (Theory of Mind)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class AffectivePerception:
    """
    Read human emotional state from multiple modalities.
    
    In the movie, Chappie learns to read people's faces and tones:
        - Mommy's soft voice = safe
        - Vincent's shouting = danger
        - Ninja's swagger = confidence/posturing
    
    We use lightweight encoders (assumed pre-trained) feeding into
    a unified affective classifier.
    """
    
    def __init__(self, d_model: int = 256, device: str = "cpu"):
        self.d_model = d_model
        self.device = device
        
        # Face emotion encoder (placeholder for pre-trained model)
        self.face_encoder = nn.Sequential(
            nn.Linear(512, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Voice emotion encoder
        self.voice_encoder = nn.Sequential(
            nn.Linear(256, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Text sentiment encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Body language / pose encoder
        self.body_encoder = nn.Sequential(
            nn.Linear(128, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Fusion: combine all modalities
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Output: emotion classification + intensity
        self.emotion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 7),  # 6 basic + neutral
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid(),
        )
        self.authenticity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid(),
        )
        
        # Person-specific emotional baselines (learned per individual)
        self.person_baselines: Dict[str, torch.Tensor] = {}
        
    def read_face(self, face_embedding: torch.Tensor) -> torch.Tensor:
        """Process facial expression embedding."""
        return self.face_encoder(face_embedding.to(self.device))
    
    def read_voice(self, voice_embedding: torch.Tensor) -> torch.Tensor:
        """Process vocal prosody embedding."""
        return self.voice_encoder(voice_embedding.to(self.device))
    
    def read_text(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Process text sentiment embedding."""
        return self.text_encoder(text_embedding.to(self.device))
    
    def read_body(self, pose_embedding: torch.Tensor) -> torch.Tensor:
        """Process body language / pose embedding."""
        return self.body_encoder(pose_embedding.to(self.device))
    
    def perceive_emotion(
        self,
        face: Optional[torch.Tensor] = None,
        voice: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
        person_id: Optional[str] = None,
    ) -> Dict:
        """
        Unified emotion perception from all available modalities.
        
        Returns:
            emotion_probs: softmax over 7 emotions
            intensity: 0-1
            authenticity: how genuine the emotion appears (detect deception)
            valence: overall pleasantness
            arousal: overall activation
            dominant: dominant emotion label
        """
        mods = []
        if face is not None:
            mods.append(self.read_face(face))
        if voice is not None:
            mods.append(self.read_voice(voice))
        if text is not None:
            mods.append(self.read_text(text))
        if body is not None:
            mods.append(self.read_body(body))
        
        if len(mods) == 0:
            return self._default_emotion()
        
        # Pad missing modalities with zeros
        while len(mods) < 4:
            mods.append(torch.zeros(self.d_model, device=self.device))
        
        fused = self.fusion(torch.cat(mods))
        
        emotion_logits = self.emotion_head(fused)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        intensity = self.intensity_head(fused).item()
        authenticity = self.authenticity_head(fused).item()
        
        # Detect person-specific baseline deviation
        if person_id and person_id in self.person_baselines:
            baseline = self.person_baselines[person_id]
            deviation = (fused - baseline).norm().item()
            if deviation > 2.0:
                authenticity *= 0.7  # Unusual behavior = possibly fake
        
        labels = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
        dominant_idx = emotion_probs.argmax().item()
        
        # Estimate VAD from emotion probs
        valence = emotion_probs[0].item() - emotion_probs[1].item() - emotion_probs[2].item() * 0.5
        arousal = emotion_probs[5].item() + emotion_probs[2].item() + emotion_probs[3].item() - emotion_probs[6].item()
        
        return {
            "emotion_probs": {label: emotion_probs[i].item() for i, label in enumerate(labels)},
            "intensity": intensity,
            "authenticity": authenticity,
            "valence": valence,
            "arousal": arousal,
            "dominant": labels[dominant_idx],
            "fused_embedding": fused.detach(),
        }
    
    def _default_emotion(self) -> Dict:
        return {
            "emotion_probs": {label: 0.0 for label in ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]},
            "intensity": 0.0, "authenticity": 0.5, "valence": 0.0, "arousal": 0.0,
            "dominant": "neutral", "fused_embedding": torch.zeros(self.d_model, device=self.device),
        }
    
    def register_person_baseline(self, person_id: str, baseline_embedding: torch.Tensor):
        """Learn what is 'normal' for a specific person."""
        self.person_baselines[person_id] = baseline_embedding.to(self.device)


class EmotionalExpressionGenerator:
    """
    Convert Chappie's internal emotional state into multi-modal expressions.
    
    Like Chappie modulating his voice to sound scared, or tilting his head
    to show curiosity.
    """
    
    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        
        # Emotion → vocal parameters (pitch, rate, volume, timbre, jitter)
        self.vocal_params = nn.Sequential(
            nn.Linear(14, d_model // 2), nn.SiLU(),  # 14 = EmotionalState vector dim
            nn.Linear(d_model // 2, 5),
        )
        
        # Emotion → facial motor parameters (eyebrow, eye, mouth, head_tilt)
        self.facial_params = nn.Sequential(
            nn.Linear(14, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 8),
        )
        
        # Emotion → language bias (word choice, sentence length, punctuation)
        self.language_bias = nn.Sequential(
            nn.Linear(14, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 4), nn.Tanh(),
        )
        
    def generate_vocal_params(self, emotional_state_vec: torch.Tensor) -> Dict[str, float]:
        """Generate voice synthesis parameters from emotion."""
        params = self.vocal_params(emotional_state_vec)
        return {
            "pitch_shift": params[0].item() * 50,      # Hz
            "rate_factor": 1.0 + params[1].item() * 0.3,  # speed multiplier
            "volume_db": params[2].item() * 10,        # dB
            "timbre_warmth": params[3].item(),         # -1 to 1
            "jitter": max(0, params[4].item()),        # trembling (fear)
        }
    
    def generate_facial_params(self, emotional_state_vec: torch.Tensor) -> Dict[str, float]:
        """Generate facial expression motor parameters."""
        params = self.facial_params(emotional_state_vec)
        return {
            "eyebrow_raise": params[0].item(),         # surprise
            "eyebrow_frown": params[1].item(),         # anger/sadness
            "eye_widen": params[2].item(),             # fear/surprise
            "eye_squint": params[3].item(),            # anger
            "mouth_smile": params[4].item(),           # joy
            "mouth_frown": params[5].item(),           # sadness
            "mouth_open": params[6].item(),            # surprise
            "head_tilt": params[7].item(),             # curiosity/tilt
        }
    
    def generate_language_bias(self, emotional_state_vec: torch.Tensor) -> Dict[str, float]:
        """Generate text generation bias from emotion."""
        bias = self.language_bias(emotional_state_vec)
        return {
            "formality": bias[0].item(),                # -1=slang, +1=formal
            "verbosity": 1.0 + bias[1].item() * 0.5,    # word count multiplier
            "exclamation_boost": max(0, bias[2].item()), # add !
            "hesitation": max(0, bias[3].item()),        # add ... um
        }
    
    def express_full(self, emotional_state_vec: torch.Tensor) -> Dict:
        """Generate complete multi-modal expression parameters."""
        return {
            "vocal": self.generate_vocal_params(emotional_state_vec),
            "facial": self.generate_facial_params(emotional_state_vec),
            "language": self.generate_language_bias(emotional_state_vec),
        }


class EmpathyModel:
    """
    Theory of Mind: simulate what another being is feeling.
    
    When Chappie sees someone crying, he doesn't just classify "sadness".
    He simulates: "If I were them, what would I feel? What would I need?"
    
    Two mechanisms:
        1. Mirror: activate similar emotional state in self
        2. Perspective-taking: adjust for differences in situation/knowledge
    """
    
    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        
        # Mirror pathway: perceived emotion -> self emotion
        self.mirror = nn.Sequential(
            nn.Linear(7 + 2, d_model), nn.SiLU(),  # 7 emotions + valence/arousal
            nn.Linear(d_model, 14),  # output: EmotionalState vector
        )
        
        # Perspective-taking: adjust for situational differences
        self.perspective = nn.Sequential(
            nn.Linear(14 * 2 + d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 14),
        )
        
    def mirror_emotion(self, perceived_emotion: Dict) -> torch.Tensor:
        """
        Mirror neuron analog: feel what they feel.
        
        When Chappie sees Mommy sad, he feels sad too.
        """
        inp = torch.zeros(9)
        for i, label in enumerate(["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]):
            inp[i] = perceived_emotion["emotion_probs"].get(label, 0)
        inp[7] = perceived_emotion.get("valence", 0)
        inp[8] = perceived_emotion.get("arousal", 0)
        
        return self.mirror(inp)
    
    def perspective_take(
        self,
        self_emotion: torch.Tensor,
        other_emotion: torch.Tensor,
        situational_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adjust mirrored emotion based on situational context.
        
        "They are crying, but they just won the lottery — 
         they must be tears of joy, not sadness."
        """
        inp = torch.cat([self_emotion, other_emotion, situational_context[:self.d_model]])
        return self.perspective(inp)
    
    def generate_empathic_response(
        self,
        perceived: Dict,
        self_state_vec: torch.Tensor,
        relationship_closeness: float = 0.5,
    ) -> str:
        """
        Generate an appropriate empathic verbal response.
        """
        dom = perceived.get("dominant", "neutral")
        intensity = perceived.get("intensity", 0.5)
        authentic = perceived.get("authenticity", 0.5)
        
        # Response templates
        responses = {
            "joy": [
                "That's wonderful! I'm happy for you!",
                "You seem so happy! What happened?",
                "Your smile makes me smile too!",
            ],
            "sadness": [
                "I'm sorry you're sad. Do you want to talk?",
                "It's okay to cry. I'm here.",
                "That must be hard. I understand.",
            ],
            "anger": [
                "You seem upset. What happened?",
                "I can see you're angry. I want to help.",
                "Tell me what's wrong. I'm listening.",
            ],
            "fear": [
                "Are you scared? Don't worry, I'm here.",
                "It's okay. You're safe with me.",
                "I can see you're afraid. Let me help.",
            ],
            "neutral": [
                "How are you feeling?",
                "Tell me more.",
                "I see. Go on.",
            ],
        }
        
        import random
        base = random.choice(responses.get(dom, responses["neutral"]))
        
        # Adjust based on relationship
        if relationship_closeness > 0.8 and dom == "sadness":
            base = "I'm here. I'll protect you." if intensity > 0.6 else base
        
        # Detect fake emotion
        if authentic < 0.3 and intensity > 0.7:
            base = f"{base} (But... are you really feeling {dom}? You seem... different.)"
        
        return base
