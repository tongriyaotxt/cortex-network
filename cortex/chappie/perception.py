"""
CHAPPIE Phase 3: Multimodal Perception

Vision, audition, tactile, proprioception — all feeding into unified
perceptual tokens for the CORTEX input layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class VisionEncoder(nn.Module):
    """
    Process visual input into semantic tokens.
    
    In production, wraps a pre-trained vision model (MobileViT, CLIP).
    Here we provide the interface and a lightweight fallback.
    """
    
    def __init__(self, d_model: int = 512, n_objects: int = 100, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device
        
        # Simulated visual feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(2048, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Object classifier
        self.object_head = nn.Linear(d_model, n_objects)
        
        # Face embedding extractor
        self.face_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 128),
        )
        
        # Scene context
        self.scene_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 32),
        )
        
        # Text OCR (simplified)
        self.ocr_head = nn.Linear(d_model, 128)
        
        self.known_faces: Dict[str, torch.Tensor] = {}
        
    def forward(self, visual_features: torch.Tensor) -> Dict:
        """
        Args:
            visual_features: (batch, 2048) from pre-trained vision backbone
        Returns:
            Dict with object_logits, face_embedding, scene_embedding, tokens
        """
        x = self.backbone(visual_features.to(self.device))
        
        return {
            "embedding": x,
            "object_logits": self.object_head(x),
            "face_embedding": self.face_head(x),
            "scene_embedding": self.scene_head(x),
            "ocr_embedding": self.ocr_head(x),
            "tokens": x.unsqueeze(1),  # (batch, 1, d_model) for CORTEX input
        }
    
    def identify_face(self, face_embedding: torch.Tensor) -> Optional[str]:
        """Match face against known identities."""
        if not self.known_faces:
            return None
        best_match = None
        best_sim = -1
        for name, emb in self.known_faces.items():
            sim = F.cosine_similarity(face_embedding, emb, dim=-1).item()
            if sim > best_sim and sim > 0.7:
                best_sim = sim
                best_match = name
        return best_match
    
    def register_face(self, name: str, face_embedding: torch.Tensor):
        """Register a new known face."""
        self.known_faces[name] = face_embedding.to(self.device)


class AuditionEncoder(nn.Module):
    """
    Process audio into semantic tokens.
    
    Speech recognition, speaker ID, emotion prosody, sound events.
    """
    
    def __init__(self, d_model: int = 512, n_speakers: int = 50, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device
        
        self.backbone = nn.Sequential(
            nn.Linear(768, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Speaker identification
        self.speaker_head = nn.Linear(d_model, n_speakers)
        
        # Speech content (phoneme/logit level)
        self.phoneme_head = nn.Linear(d_model, 100)
        
        # Prosody (pitch, energy, tempo patterns)
        self.prosody_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 16),
        )
        
        # Sound event detection (gunshot, siren, glass break, etc.)
        self.event_head = nn.Linear(d_model, 20)
        
        self.known_speakers: Dict[str, torch.Tensor] = {}
        
    def forward(self, audio_features: torch.Tensor) -> Dict:
        x = self.backbone(audio_features.to(self.device))
        return {
            "embedding": x,
            "speaker_logits": self.speaker_head(x),
            "phoneme_logits": self.phoneme_head(x),
            "prosody": self.prosody_head(x),
            "sound_event_logits": self.event_head(x),
            "tokens": x.unsqueeze(1),
        }
    
    def identify_speaker(self, speaker_embedding: torch.Tensor) -> Optional[str]:
        if not self.known_speakers:
            return None
        best_match = None
        best_sim = -1
        for name, emb in self.known_speakers.items():
            sim = F.cosine_similarity(speaker_embedding, emb, dim=-1).item()
            if sim > best_sim and sim > 0.7:
                best_sim = sim
                best_match = name
        return best_match
    
    def register_speaker(self, name: str, speaker_embedding: torch.Tensor):
        self.known_speakers[name] = speaker_embedding.to(self.device)


class TactileEncoder(nn.Module):
    """
    Process tactile sensor input.
    
    Pressure, temperature, texture, pain.
    Maps to interoceptive signals.
    """
    
    def __init__(self, d_tactile: int = 64, d_model: int = 256, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        
        # Pressure map encoder
        self.pressure_encoder = nn.Sequential(
            nn.Linear(d_tactile * d_tactile, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
        )
        
        # Temperature scalar
        self.temp_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4), nn.SiLU(),
        )
        
        # Pain signal (high pressure + temperature extremes)
        self.pain_head = nn.Sequential(
            nn.Linear(d_model // 2 + d_model // 4, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid(),
        )
        
    def forward(self, pressure_map: torch.Tensor, temperature: torch.Tensor) -> Dict:
        p = pressure_map.flatten(-2)
        p_emb = self.pressure_encoder(p)
        t_emb = self.temp_encoder(temperature.unsqueeze(-1) if temperature.dim() == 0 else temperature)
        
        combined = torch.cat([p_emb, t_emb], dim=-1)
        pain = self.pain_head(combined)
        
        return {
            "pressure_embedding": p_emb,
            "temperature": temperature,
            "pain": pain.item(),
            "tokens": combined.unsqueeze(1),
        }


class ProprioceptionEncoder(nn.Module):
    """
    Body state awareness — joint angles, velocities, contact states.
    
    "I know where my hand is without looking at it."
    """
    
    def __init__(self, n_joints: int = 20, d_model: int = 256):
        super().__init__()
        self.n_joints = n_joints
        
        self.encoder = nn.Sequential(
            nn.Linear(n_joints * 3, d_model), nn.SiLU(),  # angle, velocity, torque per joint
            nn.Linear(d_model, d_model // 2),
        )
        
        # Body integrity check (damage detection)
        self.integrity_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, n_joints), nn.Sigmoid(),
        )
        
    def forward(self, joint_states: torch.Tensor) -> Dict:
        """
        Args:
            joint_states: (n_joints, 3) = [angle, velocity, torque]
        """
        x = joint_states.flatten()
        emb = self.encoder(x)
        integrity = self.integrity_head(emb)
        
        damaged_joints = (integrity < 0.5).nonzero(as_tuple=True)[0].tolist()
        
        return {
            "body_embedding": emb,
            "joint_integrity": integrity,
            "damaged_joints": damaged_joints,
            "tokens": emb.unsqueeze(0).unsqueeze(0),
        }


class MultimodalFusion(nn.Module):
    """
    Fuse vision, audio, tactile, proprioception into unified perceptual tokens.
    
    Temporal alignment + cross-modal attention.
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Cross-modal attention layers
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Temporal integration
        self.temporal_gru = nn.GRU(d_model, d_model // 2, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(d_model // 2, d_model)
        
    def forward(self, modality_tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_tokens: list of (batch, seq_len, d_model) from each sensor
        Returns:
            fused_tokens: (batch, total_seq_len, d_model)
        """
        if len(modality_tokens) == 0:
            return torch.zeros(1, 1, self.d_model)
        
        # Concatenate all modalities
        combined = torch.cat(modality_tokens, dim=1)  # (batch, total_seq, d_model)
        
        # Self-attention across modalities
        attended, _ = self.cross_attn(combined, combined, combined)
        
        # Temporal integration (if seq_len > 1)
        if attended.size(1) > 1:
            out, _ = self.temporal_gru(attended)
        else:
            out = attended[:, :, :self.d_model // 2]
        
        return self.output_proj(out)
