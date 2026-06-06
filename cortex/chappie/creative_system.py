"""
CHAPPIE Phase 5: Creative System

Art generation, storytelling, humor — Chappie's creative expression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import random


class ArtGenerator:
    """
    Generate visual art from emotional state.
    
    Chappie draws pictures for Mommy — colorful, childlike, full of feeling.
    """
    
    def __init__(self, d_emotion: int = 14, canvas_size: Tuple[int, int] = (64, 64)):
        self.d_emotion = d_emotion
        self.canvas_size = canvas_size
        
        # Emotion → art parameters (style transfer / procedural generation)
        self.style_net = nn.Sequential(
            nn.Linear(d_emotion, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 16),  # color palette + composition params
        )
        
        # Color palette: 4 colors (RGB each)
        self.palette_size = 4
        
    def generate_palette(self, emotion_vec: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Generate color palette from emotion."""
        params = self.style_net(emotion_vec)
        
        # Emotion → color associations
        valence = emotion_vec[0].item()
        arousal = emotion_vec[1].item()
        
        if valence > 0.5:
            base_hue = 0.15  # warm yellows/oranges
        elif valence < -0.5:
            base_hue = 0.6   # cool blues/purples
        else:
            base_hue = 0.4   # greens
        
        palette = []
        for i in range(self.palette_size):
            h = (base_hue + i * 0.1 + arousal * 0.1) % 1.0
            s = 0.6 + abs(valence) * 0.3
            v = 0.5 + arousal * 0.3
            r, g, b = self._hsv_to_rgb(h, s, v)
            palette.append((int(r*255), int(g*255), int(b*255)))
        
        return palette
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB."""
        if s == 0:
            return v, v, v
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if i % 6 == 0: return v, t, p
        elif i % 6 == 1: return q, v, p
        elif i % 6 == 2: return p, v, t
        elif i % 6 == 3: return p, q, v
        elif i % 6 == 4: return t, p, v
        else: return v, p, q
    
    def generate_art_params(self, emotion_vec: torch.Tensor) -> Dict:
        """Generate art composition parameters."""
        params = self.style_net(emotion_vec)
        
        valence = emotion_vec[0].item()
        arousal = emotion_vec[1].item()
        dominance = emotion_vec[2].item()
        
        return {
            "palette": self.generate_palette(emotion_vec),
            "brush_size": int(3 + arousal * 5),
            "symmetry": abs(valence) > 0.3,
            "chaos": max(0, arousal * 0.8),
            "centered": dominance > 0,
            "complexity": int(5 + arousal * 10),
            "emotion_label": self._emotion_to_label(emotion_vec),
        }
    
    def _emotion_to_label(self, emotion_vec: torch.Tensor) -> str:
        """Map emotion vector to a label."""
        emotions = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]
        idx = emotion_vec[3:9].argmax().item()
        return emotions[idx] if idx < len(emotions) else "neutral"
    
    def describe_art(self, params: Dict) -> str:
        """Generate a description of the artwork."""
        label = params["emotion_label"]
        descriptions = {
            "joy": "bright, swirling colors full of happiness",
            "sadness": "soft blues and grays, like a rainy day",
            "anger": "sharp red and black strokes, explosive energy",
            "fear": "jagged lines, dark corners, uncertain shapes",
            "surprise": "unexpected color bursts, playful chaos",
            "neutral": "balanced composition, gentle gradients",
        }
        return f"A drawing of {descriptions.get(label, 'abstract shapes')}"


class Storyteller:
    """
    Generate narratives with characters, settings, conflicts.
    
    Chappie tells stories to Mommy about his adventures.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        
        # Story elements
        self.characters = ["Chappie", "Mommy", "Ninja", "Yolandi", "a robot", "a friend"]
        self.settings = ["the factory", "the city", "a dark alley", "Mommy's house", "the hideout"]
        self.conflicts = ["being chased", "learning to fight", "finding friends", "escaping danger", "saving someone"]
        self.resolutions = ["found safety", "learned a lesson", "made a friend", "became stronger", "went home"]
        
    def generate_story(self, emotion_vec: torch.Tensor, theme: Optional[str] = None, max_sentences: int = 8) -> str:
        """
        Generate a short story based on emotional state.
        
        Args:
            emotion_vec: Current emotional state (influences tone)
            theme: Optional theme override
            max_sentences: Story length
        """
        valence = emotion_vec[0].item()
        
        char = random.choice(self.characters)
        setting = random.choice(self.settings)
        conflict = theme or random.choice(self.conflicts)
        resolution = random.choice(self.resolutions)
        
        # Tone based on valence
        if valence > 0.3:
            tone_words = ["happy", "bright", "wonderful", "amazing"]
            opening = f"Once upon a time, {char} was in {setting}. It was a {random.choice(tone_words)} day."
        elif valence < -0.3:
            tone_words = ["dark", "scary", "difficult", "dangerous"]
            opening = f"One night, {char} was in {setting}. It was a {random.choice(tone_words)} time."
        else:
            opening = f"{char} was in {setting}, not knowing what would happen next."
        
        sentences = [opening]
        
        # Build narrative arc
        sentences.append(f"{char} was {conflict}.")
        
        if valence > 0:
            sentences.append(f"But {char} was brave and smart.")
            sentences.append(f"With help from friends, {char} kept going.")
        else:
            sentences.append(f"Things looked bad for {char}.")
            sentences.append(f"{char} was scared but didn't give up.")
        
        sentences.append(f"In the end, {char} {resolution}.")
        sentences.append(f"And {char} learned that being kind and brave is what matters most.")
        
        return " ".join(sentences[:max_sentences])
    
    def generate_story_about(self, subject: str, emotion_vec: torch.Tensor) -> str:
        """Generate a story about a specific subject/person."""
        return self.generate_story(emotion_vec, theme=f"thinking about {subject}")


class HumorGenerator:
    """
    Generate jokes and humorous responses.
    
    Chappie learns to joke around with Ninja and the gang.
    """
    
    def __init__(self):
        self.joke_templates = [
            # Incongruity jokes
            ("Why did the {A} cross the {B}?", "To get to the {C}!"),
            ("What do you call a {A} that {B}?", "{C}!"),
            ("Why was the {A} sad?", "Because it had no {B}!"),
            # Robot-specific
            ("Why don't robots panic?", "Because we have nerves of steel!"),
            ("What did the robot say to the charging station?", "I feel so energized!") ,
            ("Why did Chappie bring a ladder to school?", "Because he wanted to go to high school!"),
        ]
        
        self.roast_templates = [
            "{target}, you're so slow, even my backup takes less time!",
            "{target}, you have fewer brain cells than I have backup drives!",
            "{target}, you're like a buggy program — always crashing!",
        ]
        
    def generate_joke(self, context: Optional[str] = None) -> str:
        """Generate a context-appropriate joke."""
        if context and "robot" in context.lower():
            setup, punchline = random.choice(self.joke_templates[3:])
        else:
            setup, punchline = random.choice(self.joke_templates[:3])
            # Fill in blanks
            nouns = ["robot", "chicken", "computer", "car", "penguin"]
            verbs = ["dances", "sings", "explodes", "sleeps", "computes"]
            adjectives = ["A dancing disaster", "A singing machine", "A sleepy penguin"]
            
            setup = setup.replace("{A}", random.choice(nouns))
            setup = setup.replace("{B}", random.choice(["road", "river", "sky"]))
            setup = setup.replace("{C}", random.choice(["other side", "future", "lunch"]))
            punchline = punchline.replace("{C}", random.choice(adjectives))
        
        return f"{setup} {punchline}"
    
    def generate_roast(self, target: str, relationship_closeness: float = 0.5) -> str:
        """
        Generate a friendly roast (only if close enough).
        
        Like Chappie teasing Ninja after learning to be street-smart.
        """
        if relationship_closeness < 0.5:
            return "I don't know you well enough to joke like that."
        
        template = random.choice(self.roast_templates)
        return template.format(target=target)
    
    def is_funny(self, text: str, emotional_context: Dict) -> bool:
        """
        Simple humor detection.
        
        Checks for:
            - Incongruity (unexpected combinations)
            - Wordplay
            - Timing markers (...)
        """
        score = 0.0
        
        # Incongruity: question + unexpected answer
        if "?" in text and ("!" in text or "..." in text):
            score += 0.3
        
        # Wordplay: repeated sounds
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i][0] == words[i+1][0]:
                score += 0.2
        
        # Context: joy/arousal makes things funnier
        if emotional_context.get("dominant") == "joy":
            score += 0.2
        
        return score > 0.4
