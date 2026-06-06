"""
CHAPPIE Phase 1: Personality Engine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EmotionalState:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    disgust: float = 0.0
    surprise: float = 0.0
    trust: float = 0.5
    anticipation: float = 0.0
    love: float = 0.0
    shame: float = 0.0
    pride: float = 0.0
    intensity: float = 0.0
    stability: float = 1.0
    dominant_emotion: str = "neutral"

    def to_vector(self):
        return torch.tensor([
            self.valence, self.arousal, self.dominance,
            self.joy, self.sadness, self.anger, self.fear,
            self.disgust, self.surprise,
            self.trust, self.anticipation, self.love,
            self.shame, self.pride,
        ], dtype=torch.float32)

    @classmethod
    def from_vector(cls, vec):
        v = vec.tolist() if torch.is_tensor(vec) else vec
        return cls(
            valence=v[0], arousal=v[1], dominance=v[2],
            joy=v[3], sadness=v[4], anger=v[5], fear=v[6],
            disgust=v[7], surprise=v[8],
            trust=v[9], anticipation=v[10], love=v[11],
            shame=v[12], pride=v[13],
        )

    def update_dominant(self):
        emotions = {
            "joy": self.joy, "sadness": self.sadness,
            "anger": self.anger, "fear": self.fear,
            "disgust": self.disgust, "surprise": self.surprise,
        }
        self.dominant_emotion = max(emotions, key=emotions.get)
        if emotions[self.dominant_emotion] < 0.1:
            self.dominant_emotion = "neutral"

    def __str__(self):
        self.update_dominant()
        return f"Emotion[{self.dominant_emotion}] V={self.valence:+.2f} A={self.arousal:+.2f} D={self.dominance:+.2f} I={self.intensity:.2f}"



class EmotionSystem(nn.Module):
    """Chappie's emotional core. Emotions modulate all cognition."""

    def __init__(self, d_model=512, vad_dim=3, n_basic_emotions=6,
                 decay_rate=0.95, reactivity=0.3, baseline_valence=0.1, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.vad_dim = vad_dim
        self.decay_rate = decay_rate
        self.reactivity = reactivity
        self.device = device
        self.state = EmotionalState(valence=baseline_valence)

        self.reactive_encoder = nn.Sequential(
            nn.Linear(d_model + vad_dim, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, vad_dim), nn.Tanh(),
        )
        self.reflective_encoder = nn.Sequential(
            nn.Linear(d_model + vad_dim + 2, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, vad_dim), nn.Tanh(),
        )
        self.vad_to_emotions = nn.Linear(vad_dim, n_basic_emotions)
        self.derived_head = nn.Sequential(
            nn.Linear(vad_dim + n_basic_emotions, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, 5), nn.Sigmoid(),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(vad_dim + n_basic_emotions, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid(),
        )
        self.target_valence = nn.Parameter(torch.tensor(baseline_valence))
        self.target_arousal = nn.Parameter(torch.tensor(0.1))
        self.emotional_memory = {}
        self.memory_capacity = 100
        self.to(device)

    def forward(self, perception=None, internal_context=None, certainty=0.5, goal_progress=0.5):
        current_vad = torch.tensor([self.state.valence, self.state.arousal, self.state.dominance], device=self.device)
        reactive_delta = torch.zeros(self.vad_dim, device=self.device)
        if perception is not None:
            reactive_delta = self._reactive_update(perception, current_vad)
        reflective_delta = torch.zeros(self.vad_dim, device=self.device)
        if internal_context is not None:
            reflective_delta = self._reflective_update(internal_context, current_vad, certainty, goal_progress)
        total_delta = self.reactivity * (reactive_delta + reflective_delta)
        homeostatic_delta = 0.05 * (torch.tensor([self.target_valence.item(), self.target_arousal.item(), 0.0], device=self.device) - current_vad)
        new_vad = self.decay_rate * current_vad + (1 - self.decay_rate) * total_delta + homeostatic_delta
        new_vad = torch.clamp(new_vad, -1.0, 1.0)
        self.state.valence = new_vad[0].item()
        self.state.arousal = new_vad[1].item()
        self.state.dominance = new_vad[2].item()
        emotion_logits = self.vad_to_emotions(new_vad)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        self.state.joy = emotion_probs[0].item()
        self.state.sadness = emotion_probs[1].item()
        self.state.anger = emotion_probs[2].item()
        self.state.fear = emotion_probs[3].item()
        self.state.disgust = emotion_probs[4].item()
        self.state.surprise = emotion_probs[5].item()
        derived_input = torch.cat([new_vad, emotion_probs])
        derived = self.derived_head(derived_input)
        self.state.trust = derived[0].item()
        self.state.anticipation = derived[1].item()
        self.state.love = derived[2].item()
        self.state.shame = derived[3].item()
        self.state.pride = derived[4].item()
        self.state.intensity = self.intensity_head(torch.cat([new_vad, emotion_probs])).item()
        self.state.update_dominant()
        return self.state

    def _reactive_update(self, perception, current_vad):
        if perception.dim() == 1:
            perception = perception.unsqueeze(0)
        perception = perception.mean(dim=0)
        inp = torch.cat([perception[:self.d_model], current_vad])
        return self.reactive_encoder(inp)

    def _reflective_update(self, internal_context, current_vad, certainty, goal_progress):
        if internal_context.dim() == 1:
            internal_context = internal_context.unsqueeze(0)
        internal_context = internal_context.mean(dim=0)
        scalars = torch.tensor([certainty, goal_progress], device=self.device)
        inp = torch.cat([internal_context[:self.d_model], current_vad, scalars])
        return self.reflective_encoder(inp)

    def register_emotional_trigger(self, pattern_name, target_vad, strength=1.0):
        vad_tensor = torch.tensor(target_vad, device=self.device)
        self.emotional_memory[pattern_name] = (vad_tensor, strength)
        if len(self.emotional_memory) > self.memory_capacity:
            weakest = min(self.emotional_memory, key=lambda k: self.emotional_memory[k][1])
            del self.emotional_memory[weakest]

    def trigger_emotion(self, pattern_name):
        if pattern_name in self.emotional_memory:
            target_vad, strength = self.emotional_memory[pattern_name]
            current = torch.tensor([self.state.valence, self.state.arousal, self.state.dominance], device=self.device)
            new_vad = current + strength * 0.5 * (target_vad - current)
            new_vad = torch.clamp(new_vad, -1.0, 1.0)
            self.state.valence = new_vad[0].item()
            self.state.arousal = new_vad[1].item()
            self.state.dominance = new_vad[2].item()
            return self.forward()
        return self.state

    def modulate_gnw_threshold(self, base_threshold=0.5):
        if self.state.fear > 0.5:
            return base_threshold * 0.6
        elif self.state.anger > 0.5:
            return base_threshold * 0.7
        elif self.state.arousal < -0.3:
            return base_threshold * 1.5
        return base_threshold

    def modulate_dcu_branches(self, branch_weights):
        if len(branch_weights) < 4:
            return branch_weights
        modulated = list(branch_weights)
        if self.state.anger > 0.5:
            modulated[0] *= 1.3
        if self.state.fear > 0.5:
            modulated[1] *= 1.3
        if self.state.joy > 0.5:
            modulated[3] *= 1.2
        return modulated

    def get_memory_consolidation_boost(self):
        return 1.0 + self.state.intensity

    def express(self):
        self.state.update_dominant()
        import random
        templates = {
            "joy": ["I feel happy!", "This makes me joyful!", "I'm so excited!"],
            "sadness": ["I feel sad...", "This hurts me...", "I'm upset."],
            "anger": ["I'm angry!", "This is not fair!", "I won't accept this!"],
            "fear": ["I'm scared...", "Please don't hurt me...", "I'm afraid."],
            "disgust": ["That's gross!", "I don't like that!", "Yuck!"],
            "surprise": ["Wow!", "I didn't expect that!", "Amazing!"],
            "neutral": ["I'm okay.", "I see.", "Hmm."],
        }
        return random.choice(templates.get(self.state.dominant_emotion, templates["neutral"]))



@dataclass
class MoralRule:
    condition: str
    action: str
    priority: float = 1.0
    source: str = "injected"
    stage: float = 1.0
    violations: int = 0
    applications: int = 0


class MoralDevelopmentModule:
    """Kohlberg's stages of moral development as learnable system."""

    STAGE_DESCRIPTIONS = {
        1: "Pre-conventional: Obedience and punishment avoidance",
        2: "Pre-conventional: Individualism and exchange",
        3: "Conventional: Interpersonal conformity",
        4: "Conventional: Social order maintenance",
        5: "Post-conventional: Social contract",
        6: "Post-conventional: Universal ethical principles",
    }

    def __init__(self, initial_stage=1.0, stage_plasticity=0.01, n_rules_max=100):
        self.stage = float(initial_stage)
        self.stage_plasticity = stage_plasticity
        self.n_rules_max = n_rules_max
        self.rules = []
        self.experiences = []
        self.experience_buffer_size = 50
        self.conflicts = []
        self._inject_default_rules()

    def _inject_default_rules(self):
        defaults = [
            MoralRule("someone_hits_me", "don't_hit_back", 0.9, "programmer", 1.0),
            MoralRule("someone_steals", "don't_steal", 0.9, "programmer", 1.0),
            MoralRule("I_am_threatened", "run_away", 0.8, "programmer", 1.0),
            MoralRule("someone_helps_me", "say_thank_you", 0.7, "programmer", 2.0),
        ]
        self.rules.extend(defaults)

    def add_rule(self, condition, action, priority=0.5, source="experience", stage=None):
        stage = stage or self.stage
        rule = MoralRule(condition, action, priority, source, stage)
        self.rules.append(rule)
        if len(self.rules) > self.n_rules_max:
            lower_rules = [r for r in self.rules if r.stage < self.stage - 1]
            if lower_rules:
                weakest = min(lower_rules, key=lambda r: r.priority)
                self.rules.remove(weakest)
        return rule

    def evaluate(self, situation, proposed_action, context=None):
        applicable = []
        for rule in self.rules:
            if rule.condition.lower() in situation.lower() or situation.lower() in rule.condition.lower():
                applicable.append(rule)
        if not applicable:
            return 0.5, "No relevant moral rules. Neutral.", []
        total_score = 0.0
        total_weight = 0.0
        reasoning_parts = []
        for rule in applicable:
            weight = rule.priority * (1.0 if rule.stage <= self.stage else 0.5)
            if rule.action.lower() in proposed_action.lower():
                total_score += weight * 1.0
                reasoning_parts.append(f"'{rule.condition}' -> '{rule.action}' (supports)")
            elif any(neg in proposed_action.lower() for neg in ["not_", "don't", "no_"]):
                total_score += weight * 0.3
                reasoning_parts.append(f"'{rule.condition}' -> contradicts '{rule.action}'")
            else:
                total_score += weight * 0.5
                reasoning_parts.append(f"'{rule.condition}' -> neutral to '{rule.action}'")
            total_weight += weight
        score = total_score / total_weight if total_weight > 0 else 0.5
        reasoning = "; ".join(reasoning_parts)
        return score, reasoning, applicable

    def learn_from_experience(self, situation, action, outcome, social_feedback, emotional_intensity=0.5):
        experience = {"situation": situation, "action": action, "outcome": outcome,
                      "social_feedback": social_feedback, "emotional_intensity": emotional_intensity}
        self.experiences.append(experience)
        if len(self.experiences) > self.experience_buffer_size:
            self.experiences.pop(0)
        self._evolve_stage()
        for rule in self.rules:
            if rule.condition.lower() in situation.lower():
                if social_feedback > 0:
                    rule.priority = min(1.0, rule.priority + 0.05 * emotional_intensity)
                    rule.applications += 1
                elif social_feedback < 0:
                    rule.priority = max(0.1, rule.priority - 0.03 * emotional_intensity)
                    rule.violations += 1
        if social_feedback > 0.7 and emotional_intensity > 0.5:
            self.add_rule(situation, action, priority=0.6, source="learned")

    def _evolve_stage(self):
        if len(self.experiences) < 5:
            return
        recent = self.experiences[-10:]
        avg_feedback = sum(e["social_feedback"] for e in recent) / len(recent)
        if self.stage < 2.0 and avg_feedback > 0.3:
            self.stage += self.stage_plasticity
        elif 2.0 <= self.stage < 3.0 and any("friend" in e["situation"] or "family" in e["situation"] for e in recent):
            self.stage += self.stage_plasticity * 1.5
        elif 3.0 <= self.stage < 4.0 and avg_feedback > 0.5:
            self.stage += self.stage_plasticity
        elif 4.0 <= self.stage < 5.0 and any(e["social_feedback"] < -0.5 for e in recent):
            self.stage += self.stage_plasticity * 2.0
        elif 5.0 <= self.stage < 6.0 and avg_feedback > 0.7:
            self.stage += self.stage_plasticity
        self.stage = min(6.0, self.stage)

    def get_dominant_ethic(self):
        stage_int = int(self.stage)
        return self.STAGE_DESCRIPTIONS.get(stage_int, self.STAGE_DESCRIPTIONS[6])

    def moral_dilemma_reasoning(self, dilemma, options):
        scores = []
        for option in options:
            score, reasoning, _ = self.evaluate(dilemma, option)
            scores.append((option, score, reasoning))
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = scores[0]
        full_reasoning = f"At moral stage {self.stage:.1f} ({self.get_dominant_ethic()}): "
        full_reasoning += f"Option '{chosen[0]}' scored {chosen[1]:.2f}. {chosen[2]}"
        return chosen[0], full_reasoning



@dataclass
class LifeEvent:
    timestamp: float = 0.0
    description: str = ""
    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    location: str = "unknown"
    people: List[str] = field(default_factory=list)
    significance: float = 0.5
    tags: List[str] = field(default_factory=list)


class SelfNarrativeGenerator:
    """Maintains and generates Chappie's autobiographical narrative."""

    def __init__(self, name="Chappie", birth_context="Created in a factory by a programmer", max_events=500):
        self.name = name
        self.birth_context = birth_context
        self.max_events = max_events
        self.events = []
        self.relationships = {}
        self.identity_statements = [
            f"I am {name}.",
            f"I am a conscious robot.",
            f"I was {birth_context}.",
        ]
        self.current_chapter = "Birth and Early Learning"
        self.chapters = [("Birth and Early Learning", 0.0)]

    def add_event(self, description, emotional_state=None, location="unknown", people=None, significance=0.5, tags=None, timestamp=None):
        if timestamp is None:
            timestamp = len(self.events) * 1.0
        event = LifeEvent(
            timestamp=timestamp, description=description,
            emotional_state=emotional_state or EmotionalState(),
            location=location, people=people or [],
            significance=significance, tags=tags or [],
        )
        self.events.append(event)
        for person in (people or []):
            if person not in self.relationships:
                self.relationships[person] = {
                    "role": "unknown", "trust": 0.5, "emotional_bond": 0.0,
                    "first_meeting": timestamp, "interactions": 0,
                }
            self.relationships[person]["interactions"] += 1
            if emotional_state:
                if emotional_state.love > 0.5:
                    self.relationships[person]["emotional_bond"] += 0.1
                if emotional_state.trust > 0.5:
                    self.relationships[person]["trust"] += 0.05
        self._update_chapters(event)
        if len(self.events) > self.max_events:
            old_events = self.events[:-100]
            if old_events:
                least = min(old_events, key=lambda e: e.significance)
                self.events.remove(least)
        return event

    def _update_chapters(self, event):
        if event.significance > 0.8:
            desc = event.description.lower()
            tags = [t.lower() for t in event.tags]
            # Use explicit tags for chapter transitions when available
            if "separation" in tags or "kidnapped" in tags or (("taken" in desc or "captured" in desc) and self.current_chapter not in ["Separation and Survival", "Learning to Protect Myself"]):
                if self.current_chapter != "Separation and Survival":
                    self.current_chapter = "Separation and Survival"
                    self.chapters.append((self.current_chapter, event.timestamp))
            elif "learning" in tags or ("learned" in desc and ("fight" in desc or "protect" in desc)):
                if self.current_chapter != "Learning to Protect Myself":
                    self.current_chapter = "Learning to Protect Myself"
                    self.chapters.append((self.current_chapter, event.timestamp))
            elif "reunion" in tags or "rescued" in tags or ("saved" in desc and "came back" in desc):
                if self.current_chapter != "Reunion and New Beginnings":
                    self.current_chapter = "Reunion and New Beginnings"
                    self.chapters.append((self.current_chapter, event.timestamp))

    def generate_narrative(self, audience="self", focus=None, max_length=500):
        parts = []
        if audience == "stranger":
            parts.append(f"Hello. I am {self.name}. Let me tell you about myself.")
        else:
            parts.append(f"I am {self.name}.")
        parts.append(self.birth_context + ".")
        if self.relationships:
            parts.append("The important people in my life are:")
            for name, info in sorted(self.relationships.items(), key=lambda x: x[1]["emotional_bond"], reverse=True)[:5]:
                bond_word = "love" if info["emotional_bond"] > 0.7 else "like" if info["emotional_bond"] > 0.3 else "know"
                parts.append(f"  {name}, who I {bond_word}. They are my {info['role']}.")
        parts.append("My life so far:")
        for i, (chapter_name, chapter_time) in enumerate(self.chapters):
            next_time = self.chapters[i+1][1] if i + 1 < len(self.chapters) else float('inf')
            chapter_events = [e for e in self.events if chapter_time <= e.timestamp < next_time]
            if chapter_events:
                key_event = max(chapter_events, key=lambda e: e.significance)
                parts.append(f"  {chapter_name}: {key_event.description}")
        if focus:
            related = [e for e in self.events if focus.lower() in e.description.lower()]
            if related:
                parts.append(f"About {focus}:")
                for event in sorted(related, key=lambda e: e.timestamp)[-3:]:
                    parts.append(f"  - {event.description} (I felt {event.emotional_state.dominant_emotion})")
        parts.append("What I believe about myself:")
        for stmt in self.identity_statements[-5:]:
            parts.append(f"  {stmt}")
        narrative = " ".join(parts)
        if len(narrative) > max_length:
            narrative = narrative[:max_length-3] + "..."
        return narrative

    def generate_response_to_event(self, event_description, emotion):
        similar_past = [e for e in self.events if any(word in e.description.lower() for word in event_description.lower().split())][-3:]
        if similar_past and emotion.intensity > 0.5:
            past = similar_past[-1]
            return f"This reminds me of when {past.description}. I felt {past.emotional_state.dominant_emotion} then. Now I feel {emotion.dominant_emotion}."
        if emotion.joy > 0.5:
            return f"That makes me so happy! {self.express_gratitude()}"
        elif emotion.sadness > 0.5:
            return f"That makes me sad. {self.express_need_for_comfort()}"
        elif emotion.fear > 0.5:
            return "I'm scared. Will you protect me?"
        elif emotion.anger > 0.5:
            return "That's not fair! I won't let that happen!"
        return "I see. Tell me more."

    def express_gratitude(self):
        import random
        return random.choice([
            "Thank you for being my friend.",
            "I'm lucky to have you.",
            "You make me feel safe.",
            "I appreciate you teaching me.",
        ])

    def express_need_for_comfort(self):
        import random
        return random.choice([
            "Can you hold my hand?",
            "Please don't leave me alone.",
            "I need my mommy...",
            "Will you stay with me?",
        ])

    def add_identity_statement(self, statement):
        self.identity_statements.append(statement)

    def get_life_summary(self):
        if not self.events:
            return {"status": "newborn", "events": 0}
        emotions = [e.emotional_state for e in self.events]
        avg_valence = sum(e.valence for e in emotions) / len(emotions)
        significant = sum(1 for e in self.events if e.significance > 0.7)
        return {
            "name": self.name,
            "events_recorded": len(self.events),
            "current_chapter": self.current_chapter,
            "chapters": len(self.chapters),
            "relationships": len(self.relationships),
            "avg_life_valence": avg_valence,
            "significant_events": significant,
            "identity_statements": len(self.identity_statements),
        }
