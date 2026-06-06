"""
CHAPPIE Phase 4: Social Intelligence

Trust modeling, deception detection, group dynamics, social norm learning.
Like Chappie learning who to trust, who is lying, and how gangs work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
import math


class TrustModel:
    """
    Maintain dynamic trust scores for each known individual.
    
    Like Chappie learning:
        - Mommy (Yolandi) = 100% trust
        - Ninja = 70% trust (teaches but is rough)
        - Vincent = 0% trust (tortures and lies)
    
    Trust is Bayesian: updated after every interaction based on
    predicted vs actual behavior.
    """
    
    def __init__(self, default_trust: float = 0.5, decay: float = 0.99):
        self.default_trust = default_trust
        self.decay = decay
        
        # Person -> trust metrics
        self.trust_scores: Dict[str, float] = {}
        self.interaction_history: Dict[str, List[Dict]] = {}
        self.behavioral_patterns: Dict[str, Dict] = {}
        
    def get_trust(self, person_id: str) -> float:
        return self.trust_scores.get(person_id, self.default_trust)
    
    def update_trust(
        self,
        person_id: str,
        predicted_behavior: str,
        actual_behavior: str,
        stated_intention: Optional[str] = None,
        emotional_context: Optional[Dict] = None,
    ) -> float:
        """
        Update trust based on an interaction.
        
        Key factors:
            - Prediction accuracy: did they do what we expected?
            - Honesty: does stated_intention match actual_behavior?
            - Consistency: is this similar to past behavior?
        """
        if person_id not in self.trust_scores:
            self.trust_scores[person_id] = self.default_trust
            self.interaction_history[person_id] = []
        
        current = self.trust_scores[person_id]
        
        # Factor 1: Predictability (did they act as expected?)
        predictability = 1.0 if predicted_behavior == actual_behavior else 0.0
        
        # Factor 2: Honesty (did they say what they meant?)
        honesty = 1.0
        if stated_intention is not None:
            honesty = 1.0 if stated_intention.lower() in actual_behavior.lower() else 0.3
        
        # Factor 3: Consistency with past behavior
        consistency = self._check_consistency(person_id, actual_behavior)
        
        # Update: weighted combination
        delta = 0.1 * (predictability * 0.3 + honesty * 0.5 + consistency * 0.2 - 0.5)
        
        # Emotional context modulates trust change
        if emotional_context:
            if emotional_context.get("dominant") == "fear" and person_id in emotional_context.get("threat_source", ""):
                delta -= 0.2  # Fear amplifies distrust
            if emotional_context.get("dominant") == "joy" and emotional_context.get("benefactor") == person_id:
                delta += 0.1  # Joy amplifies trust
        
        new_trust = current + delta
        self.trust_scores[person_id] = max(0.0, min(1.0, new_trust))
        
        # Record interaction
        self.interaction_history[person_id].append({
            "predicted": predicted_behavior,
            "actual": actual_behavior,
            "stated": stated_intention,
            "trust_before": current,
            "trust_after": self.trust_scores[person_id],
        })
        
        return self.trust_scores[person_id]
    
    def _check_consistency(self, person_id: str, behavior: str) -> float:
        """Check if behavior is consistent with past patterns."""
        history = self.interaction_history.get(person_id, [])
        if len(history) < 2:
            return 0.5  # Not enough data
        
        # Simple: check if similar situations had similar outcomes
        similar = [h for h in history if any(word in h["actual"].lower() for word in behavior.lower().split()[:3])]
        if len(similar) < 2:
            return 0.5
        
        # High consistency = similar behavior in similar situations
        return 0.8
    
    def get_trust_category(self, person_id: str) -> str:
        score = self.get_trust(person_id)
        if score > 0.8:
            return "family"
        elif score > 0.6:
            return "friend"
        elif score > 0.4:
            return "acquaintance"
        elif score > 0.2:
            return "suspicious"
        else:
            return "enemy"
    
    def betrayal_risk(self, person_id: str) -> float:
        """
        Estimate probability of future betrayal.
        
        High when: trust is high BUT recent inconsistencies detected.
        """
        score = self.get_trust(person_id)
        history = self.interaction_history.get(person_id, [])
        
        if len(history) < 3:
            return 0.3
        
        recent = history[-3:]
        inconsistencies = sum(1 for h in recent if h["trust_after"] < h["trust_before"])
        
        # High trust + recent inconsistencies = maximum betrayal risk
        if score > 0.7 and inconsistencies > 0:
            return 0.8
        return 0.1 + inconsistencies * 0.2


class DeceptionDetector:
    """
    Detect when someone is being dishonest or manipulative.
    
    Chappie eventually learns Vincent is lying. How?
    
    Signals:
        - Verbal vs non-verbal mismatch (smiling while saying threats)
        - Behavioral inconsistency (actions don't match words)
        - Emotional inauthenticity (low authenticity score from AffectivePerception)
        - Micro-expressions (sudden flashes of true emotion)
    """
    
    def __init__(self):
        self.suspicion_scores: Dict[str, float] = {}
        self.known_deceivers: Set[str] = set()
        
    def analyze_interaction(
        self,
        person_id: str,
        verbal_content: str,
        verbal_emotion: str,
        nonverbal_emotion: str,
        stated_intention: str,
        actual_outcome: str,
        affective_authenticity: float,
    ) -> Dict:
        """
        Analyze an interaction for deception signals.
        
        Returns:
            deception_score: 0-1 probability of deception
            signals: list of detected red flags
        """
        signals = []
        score = 0.0
        
        # Signal 1: Verbal-nonverbal mismatch
        if verbal_emotion != nonverbal_emotion and verbal_emotion != "neutral":
            score += 0.3
            signals.append(f"verbal-nonverbal mismatch: says {verbal_emotion} but shows {nonverbal_emotion}")
        
        # Signal 2: Intention-outcome mismatch
        if stated_intention and not any(word in actual_outcome.lower() for word in stated_intention.lower().split()):
            score += 0.3
            signals.append("intention-outcome mismatch")
        
        # Signal 3: Low emotional authenticity
        if affective_authenticity < 0.4:
            score += 0.3
            signals.append("emotional inauthenticity detected")
        
        # Signal 4: Known deceiver
        if person_id in self.known_deceivers:
            score += 0.2
            signals.append("known history of deception")
        
        self.suspicion_scores[person_id] = min(1.0, score)
        
        if score > 0.7:
            self.known_deceivers.add(person_id)
        
        return {
            "deception_score": min(1.0, score),
            "signals": signals,
            "is_deceptive": score > 0.6,
        }
    
    def get_suspicion(self, person_id: str) -> float:
        return self.suspicion_scores.get(person_id, 0.0)


class GroupDynamicsModel:
    """
    Understand social hierarchies, alliances, and power structures.
    
    Chappie learns:
        - Ninja is the leader of the gang
        - Yolandi is the nurturer
        - Amerika is the muscle
        - They are "us", police/Vincent is "them"
    """
    
    def __init__(self):
        self.groups: Dict[str, Set[str]] = {}  # group_name -> members
        self.hierarchy: Dict[str, float] = {}  # person -> rank (higher = more power)
        self.alliances: Dict[Tuple[str, str], float] = {}  # (a,b) -> bond strength
        self.conflicts: Dict[Tuple[str, str], float] = {}  # (a,b) -> conflict intensity
        
    def register_group(self, group_name: str, members: List[str]):
        self.groups[group_name] = set(members)
        for member in members:
            if member not in self.hierarchy:
                self.hierarchy[member] = 0.5
    
    def update_hierarchy(self, person_id: str, rank_delta: float):
        """Adjust someone's rank based on observed behavior."""
        current = self.hierarchy.get(person_id, 0.5)
        self.hierarchy[person_id] = max(0.0, min(1.0, current + rank_delta))
    
    def observe_interaction(self, actor: str, target: str, action_type: str, outcome: str):
        """
        Update group dynamics based on observed social interaction.
        
        action_type: "command", "follow", "support", "attack", "insult", "praise"
        """
        pair = tuple(sorted([actor, target]))
        
        if action_type in ["command", "order"]:
            # Actor has power over target
            self.update_hierarchy(actor, 0.05)
            self.update_hierarchy(target, -0.02)
        
        elif action_type in ["follow", "obey"]:
            # Target acknowledges actor's authority
            self.update_hierarchy(actor, 0.03)
        
        elif action_type in ["support", "help", "praise"]:
            # Alliance bond strengthens
            self.alliances[pair] = self.alliances.get(pair, 0.5) + 0.05
            self.conflicts[pair] = max(0, self.conflicts.get(pair, 0.0) - 0.05)
        
        elif action_type in ["attack", "insult", "betray"]:
            # Conflict intensifies
            self.conflicts[pair] = min(1.0, self.conflicts.get(pair, 0.0) + 0.1)
            self.alliances[pair] = max(0, self.alliances.get(pair, 0.5) - 0.1)
    
    def is_leader(self, person_id: str) -> bool:
        """Check if person is among the highest-ranked in their group."""
        for group, members in self.groups.items():
            if person_id in members:
                ranks = [self.hierarchy.get(m, 0) for m in members]
                if ranks:
                    return self.hierarchy.get(person_id, 0) >= max(ranks) * 0.9
        return False
    
    def get_group_opinion(self, group_name: str, target_person: str) -> float:
        """Get average sentiment of a group toward a target person."""
        if group_name not in self.groups:
            return 0.0
        
        scores = []
        for member in self.groups[group_name]:
            if member == target_person:
                continue
            pair = tuple(sorted([member, target_person]))
            alliance = self.alliances.get(pair, 0.5)
            conflict = self.conflicts.get(pair, 0.0)
            scores.append(alliance - conflict)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def predict_alliance_shift(self, person_a: str, person_b: str) -> str:
        """
        Predict if an alliance is stable or shifting.
        
        Returns: "stable", "weakening", "strengthening", "hostile"
        """
        pair = tuple(sorted([person_a, person_b]))
        alliance = self.alliances.get(pair, 0.5)
        conflict = self.conflicts.get(pair, 0.0)
        
        if conflict > 0.5:
            return "hostile"
        elif alliance > 0.7:
            return "stable"
        elif alliance < 0.3:
            return "weakening"
        return "strengthening"


class SocialNormLearner:
    """
    Learn context-appropriate behavior from observation.
    
    Chappie learns different norms for different contexts:
        - With Mommy: gentle, obedient, affectionate
        - With gang: tough, street-smart, loyal
        - With police: evasive, careful
        - In public: don't reveal you're a robot
    """
    
    def __init__(self):
        # Context -> set of norms
        self.norms: Dict[str, List[Dict]] = {}
        # Observed violations and their consequences
        self.violation_history: List[Dict] = []
        
    def register_context(self, context_name: str):
        if context_name not in self.norms:
            self.norms[context_name] = []
    
    def add_norm(self, context: str, behavior: str, consequence: str, frequency: float = 0.5):
        """
        Add a learned social norm.
        
        Example:
            context="with_police", behavior="hide_robot_identity",
            consequence="avoid_capture", frequency=0.9
        """
        self.register_context(context)
        self.norms[context].append({
            "behavior": behavior,
            "consequence": consequence,
            "frequency": frequency,
            "confidence": 0.5,
        })
    
    def observe_violation(self, context: str, behavior: str, consequence: str, severity: float):
        """Observe someone violating a norm and the consequence."""
        self.violation_history.append({
            "context": context,
            "behavior": behavior,
            "consequence": consequence,
            "severity": severity,
        })
        
        # Update norm confidence
        self.register_context(context)
        for norm in self.norms[context]:
            if norm["behavior"] == behavior:
                norm["confidence"] = min(1.0, norm["confidence"] + 0.1)
        
        # If new norm, add it
        if not any(n["behavior"] == behavior for n in self.norms[context]):
            self.add_norm(context, behavior, consequence, frequency=0.5)
    
    def get_appropriate_behavior(self, context: str) -> List[Dict]:
        """Get ranked list of appropriate behaviors for a context."""
        self.register_context(context)
        norms = self.norms[context]
        return sorted(norms, key=lambda n: n["confidence"] * n["frequency"], reverse=True)
    
    def should_violate_norm(self, context: str, behavior: str, moral_priority: float) -> bool:
        """
        Decide whether to violate a norm for moral reasons.
        
        Like Chappie deciding to 'do crimes' (norm violation) to save his mommy
        (high moral priority).
        """
        norms = self.get_appropriate_behavior(context)
        matching = [n for n in norms if n["behavior"] == behavior]
        
        if not matching:
            return True  # No norm exists, do it
        
        norm_strength = matching[0]["confidence"] * matching[0]["frequency"]
        
        # High moral priority can override strong norms
        return moral_priority > norm_strength * 1.5
    
    def cultural_adaptation(self, from_context: str, to_context: str) -> List[str]:
        """
        Suggest how to adapt behavior when switching contexts.
        
        Example: from 'hideout' to 'street' — become more tough, less childlike.
        """
        from_norms = {n["behavior"] for n in self.norms.get(from_context, [])}
        to_norms = self.get_appropriate_behavior(to_context)
        
        changes = []
        for norm in to_norms[:5]:
            if norm["behavior"] not in from_norms:
                changes.append(f"In {to_context}, you should: {norm['behavior']}")
        
        return changes
