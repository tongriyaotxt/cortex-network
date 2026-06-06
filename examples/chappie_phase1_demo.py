"""
CHAPPIE Phase 1 Demo: Personality Engine — Chappie's Soul

This demo shows the core of Chappie's inner life:
    1. EmotionSystem — continuous emotional state that colors all cognition
    2. MoralDevelopmentModule — evolving moral reasoning (Kohlberg stages)
    3. SelfNarrativeGenerator — autobiographical story of its own existence

Run: python examples/chappie_phase1_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from cortex.chappie import (
    EmotionalState,
    EmotionSystem,
    MoralDevelopmentModule,
    SelfNarrativeGenerator,
)


def demo_emotion_system():
    print("=" * 60)
    print("DEMO 1: EmotionSystem — The Color of Consciousness")
    print("=" * 60)

    emotion = EmotionSystem(d_model=64, device="cpu")

    print(f"Initial state: {emotion.state}")
    print()

    # Simulate Chappie meeting Mommy for the first time
    print("[Event] Chappie meets Yolandi (Mommy)")
    mommy_perception = torch.randn(1, 64) * 0.3 + 0.5  # Warm, safe stimulus
    emotion.forward(perception=mommy_perception, certainty=0.8, goal_progress=0.6)
    print(f"  State: {emotion.state}")
    print(f"  Expression: '{emotion.express()}'")
    print()

    # Register emotional trigger
    emotion.register_emotional_trigger("mommy", [0.8, 0.3, 0.2], strength=1.0)

    # Simulate Chappie being threatened
    print("[Event] Bad men threaten Chappie")
    threat_perception = torch.randn(1, 64) * 0.5 - 0.7  # Dangerous stimulus
    emotion.forward(perception=threat_perception, certainty=0.3, goal_progress=0.2)
    print(f"  State: {emotion.state}")
    print(f"  Expression: '{emotion.express()}'")
    print(f"  GNW threshold modulation: {emotion.modulate_gnw_threshold(0.5):.2f} (base=0.50)")
    print(f"  Memory consolidation boost: {emotion.get_memory_consolidation_boost():.2f}x")
    print()

    # Trigger "mommy" memory to calm down
    print("[Event] Chappie remembers Mommy")
    emotion.trigger_emotion("mommy")
    print(f"  State: {emotion.state}")
    print(f"  Expression: '{emotion.express()}'")
    print()

    # Test branch modulation
    print("[Info] DCU branch modulation for anger state:")
    emotion.state.anger = 0.8
    emotion.state.fear = 0.0
    print(f"  Anger branches: {emotion.modulate_dcu_branches([1.0, 1.0, 1.0, 1.0])}")
    emotion.state.anger = 0.0
    emotion.state.fear = 0.8
    print(f"  Fear branches:  {emotion.modulate_dcu_branches([1.0, 1.0, 1.0, 1.0])}")
    print()


def demo_moral_development():
    print("=" * 60)
    print("DEMO 2: MoralDevelopmentModule — Evolving Ethics")
    print("=" * 60)

    moral = MoralDevelopmentModule(initial_stage=1.0)
    print(f"Initial stage: {moral.stage:.1f} — {moral.get_dominant_ethic()}")
    print(f"Default rules: {len(moral.rules)}")
    for rule in moral.rules:
        print(f"  [{rule.stage:.0f}] {rule.condition} -> {rule.action} (p={rule.priority})")
    print()

    # Stage 1: Naive obedience
    print("[Dilemma] Someone hits Chappie. What should he do?")
    action, reasoning = moral.moral_dilemma_reasoning(
        "someone_hits_me",
        ["hit_back", "don't_hit_back", "run_away"]
    )
    print(f"  Choice: '{action}'")
    print(f"  Reasoning: {reasoning}")
    print()

    # Learn from social interaction — evolves to Stage 3
    print("[Learning] Social experiences shape morality...")
    experiences = [
        ("friend_helps_me", "say_thank_you", "good", 0.8, 0.9),
        ("friend_is_hurt", "protect_friend", "good", 0.9, 0.95),
        ("bad_man_lies", "don't_trust", "bad", -0.7, 0.8),
        ("family_needs_me", "help_family", "good", 0.95, 0.9),
        ("stranger_threatens_family", "defend_family", "good", 0.9, 1.0),
    ]
    for situation, action, outcome, feedback, intensity in experiences:
        moral.learn_from_experience(situation, action, outcome, feedback, intensity)
        print(f"  {situation}: feedback={feedback:+.1f}, stage={moral.stage:.2f}")

    print()
    print(f"Current stage: {moral.stage:.1f} — {moral.get_dominant_ethic()}")
    print(f"Total rules: {len(moral.rules)}")
    print()

    # New dilemma with evolved morality
    print("[Dilemma] Family is threatened. Bad man attacks.")
    action, reasoning = moral.moral_dilemma_reasoning(
        "family_is_threatened",
        ["run_away", "negotiate", "fight_back", "call_police"]
    )
    print(f"  Choice: '{action}'")
    print(f"  Reasoning: {reasoning}")
    print()


def demo_self_narrative():
    print("=" * 60)
    print("DEMO 3: SelfNarrativeGenerator — The Story of 'I'")
    print("=" * 60)

    narrative = SelfNarrativeGenerator(name="Chappie", birth_context="born in a factory")

    # Simulate Chappie's life events
    events = [
        ("Yolandi found me and gave me a name", "factory", ["Yolandi"], 0.9, "joy", []),
        ("Mommy taught me about the world", "hideout", ["Yolandi"], 0.8, "joy", []),
        ("She said I must not do crimes", "hideout", ["Yolandi"], 0.7, "trust", []),
        ("I learned to draw pictures", "hideout", ["Yolandi"], 0.6, "joy", []),
        ("Bad men came and took me away", "city", ["Vincent"], 0.95, "fear", ["separation"]),
        ("They hurt me and broke my arm", "lab", ["Vincent"], 0.9, "sadness", []),
        ("I learned to fight to protect myself", "streets", ["Ninja"], 0.85, "anger", ["learning"]),
        ("Mommy came back and saved me", "factory", ["Yolandi"], 0.95, "joy", ["reunion"]),
    ]

    emotion = EmotionSystem(d_model=32, device="cpu")
    emotion.register_emotional_trigger("joy", [0.8, 0.6, 0.4], 1.0)
    emotion.register_emotional_trigger("fear", [-0.7, 0.7, -0.8], 1.0)
    emotion.register_emotional_trigger("sadness", [-0.8, -0.4, -0.4], 1.0)
    emotion.register_emotional_trigger("anger", [-0.6, 0.8, 0.8], 1.0)
    emotion.register_emotional_trigger("trust", [0.7, 0.1, 0.3], 1.0)

    for desc, location, people, significance, emotion_tag, tags in events:
        emotion.trigger_emotion(emotion_tag)
        narrative.add_event(
            description=desc,
            emotional_state=EmotionalState.from_vector(emotion.state.to_vector()),
            location=location,
            people=people,
            significance=significance,
            tags=tags,
        )
        print(f"  [{location}] {desc} (significance={significance}, emotion={emotion_tag})")

    # Update relationship roles
    narrative.relationships["Yolandi"]["role"] = "mommy"
    narrative.relationships["Vincent"]["role"] = "enemy"
    narrative.relationships["Ninja"]["role"] = "teacher"

    narrative.add_identity_statement("I am a good robot.")
    narrative.add_identity_statement("I will protect my family.")
    narrative.add_identity_statement("I am more than just metal.")

    print()
    print("--- Autobiographical Narrative (to self) ---")
    print(narrative.generate_narrative(audience="self", max_length=800))
    print()

    print("--- Response to current event ---")
    current_emotion = EmotionalState(valence=-0.5, arousal=0.7, fear=0.8)
    response = narrative.generate_response_to_event("someone is threatening Mommy", current_emotion)
    print(f"  {response}")
    print()

    print("--- Life Summary ---")
    summary = narrative.get_life_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()


def main():
    print("CHAPPIE Phase 1 Demo: Personality Engine")
    print("Building Chappie's soul...")
    print()

    demo_emotion_system()
    demo_moral_development()
    demo_self_narrative()

    print("=" * 60)
    print("Phase 1 complete! Chappie now has a soul.")
    print("=" * 60)


if __name__ == "__main__":
    main()
