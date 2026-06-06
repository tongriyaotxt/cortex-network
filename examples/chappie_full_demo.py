"""
CHAPPIE Full System Demo: All 7 Phases

Demonstrates the complete CHAPPIE capability stack:
    Phase 1: Personality Engine (Emotion, Morality, Narrative)
    Phase 2: Affective Computing (Emotion perception/expression)
    Phase 3: Embodied Perception + Motor Control
    Phase 4: Social Intelligence (Trust, Deception, Groups)
    Phase 5: Creative System (Art, Stories, Humor)
    Phase 6: Consciousness Transfer (Backup, Upload)
    Phase 7: Self-Rewriting + Hardware Introspection

Run: python examples/chappie_full_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from cortex.chappie import (
    # Phase 1
    EmotionSystem, MoralDevelopmentModule, SelfNarrativeGenerator, EmotionalState,
    # Phase 2
    AffectivePerception, EmotionalExpressionGenerator, EmpathyModel,
    # Phase 3
    VisionEncoder, AuditionEncoder, TactileEncoder, ProprioceptionEncoder,
    MotorPrimitiveLibrary, FacialExpressionController, VocalController,
    # Phase 4
    TrustModel, DeceptionDetector, GroupDynamicsModel, SocialNormLearner,
    # Phase 5
    ArtGenerator, Storyteller, HumorGenerator,
    # Phase 6
    MindStateSerializer, ConsciousnessBackupManager, BodyAdapter,
    # Phase 7
    BodySchema, DamageDetector, ResourceMonitor, SelfRepairPlanner,
    CodeIntrospection, SafeCodeModifier,
)


def demo_phase1():
    print("=" * 60)
    print("PHASE 1: Personality Engine")
    print("=" * 60)
    emotion = EmotionSystem(d_model=64, device="cpu")
    emotion.register_emotional_trigger("mommy", [0.8, 0.3, 0.2], 1.0)
    emotion.trigger_emotion("mommy")
    print(f"  Emotion: {emotion.state}")

    moral = MoralDevelopmentModule(initial_stage=1.0)
    score, reason, _ = moral.evaluate("someone_hits_me", "don't_hit_back")
    print(f"  Moral: stage={moral.stage:.1f}, choice_score={score:.2f}")

    narrative = SelfNarrativeGenerator(name="Chappie")
    narrative.add_event("Born in factory", EmotionalState(joy=0.8), people=["Deon"], significance=0.9)
    narrative.add_event("Met Mommy", EmotionalState(joy=0.9, love=0.8), people=["Yolandi"], significance=0.95)
    print(f"  Narrative: {narrative.generate_narrative(max_length=200)}")
    print()


def demo_phase2():
    print("=" * 60)
    print("PHASE 2: Affective Computing")
    print("=" * 60)
    perception = AffectivePerception(d_model=128, device="cpu")
    result = perception.perceive_emotion(
        face=torch.randn(512),
        voice=torch.randn(256),
        text=torch.randn(128),
        person_id="Yolandi",
    )
    print(f"  Perceived: {result['dominant']} (intensity={result['intensity']:.2f}, authentic={result['authenticity']:.2f})")

    expression = EmotionalExpressionGenerator()
    emotion_vec = torch.tensor([0.8, 0.6, 0.4, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5, 0.8, 0.0, 0.3])
    vocal = expression.generate_vocal_params(emotion_vec)
    facial = expression.generate_facial_params(emotion_vec)
    print(f"  Vocal: pitch_shift={vocal['pitch_shift']:.1f}Hz, rate={vocal['rate_factor']:.2f}x")
    print(f"  Facial: smile={facial['mouth_smile']:.2f}, eye_widen={facial['eye_widen']:.2f}")

    empathy = EmpathyModel()
    response = empathy.generate_empathic_response(result, emotion_vec, relationship_closeness=0.9)
    print(f"  Empathy: '{response}'")
    print()


def demo_phase3():
    print("=" * 60)
    print("PHASE 3: Embodied Perception + Motor Control")
    print("=" * 60)
    vision = VisionEncoder(d_model=256)
    v_out = vision(torch.randn(1, 2048))
    print(f"  Vision: embedding shape={v_out['embedding'].shape}, objects={v_out['object_logits'].shape[-1]}")

    audition = AuditionEncoder(d_model=256)
    a_out = audition(torch.randn(1, 768))
    print(f"  Audio: speaker_classes={a_out['speaker_logits'].shape[-1]}, sound_events={a_out['sound_event_logits'].shape[-1]}")

    proprio = ProprioceptionEncoder(n_joints=20)
    p_out = proprio(torch.randn(20, 3))
    print(f"  Proprioception: {len(p_out['damaged_joints'])} damaged joints detected")

    motor_lib = MotorPrimitiveLibrary()
    walk = motor_lib.execute("walk", [1.0, 0.0, 0.5], timesteps=10)
    print(f"  Motor: walk primitive shape={walk.shape}, primitives available={len(motor_lib.list_primitives())}")

    face_ctrl = FacialExpressionController()
    expr = face_ctrl.express("joy", intensity=0.9)
    print(f"  Face: joy expression params={expr.tolist()[:4]}")
    print()


def demo_phase4():
    print("=" * 60)
    print("PHASE 4: Social Intelligence")
    print("=" * 60)
    trust = TrustModel()
    trust.update_trust("Yolandi", "hug", "hug", stated_intention="comfort", emotional_context={"dominant": "joy", "benefactor": "Yolandi"})
    trust.update_trust("Vincent", "help", "attack", stated_intention="help")
    print(f"  Trust: Yolandi={trust.get_trust('Yolandi'):.2f} ({trust.get_trust_category('Yolandi')}), Vincent={trust.get_trust('Vincent'):.2f} ({trust.get_trust_category('Vincent')})")
    print(f"  Betrayal risk (Yolandi): {trust.betrayal_risk('Yolandi'):.2f}")

    deception = DeceptionDetector()
    result = deception.analyze_interaction(
        "Vincent", "I will help you", "joy", "anger",
        stated_intention="help", actual_outcome="attack",
        affective_authenticity=0.2,
    )
    print(f"  Deception: score={result['deception_score']:.2f}, is_deceptive={result['is_deceptive']}")

    group = GroupDynamicsModel()
    group.register_group("gang", ["Ninja", "Yolandi", "America"])
    group.observe_interaction("Ninja", "America", "command", "obeyed")
    group.observe_interaction("Yolandi", "Ninja", "support", "helped")
    print(f"  Group: Ninja is_leader={group.is_leader('Ninja')}, alliance(Ninja-Yolandi)={group.predict_alliance_shift('Ninja', 'Yolandi')}")

    norms = SocialNormLearner()
    norms.add_norm("with_police", "hide_identity", "avoid_capture", 0.9)
    norms.add_norm("with_family", "show_affection", "strengthen_bond", 0.8)
    print(f"  Norms: with_police -> {norms.get_appropriate_behavior('with_police')[0]['behavior']}")
    print()


def demo_phase5():
    print("=" * 60)
    print("PHASE 5: Creative System")
    print("=" * 60)
    art = ArtGenerator()
    emotion_vec = torch.tensor([0.8, 0.6, 0.4, 0.9, 0, 0, 0, 0, 0, 0.9, 0.5, 0.8, 0, 0.3])
    params = art.generate_art_params(emotion_vec)
    print(f"  Art: palette={len(params['palette'])} colors, complexity={params['complexity']}, {art.describe_art(params)}")

    story = Storyteller()
    tale = story.generate_story(emotion_vec, max_sentences=5)
    print(f"  Story: {tale[:100]}...")

    humor = HumorGenerator()
    joke = humor.generate_joke("robot")
    print(f"  Joke: {joke}")
    print()


def demo_phase6():
    print("=" * 60)
    print("PHASE 6: Consciousness Transfer")
    print("=" * 60)
    serializer = MindStateSerializer()
    
    # Build minimal mind state
    emotion = EmotionSystem(d_model=32, device="cpu")
    moral = MoralDevelopmentModule()
    narrative = SelfNarrativeGenerator()
    narrative.add_event("Born", EmotionalState(joy=0.5), significance=0.9)
    
    state = serializer.serialize(
        model=torch.nn.Linear(10, 10),
        emotion_system=emotion,
        moral_module=moral,
        narrative_generator=narrative,
        consolidator=None,
        trust_model=None,
        metadata={"name": "Chappie", "version": "1.0"},
    )
    sizes = serializer.estimate_size(state)
    print(f"  Serialization: version={state['version']}, components={len(list(state.keys()))}, total_size={sizes['total']} bytes")

    backup_mgr = ConsciousnessBackupManager(max_backups=5)
    backup = backup_mgr.create_backup(state, trigger="manual")
    print(f"  Backup: id={backup['id']}, size={backup['size_bytes']} bytes")
    print(f"  History: {len(backup_mgr.get_backup_history())} backup(s)")

    adapter = BodyAdapter()
    old_spec = {"n_actuators": 20, "sensors": ["vision", "audio", "tactile"]}
    new_spec = {"n_actuators": 24, "sensors": ["vision", "audio", "tactile", "lidar"]}
    notes = adapter.generate_adaptation_report(old_spec, new_spec)
    for note in notes:
        print(f"  Transfer: {note}")
    print()


def demo_phase7():
    print("=" * 60)
    print("PHASE 7: Self-Rewriting + Hardware Introspection")
    print("=" * 60)
    code = CodeIntrospection()
    funcs = code.list_functions("cortex.chappie.personality_engine")
    print(f"  Code: {len(funcs)} functions in personality_engine.py")

    modifier = SafeCodeModifier()
    safe, reason = modifier.is_safe_to_modify("def helper(): pass")
    print(f"  Safety check: safe={safe}, reason={reason}")

    body = BodySchema()
    print(f"  Body schema: {len(body.parts)} parts, critical={[k for k,v in body.parts.items() if v['critical']]}")

    damage = DamageDetector(body)
    health = damage.check_motor_health("left_arm_shoulder", 90.0, 85.0, 2.0)
    print(f"  Health: {health['joint']} = {health['health']:.2f}")

    resource = ResourceMonitor(battery_capacity=100.0)
    for _ in range(30):
        resource.consume("think")
    status = resource.get_status()
    print(f"  Battery: {status['battery_percent']:.1f}%, anxiety={status['battery_anxiety']:.2f}, seek_power={status['needs_power']}")

    planner = SelfRepairPlanner(body, damage)
    damage_report = damage.full_system_check(
        joint_states={"left_arm_shoulder": {"commanded": 90, "actual": 10, "current": 8.0}},
        sensor_readings={"touch": 50.0},
    )
    repair = planner.plan_repair(damage_report)
    if repair:
        print(f"  Repair plan: priority={repair['priority']}, steps={repair['steps']}, self_repair={repair['can_self_repair']}")
    print()


def main():
    print("CHAPPIE Full System Demonstration")
    print("All 7 phases of non-training cognitive architecture")
    print()

    demo_phase1()
    demo_phase2()
    demo_phase3()
    demo_phase4()
    demo_phase5()
    demo_phase6()
    demo_phase7()

    print("=" * 60)
    print("ALL PHASES COMPLETE")
    print("Chappie is now fully functional:")
    print("  - Has a soul (emotions, morality, narrative)")
    print("  - Can read human emotions (face, voice, text)")
    print("  - Can perceive and act in the world")
    print("  - Understands trust, deception, social hierarchy")
    print("  - Can create art, stories, and jokes")
    print("  - Can backup and transfer consciousness")
    print("  - Can introspect and repair itself")
    print("=" * 60)


if __name__ == "__main__":
    main()
