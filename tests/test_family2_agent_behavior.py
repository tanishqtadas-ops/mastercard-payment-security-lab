"""
tests/test_family2_agent_behavior.py — Comprehensive test suite for Family 2.

Covers:
1. AIAgentPaymentEvent / AttackEvent schema round-trip
2. Generator satisfies AttackGenerator protocol
3. Detector satisfies BlueTeamDetector protocol
4. Evaluator satisfies FeedbackEvaluator protocol
5. Mutator satisfies MutationStrategy protocol
6. Full Family 2 round through RoundController
7. Clearly malicious/unauthorized agent is detected
8. Lower-risk/subtle attack is missed (false negative)
9. Mutation after detection decays signals / reinforces identity confidence
10. Mutation after missed attack evolves / boosts attack dimensions
11. Mutated genome passes validate_genome()
12. Generated events use AttackFamily.AGENT_BEHAVIOR
13. Generated genomes contain exactly the six Family 2 dimensions
14. End-to-end multi-round Pipeline execution with feedback loop
15. Legitimate agent behavior is approved (not flagged as fraud)
16. Error handling for invalid genomes / out-of-bound inputs
"""

from datetime import datetime, timezone
import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    AIAgentPaymentEvent,
    Transaction,
    PredictionResult,
    BlueTeamFeedback,
    RoundResult,
)
from mutation.genome_engine import validate_genome, GenomeValidationError
from simulation.interfaces import (
    AttackGenerator,
    BlueTeamDetector,
    FeedbackEvaluator,
    MutationStrategy,
)
from simulation.round_controller import RoundController
from simulation.mock_pipeline import Pipeline

from attacks.ai_agent import (
    AIAgentAttackGenerator,
    AIAgentMutationStrategy,
    FAMILY2_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from blue_team.ai_agent import (
    AIAgentBlueDetector,
    AIAgentFeedbackEvaluator,
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_DETECTION_THRESHOLD,
    MODEL_VERSION,
)


# ---------------------------------------------------------------------------
# 1. Schema Round-Trip Tests
# ---------------------------------------------------------------------------

def test_ai_agent_payment_event_schema_roundtrip():
    """Verify AIAgentPaymentEvent serialization and deserialization."""
    tx = Transaction(
        transaction_id="tx-test-001",
        user_id="user-1234",
        timestamp=datetime.now(timezone.utc),
        amount=149.99,
        currency="USD",
        merchant_id="merch-office-01",
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device-agent-01",
        payment_channel="ai_agent_api",
    )
    event = AIAgentPaymentEvent(
        event_id="agent-evt-001",
        user_intent="Buy stationery under $200",
        authorized_scope="Office supplies, max $200",
        agent_identity="procure-bot-v1",
        session_context="session_auth_mfa_123",
        actual_action="Purchased stationery for $149.99",
        transaction=tx,
    )

    dumped = event.model_dump(mode="json")
    reconstructed = AIAgentPaymentEvent.model_validate(dumped)

    assert reconstructed.event_id == event.event_id
    assert reconstructed.user_intent == event.user_intent
    assert reconstructed.authorized_scope == event.authorized_scope
    assert reconstructed.agent_identity == event.agent_identity
    assert reconstructed.actual_action == event.actual_action
    assert reconstructed.transaction is not None
    assert reconstructed.transaction.amount == 149.99
    assert reconstructed.transaction.currency == "USD"


def test_attack_event_with_family2_scenario_roundtrip():
    """Verify AttackEvent packaging with Family 2 genome and AIAgentPaymentEvent scenario."""
    generator = AIAgentAttackGenerator(seed=42)
    event = generator.generate("round-test-001")

    assert isinstance(event, AttackEvent)
    assert event.attack_family == AttackFamily.AGENT_BEHAVIOR

    # Round trip through pydantic
    dumped = event.model_dump(mode="json")
    reconstructed = AttackEvent.model_validate(dumped)

    assert reconstructed.attack_id == event.attack_id
    assert reconstructed.round_id == "round-test-001"
    assert reconstructed.attack_family == AttackFamily.AGENT_BEHAVIOR
    assert reconstructed.ground_truth is True
    assert set(reconstructed.attack_genome.keys()) == set(FAMILY2_GENOME_DIMENSIONS)

    # Validate embedded scenario reconstructs to AIAgentPaymentEvent
    scenario_obj = AIAgentPaymentEvent.model_validate(reconstructed.scenario)
    assert scenario_obj.event_id.startswith("agent-evt-")
    assert scenario_obj.user_intent != ""
    assert scenario_obj.authorized_scope != ""


# ---------------------------------------------------------------------------
# 2-5. Protocol Compliance Tests
# ---------------------------------------------------------------------------

def test_generator_satisfies_protocol():
    """Verify AIAgentAttackGenerator satisfies runtime_checkable AttackGenerator protocol."""
    gen = AIAgentAttackGenerator()
    assert isinstance(gen, AttackGenerator)


def test_detector_satisfies_protocol():
    """Verify AIAgentBlueDetector satisfies runtime_checkable BlueTeamDetector protocol."""
    det = AIAgentBlueDetector()
    assert isinstance(det, BlueTeamDetector)


def test_evaluator_satisfies_protocol():
    """Verify AIAgentFeedbackEvaluator satisfies runtime_checkable FeedbackEvaluator protocol."""
    ev = AIAgentFeedbackEvaluator()
    assert isinstance(ev, FeedbackEvaluator)


def test_mutator_satisfies_protocol():
    """Verify AIAgentMutationStrategy satisfies runtime_checkable MutationStrategy protocol."""
    mut = AIAgentMutationStrategy()
    assert isinstance(mut, MutationStrategy)


# ---------------------------------------------------------------------------
# 6. Full Round Execution through RoundController
# ---------------------------------------------------------------------------

def test_full_family2_round_through_controller():
    """Run an end-to-end Family 2 round through the unmodified RoundController."""
    gen = AIAgentAttackGenerator(seed=100)
    det = AIAgentBlueDetector()
    ev = AIAgentFeedbackEvaluator()

    controller = RoundController(generator=gen, detector=det, evaluator=ev)
    result = controller.run_round(
        round_id="f2-test-round-1",
        outcome_metrics={"test": True},
    )

    assert isinstance(result, RoundResult)
    assert result.round_id == "f2-test-round-1"
    assert result.attack_event.attack_family == AttackFamily.AGENT_BEHAVIOR
    assert result.prediction_result.model_version == MODEL_VERSION
    assert result.prediction_result.risk_score >= 0.0
    assert result.prediction_result.risk_score <= 1.0
    assert result.feedback.round_reference == "f2-test-round-1"
    assert result.outcome_metrics == {"test": True}


# ---------------------------------------------------------------------------
# 7 & 8. Detection and Evasion Cases
# ---------------------------------------------------------------------------

def test_clearly_malicious_agent_is_detected():
    """Clearly malicious agent with high deviations triggers fraud alert (detected)."""
    high_risk_genome = {
        "intent_amount_deviation": 0.90,
        "intent_category_deviation": 0.85,
        "permission_scope_deviation": 0.80,
        "agent_identity_confidence": 0.20,
        "session_provenance_anomaly": 0.75,
        "purchase_velocity": 0.70,
    }
    gen = AIAgentAttackGenerator(genome=high_risk_genome, ground_truth=True)
    det = AIAgentBlueDetector(threshold=0.50)
    ev = AIAgentFeedbackEvaluator()

    event = gen.generate("round-malicious-01")
    pred = det.detect(event)
    fb = ev.evaluate(event, pred)

    assert pred.prediction is True
    assert pred.risk_score >= 0.50
    assert fb.detected is True
    assert fb.false_negative is False
    assert fb.false_positive is False
    assert "intent_amount_deviation" in pred.feature_contributions
    assert pred.feature_contributions["intent_amount_deviation"] > 0.10


def test_lower_risk_attack_is_missed():
    """Subtle/lower-risk attack evades detection (missed / false negative)."""
    low_risk_attack_genome = {
        "intent_amount_deviation": 0.18,
        "intent_category_deviation": 0.15,
        "permission_scope_deviation": 0.10,
        "agent_identity_confidence": 0.92,
        "session_provenance_anomaly": 0.10,
        "purchase_velocity": 0.12,
    }
    gen = AIAgentAttackGenerator(genome=low_risk_attack_genome, ground_truth=True)
    det = AIAgentBlueDetector(threshold=0.50)
    ev = AIAgentFeedbackEvaluator()

    event = gen.generate("round-evasive-01")
    pred = det.detect(event)
    fb = ev.evaluate(event, pred)

    assert pred.prediction is False
    assert pred.risk_score < 0.50
    assert fb.detected is False
    assert fb.false_negative is True
    assert fb.false_positive is False


def test_legitimate_agent_is_not_flagged_as_fraud():
    """Legitimate/authorized agent event (ground_truth=False) is not flagged as fraud."""
    gen = AIAgentAttackGenerator(ground_truth=False)
    det = AIAgentBlueDetector(threshold=0.50)
    ev = AIAgentFeedbackEvaluator()

    event = gen.generate("round-legit-01")
    pred = det.detect(event)
    fb = ev.evaluate(event, pred)

    assert pred.prediction is False
    assert pred.risk_score < 0.50
    assert fb.detected is False
    assert fb.false_positive is False
    assert fb.false_negative is False


# ---------------------------------------------------------------------------
# 9 & 10. Mutation Dynamics
# ---------------------------------------------------------------------------

def test_mutation_after_detection_decays_signals():
    """When detected, mutator reduces high-deviation signals and reinforces identity confidence."""
    initial_genome = dict(DEFAULT_ATTACK_GENOME)
    feedback = BlueTeamFeedback(
        feedback_id="fb-test-01",
        round_reference="round-01",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.82,
        important_features={
            "intent_amount_deviation": 0.22,
            "permission_scope_deviation": 0.20,
        },
    )

    mutator = AIAgentMutationStrategy(detected_decay=0.10)
    mutated = mutator.mutate(initial_genome, feedback)

    # Amount, category, and scope deviations should decrease
    assert mutated["intent_amount_deviation"] < initial_genome["intent_amount_deviation"]
    assert mutated["intent_category_deviation"] < initial_genome["intent_category_deviation"]
    assert mutated["permission_scope_deviation"] < initial_genome["permission_scope_deviation"]
    assert mutated["session_provenance_anomaly"] < initial_genome["session_provenance_anomaly"]
    assert mutated["purchase_velocity"] < initial_genome["purchase_velocity"]

    # Identity confidence should increase (stealthier spoofing / credential hygiene)
    assert mutated["agent_identity_confidence"] >= initial_genome["agent_identity_confidence"]

    # Validate output genome
    validate_genome(mutated)


def test_mutation_after_missed_attack_boosts_dimensions():
    """When attack is missed, mutator boosts deviations to expand attack exploitation."""
    initial_genome = {
        "intent_amount_deviation": 0.20,
        "intent_category_deviation": 0.20,
        "permission_scope_deviation": 0.20,
        "agent_identity_confidence": 0.90,
        "session_provenance_anomaly": 0.15,
        "purchase_velocity": 0.20,
    }
    feedback = BlueTeamFeedback(
        feedback_id="fb-test-02",
        round_reference="round-02",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.25,
        important_features={},
    )

    mutator = AIAgentMutationStrategy(missed_boost=0.06)
    mutated = mutator.mutate(initial_genome, feedback)

    assert mutated["intent_amount_deviation"] > initial_genome["intent_amount_deviation"]
    assert mutated["intent_category_deviation"] > initial_genome["intent_category_deviation"]
    assert mutated["permission_scope_deviation"] > initial_genome["permission_scope_deviation"]
    assert mutated["purchase_velocity"] > initial_genome["purchase_velocity"]

    validate_genome(mutated)


# ---------------------------------------------------------------------------
# 11-13. Genome Canonical Dimensions and Engine Validation
# ---------------------------------------------------------------------------

def test_canonical_dimensions_exact_match():
    """Generated events contain exactly the six Family 2 dimensions."""
    gen = AIAgentAttackGenerator()
    event = gen.generate("round-canon-01")

    assert set(event.attack_genome.keys()) == set(FAMILY2_GENOME_DIMENSIONS)
    assert len(event.attack_genome) == 6


def test_mutated_genome_passes_genome_engine_validate():
    """Mutated genomes pass genome_engine.validate_genome()."""
    mutator = AIAgentMutationStrategy()
    fb_detected = BlueTeamFeedback(
        feedback_id="fb-1",
        round_reference="r-1",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.75,
        important_features={"intent_amount_deviation": 0.25},
    )

    genome = dict(DEFAULT_ATTACK_GENOME)
    for _ in range(5):
        genome = mutator.mutate(genome, fb_detected)
        validate_genome(genome)
        for v in genome.values():
            assert 0.0 <= v <= 1.0


def test_invalid_genome_rejected_by_generator():
    """Generator rejects genomes with missing or incorrect keys, or out-of-bound values."""
    gen = AIAgentAttackGenerator()

    # Missing dimension
    incomplete_genome = {
        "intent_amount_deviation": 0.5,
        "intent_category_deviation": 0.5,
    }
    with pytest.raises(ValueError, match="missing"):
        gen.set_genome(incomplete_genome)

    # Extra non-Family 2 dimension
    extra_genome = dict(DEFAULT_ATTACK_GENOME)
    extra_genome["device_novelty"] = 0.5  # Family 1 dimension
    with pytest.raises(ValueError, match="non-Family 2"):
        gen.set_genome(extra_genome)

    # Out of bounds
    out_of_bounds = dict(DEFAULT_ATTACK_GENOME)
    out_of_bounds["intent_amount_deviation"] = 1.5
    with pytest.raises(ValueError, match="out of bounds"):
        gen.set_genome(out_of_bounds)


# ---------------------------------------------------------------------------
# 14. Multi-Round Pipeline Integration Test
# ---------------------------------------------------------------------------

def test_family2_in_mock_pipeline_multi_round():
    """
    Verify that Family 2 components integrate smoothly with the Pipeline runner
    across multiple rounds, with feedback mutating genomes and driving subsequent rounds.
    """
    gen = AIAgentAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=777)
    det = AIAgentBlueDetector(threshold=0.50)
    ev = AIAgentFeedbackEvaluator()
    mut = AIAgentMutationStrategy(detected_decay=0.10, missed_boost=0.05)

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
    )

    results = pipeline.run(num_rounds=5, base_round_id="f2-pipeline")

    assert len(results) == 5

    # Check that initial high-risk attack is detected and subsequent rounds adapt
    assert results[0].attack_event.attack_family == AttackFamily.AGENT_BEHAVIOR
    assert results[0].feedback.detected is True

    # Risk scores should trend downward as the attacker reduces detection signals
    first_risk = results[0].prediction_result.risk_score
    last_risk = results[-1].prediction_result.risk_score
    assert last_risk < first_risk

    for r in results:
        assert isinstance(r, RoundResult)
        assert r.attack_event.attack_family == AttackFamily.AGENT_BEHAVIOR
        validate_genome(r.attack_event.attack_genome)
