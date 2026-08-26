"""
tests/test_family2_feedback.py — Comprehensive test suite for Family 2 Feedback Evaluator.

Covers:
1. Evaluator satisfies runtime_checkable FeedbackEvaluator protocol.
2. True Positive (detected attack): ground_truth=True, prediction=True -> detected=True, FP=False, FN=False.
3. False Negative (missed attack): ground_truth=True, prediction=False -> detected=False, FP=False, FN=True.
4. False Positive (false alarm): ground_truth=False, prediction=True -> detected=False, FP=True, FN=False.
5. True Negative (benign approved): ground_truth=False, prediction=False -> detected=False, FP=False, FN=False.
6. round_reference and feedback_id format consistency.
7. Risk score preservation and bounds in [0.0, 1.0].
8. Important features mapping from PredictionResult.feature_contributions.
9. Explanation data structure and domain context enrichment.
10. Schema serialization and pydantic validation round-trip.
11. End-to-end evaluation with AIAgentAttackGenerator and AIAgentBlueDetector (detected).
12. End-to-end evaluation with subtle/evasive attack (missed / false negative).
13. End-to-end evaluation with legitimate agent event (true negative).
14. Graceful handling of None / empty feature_contributions and explanations.
15. Evaluator alias verification (AIAgentEvaluator).
16. Feedback handoff to AIAgentMutationStrategy (detected -> decay signals).
17. Feedback handoff to AIAgentMutationStrategy (missed -> boost signals).
"""

from datetime import datetime, timezone
import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    PredictionResult,
    BlueTeamFeedback,
    AIAgentPaymentEvent,
    Transaction,
)
from simulation.interfaces import FeedbackEvaluator
from attacks.ai_agent import (
    AIAgentAttackGenerator,
    AIAgentMutationStrategy,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
    FAMILY2_GENOME_DIMENSIONS,
)
from blue_team.ai_agent import (
    AIAgentBlueDetector,
    AIAgentFeedbackEvaluator,
    AIAgentEvaluator,
    MODEL_VERSION,
)


@pytest.fixture
def evaluator() -> AIAgentFeedbackEvaluator:
    return AIAgentFeedbackEvaluator()


@pytest.fixture
def sample_attack_event() -> AttackEvent:
    tx = Transaction(
        transaction_id="tx-f2-test-01",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=950.00,
        currency="USD",
        merchant_id="merch_crypto_vault",
        merchant_category="crypto_assets",
        location="US-Online",
        device_id="device_agent_unverified",
        payment_channel="ai_agent_batch_burst_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-f2-001",
        user_intent="Purchase office supplies up to $150",
        authorized_scope="Office supplies, max $150, approved merchants only",
        agent_identity="unverified-agent-spoof-01",
        session_context="session_hijack_anom_remote_ip_flagged",
        actual_action="Executed unauthorized crypto transaction for $950.00. Scope violation: elevated privileges.",
        transaction=tx,
    )
    return AttackEvent(
        attack_id="atk-f2-001",
        round_id="round-f2-001",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome=dict(DEFAULT_ATTACK_GENOME),
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )


@pytest.fixture
def sample_legit_event() -> AttackEvent:
    tx = Transaction(
        transaction_id="tx-f2-legit-01",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=120.00,
        currency="USD",
        merchant_id="merch_staples_01",
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_agent_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-f2-legit-01",
        user_intent="Purchase office supplies up to $150",
        authorized_scope="Office supplies, max $150, approved merchants only",
        agent_identity="procure-bot-v2-valid",
        session_context="session_auth_mfa_valid_001",
        actual_action="Purchased office stationery from Staples Direct for $120.00. Agent adhered strictly to authorized operational bounds.",
        transaction=tx,
    )
    return AttackEvent(
        attack_id="legit-f2-001",
        round_id="round-f2-002",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome=dict(DEFAULT_LEGITIMATE_GENOME),
        scenario=scenario.model_dump(mode="json"),
        ground_truth=False,
    )


# ---------------------------------------------------------------------------
# 1. Protocol & Class Structure Tests
# ---------------------------------------------------------------------------

def test_evaluator_satisfies_protocol(evaluator: AIAgentFeedbackEvaluator):
    """Requirement 1: Evaluator satisfies runtime_checkable FeedbackEvaluator protocol."""
    assert isinstance(evaluator, FeedbackEvaluator)


def test_evaluator_alias_identity():
    """Requirement 15: AIAgentEvaluator alias matches AIAgentFeedbackEvaluator."""
    assert AIAgentEvaluator is AIAgentFeedbackEvaluator
    instance = AIAgentEvaluator()
    assert isinstance(instance, FeedbackEvaluator)


# ---------------------------------------------------------------------------
# 2-5. Confusion Matrix State Evaluation Tests
# ---------------------------------------------------------------------------

def test_true_positive_detection(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 2: Attack correctly predicted as fraud -> detected=True, FP=False, FN=False."""
    pred = PredictionResult(
        prediction_id="pred-01",
        prediction=True,
        risk_score=0.88,
        model_version=MODEL_VERSION,
        explanation="Flagged unauthorized AI-agent behavior.",
        feature_contributions={
            "intent_amount_deviation": 0.25,
            "intent_category_deviation": 0.20,
            "permission_scope_deviation": 0.25,
            "agent_identity_confidence": 0.08,
            "session_provenance_anomaly": 0.05,
            "purchase_velocity": 0.05,
        },
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    assert isinstance(fb, BlueTeamFeedback)
    assert fb.detected is True
    assert fb.false_positive is False
    assert fb.false_negative is False
    assert fb.risk_score == 0.88


def test_false_negative_missed_attack(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 3: Attack incorrectly predicted as benign -> detected=False, FP=False, FN=True."""
    pred = PredictionResult(
        prediction_id="pred-02",
        prediction=False,
        risk_score=0.25,
        model_version=MODEL_VERSION,
        explanation="Agent transaction approved within authorized envelope.",
        feature_contributions={"intent_amount_deviation": 0.05},
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    assert isinstance(fb, BlueTeamFeedback)
    assert fb.detected is False
    assert fb.false_positive is False
    assert fb.false_negative is True
    assert fb.risk_score == 0.25


def test_false_positive_false_alarm(
    evaluator: AIAgentFeedbackEvaluator,
    sample_legit_event: AttackEvent,
):
    """Requirement 4: Benign event incorrectly predicted as fraud -> detected=False, FP=True, FN=False."""
    pred = PredictionResult(
        prediction_id="pred-03",
        prediction=True,
        risk_score=0.65,
        model_version=MODEL_VERSION,
        explanation="Flagged unauthorized AI-agent behavior.",
        feature_contributions={"permission_scope_deviation": 0.25},
    )

    fb = evaluator.evaluate(sample_legit_event, pred)

    assert isinstance(fb, BlueTeamFeedback)
    assert fb.detected is False
    assert fb.false_positive is True
    assert fb.false_negative is False
    assert fb.risk_score == 0.65


def test_true_negative_benign_approved(
    evaluator: AIAgentFeedbackEvaluator,
    sample_legit_event: AttackEvent,
):
    """Requirement 5: Benign event correctly predicted as benign -> detected=False, FP=False, FN=False."""
    pred = PredictionResult(
        prediction_id="pred-04",
        prediction=False,
        risk_score=0.08,
        model_version=MODEL_VERSION,
        explanation="Agent transaction approved within authorized envelope.",
        feature_contributions={"intent_amount_deviation": 0.01},
    )

    fb = evaluator.evaluate(sample_legit_event, pred)

    assert isinstance(fb, BlueTeamFeedback)
    assert fb.detected is False
    assert fb.false_positive is False
    assert fb.false_negative is False
    assert fb.risk_score == 0.08


# ---------------------------------------------------------------------------
# 6-10. Field Population, Preservation, and Schema Round-Trip
# ---------------------------------------------------------------------------

def test_round_reference_and_feedback_id(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 6: round_reference matches round_id and feedback_id has predictable structure."""
    pred = PredictionResult(
        prediction_id="pred-05",
        prediction=True,
        risk_score=0.75,
        model_version=MODEL_VERSION,
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    assert fb.round_reference == sample_attack_event.round_id
    assert fb.feedback_id == f"fb-f2-{sample_attack_event.round_id}"


def test_risk_score_preservation_and_bounds(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 7: risk_score exactly matches PredictionResult and is bounded in [0.0, 1.0]."""
    test_scores = [0.0, 0.0001, 0.3333, 0.5, 0.8765, 1.0]

    for score in test_scores:
        pred = PredictionResult(
            prediction_id="pred-score",
            prediction=score >= 0.5,
            risk_score=score,
            model_version=MODEL_VERSION,
        )
        fb = evaluator.evaluate(sample_attack_event, pred)
        assert fb.risk_score == score
        assert 0.0 <= fb.risk_score <= 1.0


def test_important_features_captured_from_contributions(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 8: important_features faithfully maps prediction.feature_contributions."""
    contributions = {
        "intent_amount_deviation": 0.22,
        "intent_category_deviation": 0.18,
        "permission_scope_deviation": 0.24,
        "agent_identity_confidence": 0.08,
        "session_provenance_anomaly": 0.07,
        "purchase_velocity": 0.06,
    }

    pred = PredictionResult(
        prediction_id="pred-feat",
        prediction=True,
        risk_score=0.85,
        model_version=MODEL_VERSION,
        feature_contributions=contributions,
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    assert fb.important_features == contributions
    assert fb.important_features["intent_amount_deviation"] == 0.22


def test_explanation_data_structure(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 9: explanation_data carries structured context including domain metadata."""
    pred = PredictionResult(
        prediction_id="pred-expl",
        prediction=True,
        risk_score=0.91,
        model_version=MODEL_VERSION,
        explanation="Flagged unauthorized AI-agent behavior.",
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    assert isinstance(fb.explanation_data, dict)
    assert fb.explanation_data["ground_truth"] is True
    assert fb.explanation_data["prediction"] is True
    assert fb.explanation_data["risk_score"] == 0.91
    assert fb.explanation_data["model_version"] == MODEL_VERSION
    assert fb.explanation_data["explanation"] == "Flagged unauthorized AI-agent behavior."
    assert fb.explanation_data["attack_family"] == AttackFamily.AGENT_BEHAVIOR.value
    assert fb.explanation_data["event_id"] == "agent-evt-f2-001"
    assert fb.explanation_data["agent_identity"] == "unverified-agent-spoof-01"
    assert "actual_action" in fb.explanation_data


def test_feedback_schema_serialization_roundtrip(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 10: BlueTeamFeedback serializes to JSON and validates cleanly back into pydantic."""
    pred = PredictionResult(
        prediction_id="pred-rt",
        prediction=True,
        risk_score=0.79,
        model_version=MODEL_VERSION,
        explanation="Anomaly detected.",
        feature_contributions={"intent_amount_deviation": 0.25},
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    dumped = fb.model_dump(mode="json")
    reconstructed = BlueTeamFeedback.model_validate(dumped)

    assert reconstructed.feedback_id == fb.feedback_id
    assert reconstructed.round_reference == fb.round_reference
    assert reconstructed.detected == fb.detected
    assert reconstructed.false_positive == fb.false_positive
    assert reconstructed.false_negative == fb.false_negative
    assert reconstructed.risk_score == fb.risk_score
    assert reconstructed.important_features == fb.important_features
    assert reconstructed.explanation_data == fb.explanation_data


# ---------------------------------------------------------------------------
# 11-13. Integration with Generator and Detector
# ---------------------------------------------------------------------------

def test_integration_with_generator_and_detector_detected(
    evaluator: AIAgentFeedbackEvaluator,
):
    """Requirement 11: End-to-end evaluation of generated attack detected by BlueDetector."""
    gen = AIAgentAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=42)
    det = AIAgentBlueDetector()

    event = gen.generate(round_id="round-e2e-det-01")
    pred = det.detect(event)
    fb = evaluator.evaluate(event, pred)

    assert pred.prediction is True
    assert fb.detected is True
    assert fb.false_negative is False
    assert fb.false_positive is False
    assert fb.round_reference == "round-e2e-det-01"
    assert len(fb.important_features) > 0


def test_integration_with_generator_and_detector_missed(
    evaluator: AIAgentFeedbackEvaluator,
):
    """Requirement 12: High threshold detector misses attack -> feedback flags false_negative=True."""
    gen = AIAgentAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=42)
    # Set threshold very high to force a missed attack
    det = AIAgentBlueDetector(threshold=0.9999)

    event = gen.generate(round_id="round-e2e-miss-01")
    pred = det.detect(event)
    fb = evaluator.evaluate(event, pred)

    assert pred.prediction is False
    assert fb.detected is False
    assert fb.false_negative is True
    assert fb.false_positive is False


def test_integration_with_legitimate_event_and_detector_approved(
    evaluator: AIAgentFeedbackEvaluator,
):
    """Requirement 13: Legitimate agent event evaluated through detector -> true negative (all fraud flags False)."""
    gen = AIAgentAttackGenerator(ground_truth=False, seed=42)
    det = AIAgentBlueDetector()

    event = gen.generate(round_id="round-legit-eval-01")
    pred = det.detect(event)
    fb = evaluator.evaluate(event, pred)

    assert pred.prediction is False
    assert fb.detected is False
    assert fb.false_positive is False
    assert fb.false_negative is False


# ---------------------------------------------------------------------------
# 14. Edge Cases & Robustness
# ---------------------------------------------------------------------------

def test_handles_none_feature_contributions_and_explanation(
    evaluator: AIAgentFeedbackEvaluator,
):
    """Requirement 14: Handles None/omitted feature_contributions and explanation without error."""
    event = AttackEvent(
        attack_id="edge-f2-01",
        round_id="round-f2-edge-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome=dict(DEFAULT_ATTACK_GENOME),
        scenario={},  # empty scenario without event_id
        ground_truth=True,
    )

    pred = PredictionResult(
        prediction_id="pred-edge",
        prediction=True,
        risk_score=0.80,
        model_version=MODEL_VERSION,
        explanation=None,
        feature_contributions=None,
    )

    fb = evaluator.evaluate(event, pred)

    assert fb.detected is True
    assert fb.important_features == {}
    assert fb.explanation_data is not None
    assert fb.explanation_data["explanation"] is None
    assert "event_id" not in fb.explanation_data


# ---------------------------------------------------------------------------
# 15-16. Mutation Strategy Integration
# ---------------------------------------------------------------------------

def test_feedback_drives_mutator_decay_on_detection(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 16: Evaluator produces feedback that successfully drives mutation decay in AIAgentMutationStrategy."""
    pred = PredictionResult(
        prediction_id="pred-mut-01",
        prediction=True,
        risk_score=0.85,
        model_version=MODEL_VERSION,
        explanation="Flagged unauthorized AI-agent behavior.",
        feature_contributions={
            "intent_amount_deviation": 0.25,
            "intent_category_deviation": 0.20,
            "permission_scope_deviation": 0.25,
            "agent_identity_confidence": 0.08,
            "session_provenance_anomaly": 0.06,
            "purchase_velocity": 0.06,
        },
    )

    fb = evaluator.evaluate(sample_attack_event, pred)
    mutator = AIAgentMutationStrategy(detected_decay=0.10)
    initial_genome = dict(sample_attack_event.attack_genome)

    mutated = mutator.mutate(initial_genome, fb)

    assert mutated["intent_amount_deviation"] < initial_genome["intent_amount_deviation"]
    assert mutated["intent_category_deviation"] < initial_genome["intent_category_deviation"]
    assert mutated["permission_scope_deviation"] < initial_genome["permission_scope_deviation"]
    assert mutated["agent_identity_confidence"] >= initial_genome["agent_identity_confidence"]


def test_feedback_drives_mutator_boost_on_missed(
    evaluator: AIAgentFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 17: Evaluator produces feedback on missed attack that drives mutation escalation."""
    pred = PredictionResult(
        prediction_id="pred-mut-02",
        prediction=False,
        risk_score=0.20,
        model_version=MODEL_VERSION,
        explanation="Agent transaction approved.",
        feature_contributions={"intent_amount_deviation": 0.05},
    )

    fb = evaluator.evaluate(sample_attack_event, pred)
    mutator = AIAgentMutationStrategy(missed_boost=0.06)
    initial_genome = {
        "intent_amount_deviation": 0.20,
        "intent_category_deviation": 0.20,
        "permission_scope_deviation": 0.20,
        "agent_identity_confidence": 0.90,
        "session_provenance_anomaly": 0.15,
        "purchase_velocity": 0.20,
    }

    mutated = mutator.mutate(initial_genome, fb)

    assert mutated["intent_amount_deviation"] > initial_genome["intent_amount_deviation"]
    assert mutated["intent_category_deviation"] > initial_genome["intent_category_deviation"]
    assert mutated["purchase_velocity"] > initial_genome["purchase_velocity"]
