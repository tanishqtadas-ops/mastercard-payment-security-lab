"""
tests/test_family3_feedback.py — Comprehensive test suite for Family 3 Feedback Evaluator.

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
11. End-to-end evaluation with SyntheticIdentityAttackGenerator and SyntheticIdentityBlueDetector (detected).
12. End-to-end evaluation with subtle/evasive attack (missed / false negative).
13. End-to-end evaluation with LegitimateIdentityGenerator (true negative).
14. Graceful handling of None / empty feature_contributions and explanations.
15. Evaluator alias verification (SyntheticIdentityEvaluator).
"""

import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    PredictionResult,
    BlueTeamFeedback,
    SyntheticIdentity,
)
from simulation.interfaces import FeedbackEvaluator
from data.generators.identity_generator import LegitimateIdentityGenerator
from attacks.synthetic_identity import (
    SyntheticIdentityAttackGenerator,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from blue_team.synthetic_identity import (
    SyntheticIdentityBlueDetector,
    SyntheticIdentityFeedbackEvaluator,
    SyntheticIdentityEvaluator,
    MODEL_VERSION,
)


@pytest.fixture
def evaluator() -> SyntheticIdentityFeedbackEvaluator:
    return SyntheticIdentityFeedbackEvaluator()


@pytest.fixture
def sample_attack_event() -> AttackEvent:
    return AttackEvent(
        attack_id="atk-f3-001",
        round_id="round-f3-001",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome=DEFAULT_ATTACK_GENOME,
        scenario={
            "identity_id": "ident_f3_001",
            "identity_attributes": {"first_name": "Alex", "last_name": "Morgan"},
        },
        ground_truth=True,
    )


@pytest.fixture
def sample_legit_event() -> AttackEvent:
    return AttackEvent(
        attack_id="legit-f3-001",
        round_id="round-f3-002",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome=DEFAULT_LEGITIMATE_GENOME,
        scenario={
            "identity_id": "ident_0042_00001",
            "identity_attributes": {"first_name": "Jane", "last_name": "Doe"},
        },
        ground_truth=False,
    )


# ---------------------------------------------------------------------------
# 1. Protocol & Class Structure Tests
# ---------------------------------------------------------------------------

def test_evaluator_satisfies_protocol(evaluator: SyntheticIdentityFeedbackEvaluator):
    """Requirement 1: Evaluator satisfies runtime_checkable FeedbackEvaluator protocol."""
    assert isinstance(evaluator, FeedbackEvaluator)


def test_evaluator_alias_identity():
    """Requirement 15: SyntheticIdentityEvaluator alias matches SyntheticIdentityFeedbackEvaluator."""
    assert SyntheticIdentityEvaluator is SyntheticIdentityFeedbackEvaluator
    instance = SyntheticIdentityEvaluator()
    assert isinstance(instance, FeedbackEvaluator)


# ---------------------------------------------------------------------------
# 2-5. Confusion Matrix State Evaluation Tests
# ---------------------------------------------------------------------------

def test_true_positive_detection(
    evaluator: SyntheticIdentityFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 2: Attack correctly predicted as fraud -> detected=True, FP=False, FN=False."""
    pred = PredictionResult(
        prediction_id="pred-01",
        prediction=True,
        risk_score=0.88,
        model_version=MODEL_VERSION,
        explanation="High synthetic identity risk flagged.",
        feature_contributions={"cross_field_consistency": 0.35, "disposable_email_flag": 0.45},
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    assert isinstance(fb, BlueTeamFeedback)
    assert fb.detected is True
    assert fb.false_positive is False
    assert fb.false_negative is False
    assert fb.risk_score == 0.88


def test_false_negative_missed_attack(
    evaluator: SyntheticIdentityFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 3: Attack incorrectly predicted as benign -> detected=False, FP=False, FN=True."""
    pred = PredictionResult(
        prediction_id="pred-02",
        prediction=False,
        risk_score=0.25,
        model_version=MODEL_VERSION,
        explanation="Identity profile verified as legitimate.",
        feature_contributions={"cross_field_consistency": 0.05},
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    assert isinstance(fb, BlueTeamFeedback)
    assert fb.detected is False
    assert fb.false_positive is False
    assert fb.false_negative is True
    assert fb.risk_score == 0.25


def test_false_positive_false_alarm(
    evaluator: SyntheticIdentityFeedbackEvaluator,
    sample_legit_event: AttackEvent,
):
    """Requirement 4: Benign identity incorrectly predicted as fraud -> detected=False, FP=True, FN=False."""
    pred = PredictionResult(
        prediction_id="pred-03",
        prediction=True,
        risk_score=0.65,
        model_version=MODEL_VERSION,
        explanation="Flagged synthetic identity risk.",
        feature_contributions={"disposable_email_flag": 0.50},
    )

    fb = evaluator.evaluate(sample_legit_event, pred)

    assert isinstance(fb, BlueTeamFeedback)
    assert fb.detected is False
    assert fb.false_positive is True
    assert fb.false_negative is False
    assert fb.risk_score == 0.65


def test_true_negative_benign_approved(
    evaluator: SyntheticIdentityFeedbackEvaluator,
    sample_legit_event: AttackEvent,
):
    """Requirement 5: Benign identity correctly predicted as benign -> detected=False, FP=False, FN=False."""
    pred = PredictionResult(
        prediction_id="pred-04",
        prediction=False,
        risk_score=0.12,
        model_version=MODEL_VERSION,
        explanation="Identity profile verified as legitimate and consistent.",
        feature_contributions={"cross_field_consistency": 0.01},
    )

    fb = evaluator.evaluate(sample_legit_event, pred)

    assert isinstance(fb, BlueTeamFeedback)
    assert fb.detected is False
    assert fb.false_positive is False
    assert fb.false_negative is False
    assert fb.risk_score == 0.12


# ---------------------------------------------------------------------------
# 6-10. Field Population, Preservation, and Schema Round-Trip
# ---------------------------------------------------------------------------

def test_round_reference_and_feedback_id(
    evaluator: SyntheticIdentityFeedbackEvaluator,
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
    assert fb.feedback_id == f"fb-f3-{sample_attack_event.round_id}"


def test_risk_score_preservation_and_bounds(
    evaluator: SyntheticIdentityFeedbackEvaluator,
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
    evaluator: SyntheticIdentityFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 8: important_features faithfully maps prediction.feature_contributions."""
    contributions = {
        "cross_field_consistency": 0.42,
        "contact_consistency": -0.15,
        "device_history_score": 0.68,
        "emulator_detected": 0.90,
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
    assert fb.important_features["emulator_detected"] == 0.90


def test_explanation_data_structure(
    evaluator: SyntheticIdentityFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 9: explanation_data carries structured context including domain metadata."""
    pred = PredictionResult(
        prediction_id="pred-expl",
        prediction=True,
        risk_score=0.91,
        model_version=MODEL_VERSION,
        explanation="Flagged synthetic identity risk.",
    )

    fb = evaluator.evaluate(sample_attack_event, pred)

    assert isinstance(fb.explanation_data, dict)
    assert fb.explanation_data["ground_truth"] is True
    assert fb.explanation_data["prediction"] is True
    assert fb.explanation_data["risk_score"] == 0.91
    assert fb.explanation_data["model_version"] == MODEL_VERSION
    assert fb.explanation_data["explanation"] == "Flagged synthetic identity risk."
    assert fb.explanation_data["attack_family"] == AttackFamily.SYNTHETIC_IDENTITY.value
    assert fb.explanation_data["identity_id"] == "ident_f3_001"


def test_feedback_schema_serialization_roundtrip(
    evaluator: SyntheticIdentityFeedbackEvaluator,
    sample_attack_event: AttackEvent,
):
    """Requirement 10: BlueTeamFeedback serializes to JSON and validates cleanly back into pydantic."""
    pred = PredictionResult(
        prediction_id="pred-rt",
        prediction=True,
        risk_score=0.79,
        model_version=MODEL_VERSION,
        explanation="Anomaly detected.",
        feature_contributions={"risk_event_count": 0.60},
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
    evaluator: SyntheticIdentityFeedbackEvaluator,
):
    """Requirement 11: End-to-end evaluation of generated attack detected by BlueDetector."""
    gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=42)
    det = SyntheticIdentityBlueDetector()

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
    evaluator: SyntheticIdentityFeedbackEvaluator,
):
    """Requirement 12: High threshold detector misses attack -> feedback flags false_negative=True."""
    gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=42)
    # Set threshold very high to force a missed attack
    det = SyntheticIdentityBlueDetector(threshold=0.9999)

    event = gen.generate(round_id="round-e2e-miss-01")
    pred = det.detect(event)
    fb = evaluator.evaluate(event, pred)

    assert pred.prediction is False
    assert fb.detected is False
    assert fb.false_negative is True
    assert fb.false_positive is False


def test_integration_with_legitimate_data_and_detector_approved(
    evaluator: SyntheticIdentityFeedbackEvaluator,
):
    """Requirement 13: Legitimate identity evaluated through detector -> true negative (all fraud flags False)."""
    gen_legit = LegitimateIdentityGenerator(seed=42)
    legit_ident = gen_legit.generate_identity(index=0)
    det = SyntheticIdentityBlueDetector()

    event = AttackEvent(
        attack_id="legit-round-01",
        round_id="round-legit-eval-01",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome=DEFAULT_LEGITIMATE_GENOME,
        scenario=legit_ident.model_dump(),
        ground_truth=False,
    )

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
    evaluator: SyntheticIdentityFeedbackEvaluator,
):
    """Requirement 14: Handles None/omitted feature_contributions and explanation without error."""
    event = AttackEvent(
        attack_id="edge-atk-01",
        round_id="round-edge-01",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome=DEFAULT_ATTACK_GENOME,
        scenario={},  # empty scenario without identity_id
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
    assert fb.explanation_data["explanation"] is None
    assert "identity_id" not in fb.explanation_data
