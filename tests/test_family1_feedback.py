"""
tests/test_family1_feedback.py — Unit tests for Family 1 TransactionFeedbackEvaluator.
"""

import copy
import pytest

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.transaction import Transaction
from simulation.interfaces import FeedbackEvaluator
from attacks.transaction_evasion.generator import (
    TransactionAttackGenerator,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from blue_team.transaction.detector import (
    TransactionBlueDetector,
    MODEL_VERSION,
)
from blue_team.transaction.evaluator import (
    TransactionFeedbackEvaluator,
    TransactionEvaluator,
)


@pytest.fixture
def evaluator() -> TransactionFeedbackEvaluator:
    return TransactionFeedbackEvaluator()


@pytest.fixture
def sample_attack_event() -> AttackEvent:
    return AttackEvent(
        attack_id="atk-f1-001",
        round_id="round-f1-001",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome=DEFAULT_ATTACK_GENOME,
        scenario={
            "transaction": {
                "transaction_id": "tx_test_01",
                "user_id": "usr_retail_101",
                "amount": 1500.0,
                "currency": "USD",
                "merchant_id": "merch_crypto_01",
                "merchant_category": "cryptocurrency_onramp",
                "location": "Lagos, NG",
                "device_id": "dev_emulator_01",
                "payment_channel": "online_card_not_present",
                "timestamp": "2026-08-28T03:00:00Z",
            }
        },
        ground_truth=True,
    )


@pytest.fixture
def sample_legit_event() -> AttackEvent:
    return AttackEvent(
        attack_id="legit-f1-001",
        round_id="round-f1-002",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome=DEFAULT_LEGITIMATE_GENOME,
        scenario={
            "transaction": {
                "transaction_id": "tx_test_02",
                "user_id": "usr_retail_101",
                "amount": 42.50,
                "currency": "USD",
                "merchant_id": "merch_gro_01",
                "merchant_category": "grocery",
                "location": "New York, US",
                "device_id": "dev_ios_iphone15_a1",
                "payment_channel": "pos_contactless",
                "timestamp": "2026-08-28T14:30:00Z",
            }
        },
        ground_truth=False,
    )


# ---------------------------------------------------------------------------
# 1. Protocol Satisfaction & Class Structure
# ---------------------------------------------------------------------------

def test_evaluator_satisfies_protocol(evaluator: TransactionFeedbackEvaluator):
    """Requirement 1: Evaluator satisfies runtime_checkable FeedbackEvaluator protocol."""
    assert isinstance(evaluator, FeedbackEvaluator)


def test_evaluator_alias_identity():
    """Verify TransactionEvaluator alias matches TransactionFeedbackEvaluator."""
    assert TransactionEvaluator is TransactionFeedbackEvaluator
    instance = TransactionEvaluator()
    assert isinstance(instance, FeedbackEvaluator)


# ---------------------------------------------------------------------------
# 2-5. Confusion Matrix Outcome Flag Tests
# ---------------------------------------------------------------------------

def test_true_positive_detected_attack(evaluator: TransactionFeedbackEvaluator, sample_attack_event: AttackEvent):
    """Requirement 2: True positive (attack detected)."""
    prediction = PredictionResult(
        prediction_id="pred-f1-001",
        prediction=True,
        risk_score=0.88,
        model_version=MODEL_VERSION,
        explanation="Flagged high amount anomaly",
        feature_contributions={"amount_deviation": 0.25, "location_deviation": 0.15},
    )

    feedback = evaluator.evaluate(sample_attack_event, prediction)

    assert isinstance(feedback, BlueTeamFeedback)
    assert feedback.detected is True
    assert feedback.false_negative is False
    assert feedback.false_positive is False
    assert feedback.risk_score == 0.88


def test_false_negative_missed_attack(evaluator: TransactionFeedbackEvaluator, sample_attack_event: AttackEvent):
    """Requirement 3: False negative (attack missed)."""
    prediction = PredictionResult(
        prediction_id="pred-f1-002",
        prediction=False,
        risk_score=0.35,
        model_version=MODEL_VERSION,
        explanation="Deemed legitimate",
        feature_contributions={"amount_deviation": 0.05},
    )

    feedback = evaluator.evaluate(sample_attack_event, prediction)

    assert feedback.detected is False
    assert feedback.false_negative is True
    assert feedback.false_positive is False
    assert feedback.risk_score == 0.35


def test_false_positive_benign_flagged(evaluator: TransactionFeedbackEvaluator, sample_legit_event: AttackEvent):
    """Requirement 4: False positive (benign transaction falsely flagged)."""
    prediction = PredictionResult(
        prediction_id="pred-f1-003",
        prediction=True,
        risk_score=0.72,
        model_version=MODEL_VERSION,
        explanation="False alarm anomaly",
        feature_contributions={"time_deviation": 0.20},
    )

    feedback = evaluator.evaluate(sample_legit_event, prediction)

    assert feedback.detected is False
    assert feedback.false_negative is False
    assert feedback.false_positive is True
    assert feedback.risk_score == 0.72


def test_true_negative_correct_benign(evaluator: TransactionFeedbackEvaluator, sample_legit_event: AttackEvent):
    """Requirement 5: True negative (benign transaction correctly approved)."""
    prediction = PredictionResult(
        prediction_id="pred-f1-004",
        prediction=False,
        risk_score=0.10,
        model_version=MODEL_VERSION,
        explanation="Transaction verified as legitimate",
        feature_contributions={"amount_deviation": 0.02},
    )

    feedback = evaluator.evaluate(sample_legit_event, prediction)

    assert feedback.detected is False
    assert feedback.false_negative is False
    assert feedback.false_positive is False
    assert feedback.risk_score == 0.10


# ---------------------------------------------------------------------------
# 6-9. Risk, Feature Contributions, Metadata, & Immutability
# ---------------------------------------------------------------------------

def test_risk_score_and_important_features_propagation(evaluator: TransactionFeedbackEvaluator, sample_attack_event: AttackEvent):
    """Requirement 6 & 7: Verify risk score and feature contributions propagation."""
    contributions = {
        "amount_deviation": 0.22,
        "velocity_deviation": 0.18,
        "device_novelty": 0.14,
        "location_deviation": 0.15,
        "time_deviation": 0.08,
        "sequence_anomaly": 0.12,
    }
    prediction = PredictionResult(
        prediction_id="pred-f1-005",
        prediction=True,
        risk_score=0.89,
        model_version=MODEL_VERSION,
        explanation="High composite risk",
        feature_contributions=contributions,
    )

    feedback = evaluator.evaluate(sample_attack_event, prediction)

    assert feedback.risk_score == 0.89
    assert feedback.important_features == contributions


def test_round_reference_and_feedback_id_formatting(evaluator: TransactionFeedbackEvaluator, sample_attack_event: AttackEvent):
    """Requirement 8: Verify round_reference matches round_id and feedback_id format."""
    prediction = PredictionResult(
        prediction_id="pred-f1-006",
        prediction=True,
        risk_score=0.80,
        model_version=MODEL_VERSION,
    )

    feedback = evaluator.evaluate(sample_attack_event, prediction)

    assert feedback.round_reference == sample_attack_event.round_id
    assert feedback.feedback_id == f"fb-f1-{sample_attack_event.round_id}"


def test_evaluator_does_not_mutate_inputs(evaluator: TransactionFeedbackEvaluator, sample_attack_event: AttackEvent):
    """Requirement 9: Verify evaluator does not modify the input event or prediction objects."""
    prediction = PredictionResult(
        prediction_id="pred-f1-007",
        prediction=True,
        risk_score=0.85,
        model_version=MODEL_VERSION,
        feature_contributions={"amount_deviation": 0.25},
    )

    event_copy = copy.deepcopy(sample_attack_event)
    pred_copy = copy.deepcopy(prediction)

    _ = evaluator.evaluate(sample_attack_event, prediction)

    assert sample_attack_event == event_copy
    assert prediction == pred_copy


def test_explanation_data_enrichment(evaluator: TransactionFeedbackEvaluator, sample_attack_event: AttackEvent):
    """Verify explanation_data contains domain context and classification metadata."""
    prediction = PredictionResult(
        prediction_id="pred-f1-008",
        prediction=True,
        risk_score=0.85,
        model_version=MODEL_VERSION,
        explanation="High amount and foreign location",
    )

    feedback = evaluator.evaluate(sample_attack_event, prediction)

    exp_data = feedback.explanation_data
    assert isinstance(exp_data, dict)
    assert exp_data["ground_truth"] is True
    assert exp_data["prediction"] is True
    assert exp_data["risk_score"] == 0.85
    assert exp_data["model_version"] == MODEL_VERSION
    assert exp_data["explanation"] == "High amount and foreign location"
    assert exp_data["attack_family"] == AttackFamily.ADAPTIVE_EVASION.value
    assert exp_data["transaction_id"] == "tx_test_01"
    assert exp_data["user_id"] == "usr_retail_101"
    assert exp_data["amount"] == 1500.0


def test_graceful_handling_of_none_feature_contributions_and_explanations(evaluator: TransactionFeedbackEvaluator, sample_attack_event: AttackEvent):
    """Verify handling when feature_contributions or explanation are None."""
    prediction = PredictionResult(
        prediction_id="pred-f1-009",
        prediction=True,
        risk_score=0.80,
        model_version=MODEL_VERSION,
        explanation=None,
        feature_contributions=None,
    )

    feedback = evaluator.evaluate(sample_attack_event, prediction)

    assert feedback.important_features == {}
    assert feedback.explanation_data["explanation"] is None


# ---------------------------------------------------------------------------
# 10. End-to-End Pipeline Integration with Family 1 Components
# ---------------------------------------------------------------------------

def test_end_to_end_generator_detector_evaluator(evaluator: TransactionFeedbackEvaluator):
    """Verify end-to-end integration: generator -> detector -> evaluator."""
    gen_attack = TransactionAttackGenerator(ground_truth=True, seed=42)
    event_attack = gen_attack.generate("round-e2e-01")

    detector = TransactionBlueDetector()
    prediction = detector.detect(event_attack)

    feedback = evaluator.evaluate(event_attack, prediction)

    assert feedback.round_reference == "round-e2e-01"
    assert feedback.feedback_id == "fb-f1-round-e2e-01"
    assert feedback.detected is True
    assert feedback.false_negative is False
    assert feedback.false_positive is False
    assert feedback.risk_score == prediction.risk_score
    assert len(feedback.important_features) == 6
