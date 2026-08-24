"""
tests/test_family3_detector.py — Comprehensive test suite for Family 3 Blue-Team Detector.

Covers:
1. Detector satisfies runtime_checkable BlueTeamDetector protocol.
2. Legitimate Family 3 scenario produces a valid low-risk PredictionResult.
3. Obvious synthetic identity attack is detected with high risk score.
4. Risk score is strictly within [0.0, 1.0].
5. Boolean prediction is strictly consistent with the decision threshold.
6. model_version is populated and traceable.
7. Human-readable explanation is populated.
8. feature_contributions are populated and meaningful.
9. Detector does NOT use ground_truth as a prediction feature.
10. Detector evaluates events from Task 3 SyntheticIdentityAttackGenerator.
11. Different Family 3 risk profiles produce sensible risk ordering.
12. Training uses legitimate training baseline and NEVER touches the held-out evaluation dataset.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    PredictionResult,
    SyntheticIdentity,
)
from simulation.interfaces import BlueTeamDetector
from data.generators.identity_generator import LegitimateIdentityGenerator
from attacks.synthetic_identity import (
    SyntheticIdentityAttackGenerator,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from blue_team.synthetic_identity import (
    SyntheticIdentityBlueDetector,
    DEFAULT_DETECTION_THRESHOLD,
    MODEL_VERSION,
    FEATURE_NAMES,
    extract_identity_features,
)


@pytest.fixture(scope="module")
def detector() -> SyntheticIdentityBlueDetector:
    """Instantiate a shared detector instance for test cases."""
    return SyntheticIdentityBlueDetector()


def test_detector_satisfies_protocol(detector: SyntheticIdentityBlueDetector):
    """Requirement 1: Detector satisfies runtime_checkable BlueTeamDetector protocol."""
    assert isinstance(detector, BlueTeamDetector)


def test_legitimate_scenario_produces_low_risk(detector: SyntheticIdentityBlueDetector):
    """Requirement 2: Legitimate identity profile produces low risk score and is not flagged as fraud."""
    gen_legit = LegitimateIdentityGenerator(seed=42)
    legit_ident = gen_legit.generate_identity(index=0)

    event = AttackEvent(
        attack_id="test-legit-01",
        round_id="round-legit-01",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome=DEFAULT_LEGITIMATE_GENOME,
        scenario=legit_ident.model_dump(),
        ground_truth=False,
    )

    pred = detector.detect(event)

    assert isinstance(pred, PredictionResult)
    assert pred.prediction is False
    assert pred.risk_score < 0.50
    assert "legitimate" in pred.explanation.lower() or "consistent" in pred.explanation.lower()


def test_obvious_synthetic_identity_attack_detected(detector: SyntheticIdentityBlueDetector):
    """Requirement 3: Obvious synthetic identity attack is detected with high risk score."""
    gen_attack = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=777)
    event = gen_attack.generate(round_id="round-atk-01")

    pred = detector.detect(event)

    assert isinstance(pred, PredictionResult)
    assert pred.prediction is True
    assert pred.risk_score >= 0.50
    assert "flagged" in pred.explanation.lower() or "risk" in pred.explanation.lower()


def test_risk_score_is_bounded(detector: SyntheticIdentityBlueDetector):
    """Requirement 4: Risk score is strictly within [0.0, 1.0]."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    for i in range(5):
        event = gen.generate(round_id=f"round-bound-{i}")
        pred = detector.detect(event)
        assert 0.0 <= pred.risk_score <= 1.0


def test_prediction_agrees_with_threshold():
    """Requirement 5: Prediction boolean strictly matches risk_score >= threshold."""
    det_low = SyntheticIdentityBlueDetector(threshold=0.20)
    det_high = SyntheticIdentityBlueDetector(threshold=0.85)

    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="round-thresh-01")

    pred_low = det_low.detect(event)
    pred_high = det_high.detect(event)

    assert pred_low.prediction == (pred_low.risk_score >= 0.20)
    assert pred_high.prediction == (pred_high.risk_score >= 0.85)


def test_model_version_populated(detector: SyntheticIdentityBlueDetector):
    """Requirement 6: model_version is populated and matches constant."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="round-ver-01")
    pred = detector.detect(event)

    assert pred.model_version == MODEL_VERSION
    assert pred.model_version == "family3-xgb-v1"


def test_explanation_populated(detector: SyntheticIdentityBlueDetector):
    """Requirement 7: Explanation is a non-empty descriptive string."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="round-exp-01")
    pred = detector.detect(event)

    assert isinstance(pred.explanation, str)
    assert len(pred.explanation) > 10


def test_feature_contributions_populated(detector: SyntheticIdentityBlueDetector):
    """Requirement 8: feature_contributions contains all extracted feature dimensions."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="round-contrib-01")
    pred = detector.detect(event)

    assert isinstance(pred.feature_contributions, dict)
    assert len(pred.feature_contributions) == len(FEATURE_NAMES)
    for feat in FEATURE_NAMES:
        assert feat in pred.feature_contributions
        assert isinstance(pred.feature_contributions[feat], (int, float))


def test_detector_does_not_use_ground_truth_feature(detector: SyntheticIdentityBlueDetector):
    """Requirement 9: Inverting ground_truth flag produces the exact same risk score (zero leakage)."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event_true = gen.generate(round_id="round-leak-01")

    # Clone event with ground_truth inverted
    event_false = AttackEvent(
        attack_id=event_true.attack_id,
        round_id=event_true.round_id,
        attack_family=event_true.attack_family,
        attack_genome=event_true.attack_genome,
        scenario=event_true.scenario,
        ground_truth=False,  # Flipped
    )

    pred_true = detector.detect(event_true)
    pred_false = detector.detect(event_false)

    assert pred_true.risk_score == pred_false.risk_score
    assert pred_true.prediction == pred_false.prediction
    assert pred_true.feature_contributions == pred_false.feature_contributions


def test_detector_handles_task3_generated_events(detector: SyntheticIdentityBlueDetector):
    """Requirement 10: Detector evaluates events from Task 3 SyntheticIdentityAttackGenerator."""
    gen = SyntheticIdentityAttackGenerator(seed=999)
    for rid in ["round-alpha", "round-beta", "round-gamma"]:
        event = gen.generate(round_id=rid)
        pred = detector.detect(event)
        assert isinstance(pred, PredictionResult)
        assert pred.prediction_id == f"pred-f3-{rid}"


def test_risk_ordering_across_profiles(detector: SyntheticIdentityBlueDetector):
    """Requirement 11: Obvious attack produces higher risk score than legitimate baseline."""
    gen_attack = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=123)
    gen_legit = LegitimateIdentityGenerator(seed=123)

    event_attack = gen_attack.generate(round_id="round-order-atk")
    legit_ident = gen_legit.generate_identity(index=0)
    event_legit = AttackEvent(
        attack_id="order-legit-01",
        round_id="round-order-legit",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome=DEFAULT_LEGITIMATE_GENOME,
        scenario=legit_ident.model_dump(),
        ground_truth=False,
    )

    pred_attack = detector.detect(event_attack)
    pred_legit = detector.detect(event_legit)

    assert pred_attack.risk_score > pred_legit.risk_score
    assert pred_attack.prediction is True
    assert pred_legit.prediction is False


def test_heldout_dataset_is_not_used_during_training():
    """Requirement 12: Verifies held-out evaluation dataset (data/held_out/) is NEVER loaded during training."""
    heldout_file = "heldout_identities.json"
    accessed_files = []

    original_open = open

    def tracking_open(file, *args, **kwargs):
        accessed_files.append(str(file))
        return original_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=tracking_open):
        # Initialize detector which runs default model training
        det = SyntheticIdentityBlueDetector()

    # Assert held-out dataset was never opened
    heldout_accessed = [f for f in accessed_files if heldout_file in f]
    assert len(heldout_accessed) == 0, (
        f"Held-out dataset was accessed during detector training: {heldout_accessed}"
    )
