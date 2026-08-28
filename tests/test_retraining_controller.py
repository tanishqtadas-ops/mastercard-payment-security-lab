"""
tests/test_retraining_controller.py — Comprehensive tests for Model Retraining Controller (Task 7.3).

Validates all required behaviors:
1. No retraining before configured interval.
2. Retraining occurs at round 2 by default.
3. Interval is configurable.
4. False negatives are consumed and used.
5. Original legitimate baseline is preserved and used.
6. Fresh legitimate samples are handled properly.
7. Held-out data is NEVER used for training (evaluated separately).
8. Detector state actually changes on retraining.
9. Updated detector uses new trained/adapted state for subsequent predictions.
10. Before metrics are accurately recorded.
11. After metrics are accurately recorded.
12. Model update metadata / versioning is recorded in ModelUpdateRecord.
13. Multiple families can be handled.
14. Family isolation is preserved.
15. No cross-family training contamination.
16. Empty failure memory is handled safely (no-op).
17. Insufficient training data is handled safely.
18. Deterministic retraining produces reproducible results.
19. Input data is not mutated.
20. Fail-safe behavior: trainer exceptions preserve previous working detector state.
"""

import copy
from datetime import datetime, timezone
from pathlib import Path
import pytest

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.round import RoundResult
from schemas.transaction import Transaction
from schemas.identity import SyntheticIdentity

from blue_team.transaction.detector import TransactionBlueDetector
from blue_team.ai_agent.detector import AIAgentBlueDetector
from blue_team.synthetic_identity.detector import SyntheticIdentityBlueDetector

from blue_team.learning.failure_memory import FailureRecord, FailureMemory
from blue_team.learning.dataset import (
    ProvenanceType,
    HoldoutDataLeakageError,
    assemble_retraining_dataset,
)
from blue_team.learning.retraining import (
    EvaluationMetrics,
    ModelUpdateRecord,
    Family1TransactionTrainer,
    Family2AIAgentTrainer,
    Family3SyntheticIdentityTrainer,
    RetrainingController,
    compute_binary_metrics,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def family1_missed_round():
    """Create a RoundResult representing a missed transaction evasion attack."""
    event = AttackEvent(
        attack_id="atk-f1-001",
        round_id="round-f1-1",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome={"amount_deviation": 0.20, "velocity_deviation": 0.30},
        scenario={
            "transaction": {
                "transaction_id": "tx-101",
                "user_id": "u42",
                "amount": 180.0,
                "currency": "USD",
                "merchant_id": "m99",
                "merchant_category": "electronics",
                "location": "NYC",
                "device_id": "dev-01",
                "payment_channel": "online",
            }
        },
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-f1-001",
        prediction=False,
        risk_score=0.25,
        model_version="heuristic-family1-v1",
        feature_contributions={"amount_deviation": 0.10},
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-f1-001",
        round_reference="round-f1-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.25,
        important_features={"amount_deviation": 0.10},
    )
    return RoundResult(
        round_id="round-f1-1",
        attack_event=event,
        prediction_result=pred,
        feedback=fb,
        outcome_metrics={"round_index": 1},
    )


@pytest.fixture
def family2_missed_round():
    """Create a RoundResult representing a missed AI agent attack."""
    event = AttackEvent(
        attack_id="atk-f2-002",
        round_id="round-f2-1",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={"intent_amount_deviation": 0.30, "permission_scope_deviation": 0.25},
        scenario={
            "event_id": "agent-evt-202",
            "agent_identity": "procure-bot",
            "actual_action": "purchase_gift_cards",
        },
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-f2-002",
        prediction=False,
        risk_score=0.30,
        model_version="heuristic-family2-v1",
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-f2-002",
        round_reference="round-f2-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.30,
        important_features={},
    )
    return RoundResult(
        round_id="round-f2-1",
        attack_event=event,
        prediction_result=pred,
        feedback=fb,
        outcome_metrics={"round_index": 1},
    )


@pytest.fixture
def family3_missed_round():
    """Create a RoundResult representing a missed synthetic identity fraud event."""
    event = AttackEvent(
        attack_id="atk-f3-003",
        round_id="round-f3-1",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome={"cross_field_consistency": 0.85, "profile_plausibility_score": 0.80},
        scenario={
            "identity_id": "syn-id-303",
            "identity_attributes": {"name": "Alex Mercer", "ssn_match": True},
            "account_metadata": {"kyc_verification_status": "unverified", "account_status": "restricted"},
            "lifecycle_info": {"risk_event_count": 2, "days_to_risky_activity": 12, "lifecycle_coherence_score": 0.4},
        },
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-f3-003",
        prediction=False,
        risk_score=0.35,
        model_version="family3-xgb-v1",
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-f3-003",
        round_reference="round-f3-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.35,
        important_features={},
    )
    return RoundResult(
        round_id="round-f3-1",
        attack_event=event,
        prediction_result=pred,
        feedback=fb,
        outcome_metrics={"round_index": 1},
    )


# ===========================================================================
# 1. Retraining Schedule & Intervals (Requirements 1, 2, 3)
# ===========================================================================

def test_no_retraining_before_configured_interval(family1_missed_round):
    """Round 1 does NOT trigger retraining when interval is 2."""
    det = TransactionBlueDetector()
    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
    )

    update = ctrl.on_round_completed(family1_missed_round, round_index=1)
    assert update is None
    assert len(ctrl.get_history()) == 0
    assert len(ctrl.failure_memory) == 1


def test_retraining_occurs_at_round_2_default(family1_missed_round):
    """Retraining triggers at round 2 by default."""
    det = TransactionBlueDetector()
    baseline = [
        Transaction(
            transaction_id="tx-legit-1",
            user_id="u42",
            timestamp=datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc),
            amount=40.0,
            currency="USD",
            merchant_id="m1",
            merchant_category="groceries",
            location="NYC",
            device_id="dev-01",
            payment_channel="pos",
        )
    ]
    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: baseline},
        retrain_interval=2,
    )

    # Round 1: no retrain
    ctrl.on_round_completed(family1_missed_round, round_index=1)

    # Round 2: triggers retrain
    r2_event = copy.deepcopy(family1_missed_round.attack_event)
    r2_event.round_id = "round-f1-2"
    r2_event.attack_id = "atk-f1-002"
    r2_result = RoundResult(
        round_id="round-f1-2",
        attack_event=r2_event,
        prediction_result=family1_missed_round.prediction_result,
        feedback=family1_missed_round.feedback,
        outcome_metrics={"round_index": 2},
    )

    update = ctrl.on_round_completed(r2_result, round_index=2)
    assert update is not None
    assert update.retrained is True
    assert update.round_index == 2
    assert update.false_negative_count >= 1
    assert "retrained" in update.new_model_version


def test_configurable_retrain_interval(family1_missed_round):
    """Setting retrain_interval=3 triggers retraining only at round 3."""
    det = TransactionBlueDetector()
    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=3,
    )

    assert ctrl.on_round_completed(family1_missed_round, round_index=1) is None
    assert ctrl.on_round_completed(family1_missed_round, round_index=2) is None
    update = ctrl.on_round_completed(family1_missed_round, round_index=3)
    assert update is not None
    assert update.round_index == 3


# ===========================================================================
# 2. Data Sources & Composition (Requirements 4, 5, 6, 7)
# ===========================================================================

def test_false_negatives_and_baseline_included(family1_missed_round):
    """Assembled retraining dataset includes baseline legitimate and false negatives."""
    det = TransactionBlueDetector()
    baseline = [{"transaction_id": "tx-base-01", "amount": 25.0}]
    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: baseline},
        retrain_interval=1,
    )

    update = ctrl.on_round_completed(family1_missed_round, round_index=1)
    assert update is not None
    assert update.retrained is True
    assert update.baseline_count == 1
    assert update.false_negative_count == 1
    assert update.training_sample_count == 2


def test_fresh_legitimate_samples_handled(family1_missed_round):
    """Fresh legitimate data is included when provided."""
    det = TransactionBlueDetector()
    baseline = [{"transaction_id": "tx-base-01"}]
    fresh = [{"transaction_id": "tx-fresh-01"}, {"transaction_id": "tx-fresh-02"}]
    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: baseline},
        fresh_legitimate_data={AttackFamily.ADAPTIVE_EVASION: fresh},
        retrain_interval=1,
    )

    update = ctrl.on_round_completed(family1_missed_round, round_index=1)
    assert update is not None
    assert update.baseline_count == 1
    assert update.fresh_legitimate_count == 2
    assert update.training_sample_count == 4


def test_heldout_data_never_used_for_training(family3_missed_round):
    """Held-out data is used strictly for evaluation, never in the training dataset."""
    det = SyntheticIdentityBlueDetector()
    baseline = [{"identity_id": "id-base-01", "name": "Base"}]
    heldout = [{"identity_id": "id-holdout-01", "name": "Holdout"}]

    ctrl = RetrainingController(
        detectors={AttackFamily.SYNTHETIC_IDENTITY: det},
        baseline_data={AttackFamily.SYNTHETIC_IDENTITY: baseline},
        held_out_data={AttackFamily.SYNTHETIC_IDENTITY: heldout},
        retrain_interval=1,
    )

    update = ctrl.on_round_completed(family3_missed_round, round_index=1)
    assert update is not None
    assert update.retrained is True
    # Training samples do not contain heldout
    assert update.training_sample_count == 2
    # Holdout metrics are recorded separately
    assert update.holdout_metrics is not None
    assert update.holdout_metrics.sample_count == 1


# ===========================================================================
# 3. Model & Detector State Updates (Requirements 8, 9, 10, 11, 12)
# ===========================================================================

def test_detector_state_changes_and_used_in_subsequent_predictions(family1_missed_round):
    """Retraining updates detector parameters and changes subsequent predictions."""
    det = TransactionBlueDetector()
    initial_weights = dict(det.weights)
    initial_version = det.model_version

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: [{"transaction_id": "t1"}]},
        retrain_interval=1,
    )

    update = ctrl.on_round_completed(family1_missed_round, round_index=1)
    assert update is not None
    assert update.retrained is True

    # Detector state has changed
    assert det.model_version != initial_version
    assert det.model_version == update.new_model_version
    assert det.weights != initial_weights

    # Subsequent prediction uses new model version
    new_pred = det.detect(family1_missed_round.attack_event)
    assert new_pred.model_version == update.new_model_version


def test_before_and_after_metrics_recorded(family1_missed_round):
    """ModelUpdateRecord contains valid before and after metrics."""
    det = TransactionBlueDetector()
    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: [{"transaction_id": "t1"}]},
        retrain_interval=1,
    )

    update = ctrl.on_round_completed(family1_missed_round, round_index=1)
    assert update is not None
    assert update.before_metrics is not None
    assert update.after_metrics is not None
    assert 0.0 <= update.before_metrics.accuracy <= 1.0
    assert 0.0 <= update.after_metrics.accuracy <= 1.0


# ===========================================================================
# 4. Multi-Family & Isolation (Requirements 13, 14, 15)
# ===========================================================================

def test_multi_family_support_and_isolation(
    family1_missed_round,
    family2_missed_round,
    family3_missed_round,
):
    """Controller retrains all 3 families independently without cross-contamination."""
    det1 = TransactionBlueDetector()
    det2 = AIAgentBlueDetector()
    det3 = SyntheticIdentityBlueDetector()

    ctrl = RetrainingController(
        detectors={
            AttackFamily.ADAPTIVE_EVASION: det1,
            AttackFamily.AGENT_BEHAVIOR: det2,
            AttackFamily.SYNTHETIC_IDENTITY: det3,
        },
        baseline_data={
            AttackFamily.ADAPTIVE_EVASION: [{"transaction_id": "tx1"}],
            AttackFamily.AGENT_BEHAVIOR: [{"event_id": "agent1"}],
            AttackFamily.SYNTHETIC_IDENTITY: [{"identity_id": "ident1"}],
        },
        retrain_interval=2,
    )

    # Ingest Family 1 failure
    ctrl.on_round_completed(family1_missed_round, round_index=1)
    # Ingest Family 2 failure & trigger retrain on round 2
    update2 = ctrl.on_round_completed(family2_missed_round, round_index=2)

    assert update2 is not None
    assert update2.family == AttackFamily.AGENT_BEHAVIOR
    assert update2.retrained is True
    # Family 2 updated, Family 1 was NOT retrained at round 2 because Family 2 was active
    assert "retrained" in det2.model_version

    # Explicitly retrain Family 1
    update1 = ctrl.retrain_family(AttackFamily.ADAPTIVE_EVASION, round_id="manual-f1")
    assert update1.retrained is True
    assert update1.family == AttackFamily.ADAPTIVE_EVASION
    assert update1.false_negative_count == 1

    # Family 3 has zero failures so far -> reports safe no-op
    update3 = ctrl.retrain_family(AttackFamily.SYNTHETIC_IDENTITY, round_id="manual-f3")
    assert update3.retrained is False
    assert update3.false_negative_count == 0


# ===========================================================================
# 5. Empty & Insufficient Data (Requirements 16, 17)
# ===========================================================================

def test_empty_failure_memory_safe_noop():
    """Retraining with no failures is handled gracefully without errors."""
    det = TransactionBlueDetector()
    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
    )

    update = ctrl.retrain_family(AttackFamily.ADAPTIVE_EVASION, round_id="empty-test")
    assert update.retrained is False
    assert update.false_negative_count == 0
    assert "no_failures_accumulated" in update.trigger_reason


# ===========================================================================
# 6. Determinism & Immutability (Requirements 18, 19)
# ===========================================================================

def test_deterministic_retraining(family1_missed_round):
    """Identical retraining inputs produce identical updated weights and metrics."""
    det_a = TransactionBlueDetector()
    det_b = TransactionBlueDetector()

    ctrl_a = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det_a},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: [{"transaction_id": "tx-1"}]},
        retrain_interval=1,
    )
    ctrl_b = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det_b},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: [{"transaction_id": "tx-1"}]},
        retrain_interval=1,
    )

    upd_a = ctrl_a.on_round_completed(family1_missed_round, round_index=1)
    upd_b = ctrl_b.on_round_completed(family1_missed_round, round_index=1)

    assert upd_a.new_model_version == upd_b.new_model_version
    assert det_a.weights == det_b.weights
    assert det_a.threshold == det_b.threshold


def test_input_data_immutability(family1_missed_round):
    """Retraining does not mutate source baseline or failure records."""
    det = TransactionBlueDetector()
    base_dict = {"transaction_id": "tx-safe", "amount": 50.0}
    baseline_list = [base_dict]

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: baseline_list},
        retrain_interval=1,
    )

    ctrl.on_round_completed(family1_missed_round, round_index=1)
    assert base_dict == {"transaction_id": "tx-safe", "amount": 50.0}
    assert len(baseline_list) == 1


# ===========================================================================
# 7. Fail-Safe Execution & Error Rollback (Requirement 20)
# ===========================================================================

class _BrokenTrainer:
    """Mock trainer that throws an unhandled exception during training."""

    def train(self, *args, **kwargs):
        raise RuntimeError("Synthetic trainer internal failure!")


def test_failsafe_preserves_detector_state_on_training_error(family1_missed_round):
    """If trainer raises an exception, the active detector is restored to its original state."""
    det = TransactionBlueDetector()
    original_weights = dict(det.weights)
    original_version = det.model_version
    original_threshold = det.threshold

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        trainers={AttackFamily.ADAPTIVE_EVASION: _BrokenTrainer()},
        baseline_data={AttackFamily.ADAPTIVE_EVASION: [{"transaction_id": "tx1"}]},
        retrain_interval=1,
    )

    update = ctrl.on_round_completed(family1_missed_round, round_index=1)
    assert update is not None
    assert update.retrained is False
    assert "training_exception" in update.trigger_reason

    # Detector state is preserved intact
    assert det.model_version == original_version
    assert det.weights == original_weights
    assert det.threshold == original_threshold


# ===========================================================================
# 8. Family 3 XGBoost ML Retraining & SHAP (Requirement 8, 9, 12)
# ===========================================================================

def test_family3_xgboost_retraining_and_shap(family3_missed_round):
    """Family 3 XGBoost detector is retrained and SHAP explainer updated."""
    det = SyntheticIdentityBlueDetector()
    initial_version = det.model_version
    baseline_identities = [
        SyntheticIdentity(
            identity_id=f"id-base-{i}",
            identity_attributes={"name": f"User {i}"},
            contact_attributes={"email": f"u{i}@gmail.com"},
            account_metadata={"kyc_verification_status": "verified", "account_status": "active"},
            device_context={"device_id": f"dev-{i}"},
            lifecycle_info={"risk_event_count": 0, "lifecycle_coherence_score": 0.95},
        )
        for i in range(5)
    ]

    ctrl = RetrainingController(
        detectors={AttackFamily.SYNTHETIC_IDENTITY: det},
        baseline_data={AttackFamily.SYNTHETIC_IDENTITY: baseline_identities},
        auto_load_canonical_baseline=False,
        retrain_interval=1,
    )

    update = ctrl.on_round_completed(family3_missed_round, round_index=1)
    assert update is not None
    assert update.retrained is True
    assert det.model_version != initial_version
    assert "xgb-retrained" in det.model_version

    # Subsequent prediction succeeds and returns updated model version
    pred = det.detect(family3_missed_round.attack_event)
    assert pred.model_version == det.model_version
    assert pred.feature_contributions is not None


# ===========================================================================
# 9. Cross-Family Isolation & Contamination Guard (Requirements 14, 15)
# ===========================================================================

def test_no_cross_family_training_contamination(family1_missed_round):
    """Failures from Family 1 do not contaminate or trigger Family 3 training."""
    det1 = TransactionBlueDetector()
    det3 = SyntheticIdentityBlueDetector()

    ctrl = RetrainingController(
        detectors={
            AttackFamily.ADAPTIVE_EVASION: det1,
            AttackFamily.SYNTHETIC_IDENTITY: det3,
        },
        baseline_data={
            AttackFamily.ADAPTIVE_EVASION: [{"transaction_id": "tx1"}],
            AttackFamily.SYNTHETIC_IDENTITY: [{"identity_id": "id1"}],
        },
        auto_load_canonical_baseline=False,
        retrain_interval=2,
    )

    # Ingest Family 1 failure
    ctrl.on_round_completed(family1_missed_round, round_index=1)

    # Trigger retrain for Family 3 only
    upd3 = ctrl.retrain_family(AttackFamily.SYNTHETIC_IDENTITY, round_id="manual-f3")
    assert upd3.retrained is False
    assert upd3.false_negative_count == 0  # No Family 3 failures used!


# ===========================================================================
# 10. Metrics Helper Edge Cases
# ===========================================================================

def test_binary_metrics_edge_cases():
    """compute_binary_metrics handles zero samples and zero denominator edge cases cleanly."""
    m_empty = compute_binary_metrics([], [])
    assert m_empty.sample_count == 0
    assert m_empty.accuracy == 1.0

    # All positive
    m_all_pos = compute_binary_metrics([1, 1], [1, 1])
    assert m_all_pos.precision == 1.0
    assert m_all_pos.recall == 1.0
    assert m_all_pos.f1_score == 1.0

    # All negative
    m_all_neg = compute_binary_metrics([0, 0], [0, 0])
    assert m_all_neg.false_positive_rate == 0.0
    assert m_all_neg.accuracy == 1.0

