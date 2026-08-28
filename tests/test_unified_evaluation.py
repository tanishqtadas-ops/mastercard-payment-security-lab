"""
tests/test_unified_evaluation.py — Comprehensive Test Suite for Unified Cross-Family Evaluation (Task 8.1).

Covers all 16 required evaluation behaviors:
1. Family 1 evaluation (Transaction Evasion)
2. Family 2 evaluation (AI-Agent Behavior)
3. Family 3 evaluation (Synthetic Identity)
4. Per-family metric calculation
5. Consolidated report construction across all 3 families
6. Confusion matrix counts (TP, FP, TN, FN)
7. Precision, recall / detection rate, FPR, and F1 calculations
8. Clean held-out evaluation (clean pass rate / false alarm rate)
9. Before vs after learning / retraining comparison
10. Missing / unsupported metric handling (no fabrication of undefined metrics)
11. Deterministic output
12. Family isolation (metrics are segregated per family)
13. Empty-data handling (graceful no-op with 0 samples)
14. Malformed input handling
15. No training occurs during evaluation (detectors remain unmodified)
16. Held-out data is never passed into training pathways
"""

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import pytest

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.round import RoundResult
from schemas.transaction import Transaction
from schemas.identity import SyntheticIdentity

# Blue team detectors
from blue_team.transaction.detector import TransactionBlueDetector
from blue_team.ai_agent.detector import AIAgentBlueDetector
from blue_team.synthetic_identity.detector import SyntheticIdentityBlueDetector

# Learning records and dataset models
from blue_team.learning.retraining import ModelUpdateRecord, EvaluationMetrics
from blue_team.learning.dataset import (
    DatasetSample,
    ProvenanceType,
    HoldoutDataLeakageError,
    assemble_retraining_dataset,
)

# Unified evaluation suite
from evaluation import (
    ClassificationMetrics,
    ConfusionMatrix,
    RiskMetrics,
    FamilyEvaluationResult,
    ConsolidatedMetrics,
    HoldoutEvaluationResult,
    BeforeAfterComparison,
    RecoverySummary,
    UnifiedEvaluationReport,
    Family1EvaluationAdapter,
    Family2EvaluationAdapter,
    Family3EvaluationAdapter,
    UnifiedEvaluator,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_risk_metrics,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_round_results() -> List[RoundResult]:
    """Create a diverse set of simulation RoundResults spanning Families 1, 2, and 3."""
    # Family 1: 1 detected attack, 1 missed attack, 1 legitimate (TN)
    f1_r1 = RoundResult(
        round_id="f1-r1",
        attack_event=AttackEvent(
            attack_id="atk-f1-1",
            round_id="f1-r1",
            attack_family=AttackFamily.ADAPTIVE_EVASION,
            attack_genome={"amount_deviation": 0.8},
            scenario={"transaction": {"amount": 500.0, "location": "Unknown"}},
            ground_truth=True,
        ),
        prediction_result=PredictionResult(
            prediction_id="p-f1-1",
            prediction=True,
            risk_score=0.85,
            model_version="heuristic-family1-v1",
        ),
        feedback=BlueTeamFeedback(
            feedback_id="fb-1",
            round_reference="f1-r1",
            detected=True,
            false_positive=False,
            false_negative=False,
            risk_score=0.85,
            important_features={"amount_deviation": 0.8},
        ),
    )
    f1_r2 = RoundResult(
        round_id="f1-r2",
        attack_event=AttackEvent(
            attack_id="atk-f1-2",
            round_id="f1-r2",
            attack_family=AttackFamily.ADAPTIVE_EVASION,
            attack_genome={"amount_deviation": 0.1},
            scenario={"transaction": {"amount": 50.0, "location": "NYC"}},
            ground_truth=True,
        ),
        prediction_result=PredictionResult(
            prediction_id="p-f1-2",
            prediction=False,
            risk_score=0.20,
            model_version="heuristic-family1-v1",
        ),
        feedback=BlueTeamFeedback(
            feedback_id="fb-2",
            round_reference="f1-r2",
            detected=False,
            false_positive=False,
            false_negative=True,
            risk_score=0.20,
            important_features={},
        ),
    )
    f1_r3 = RoundResult(
        round_id="f1-r3",
        attack_event=AttackEvent(
            attack_id="atk-f1-3",
            round_id="f1-r3",
            attack_family=AttackFamily.ADAPTIVE_EVASION,
            attack_genome={"amount_deviation": 0.05},
            scenario={"transaction": {"amount": 25.0, "location": "NYC"}},
            ground_truth=False,
        ),
        prediction_result=PredictionResult(
            prediction_id="p-f1-3",
            prediction=False,
            risk_score=0.10,
            model_version="heuristic-family1-v1",
        ),
        feedback=BlueTeamFeedback(
            feedback_id="fb-3",
            round_reference="f1-r3",
            detected=False,
            false_positive=False,
            false_negative=False,
            risk_score=0.10,
            important_features={},
        ),
    )

    # Family 2: 2 detected attacks, 0 misses
    f2_r1 = RoundResult(
        round_id="f2-r1",
        attack_event=AttackEvent(
            attack_id="atk-f2-1",
            round_id="f2-r1",
            attack_family=AttackFamily.AGENT_BEHAVIOR,
            attack_genome={"intent_amount_deviation": 0.9},
            scenario={"event_id": "e1", "actual_action": "unauthorized_crypto_transfer"},
            ground_truth=True,
        ),
        prediction_result=PredictionResult(
            prediction_id="p-f2-1",
            prediction=True,
            risk_score=0.90,
            model_version="heuristic-family2-v1",
        ),
        feedback=BlueTeamFeedback(
            feedback_id="fb-4",
            round_reference="f2-r1",
            detected=True,
            false_positive=False,
            false_negative=False,
            risk_score=0.90,
            important_features={},
        ),
    )
    f2_r2 = RoundResult(
        round_id="f2-r2",
        attack_event=AttackEvent(
            attack_id="atk-f2-2",
            round_id="f2-r2",
            attack_family=AttackFamily.AGENT_BEHAVIOR,
            attack_genome={"intent_amount_deviation": 0.7},
            scenario={"event_id": "e2", "actual_action": "scope_breach"},
            ground_truth=True,
        ),
        prediction_result=PredictionResult(
            prediction_id="p-f2-2",
            prediction=True,
            risk_score=0.75,
            model_version="heuristic-family2-v1",
        ),
        feedback=BlueTeamFeedback(
            feedback_id="fb-5",
            round_reference="f2-r2",
            detected=True,
            false_positive=False,
            false_negative=False,
            risk_score=0.75,
            important_features={},
        ),
    )

    # Family 3: 1 detected attack, 1 missed attack
    f3_r1 = RoundResult(
        round_id="f3-r1",
        attack_event=AttackEvent(
            attack_id="atk-f3-1",
            round_id="f3-r1",
            attack_family=AttackFamily.SYNTHETIC_IDENTITY,
            attack_genome={"cross_field_consistency": 0.2},
            scenario={"identity_id": "syn-1", "identity_attributes": {"name": "Bot One"}},
            ground_truth=True,
        ),
        prediction_result=PredictionResult(
            prediction_id="p-f3-1",
            prediction=True,
            risk_score=0.80,
            model_version="family3-xgb-v1",
        ),
        feedback=BlueTeamFeedback(
            feedback_id="fb-6",
            round_reference="f3-r1",
            detected=True,
            false_positive=False,
            false_negative=False,
            risk_score=0.80,
            important_features={},
        ),
    )
    f3_r2 = RoundResult(
        round_id="f3-r2",
        attack_event=AttackEvent(
            attack_id="atk-f3-2",
            round_id="f3-r2",
            attack_family=AttackFamily.SYNTHETIC_IDENTITY,
            attack_genome={"cross_field_consistency": 0.9},
            scenario={"identity_id": "syn-2", "identity_attributes": {"name": "Bot Two"}},
            ground_truth=True,
        ),
        prediction_result=PredictionResult(
            prediction_id="p-f3-2",
            prediction=False,
            risk_score=0.30,
            model_version="family3-xgb-v1",
        ),
        feedback=BlueTeamFeedback(
            feedback_id="fb-7",
            round_reference="f3-r2",
            detected=False,
            false_positive=False,
            false_negative=True,
            risk_score=0.30,
            important_features={},
        ),
    )

    return [f1_r1, f1_r2, f1_r3, f2_r1, f2_r2, f3_r1, f3_r2]


# ===========================================================================
# 1. Family 1 Evaluation (Req 1)
# ===========================================================================

def test_family1_evaluation_adapter():
    """Requirement 1: Family 1 adapter correctly evaluates TransactionBlueDetector."""
    det = TransactionBlueDetector()
    adapter = Family1EvaluationAdapter()

    samples = [
        # Positive (Attack)
        DatasetSample(
            sample_id="f1-s1",
            label=1,
            provenance=ProvenanceType.FALSE_NEGATIVE,
            family=AttackFamily.ADAPTIVE_EVASION,
            data={"scenario": {"transaction": {"amount": 600.0, "location": "Unknown"}}},
        ),
        # Negative (Legitimate)
        DatasetSample(
            sample_id="f1-s2",
            label=0,
            provenance=ProvenanceType.BASELINE_LEGITIMATE,
            family=AttackFamily.ADAPTIVE_EVASION,
            data={"scenario": {"transaction": {"amount": 20.0, "location": "New York, US"}}},
        ),
    ]

    result = adapter.evaluate_detector(det, samples)
    assert isinstance(result, FamilyEvaluationResult)
    assert result.family == AttackFamily.ADAPTIVE_EVASION.value
    assert result.sample_count == 2
    assert result.attack_count == 1
    assert result.legitimate_count == 1
    assert result.metrics.accuracy == 1.0
    assert result.metrics.recall == 1.0


# ===========================================================================
# 2. Family 2 Evaluation (Req 2)
# ===========================================================================

def test_family2_evaluation_adapter():
    """Requirement 2: Family 2 adapter correctly evaluates AIAgentBlueDetector."""
    det = AIAgentBlueDetector()
    adapter = Family2EvaluationAdapter()

    samples = [
        # Malicious attack
        DatasetSample(
            sample_id="f2-s1",
            label=1,
            provenance=ProvenanceType.FALSE_NEGATIVE,
            family=AttackFamily.AGENT_BEHAVIOR,
            data={"scenario": {"event_id": "e1", "actual_action": "crypto_purchase"}},
        ),
        # Authorized legitimate
        DatasetSample(
            sample_id="f2-s2",
            label=0,
            provenance=ProvenanceType.BASELINE_LEGITIMATE,
            family=AttackFamily.AGENT_BEHAVIOR,
            data={"scenario": {"event_id": "e2", "actual_action": "office_supplies"}},
        ),
    ]

    result = adapter.evaluate_detector(det, samples)
    assert isinstance(result, FamilyEvaluationResult)
    assert result.family == AttackFamily.AGENT_BEHAVIOR.value
    assert result.sample_count == 2
    assert result.attack_count == 1
    assert result.legitimate_count == 1
    assert result.confusion_matrix.total_samples == 2


# ===========================================================================
# 3. Family 3 Evaluation (Req 3)
# ===========================================================================

def test_family3_evaluation_adapter():
    """Requirement 3: Family 3 adapter correctly evaluates SyntheticIdentityBlueDetector."""
    det = SyntheticIdentityBlueDetector()
    adapter = Family3EvaluationAdapter()

    samples = [
        # Legitimate identity
        SyntheticIdentity(
            identity_id="legit-1",
            identity_attributes={"name": "Alice Legit"},
            contact_attributes={"email": "alice@legit.com"},
            account_metadata={"kyc_verification_status": "verified", "account_status": "active"},
            device_context={"device_id": "dev-01"},
            lifecycle_info={"risk_event_count": 0, "lifecycle_coherence_score": 0.95},
        ),
    ]

    result = adapter.evaluate_detector(det, samples)
    assert isinstance(result, FamilyEvaluationResult)
    assert result.family == AttackFamily.SYNTHETIC_IDENTITY.value
    assert result.sample_count == 1
    assert result.legitimate_count == 1
    assert result.attack_count == 0


# ===========================================================================
# 4 & 5. Per-Family Metrics and Consolidated Report (Req 4, 5)
# ===========================================================================

def test_unified_evaluator_round_results_and_consolidation(sample_round_results):
    """Requirement 4, 5: UnifiedEvaluator builds per-family results and consolidated metrics."""
    evaluator = UnifiedEvaluator()
    report = evaluator.evaluate_round_results(sample_round_results, evaluation_id="test-run-01")

    assert isinstance(report, UnifiedEvaluationReport)
    assert report.evaluation_id == "test-run-01"
    assert len(report.per_family_results) == 3
    assert AttackFamily.ADAPTIVE_EVASION.name in report.per_family_results
    assert AttackFamily.AGENT_BEHAVIOR.name in report.per_family_results
    assert AttackFamily.SYNTHETIC_IDENTITY.name in report.per_family_results

    # Consolidated metrics check
    assert report.consolidated_metrics is not None
    cm = report.consolidated_metrics
    assert cm.total_samples == 7
    assert cm.total_attacks == 6
    assert cm.total_legitimate == 1
    # 4 TP (1 in F1, 2 in F2, 1 in F3), 2 FN (1 in F1, 1 in F3), 1 TN (in F1), 0 FP
    assert cm.confusion_matrix.true_positives == 4
    assert cm.confusion_matrix.false_negatives == 2
    assert cm.confusion_matrix.true_negatives == 1
    assert cm.confusion_matrix.false_positives == 0
    assert cm.overall_accuracy == round(5 / 7, 4)
    assert cm.overall_detection_rate == round(4 / 6, 4)
    assert cm.overall_precision == 1.0


# ===========================================================================
# 6 & 7. Confusion Matrix and Metric Calculations (Req 6, 7)
# ===========================================================================

def test_confusion_matrix_and_metrics_math():
    """Requirement 6, 7: Rigorous math for accuracy, precision, recall, FPR, and F1."""
    # TP=40, FP=10, TN=40, FN=10 (Total=100)
    y_true = [1] * 50 + [0] * 50
    y_pred = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40

    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.true_positives == 40
    assert cm.false_negatives == 10
    assert cm.false_positives == 10
    assert cm.true_negatives == 40
    assert cm.total_samples == 100

    metrics = compute_classification_metrics(cm)
    assert metrics.accuracy == 0.80
    assert metrics.recall == 0.80        # 40 / 50
    assert metrics.precision == 0.80     # 40 / 50
    assert metrics.false_positive_rate == 0.20  # 10 / 50
    assert metrics.f1_score == 0.80


# ===========================================================================
# 8. Clean Held-Out Evaluation (Req 8)
# ===========================================================================

def test_clean_held_out_evaluation():
    """Requirement 8: Clean held-out evaluation measures pass rate without false alarms."""
    evaluator = UnifiedEvaluator()

    det1 = TransactionBlueDetector()
    det2 = AIAgentBlueDetector()
    det3 = SyntheticIdentityBlueDetector()

    held_out_f1 = [
        Transaction(
            transaction_id=f"tx-h-{i}",
            user_id="usr-1",
            timestamp=datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc),
            amount=30.0 + i,
            currency="USD",
            merchant_id="m-1",
            merchant_category="groceries",
            location="New York, US",
            device_id="dev-1",
            payment_channel="pos",
        )
        for i in range(5)
    ]

    held_out_f3 = [
        SyntheticIdentity(
            identity_id=f"syn-h-{i}",
            identity_attributes={"name": f"Legit Identity {i}"},
            contact_attributes={"email": f"legit{i}@test.com"},
            account_metadata={"kyc_verification_status": "verified", "account_status": "active"},
            device_context={"device_id": f"dev-{i}"},
            lifecycle_info={"risk_event_count": 0, "lifecycle_coherence_score": 0.95},
        )
        for i in range(5)
    ]

    held_out_map = {
        AttackFamily.ADAPTIVE_EVASION: held_out_f1,
        AttackFamily.SYNTHETIC_IDENTITY: held_out_f3,
    }
    detectors = {
        AttackFamily.ADAPTIVE_EVASION: det1,
        AttackFamily.AGENT_BEHAVIOR: det2,
        AttackFamily.SYNTHETIC_IDENTITY: det3,
    }

    report = evaluator.evaluate_detectors(
        detectors=detectors,
        held_out_data=held_out_map,
        evaluation_id="test-holdout-run",
    )

    assert AttackFamily.ADAPTIVE_EVASION.name in report.holdout_results
    assert AttackFamily.SYNTHETIC_IDENTITY.name in report.holdout_results

    f1_holdout = report.holdout_results[AttackFamily.ADAPTIVE_EVASION.name]
    assert f1_holdout.sample_count == 5
    assert f1_holdout.clean_pass_rate == 1.0
    assert f1_holdout.false_positive_count == 0


# ===========================================================================
# 9. Before vs After Learning Comparison (Req 9)
# ===========================================================================

def test_before_after_learning_comparison():
    """Requirement 9: Compare metrics before and after retraining from ModelUpdateRecord."""
    evaluator = UnifiedEvaluator()

    rec = ModelUpdateRecord(
        retrained=True,
        trigger_reason="scheduled_interval",
        round_id="r2",
        round_index=2,
        family=AttackFamily.SYNTHETIC_IDENTITY,
        previous_model_version="family3-xgb-v1",
        new_model_version="family3-xgb-retrained-v1",
        training_sample_count=10,
        false_negative_count=2,
        before_metrics=EvaluationMetrics(
            accuracy=0.60,
            precision=0.60,
            recall=0.50,
            false_positive_rate=0.10,
            f1_score=0.5455,
            sample_count=10,
        ),
        after_metrics=EvaluationMetrics(
            accuracy=0.90,
            precision=0.90,
            recall=0.85,
            false_positive_rate=0.05,
            f1_score=0.8743,
            sample_count=10,
        ),
        details={"trees": 40},
    )

    comparisons = evaluator.compare_learning([rec])
    assert len(comparisons) == 1
    comp = comparisons[0]
    assert comp.family == AttackFamily.SYNTHETIC_IDENTITY.value
    assert comp.previous_model_version == "family3-xgb-v1"
    assert comp.new_model_version == "family3-xgb-retrained-v1"
    assert comp.accuracy_delta == 0.30
    assert comp.detection_rate_delta == 0.35
    assert comp.false_positive_rate_delta == -0.05
    assert comp.false_negatives_used == 2


# ===========================================================================
# 10. Missing / Unsupported Metric Handling (Req 10)
# ===========================================================================

def test_undefined_metrics_are_not_fabricated():
    """Requirement 10: Zero positive or zero negative samples cleanly result in None metrics."""
    # All negatives: Precision and Recall are undefined because TP+FP=0 and TP+FN=0
    cm_all_neg = ConfusionMatrix(true_positives=0, false_positives=0, true_negatives=10, false_negatives=0, total_samples=10)
    m_neg = compute_classification_metrics(cm_all_neg)
    assert m_neg.accuracy == 1.0
    assert m_neg.recall is None
    assert m_neg.precision is None
    assert m_neg.f1_score is None
    assert m_neg.false_positive_rate == 0.0

    # All positives: FPR is undefined because FP+TN=0
    cm_all_pos = ConfusionMatrix(true_positives=10, false_positives=0, true_negatives=0, false_negatives=0, total_samples=10)
    m_pos = compute_classification_metrics(cm_all_pos)
    assert m_pos.accuracy == 1.0
    assert m_pos.recall == 1.0
    assert m_pos.precision == 1.0
    assert m_pos.false_positive_rate is None


# ===========================================================================
# 11. Deterministic Output (Req 11)
# ===========================================================================

def test_evaluation_is_deterministic(sample_round_results):
    """Requirement 11: Evaluating the same inputs produces identical reports."""
    evaluator = UnifiedEvaluator()
    r1 = evaluator.evaluate_round_results(sample_round_results, evaluation_id="const-id")
    r2 = evaluator.evaluate_round_results(sample_round_results, evaluation_id="const-id")

    assert r1.consolidated_metrics == r2.consolidated_metrics
    assert r1.per_family_results == r2.per_family_results


# ===========================================================================
# 12. Family Isolation (Req 12)
# ===========================================================================

def test_family_isolation_in_evaluation():
    """Requirement 12: Family 1 rounds only affect Family 1 metrics."""
    evaluator = UnifiedEvaluator()

    f1_only = [
        RoundResult(
            round_id="f1-r1",
            attack_event=AttackEvent(
                attack_id="atk-f1-1",
                round_id="f1-r1",
                attack_family=AttackFamily.ADAPTIVE_EVASION,
                attack_genome={"amount_deviation": 0.5},
                scenario={},
                ground_truth=True,
            ),
            prediction_result=PredictionResult(
                prediction_id="p-1",
                prediction=True,
                risk_score=0.90,
                model_version="f1-v1",
            ),
            feedback=BlueTeamFeedback(
                feedback_id="fb-1",
                round_reference="f1-r1",
                detected=True,
                false_positive=False,
                false_negative=False,
                risk_score=0.90,
                important_features={},
            ),
        )
    ]

    report = evaluator.evaluate_round_results(f1_only)
    assert AttackFamily.ADAPTIVE_EVASION.name in report.per_family_results
    assert AttackFamily.AGENT_BEHAVIOR.name not in report.per_family_results
    assert AttackFamily.SYNTHETIC_IDENTITY.name not in report.per_family_results
    assert report.per_family_results[AttackFamily.ADAPTIVE_EVASION.name].sample_count == 1


# ===========================================================================
# 13. Empty-Data Handling (Req 13)
# ===========================================================================

def test_empty_data_handling():
    """Requirement 13: Empty round lists or datasets are handled gracefully."""
    evaluator = UnifiedEvaluator()
    report = evaluator.evaluate_round_results([])

    assert report.per_family_results == {}
    assert report.consolidated_metrics is None
    assert report.recovery_summary is None


# ===========================================================================
# 14. Malformed Input Handling (Req 14)
# ===========================================================================

def test_malformed_input_handling():
    """Requirement 14: compute_confusion_matrix raises ValueError on length mismatch."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_confusion_matrix([1, 0, 1], [1, 0])


# ===========================================================================
# 15. No Training Occurs During Evaluation (Req 15)
# ===========================================================================

def test_no_training_occurs_during_evaluation():
    """Requirement 15: Evaluation is strictly read-only; detectors are not mutated."""
    det = TransactionBlueDetector()
    orig_version = det.model_version
    orig_weights = copy.deepcopy(det.weights)
    orig_thresh = det.threshold

    evaluator = UnifiedEvaluator()
    sample = DatasetSample(
        sample_id="f1-s",
        label=1,
        provenance=ProvenanceType.FALSE_NEGATIVE,
        family=AttackFamily.ADAPTIVE_EVASION,
        data={"scenario": {"transaction": {"amount": 800.0}}},
    )

    evaluator.evaluate_detectors(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        test_datasets={AttackFamily.ADAPTIVE_EVASION: [sample]},
    )

    assert det.model_version == orig_version
    assert det.weights == orig_weights
    assert det.threshold == orig_thresh


# ===========================================================================
# 16. Held-Out Data Never Passed into Training (Req 16)
# ===========================================================================

def test_held_out_data_protection_intact():
    """Requirement 16: assemble_retraining_dataset detects and prevents holdout training attempts."""
    with pytest.raises(HoldoutDataLeakageError):
        assemble_retraining_dataset(
            baseline_data=["data/held_out/heldout_identities.json"],
        )
