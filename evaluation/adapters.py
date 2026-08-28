"""
evaluation/adapters.py — Family Evaluation Adapters.

Provides decoupled, family-specific evaluation adapters conforming to a common
FamilyEvaluationAdapter protocol. This preserves family-specific domain semantics
(transactions vs AI mandates vs identity lifecycles) while delivering a unified interface
for the cross-family evaluator.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.round import RoundResult
from schemas.transaction import Transaction
from schemas.identity import SyntheticIdentity

from blue_team.learning.dataset import DatasetSample, validate_no_holdout_leakage
from .metrics import (
    ConfusionMatrix,
    ClassificationMetrics,
    RiskMetrics,
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_risk_metrics,
)
from .report import FamilyEvaluationResult, HoldoutEvaluationResult


@runtime_checkable
class FamilyEvaluationAdapter(Protocol):
    """Protocol for family-specific evaluation adapters."""

    @property
    def family(self) -> AttackFamily:
        """The specific AttackFamily handled by this adapter."""
        ...

    def evaluate_detector(
        self,
        detector: Any,
        samples: Sequence[Any],
    ) -> FamilyEvaluationResult:
        """Evaluate a detector over an arbitrary sequence of family test samples."""
        ...

    def evaluate_round_results(
        self,
        results: Sequence[RoundResult],
    ) -> FamilyEvaluationResult:
        """Evaluate performance directly from completed simulation RoundResults."""
        ...

    def evaluate_held_out(
        self,
        detector: Any,
        held_out_data: Sequence[Any],
    ) -> HoldoutEvaluationResult:
        """Evaluate false positives on strictly isolated held-out legitimate data."""
        ...


# ===========================================================================
# Base Adapter Helper
# ===========================================================================

def _extract_family_key(family: Union[AttackFamily, str]) -> str:
    """Normalize AttackFamily to canonical string."""
    if isinstance(family, AttackFamily):
        return family.name
    fam_str = str(family).upper()
    for member in AttackFamily:
        if member.name == fam_str or member.value == str(family):
            return member.name
    return fam_str


class BaseFamilyEvaluationAdapter:
    """Common evaluation logic for packaging results, computing metrics, and holdout tests."""

    def __init__(self, family: AttackFamily) -> None:
        self._family = family

    @property
    def family(self) -> AttackFamily:
        return self._family

    def _build_family_result(
        self,
        model_version: str,
        y_true: List[int],
        y_pred: List[int],
        risk_scores: List[float],
        details: Optional[Dict[str, Any]] = None,
    ) -> FamilyEvaluationResult:
        cm = compute_confusion_matrix(y_true, y_pred)
        metrics = compute_classification_metrics(cm)
        risk_metrics = compute_risk_metrics(risk_scores)

        # False negative rate = FN / (TP + FN) if positives exist
        fnr: Optional[float] = None
        if cm.total_positives > 0:
            fnr = round(cm.false_negatives / cm.total_positives, 4)

        return FamilyEvaluationResult(
            family=self.family.value,
            model_version=model_version,
            sample_count=cm.total_samples,
            attack_count=cm.total_positives,
            legitimate_count=cm.total_negatives,
            confusion_matrix=cm,
            metrics=metrics,
            risk_metrics=risk_metrics,
            false_negative_rate=fnr,
            details=details or {},
        )

    def evaluate_round_results(
        self,
        results: Sequence[RoundResult],
    ) -> FamilyEvaluationResult:
        """Evaluate metrics directly from a list of completed simulation round results."""
        fam_key = _extract_family_key(self.family)
        filtered = [
            r for r in results
            if _extract_family_key(r.attack_event.attack_family) == fam_key
        ]

        if not filtered:
            return FamilyEvaluationResult(
                family=self.family.value,
                model_version="unknown",
                sample_count=0,
                attack_count=0,
                legitimate_count=0,
                confusion_matrix=ConfusionMatrix(),
                metrics=ClassificationMetrics(
                    accuracy=1.0,
                    precision=None,
                    recall=None,
                    false_positive_rate=None,
                    f1_score=None,
                    sample_count=0,
                ),
                risk_metrics=None,
                false_negative_rate=None,
                details={"reason": "no_matching_rounds"},
            )

        y_true = [1 if r.attack_event.ground_truth else 0 for r in filtered]
        y_pred = [1 if r.prediction_result.prediction else 0 for r in filtered]
        risk_scores = [float(r.prediction_result.risk_score) for r in filtered]
        last_version = filtered[-1].prediction_result.model_version or "unknown"

        return self._build_family_result(
            model_version=last_version,
            y_true=y_true,
            y_pred=y_pred,
            risk_scores=risk_scores,
            details={"rounds_evaluated": len(filtered)},
        )


# ===========================================================================
# Family 1 Adapter: Transaction Evasion
# ===========================================================================

class Family1EvaluationAdapter(BaseFamilyEvaluationAdapter):
    """Evaluation adapter for Family 1 (Adaptive Transaction-Pattern Evasion)."""

    def __init__(self) -> None:
        super().__init__(AttackFamily.ADAPTIVE_EVASION)

    def evaluate_detector(
        self,
        detector: Any,
        samples: Sequence[Any],
    ) -> FamilyEvaluationResult:
        """Evaluate TransactionBlueDetector over dataset samples or events."""
        y_true: List[int] = []
        y_pred: List[int] = []
        risk_scores: List[float] = []

        model_version = getattr(detector, "model_version", "heuristic-family1-v1")

        for idx, item in enumerate(samples):
            if isinstance(item, AttackEvent):
                event = item
            elif isinstance(item, DatasetSample):
                scenario = item.data.get("scenario", item.data) if isinstance(item.data, dict) else item.data
                event = AttackEvent(
                    attack_id=f"eval-f1-{idx}",
                    round_id=f"eval-f1-r{idx}",
                    attack_family=AttackFamily.ADAPTIVE_EVASION,
                    attack_genome=item.features or {},
                    scenario=scenario if isinstance(scenario, dict) else {"transaction": scenario},
                    ground_truth=item.is_attack,
                )
            elif isinstance(item, Transaction):
                event = AttackEvent(
                    attack_id=f"eval-f1-{idx}",
                    round_id=f"eval-f1-r{idx}",
                    attack_family=AttackFamily.ADAPTIVE_EVASION,
                    attack_genome={},
                    scenario={"transaction": item.model_dump()},
                    ground_truth=False,
                )
            elif isinstance(item, dict):
                gt = item.get("ground_truth", item.get("is_attack", False))
                scen = item.get("scenario", item)
                event = AttackEvent(
                    attack_id=f"eval-f1-{idx}",
                    round_id=f"eval-f1-r{idx}",
                    attack_family=AttackFamily.ADAPTIVE_EVASION,
                    attack_genome=item.get("genome", {}),
                    scenario=scen if isinstance(scen, dict) else {"transaction": scen},
                    ground_truth=bool(gt),
                )
            else:
                # Fallback object
                event = AttackEvent(
                    attack_id=f"eval-f1-{idx}",
                    round_id=f"eval-f1-r{idx}",
                    attack_family=AttackFamily.ADAPTIVE_EVASION,
                    scenario={"raw_data": str(item)},
                    ground_truth=False,
                )

            pred = detector.detect(event)
            y_true.append(1 if event.ground_truth else 0)
            y_pred.append(1 if pred.prediction else 0)
            risk_scores.append(float(pred.risk_score))

        return self._build_family_result(
            model_version=model_version,
            y_true=y_true,
            y_pred=y_pred,
            risk_scores=risk_scores,
            details={"detector_type": type(detector).__name__},
        )

    def evaluate_held_out(
        self,
        detector: Any,
        held_out_data: Sequence[Any],
    ) -> HoldoutEvaluationResult:
        """Evaluate clean pass rate on held-out legitimate transactions."""
        model_version = getattr(detector, "model_version", "heuristic-family1-v1")
        total = len(held_out_data)
        if total == 0:
            return HoldoutEvaluationResult(
                family=self.family.value,
                sample_count=0,
                true_negative_count=0,
                false_positive_count=0,
                clean_pass_rate=1.0,
                false_positive_rate=0.0,
                model_version=model_version,
                details={"reason": "empty_held_out_set"},
            )

        fp_count = 0
        tn_count = 0

        for idx, item in enumerate(held_out_data):
            scenario = item.model_dump() if hasattr(item, "model_dump") else item
            event = AttackEvent(
                attack_id=f"holdout-f1-{idx}",
                round_id=f"holdout-r{idx}",
                attack_family=AttackFamily.ADAPTIVE_EVASION,
                attack_genome={},
                scenario=scenario if isinstance(scenario, dict) else {"transaction": scenario},
                ground_truth=False,  # Held out is strictly legitimate
            )
            pred = detector.detect(event)
            if pred.prediction:
                fp_count += 1
            else:
                tn_count += 1

        pass_rate = round(tn_count / total, 4)
        fpr = round(fp_count / total, 4)

        return HoldoutEvaluationResult(
            family=self.family.value,
            sample_count=total,
            true_negative_count=tn_count,
            false_positive_count=fp_count,
            clean_pass_rate=pass_rate,
            false_positive_rate=fpr,
            model_version=model_version,
            details={"held_out_type": "legitimate_transactions"},
        )


# ===========================================================================
# Family 2 Adapter: AI-Agent Payment Behavior
# ===========================================================================

class Family2EvaluationAdapter(BaseFamilyEvaluationAdapter):
    """Evaluation adapter for Family 2 (AI-Agent Payment Behavior)."""

    def __init__(self) -> None:
        super().__init__(AttackFamily.AGENT_BEHAVIOR)

    def evaluate_detector(
        self,
        detector: Any,
        samples: Sequence[Any],
    ) -> FamilyEvaluationResult:
        """Evaluate AIAgentBlueDetector over agent payment event samples."""
        y_true: List[int] = []
        y_pred: List[int] = []
        risk_scores: List[float] = []

        model_version = getattr(detector, "model_version", "heuristic-family2-v1")

        for idx, item in enumerate(samples):
            if isinstance(item, AttackEvent):
                event = item
            elif isinstance(item, DatasetSample):
                scenario = item.data.get("scenario", item.data) if isinstance(item.data, dict) else item.data
                event = AttackEvent(
                    attack_id=f"eval-f2-{idx}",
                    round_id=f"eval-f2-r{idx}",
                    attack_family=AttackFamily.AGENT_BEHAVIOR,
                    attack_genome=item.features or {},
                    scenario=scenario if isinstance(scenario, dict) else {"event": scenario},
                    ground_truth=item.is_attack,
                )
            elif isinstance(item, dict):
                gt = item.get("ground_truth", item.get("is_attack", False))
                scen = item.get("scenario", item)
                event = AttackEvent(
                    attack_id=f"eval-f2-{idx}",
                    round_id=f"eval-f2-r{idx}",
                    attack_family=AttackFamily.AGENT_BEHAVIOR,
                    attack_genome=item.get("genome", {}),
                    scenario=scen if isinstance(scen, dict) else {"event": scen},
                    ground_truth=bool(gt),
                )
            else:
                scenario = item.model_dump() if hasattr(item, "model_dump") else item
                event = AttackEvent(
                    attack_id=f"eval-f2-{idx}",
                    round_id=f"eval-f2-r{idx}",
                    attack_family=AttackFamily.AGENT_BEHAVIOR,
                    attack_genome={},
                    scenario=scenario if isinstance(scenario, dict) else {"event": scenario},
                    ground_truth=False,
                )

            pred = detector.detect(event)
            y_true.append(1 if event.ground_truth else 0)
            y_pred.append(1 if pred.prediction else 0)
            risk_scores.append(float(pred.risk_score))

        return self._build_family_result(
            model_version=model_version,
            y_true=y_true,
            y_pred=y_pred,
            risk_scores=risk_scores,
            details={"detector_type": type(detector).__name__},
        )

    def evaluate_held_out(
        self,
        detector: Any,
        held_out_data: Sequence[Any],
    ) -> HoldoutEvaluationResult:
        """Evaluate clean pass rate on held-out authorized AI-agent payment events."""
        model_version = getattr(detector, "model_version", "heuristic-family2-v1")
        total = len(held_out_data)
        if total == 0:
            return HoldoutEvaluationResult(
                family=self.family.value,
                sample_count=0,
                true_negative_count=0,
                false_positive_count=0,
                clean_pass_rate=1.0,
                false_positive_rate=0.0,
                model_version=model_version,
                details={"reason": "empty_held_out_set"},
            )

        fp_count = 0
        tn_count = 0

        for idx, item in enumerate(held_out_data):
            scenario = item.model_dump() if hasattr(item, "model_dump") else item
            event = AttackEvent(
                attack_id=f"holdout-f2-{idx}",
                round_id=f"holdout-r{idx}",
                attack_family=AttackFamily.AGENT_BEHAVIOR,
                attack_genome={},
                scenario=scenario if isinstance(scenario, dict) else {"event": scenario},
                ground_truth=False,
            )
            pred = detector.detect(event)
            if pred.prediction:
                fp_count += 1
            else:
                tn_count += 1

        pass_rate = round(tn_count / total, 4)
        fpr = round(fp_count / total, 4)

        return HoldoutEvaluationResult(
            family=self.family.value,
            sample_count=total,
            true_negative_count=tn_count,
            false_positive_count=fp_count,
            clean_pass_rate=pass_rate,
            false_positive_rate=fpr,
            model_version=model_version,
            details={"held_out_type": "authorized_agent_events"},
        )


# ===========================================================================
# Family 3 Adapter: Synthetic Identity
# ===========================================================================

class Family3EvaluationAdapter(BaseFamilyEvaluationAdapter):
    """Evaluation adapter for Family 3 (Synthetic Identity)."""

    def __init__(self) -> None:
        super().__init__(AttackFamily.SYNTHETIC_IDENTITY)

    def evaluate_detector(
        self,
        detector: Any,
        samples: Sequence[Any],
    ) -> FamilyEvaluationResult:
        """Evaluate SyntheticIdentityBlueDetector over synthetic identity test samples."""
        y_true: List[int] = []
        y_pred: List[int] = []
        risk_scores: List[float] = []

        model_version = getattr(detector, "model_version", "family3-xgb-v1")

        for idx, item in enumerate(samples):
            if isinstance(item, AttackEvent):
                event = item
            elif isinstance(item, DatasetSample):
                scenario = item.data.get("scenario", item.data) if isinstance(item.data, dict) else item.data
                event = AttackEvent(
                    attack_id=f"eval-f3-{idx}",
                    round_id=f"eval-f3-r{idx}",
                    attack_family=AttackFamily.SYNTHETIC_IDENTITY,
                    attack_genome=item.features or {},
                    scenario=scenario,
                    ground_truth=item.is_attack,
                )
            elif isinstance(item, SyntheticIdentity):
                event = AttackEvent(
                    attack_id=f"eval-f3-{idx}",
                    round_id=f"eval-f3-r{idx}",
                    attack_family=AttackFamily.SYNTHETIC_IDENTITY,
                    attack_genome={},
                    scenario=item.model_dump(),
                    ground_truth=False,
                )
            elif isinstance(item, dict):
                gt = item.get("ground_truth", item.get("is_attack", False))
                scen = item.get("scenario", item)
                event = AttackEvent(
                    attack_id=f"eval-f3-{idx}",
                    round_id=f"eval-f3-r{idx}",
                    attack_family=AttackFamily.SYNTHETIC_IDENTITY,
                    attack_genome=item.get("genome", {}),
                    scenario=scen,
                    ground_truth=bool(gt),
                )
            else:
                scenario = item.model_dump() if hasattr(item, "model_dump") else item
                event = AttackEvent(
                    attack_id=f"eval-f3-{idx}",
                    round_id=f"eval-f3-r{idx}",
                    attack_family=AttackFamily.SYNTHETIC_IDENTITY,
                    attack_genome={},
                    scenario=scenario,
                    ground_truth=False,
                )

            pred = detector.detect(event)
            y_true.append(1 if event.ground_truth else 0)
            y_pred.append(1 if pred.prediction else 0)
            risk_scores.append(float(pred.risk_score))

        return self._build_family_result(
            model_version=model_version,
            y_true=y_true,
            y_pred=y_pred,
            risk_scores=risk_scores,
            details={"detector_type": type(detector).__name__},
        )

    def evaluate_held_out(
        self,
        detector: Any,
        held_out_data: Sequence[Any],
    ) -> HoldoutEvaluationResult:
        """Evaluate clean pass rate on strictly isolated legitimate held-out identities."""
        model_version = getattr(detector, "model_version", "family3-xgb-v1")
        total = len(held_out_data)
        if total == 0:
            return HoldoutEvaluationResult(
                family=self.family.value,
                sample_count=0,
                true_negative_count=0,
                false_positive_count=0,
                clean_pass_rate=1.0,
                false_positive_rate=0.0,
                model_version=model_version,
                details={"reason": "empty_held_out_set"},
            )

        fp_count = 0
        tn_count = 0

        for idx, item in enumerate(held_out_data):
            scenario = item.model_dump() if hasattr(item, "model_dump") else item
            event = AttackEvent(
                attack_id=f"holdout-f3-{idx}",
                round_id=f"holdout-r{idx}",
                attack_family=AttackFamily.SYNTHETIC_IDENTITY,
                attack_genome={},
                scenario=scenario,
                ground_truth=False,  # Held out is strictly legitimate
            )
            pred = detector.detect(event)
            if pred.prediction:
                fp_count += 1
            else:
                tn_count += 1

        pass_rate = round(tn_count / total, 4)
        fpr = round(fp_count / total, 4)

        return HoldoutEvaluationResult(
            family=self.family.value,
            sample_count=total,
            true_negative_count=tn_count,
            false_positive_count=fp_count,
            clean_pass_rate=pass_rate,
            false_positive_rate=fpr,
            model_version=model_version,
            details={"held_out_type": "legitimate_identities"},
        )
