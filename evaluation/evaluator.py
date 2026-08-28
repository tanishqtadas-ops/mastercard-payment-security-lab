"""
evaluation/evaluator.py — Unified Cross-Family Evaluator.

Provides the core UnifiedEvaluator engine to orchestrate family-agnostic, multi-family
evaluation across simulation round histories, benchmark test datasets, held-out evaluation
sets, and Blue-Team model retraining update records.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

from schemas.common import AttackFamily
from schemas.round import RoundResult
from blue_team.learning.retraining import ModelUpdateRecord
from blue_team.learning.dataset import HoldoutDataLeakageError, validate_no_holdout_leakage

from .metrics import (
    ClassificationMetrics,
    ConfusionMatrix,
    RiskMetrics,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_risk_metrics,
)
from .report import (
    BeforeAfterComparison,
    ConsolidatedMetrics,
    FamilyEvaluationResult,
    HoldoutEvaluationResult,
    RecoverySummary,
    UnifiedEvaluationReport,
)
from .adapters import (
    FamilyEvaluationAdapter,
    Family1EvaluationAdapter,
    Family2EvaluationAdapter,
    Family3EvaluationAdapter,
    _extract_family_key,
)


class UnifiedEvaluator:
    """
    Coordinates and computes unified, cross-family evaluation reports across all three
    attack families without hardcoded branching.
    """

    def __init__(
        self,
        adapters: Optional[Dict[str, FamilyEvaluationAdapter]] = None,
    ) -> None:
        """
        Initialize the UnifiedEvaluator.

        Args:
            adapters: Optional custom mapping of family names to adapter instances.
                      Defaults to registering Family 1, 2, and 3 standard adapters.
        """
        self._adapters: Dict[str, FamilyEvaluationAdapter] = {}

        if adapters is not None:
            for k, adapter in adapters.items():
                self.register_adapter(adapter)
        else:
            # Register canonical adapters by default
            self.register_adapter(Family1EvaluationAdapter())
            self.register_adapter(Family2EvaluationAdapter())
            self.register_adapter(Family3EvaluationAdapter())

    def register_adapter(self, adapter: FamilyEvaluationAdapter) -> None:
        """Register or override an evaluation adapter for a specific family."""
        key = _extract_family_key(adapter.family)
        self._adapters[key] = adapter

    def get_adapter(self, family: Union[AttackFamily, str]) -> Optional[FamilyEvaluationAdapter]:
        """Retrieve the registered adapter for a family, if available."""
        key = _extract_family_key(family)
        return self._adapters.get(key)

    @property
    def registered_families(self) -> List[str]:
        """Return list of canonical family keys with registered adapters."""
        return list(self._adapters.keys())

    # =======================================================================
    # 1. Evaluate from Simulation Round Results
    # =======================================================================

    def evaluate_round_results(
        self,
        results: Sequence[RoundResult],
        evaluation_id: Optional[str] = None,
        update_records: Optional[Sequence[ModelUpdateRecord]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedEvaluationReport:
        """
        Compute a comprehensive evaluation report from a sequence of completed simulation rounds.

        Args:
            results: Sequence of completed RoundResult objects across any families.
            evaluation_id: Optional custom evaluation identifier (defaults to uuid4).
            update_records: Optional list of ModelUpdateRecords captured during the run.
            metadata: Optional additional metadata dict.

        Returns:
            UnifiedEvaluationReport containing per-family and consolidated metrics.
        """
        eval_id = evaluation_id or f"eval-rounds-{uuid.uuid4().hex[:8]}"
        per_family_results: Dict[str, FamilyEvaluationResult] = {}
        model_versions: Dict[str, str] = {}

        # Group rounds by family and evaluate via registered adapters
        grouped_rounds: Dict[str, List[RoundResult]] = {}
        for r in results:
            fam_key = _extract_family_key(r.attack_event.attack_family)
            grouped_rounds.setdefault(fam_key, []).append(r)
            if r.prediction_result.model_version:
                model_versions[fam_key] = r.prediction_result.model_version

        for fam_key, family_rounds in grouped_rounds.items():
            adapter = self._adapters.get(fam_key)
            if adapter is not None:
                per_family_results[fam_key] = adapter.evaluate_round_results(family_rounds)
            else:
                # Generic fallback if no specific adapter is registered
                y_true = [1 if r.attack_event.ground_truth else 0 for r in family_rounds]
                y_pred = [1 if r.prediction_result.prediction else 0 for r in family_rounds]
                risk_scores = [float(r.prediction_result.risk_score) for r in family_rounds]
                cm = compute_confusion_matrix(y_true, y_pred)
                metrics = compute_classification_metrics(cm)
                risk_m = compute_risk_metrics(risk_scores)
                ver = family_rounds[-1].prediction_result.model_version or "unknown"
                per_family_results[fam_key] = FamilyEvaluationResult(
                    family=fam_key,
                    model_version=ver,
                    sample_count=len(family_rounds),
                    attack_count=cm.total_positives,
                    legitimate_count=cm.total_negatives,
                    confusion_matrix=cm,
                    metrics=metrics,
                    risk_metrics=risk_m,
                    details={"rounds_count": len(family_rounds)},
                )

        consolidated = self.consolidate_metrics(list(per_family_results.values()))
        comparisons = self.compare_learning(update_records or [])
        recovery = self.compute_recovery_summary(results)

        return UnifiedEvaluationReport(
            evaluation_id=eval_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            per_family_results=per_family_results,
            consolidated_metrics=consolidated,
            holdout_results={},
            before_after_comparisons=comparisons,
            recovery_summary=recovery,
            model_versions=model_versions,
            metadata=metadata or {"total_rounds_ingested": len(results)},
        )

    # =======================================================================
    # 2. Evaluate Live Detectors Over Test Datasets & Held-Out Data
    # =======================================================================

    def evaluate_detectors(
        self,
        detectors: Dict[Union[AttackFamily, str], Any],
        test_datasets: Optional[Dict[Union[AttackFamily, str], Sequence[Any]]] = None,
        held_out_data: Optional[Dict[Union[AttackFamily, str], Sequence[Any]]] = None,
        update_records: Optional[Sequence[ModelUpdateRecord]] = None,
        evaluation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedEvaluationReport:
        """
        Evaluate active detector instances directly across family test datasets and held-out data.

        Args:
            detectors: Mapping of family to active BlueTeamDetector instances.
            test_datasets: Mapping of family to evaluation test samples.
            held_out_data: Mapping of family to strictly isolated clean legitimate holdout sets.
            update_records: Optional retraining model update records to compare.
            evaluation_id: Optional identifier.
            metadata: Optional metadata.

        Returns:
            UnifiedEvaluationReport containing dataset evaluation and holdout results.
        """
        eval_id = evaluation_id or f"eval-detectors-{uuid.uuid4().hex[:8]}"
        per_family_results: Dict[str, FamilyEvaluationResult] = {}
        holdout_results: Dict[str, HoldoutEvaluationResult] = {}
        model_versions: Dict[str, str] = {}

        # 1. Dataset evaluations
        if test_datasets:
            for family_key_raw, samples in test_datasets.items():
                fam_key = _extract_family_key(family_key_raw)
                detector = self._find_detector(detectors, fam_key)
                if detector is None:
                    continue

                ver = getattr(detector, "model_version", "unknown")
                model_versions[fam_key] = ver

                adapter = self._adapters.get(fam_key)
                if adapter is not None:
                    per_family_results[fam_key] = adapter.evaluate_detector(detector, samples)

        # 2. Held-out evaluations (strictly clean legitimate data)
        if held_out_data:
            holdout_evals = self.evaluate_held_out(detectors, held_out_data)
            holdout_results.update(holdout_evals)

        for fam_raw, det in detectors.items():
            k = _extract_family_key(fam_raw)
            if k not in model_versions:
                model_versions[k] = getattr(det, "model_version", "unknown")

        consolidated = self.consolidate_metrics(list(per_family_results.values()))
        comparisons = self.compare_learning(update_records or [])

        return UnifiedEvaluationReport(
            evaluation_id=eval_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            per_family_results=per_family_results,
            consolidated_metrics=consolidated,
            holdout_results=holdout_results,
            before_after_comparisons=comparisons,
            recovery_summary=None,
            model_versions=model_versions,
            metadata=metadata or {},
        )

    # =======================================================================
    # 3. Clean Held-Out Generalization Evaluation
    # =======================================================================

    def evaluate_held_out(
        self,
        detectors: Dict[Union[AttackFamily, str], Any],
        held_out_data: Dict[Union[AttackFamily, str], Sequence[Any]],
    ) -> Dict[str, HoldoutEvaluationResult]:
        """
        Evaluate detectors strictly on held-out clean legitimate samples.

        Ensures zero contamination or training on held-out datasets.
        """
        results: Dict[str, HoldoutEvaluationResult] = {}

        for family_key_raw, items in held_out_data.items():
            fam_key = _extract_family_key(family_key_raw)
            detector = self._find_detector(detectors, fam_key)
            if detector is None:
                continue

            adapter = self._adapters.get(fam_key)
            if adapter is not None:
                results[fam_key] = adapter.evaluate_held_out(detector, items)
            else:
                # Generic holdout pass-rate calculation
                total = len(items)
                fp = 0
                for item in items:
                    scen = item.model_dump() if hasattr(item, "model_dump") else item
                    dummy_event = AttackEvent(
                        attack_id="holdout-generic",
                        round_id="holdout-r",
                        attack_family=fam_key,
                        scenario=scen if isinstance(scen, dict) else {"data": scen},
                        ground_truth=False,
                    )
                    pred = detector.detect(dummy_event)
                    if pred.prediction:
                        fp += 1
                tn = total - fp
                pass_rate = round(tn / total, 4) if total > 0 else 1.0
                fpr = round(fp / total, 4) if total > 0 else 0.0
                ver = getattr(detector, "model_version", "unknown")
                results[fam_key] = HoldoutEvaluationResult(
                    family=fam_key,
                    sample_count=total,
                    true_negative_count=tn,
                    false_positive_count=fp,
                    clean_pass_rate=pass_rate,
                    false_positive_rate=fpr,
                    model_version=ver,
                    details={"held_out_type": "generic_legitimate"},
                )

        return results

    # =======================================================================
    # 4. Learning & Retraining Progression Comparison
    # =======================================================================

    def compare_learning(
        self,
        update_records: Sequence[ModelUpdateRecord],
    ) -> List[BeforeAfterComparison]:
        """
        Extract before vs after performance comparison from ModelUpdateRecords.
        """
        comparisons: List[BeforeAfterComparison] = []

        for rec in update_records:
            fam_name = str(rec.family.value if isinstance(rec.family, AttackFamily) else rec.family or "unknown")

            before_metrics: Optional[ClassificationMetrics] = None
            if rec.before_metrics:
                before_metrics = ClassificationMetrics(
                    accuracy=rec.before_metrics.accuracy,
                    precision=rec.before_metrics.precision,
                    recall=rec.before_metrics.recall,
                    false_positive_rate=rec.before_metrics.false_positive_rate,
                    f1_score=rec.before_metrics.f1_score,
                    sample_count=rec.before_metrics.sample_count,
                )

            after_metrics: Optional[ClassificationMetrics] = None
            if rec.after_metrics:
                after_metrics = ClassificationMetrics(
                    accuracy=rec.after_metrics.accuracy,
                    precision=rec.after_metrics.precision,
                    recall=rec.after_metrics.recall,
                    false_positive_rate=rec.after_metrics.false_positive_rate,
                    f1_score=rec.after_metrics.f1_score,
                    sample_count=rec.after_metrics.sample_count,
                )

            acc_delta = None
            rec_delta = None
            fpr_delta = None

            if before_metrics and after_metrics:
                acc_delta = round(after_metrics.accuracy - before_metrics.accuracy, 4)
                if before_metrics.recall is not None and after_metrics.recall is not None:
                    rec_delta = round(after_metrics.recall - before_metrics.recall, 4)
                if before_metrics.false_positive_rate is not None and after_metrics.false_positive_rate is not None:
                    fpr_delta = round(after_metrics.false_positive_rate - before_metrics.false_positive_rate, 4)

            comparisons.append(
                BeforeAfterComparison(
                    family=fam_name,
                    previous_model_version=rec.previous_model_version,
                    new_model_version=rec.new_model_version,
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    accuracy_delta=acc_delta,
                    detection_rate_delta=rec_delta,
                    false_positive_rate_delta=fpr_delta,
                    false_negatives_used=rec.false_negative_count,
                    details=rec.details,
                )
            )

        return comparisons

    # =======================================================================
    # 5. Recovery & Arms-Race Summary
    # =======================================================================

    def compute_recovery_summary(
        self,
        results: Sequence[RoundResult],
    ) -> Optional[RecoverySummary]:
        """
        Calculate adversarial evasion vs Blue-Team recovery metrics across chronological rounds.
        """
        if not results:
            return None

        total_evasions = 0
        total_recoveries = 0
        recovery_durations: List[int] = []

        last_missed_idx: Optional[int] = None

        for idx, r in enumerate(results, start=1):
            is_miss = bool(r.attack_event.ground_truth and not r.prediction_result.prediction)
            is_detected = bool(r.attack_event.ground_truth and r.prediction_result.prediction)

            if is_miss:
                total_evasions += 1
                if last_missed_idx is None:
                    last_missed_idx = idx
            elif is_detected and last_missed_idx is not None:
                total_recoveries += 1
                recovery_durations.append(idx - last_missed_idx)
                last_missed_idx = None

        rate = (total_recoveries / total_evasions) if total_evasions > 0 else 1.0
        avg_dur = (sum(recovery_durations) / len(recovery_durations)) if recovery_durations else None

        return RecoverySummary(
            total_evasions=total_evasions,
            total_recoveries=total_recoveries,
            recovery_rate=round(min(max(rate, 0.0), 1.0), 4),
            average_rounds_to_recover=round(avg_dur, 2) if avg_dur is not None else None,
            details={"durations": recovery_durations},
        )

    # =======================================================================
    # 6. Cross-Family Metric Consolidation
    # =======================================================================

    def consolidate_metrics(
        self,
        family_results: Sequence[FamilyEvaluationResult],
    ) -> Optional[ConsolidatedMetrics]:
        """
        Aggregate confusion matrices and compute mathematically sound overall metrics.
        """
        if not family_results:
            return None

        total_tp = sum(r.confusion_matrix.true_positives for r in family_results)
        total_fp = sum(r.confusion_matrix.false_positives for r in family_results)
        total_tn = sum(r.confusion_matrix.true_negatives for r in family_results)
        total_fn = sum(r.confusion_matrix.false_negatives for r in family_results)
        total_samples = sum(r.sample_count for r in family_results)
        total_attacks = sum(r.attack_count for r in family_results)
        total_legit = sum(r.legitimate_count for r in family_results)

        if total_samples == 0:
            return ConsolidatedMetrics(
                total_samples=0,
                total_attacks=0,
                total_legitimate=0,
                confusion_matrix=ConfusionMatrix(),
                overall_accuracy=1.0,
            )

        cm = ConfusionMatrix(
            true_positives=total_tp,
            false_positives=total_fp,
            true_negatives=total_tn,
            false_negatives=total_fn,
            total_samples=total_samples,
        )
        metrics = compute_classification_metrics(cm)

        # Average risk weighted by sample count
        risk_weighted_sum = 0.0
        risk_weight_total = 0
        for r in family_results:
            if r.risk_metrics and r.sample_count > 0:
                risk_weighted_sum += r.risk_metrics.average_risk * r.sample_count
                risk_weight_total += r.sample_count

        avg_risk = round(risk_weighted_sum / risk_weight_total, 4) if risk_weight_total > 0 else None

        return ConsolidatedMetrics(
            total_samples=total_samples,
            total_attacks=total_attacks,
            total_legitimate=total_legit,
            confusion_matrix=cm,
            overall_accuracy=metrics.accuracy,
            overall_detection_rate=metrics.recall,
            overall_precision=metrics.precision,
            overall_false_positive_rate=metrics.false_positive_rate,
            overall_f1=metrics.f1_score,
            average_risk=avg_risk,
        )

    # =======================================================================
    # Internal Helpers
    # =======================================================================

    def _find_detector(
        self,
        detectors: Dict[Union[AttackFamily, str], Any],
        target_key: str,
    ) -> Optional[Any]:
        """Locate a detector matching the canonical family key."""
        for k, det in detectors.items():
            if _extract_family_key(k) == target_key:
                return det
        return None
