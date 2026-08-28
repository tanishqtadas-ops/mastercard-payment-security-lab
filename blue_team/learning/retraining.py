"""
blue_team/learning/retraining.py — Blue-Team Model Retraining Controller & Family Adapters.

Orchestrates periodic Blue-Team model retraining / updates across all three attack families
in accordance with MASTER_SPEC.md § 5 ("Every 2 rounds: Retrain/update the corresponding Blue-Team model..."):
- Consumes accumulated false-negative failures from FailureMemory.
- Assembles holdout-safe training datasets using RetrainingDatasetAssembler.
- Executes family-specific adaptation/retraining mechanisms:
    * Family 1 (Transaction Evasion): Heuristic detector parameter adaptation / weight-threshold update
      (NOT supervised ML model fitting; adapts observable deviation weights towards missed signals).
    * Family 2 (AI Agent Behavior): Heuristic mandate-envelope parameter adaptation / sensitivity tuning
      (NOT supervised ML model fitting; adapts mandate weights towards missed agent evasion signals).
    * Family 3 (Synthetic Identity): Genuine supervised ML model refitting (fits an XGBoost classifier
      and initializes a SHAP TreeExplainer on legitimate baseline identities and missed synthetic fraud).
- Evaluates detectors before and after retraining using clean held-out evaluation datasets.
- Updates active detector instances in-place with new model versions and fail-safe fallbacks.
- Produces structured ModelUpdateRecords for traceability and dashboard progression analytics.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple, Union, runtime_checkable
import numpy as np
from pydantic import BaseModel, Field

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.round import RoundResult
from schemas.identity import SyntheticIdentity
from schemas.transaction import Transaction
from schemas.agent_event import AIAgentPaymentEvent

from blue_team.learning.failure_memory import FailureRecord, FailureMemory
from blue_team.learning.dataset import (
    ProvenanceType,
    DatasetSample,
    RetrainingDataset,
    assemble_retraining_dataset,
    validate_no_holdout_leakage,
    HoldoutDataLeakageError,
)

# Family detectors and feature extractors
from blue_team.transaction.detector import TransactionBlueDetector
from blue_team.transaction.feature_extractor import (
    FEATURE_NAMES as FAMILY1_FEATURE_NAMES,
    extract_transaction_features,
)
from blue_team.ai_agent.detector import AIAgentBlueDetector, extract_mandate_features
from blue_team.synthetic_identity.detector import (
    SyntheticIdentityBlueDetector,
    FEATURE_NAMES as FAMILY3_FEATURE_NAMES,
    DEFAULT_BASELINE_PATH as FAMILY3_DEFAULT_BASELINE_PATH,
    extract_identity_features,
)
from data.generators.identity_generator import load_dataset, LegitimateIdentityGenerator
from attacks.transaction_evasion.generator import (
    TransactionAttackGenerator,
    FAMILY1_GENOME_DIMENSIONS,
)
from attacks.ai_agent.generator import (
    AIAgentAttackGenerator,
    FAMILY2_GENOME_DIMENSIONS,
)


# ===========================================================================
# Structured Metrics & Model Update Records
# ===========================================================================

class EvaluationMetrics(BaseModel):
    """Evaluation metrics computed over an evaluation or training dataset."""

    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    false_positive_rate: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    sample_count: int = Field(..., ge=0)


class ModelUpdateRecord(BaseModel):
    """
    Structured record of a Blue-Team model update / retraining operation.
    """

    retrained: bool
    trigger_reason: str
    round_id: str
    round_index: int
    family: Optional[Union[AttackFamily, str]] = None
    previous_model_version: str
    new_model_version: str
    training_sample_count: int = 0
    false_negative_count: int = 0
    baseline_count: int = 0
    fresh_legitimate_count: int = 0
    before_metrics: Optional[EvaluationMetrics] = None
    after_metrics: Optional[EvaluationMetrics] = None
    holdout_metrics: Optional[EvaluationMetrics] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# Metric Computation Utilities
# ===========================================================================

def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> EvaluationMetrics:
    """Compute standard classification evaluation metrics."""
    total = len(y_true)
    if total == 0:
        return EvaluationMetrics(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            false_positive_rate=0.0,
            f1_score=1.0,
            sample_count=0,
        )

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / total
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return EvaluationMetrics(
        accuracy=round(float(accuracy), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        false_positive_rate=round(float(fpr), 4),
        f1_score=round(float(f1), 4),
        sample_count=total,
    )


# ===========================================================================
# Family Trainer Protocol & Implementations
# ===========================================================================

@runtime_checkable
class DetectorTrainer(Protocol):
    """Protocol for family-specific model retraining adapters."""

    def train(
        self,
        dataset: RetrainingDataset,
        detector: Any,
        held_out_data: Optional[Sequence[Any]] = None,
        retrain_count: int = 1,
    ) -> Tuple[bool, str, Optional[EvaluationMetrics], Optional[EvaluationMetrics], Optional[EvaluationMetrics], Dict[str, Any]]:
        """
        Execute model retraining / adaptation.

        Returns:
            (success, new_model_version, before_metrics, after_metrics, holdout_metrics, details)
        """
        ...


class Family1TransactionTrainer:
    """
    Parameter adaptation and weight update for Family 1 (Transaction Evasion).

    NOTE: Family 1 uses a heuristic behavioral risk scorer (TransactionBlueDetector).
    This trainer performs heuristic parameter adaptation (NOT supervised ML model fitting)
    by shifting feature weights towards missed evasion dimensions and calibrating the
    decision threshold.
    """

    def __init__(self, learning_rate: float = 0.15) -> None:
        self.learning_rate = learning_rate

    def evaluate_detector(
        self,
        detector: TransactionBlueDetector,
        samples: Sequence[DatasetSample],
    ) -> EvaluationMetrics:
        """Evaluate a TransactionBlueDetector over a sequence of DatasetSamples."""
        y_true: List[int] = []
        y_pred: List[int] = []

        for idx, s in enumerate(samples):
            scenario = s.data.get("scenario", s.data) if isinstance(s.data, dict) else s.data
            event = AttackEvent(
                attack_id=f"eval-f1-{idx}",
                round_id=f"eval-round-{idx}",
                attack_family=AttackFamily.ADAPTIVE_EVASION,
                attack_genome=s.features or {},
                scenario=scenario if isinstance(scenario, dict) else {"transaction": scenario},
                ground_truth=s.is_attack,
            )
            pred = detector.detect(event)
            y_true.append(s.label)
            y_pred.append(1 if pred.prediction else 0)

        return compute_binary_metrics(y_true, y_pred)

    def train(
        self,
        dataset: RetrainingDataset,
        detector: TransactionBlueDetector,
        held_out_data: Optional[Sequence[Any]] = None,
        retrain_count: int = 1,
    ) -> Tuple[bool, str, Optional[EvaluationMetrics], Optional[EvaluationMetrics], Optional[EvaluationMetrics], Dict[str, Any]]:
        """
        Adapt feature weights towards missed evasion dimensions and calibrate threshold.
        """
        f1_samples = dataset.get_by_family(AttackFamily.ADAPTIVE_EVASION) or dataset.samples
        if not f1_samples:
            return False, detector.model_version, None, None, None, {"reason": "no_samples"}

        before_metrics = self.evaluate_detector(detector, f1_samples)

        # Collect missed attack feature dimensions
        fn_samples = [s for s in f1_samples if s.is_attack]
        if not fn_samples:
            return False, detector.model_version, before_metrics, before_metrics, None, {"reason": "no_false_negatives"}

        # Calculate average signal across missed attacks
        signal_sums: Dict[str, float] = {k: 0.0 for k in FAMILY1_FEATURE_NAMES}
        for s in fn_samples:
            scenario = s.data.get("scenario", s.data) if isinstance(s.data, dict) else s.data
            extracted = extract_transaction_features(scenario)
            for k in FAMILY1_FEATURE_NAMES:
                signal_sums[k] += extracted.get(k, 0.0)

        fn_count = len(fn_samples)
        avg_signals = {k: signal_sums[k] / fn_count for k in FAMILY1_FEATURE_NAMES}

        # Adapt weights: increase weights for features with higher missed signal
        current_weights = detector.weights
        updated_weights: Dict[str, float] = {}
        for k in FAMILY1_FEATURE_NAMES:
            w_curr = current_weights.get(k, 1.0 / len(FAMILY1_FEATURE_NAMES))
            sig = avg_signals.get(k, 0.0)
            updated_weights[k] = w_curr + self.learning_rate * sig

        # Normalize weights to sum to 1.0
        total_w = sum(updated_weights.values())
        if total_w > 0:
            updated_weights = {k: round(v / total_w, 4) for k, v in updated_weights.items()}
            # Ensure exact 1.0 sum
            diff = 1.0 - sum(updated_weights.values())
            first_key = FAMILY1_FEATURE_NAMES[0]
            updated_weights[first_key] = round(updated_weights[first_key] + diff, 4)

        # Calibrate threshold slightly if needed to improve recall on subtle evasions
        new_threshold = round(max(0.35, min(detector.threshold, 0.45)), 4)
        new_version = f"heuristic-family1-retrained-v{retrain_count}"

        # Apply update
        detector._weights = dict(updated_weights)
        detector.threshold = new_threshold
        detector.model_version = new_version

        after_metrics = self.evaluate_detector(detector, f1_samples)

        # Evaluate on held-out data if provided
        holdout_metrics = None
        if held_out_data:
            holdout_samples = [
                DatasetSample(
                    sample_id=f"heldout-{i}",
                    label=0,
                    provenance=ProvenanceType.BASELINE_LEGITIMATE,
                    family=AttackFamily.ADAPTIVE_EVASION,
                    data=item if isinstance(item, dict) else {"transaction": item},
                )
                for i, item in enumerate(held_out_data)
            ]
            holdout_metrics = self.evaluate_detector(detector, holdout_samples)

        details = {
            "adapted_weights": updated_weights,
            "new_threshold": new_threshold,
            "false_negatives_used": fn_count,
        }
        return True, new_version, before_metrics, after_metrics, holdout_metrics, details


class Family2AIAgentTrainer:
    """
    Parameter adaptation and sensitivity tuning for Family 2 (AI Agent Behavior).

    NOTE: Family 2 uses a mandate-envelope evaluator (AIAgentBlueDetector).
    This trainer performs heuristic parameter adaptation (NOT supervised ML model fitting)
    by tuning mandate deviation sensitivity weights and decision thresholds.
    """

    def __init__(self, learning_rate: float = 0.15) -> None:
        self.learning_rate = learning_rate

    def evaluate_detector(
        self,
        detector: AIAgentBlueDetector,
        samples: Sequence[DatasetSample],
    ) -> EvaluationMetrics:
        """Evaluate an AIAgentBlueDetector over a sequence of DatasetSamples."""
        y_true: List[int] = []
        y_pred: List[int] = []

        for idx, s in enumerate(samples):
            scenario = s.data.get("scenario", s.data) if isinstance(s.data, dict) else s.data
            event = AttackEvent(
                attack_id=f"eval-f2-{idx}",
                round_id=f"eval-round-{idx}",
                attack_family=AttackFamily.AGENT_BEHAVIOR,
                attack_genome=s.features or {},
                scenario=scenario if isinstance(scenario, dict) else {"event": scenario},
                ground_truth=s.is_attack,
            )
            pred = detector.detect(event)
            y_true.append(s.label)
            y_pred.append(1 if pred.prediction else 0)

        return compute_binary_metrics(y_true, y_pred)

    def train(
        self,
        dataset: RetrainingDataset,
        detector: AIAgentBlueDetector,
        held_out_data: Optional[Sequence[Any]] = None,
        retrain_count: int = 1,
    ) -> Tuple[bool, str, Optional[EvaluationMetrics], Optional[EvaluationMetrics], Optional[EvaluationMetrics], Dict[str, Any]]:
        """
        Adapt feature weights towards missed agent evasion dimensions.
        """
        f2_samples = dataset.get_by_family(AttackFamily.AGENT_BEHAVIOR) or dataset.samples
        if not f2_samples:
            return False, detector.model_version, None, None, None, {"reason": "no_samples"}

        before_metrics = self.evaluate_detector(detector, f2_samples)

        fn_samples = [s for s in f2_samples if s.is_attack]
        if not fn_samples:
            return False, detector.model_version, before_metrics, before_metrics, None, {"reason": "no_false_negatives"}

        # Calculate average signals across missed agent attacks
        signal_sums: Dict[str, float] = {k: 0.0 for k in FAMILY2_GENOME_DIMENSIONS}
        for s in fn_samples:
            scenario = s.data.get("scenario", s.data) if isinstance(s.data, dict) else s.data
            dummy_event = AttackEvent(
                attack_id=s.sample_id,
                round_id="dummy",
                attack_family=AttackFamily.AGENT_BEHAVIOR,
                attack_genome=s.features or {},
                scenario=scenario,
                ground_truth=True,
            )
            extracted = extract_mandate_features(dummy_event, mandate=detector.mandate)
            for k in FAMILY2_GENOME_DIMENSIONS:
                signal_sums[k] += extracted.get(k, 0.0)

        fn_count = len(fn_samples)
        avg_signals = {k: signal_sums[k] / fn_count for k in FAMILY2_GENOME_DIMENSIONS}

        current_weights = detector.weights
        updated_weights: Dict[str, float] = {}
        for k in FAMILY2_GENOME_DIMENSIONS:
            w_curr = current_weights.get(k, 1.0 / len(FAMILY2_GENOME_DIMENSIONS))
            sig = avg_signals.get(k, 0.0)
            updated_weights[k] = w_curr + self.learning_rate * sig

        total_w = sum(updated_weights.values())
        if total_w > 0:
            updated_weights = {k: round(v / total_w, 4) for k, v in updated_weights.items()}
            diff = 1.0 - sum(updated_weights.values())
            first_key = FAMILY2_GENOME_DIMENSIONS[0]
            updated_weights[first_key] = round(updated_weights[first_key] + diff, 4)

        new_threshold = round(max(0.35, min(detector.threshold, 0.45)), 4)
        new_version = f"heuristic-family2-retrained-v{retrain_count}"

        detector.weights = dict(updated_weights)
        detector.threshold = new_threshold
        detector.model_version = new_version

        after_metrics = self.evaluate_detector(detector, f2_samples)

        holdout_metrics = None
        if held_out_data:
            holdout_samples = [
                DatasetSample(
                    sample_id=f"heldout-{i}",
                    label=0,
                    provenance=ProvenanceType.BASELINE_LEGITIMATE,
                    family=AttackFamily.AGENT_BEHAVIOR,
                    data=item if isinstance(item, dict) else {"event": item},
                )
                for i, item in enumerate(held_out_data)
            ]
            holdout_metrics = self.evaluate_detector(detector, holdout_samples)

        details = {
            "adapted_weights": updated_weights,
            "new_threshold": new_threshold,
            "false_negatives_used": fn_count,
        }
        return True, new_version, before_metrics, after_metrics, holdout_metrics, details


class Family3SyntheticIdentityTrainer:
    """
    Supervised ML Model Retraining for Family 3 (Synthetic Identity).

    NOTE: Family 3 uses a supervised XGBoost classifier (SyntheticIdentityBlueDetector).
    This trainer performs genuine supervised ML model refitting on the assembled dataset
    (baseline legitimate identities + missed synthetic identity fraud) and re-initializes
    the SHAP TreeExplainer.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def evaluate_detector(
        self,
        detector: SyntheticIdentityBlueDetector,
        samples: Sequence[Union[DatasetSample, SyntheticIdentity, Dict[str, Any]]],
    ) -> EvaluationMetrics:
        """Evaluate a SyntheticIdentityBlueDetector over samples."""
        y_true: List[int] = []
        y_pred: List[int] = []

        for idx, s in enumerate(samples):
            if isinstance(s, DatasetSample):
                scenario = s.data.get("scenario", s.data) if isinstance(s.data, dict) else s.data
                ground_truth = s.is_attack
                label = s.label
            else:
                scenario = s.model_dump() if hasattr(s, "model_dump") else s
                ground_truth = False
                label = 0

            event = AttackEvent(
                attack_id=f"eval-f3-{idx}",
                round_id=f"eval-round-{idx}",
                attack_family=AttackFamily.SYNTHETIC_IDENTITY,
                attack_genome={},
                scenario=scenario,
                ground_truth=ground_truth,
            )
            pred = detector.detect(event)
            y_true.append(label)
            y_pred.append(1 if pred.prediction else 0)

        return compute_binary_metrics(y_true, y_pred)

    def train(
        self,
        dataset: RetrainingDataset,
        detector: SyntheticIdentityBlueDetector,
        held_out_data: Optional[Sequence[Any]] = None,
        retrain_count: int = 1,
    ) -> Tuple[bool, str, Optional[EvaluationMetrics], Optional[EvaluationMetrics], Optional[EvaluationMetrics], Dict[str, Any]]:
        """
        Retrain XGBoost classifier on assembled dataset and evaluate on clean held-out identities.
        """
        import xgboost as xgb
        import shap

        f3_samples = dataset.get_by_family(AttackFamily.SYNTHETIC_IDENTITY) or dataset.samples
        if not f3_samples:
            return False, detector.model_version, None, None, None, {"reason": "no_samples"}

        before_metrics = self.evaluate_detector(detector, f3_samples)

        # Build feature matrix X and label vector y
        feature_rows: List[List[float]] = []
        labels: List[int] = []

        for s in f3_samples:
            scenario = s.data.get("scenario", s.data) if isinstance(s.data, dict) else s.data
            feat_dict = extract_identity_features(scenario)
            feature_rows.append([feat_dict[k] for k in FAMILY3_FEATURE_NAMES])
            labels.append(s.label)

        if not feature_rows or len(set(labels)) < 1:
            return False, detector.model_version, before_metrics, before_metrics, None, {"reason": "insufficient_classes"}

        X = np.array(feature_rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Train XGBoost with deterministic hyperparameters
        model = xgb.XGBClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        model.fit(X, y)

        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None

        new_version = f"family3-xgb-retrained-v{retrain_count}"

        # Update detector state
        detector._model = model
        detector._explainer = explainer
        detector.model_version = new_version

        after_metrics = self.evaluate_detector(detector, f3_samples)

        # Evaluate strictly on held-out identities if provided
        holdout_metrics = None
        if held_out_data:
            holdout_metrics = self.evaluate_detector(detector, held_out_data)

        fn_count = sum(1 for s in f3_samples if s.is_attack)
        details = {
            "training_samples": len(feature_rows),
            "false_negatives_used": fn_count,
            "trees": 40,
        }
        return True, new_version, before_metrics, after_metrics, holdout_metrics, details


# ===========================================================================
# Model Retraining Controller
# ===========================================================================

class RetrainingController:
    """
    Coordinates Blue-Team failure consumption, retraining triggers, dataset assembly,
    model updates, and held-out evaluation across simulation rounds.
    """

    def __init__(
        self,
        failure_memory: Optional[FailureMemory] = None,
        detectors: Optional[Dict[Union[AttackFamily, str], Any]] = None,
        trainers: Optional[Dict[Union[AttackFamily, str], DetectorTrainer]] = None,
        retrain_interval: int = 2,
        baseline_data: Optional[Dict[Union[AttackFamily, str], Sequence[Any]]] = None,
        fresh_legitimate_data: Optional[Dict[Union[AttackFamily, str], Sequence[Any]]] = None,
        held_out_data: Optional[Dict[Union[AttackFamily, str], Sequence[Any]]] = None,
        auto_load_canonical_baseline: bool = True,
    ) -> None:
        """
        Initialize the Retraining Controller.

        Args:
            failure_memory: FailureMemory store for collecting missed attacks.
            detectors: Mapping of attack family to active detector instances.
            trainers: Mapping of attack family to family-specific trainer instances.
            retrain_interval: Number of rounds between retraining triggers (default: 2).
            baseline_data: Canonical baseline training data per family.
            fresh_legitimate_data: Optional fresh legitimate training data per family.
            held_out_data: Clean held-out evaluation datasets per family (evaluation only!).
            auto_load_canonical_baseline: If True, automatically loads baseline/holdout data
                                          for Family 3 from canonical disk paths if not provided.
        """
        self._failure_memory = failure_memory or FailureMemory()
        self._detectors: Dict[str, Any] = {}
        self._trainers: Dict[str, DetectorTrainer] = {}
        self._baseline_data: Dict[str, Sequence[Any]] = {}
        self._fresh_legitimate_data: Dict[str, Sequence[Any]] = {}
        self._held_out_data: Dict[str, Sequence[Any]] = {}
        self._retrain_interval = max(1, retrain_interval)
        self._history: List[ModelUpdateRecord] = []
        self._retrain_counts: Dict[str, int] = {}

        # Register default trainers
        self._register_default_trainers()

        # Register custom detectors / trainers
        if detectors:
            for fam, det in detectors.items():
                self.register_detector(fam, det)

        if trainers:
            for fam, tr in trainers.items():
                self.register_trainer(fam, tr)

        if baseline_data:
            for fam, data in baseline_data.items():
                self.set_baseline_data(fam, data)

        if fresh_legitimate_data:
            for fam, data in fresh_legitimate_data.items():
                self.set_fresh_legitimate_data(fam, data)

        if held_out_data:
            for fam, data in held_out_data.items():
                self.set_held_out_data(fam, data)

        # Auto-load canonical Family 3 datasets from disk if requested
        if auto_load_canonical_baseline:
            self._auto_load_family3_defaults()

    @property
    def failure_memory(self) -> FailureMemory:
        """Return the underlying FailureMemory store."""
        return self._failure_memory

    @property
    def retrain_interval(self) -> int:
        """Return the configured retraining interval in rounds."""
        return self._retrain_interval

    @retrain_interval.setter
    def retrain_interval(self, val: int) -> None:
        self._retrain_interval = max(1, val)

    def _family_key(self, family: Union[AttackFamily, str]) -> str:
        """Normalize family identifier to string representation."""
        return family.value if isinstance(family, AttackFamily) else str(family)

    def register_detector(self, family: Union[AttackFamily, str], detector: Any) -> None:
        """Register an active detector instance for a specific family."""
        self._detectors[self._family_key(family)] = detector

    def register_trainer(self, family: Union[AttackFamily, str], trainer: DetectorTrainer) -> None:
        """Register a trainer implementation for a specific family."""
        self._trainers[self._family_key(family)] = trainer

    def set_baseline_data(self, family: Union[AttackFamily, str], data: Sequence[Any]) -> None:
        """Set canonical baseline legitimate training data for a family."""
        # Safety guard: ensure baseline path does not point to held_out
        for item in data:
            if isinstance(item, (str, Path)) and "held_out" in str(item):
                raise HoldoutDataLeakageError(
                    f"Cannot configure held-out evaluation path as training baseline for {family}"
                )
        self._baseline_data[self._family_key(family)] = list(data)

    def set_fresh_legitimate_data(self, family: Union[AttackFamily, str], data: Sequence[Any]) -> None:
        """Set optional fresh legitimate training data for a family."""
        self._fresh_legitimate_data[self._family_key(family)] = list(data)

    def set_held_out_data(self, family: Union[AttackFamily, str], data: Sequence[Any]) -> None:
        """Set clean held-out evaluation data for a family (strictly for evaluation!)."""
        self._held_out_data[self._family_key(family)] = list(data)

    def _register_default_trainers(self) -> None:
        """Register default trainer adapters for all three families."""
        self.register_trainer(AttackFamily.ADAPTIVE_EVASION, Family1TransactionTrainer())
        self.register_trainer(AttackFamily.AGENT_BEHAVIOR, Family2AIAgentTrainer())
        self.register_trainer(AttackFamily.SYNTHETIC_IDENTITY, Family3SyntheticIdentityTrainer())

    def _auto_load_family3_defaults(self) -> None:
        """Auto-load canonical baseline and held-out datasets for Family 3 if on disk."""
        fam3_key = self._family_key(AttackFamily.SYNTHETIC_IDENTITY)

        if fam3_key not in self._baseline_data:
            baseline_path = Path("data/legitimate/baseline_identities.json")
            if baseline_path.exists():
                try:
                    self._baseline_data[fam3_key] = load_dataset(baseline_path)
                except Exception:
                    pass

        if fam3_key not in self._held_out_data:
            heldout_path = Path("data/held_out/heldout_identities.json")
            if heldout_path.exists():
                try:
                    self._held_out_data[fam3_key] = load_dataset(heldout_path)
                except Exception:
                    pass

    def on_round_completed(
        self,
        round_result: RoundResult,
        round_index: int,
    ) -> Optional[ModelUpdateRecord]:
        """
        Process a completed simulation round.

        1. Ingests round into FailureMemory (records if it's a false negative).
        2. Evaluates if round_index triggers scheduled retraining (round_index % interval == 0).
        3. If triggered, retrains the detector for the active family and returns ModelUpdateRecord.

        Args:
            round_result: Completed RoundResult from simulation.
            round_index: 1-indexed round number.

        Returns:
            ModelUpdateRecord if retraining was triggered, else None.
        """
        # Step 1: Accumulate failure if missed
        self._failure_memory.record_round(round_result)

        # Step 2: Check schedule
        if round_index <= 0 or round_index % self._retrain_interval != 0:
            return None

        # Step 3: Trigger retraining for active family
        family = round_result.attack_event.attack_family
        record = self.retrain_family(
            family=family,
            round_id=round_result.round_id,
            round_index=round_index,
            trigger_reason="scheduled_interval",
        )
        return record

    def retrain_family(
        self,
        family: Union[AttackFamily, str],
        round_id: str = "manual",
        round_index: int = 0,
        trigger_reason: str = "manual_trigger",
    ) -> ModelUpdateRecord:
        """
        Execute model retraining for a specific attack family.

        Implements fail-safe execution: if retraining or evaluation fails, the detector
        preserves its previous working state without corruption.
        """
        fam_key = self._family_key(family)
        detector = self._detectors.get(fam_key)
        trainer = self._trainers.get(fam_key)

        if detector is None:
            record = ModelUpdateRecord(
                retrained=False,
                trigger_reason=f"{trigger_reason}: detector_not_registered",
                round_id=round_id,
                round_index=round_index,
                family=family,
                previous_model_version="unknown",
                new_model_version="unknown",
                details={"error": f"No detector registered for {fam_key}"},
            )
            self._history.append(record)
            return record

        if trainer is None:
            record = ModelUpdateRecord(
                retrained=False,
                trigger_reason=f"{trigger_reason}: trainer_not_registered",
                round_id=round_id,
                round_index=round_index,
                family=family,
                previous_model_version=getattr(detector, "model_version", "unknown"),
                new_model_version=getattr(detector, "model_version", "unknown"),
                details={"error": f"No trainer registered for {fam_key}"},
            )
            self._history.append(record)
            return record

        prev_version = getattr(detector, "model_version", "initial")
        retrain_count = self._retrain_counts.get(fam_key, 0) + 1

        # Retrieve family-specific failures and baseline data
        family_failures = self._failure_memory.get_by_family(family)
        baseline_data = self._baseline_data.get(fam_key, [])
        fresh_data = self._fresh_legitimate_data.get(fam_key, [])
        held_out_data = self._held_out_data.get(fam_key, [])

        # If no failures accumulated, report safe no-op
        if not family_failures:
            record = ModelUpdateRecord(
                retrained=False,
                trigger_reason=f"{trigger_reason}: no_failures_accumulated",
                round_id=round_id,
                round_index=round_index,
                family=family,
                previous_model_version=prev_version,
                new_model_version=prev_version,
                training_sample_count=len(baseline_data) + len(fresh_data),
                false_negative_count=0,
                baseline_count=len(baseline_data),
                fresh_legitimate_count=len(fresh_data),
                details={"message": "No new false negatives to learn from."},
            )
            self._history.append(record)
            return record

        # Assemble holdout-safe retraining dataset
        try:
            dataset = assemble_retraining_dataset(
                baseline_data=baseline_data,
                failure_memory=family_failures,
                fresh_legitimate_data=fresh_data,
                family=family,
                held_out_data=held_out_data,
                deduplicate_failures=True,
                name=f"retrain_ds_{fam_key}_{round_index}",
            )
        except Exception as exc:
            record = ModelUpdateRecord(
                retrained=False,
                trigger_reason=f"{trigger_reason}: dataset_assembly_error ({exc})",
                round_id=round_id,
                round_index=round_index,
                family=family,
                previous_model_version=prev_version,
                new_model_version=prev_version,
                details={"error": str(exc)},
            )
            self._history.append(record)
            return record

        # Fail-Safe Execution: Preserve detector snapshot in case training throws
        snapshot = self._capture_detector_state(detector)

        try:
            success, new_version, before_m, after_m, holdout_m, details = trainer.train(
                dataset=dataset,
                detector=detector,
                held_out_data=held_out_data,
                retrain_count=retrain_count,
            )

            if success:
                self._retrain_counts[fam_key] = retrain_count
                record = ModelUpdateRecord(
                    retrained=True,
                    trigger_reason=trigger_reason,
                    round_id=round_id,
                    round_index=round_index,
                    family=family,
                    previous_model_version=prev_version,
                    new_model_version=new_version,
                    training_sample_count=dataset.total_count,
                    false_negative_count=dataset.false_negative_count,
                    baseline_count=dataset.baseline_count,
                    fresh_legitimate_count=dataset.fresh_legitimate_count,
                    before_metrics=before_m,
                    after_metrics=after_m,
                    holdout_metrics=holdout_m,
                    details=details,
                )
            else:
                self._restore_detector_state(detector, snapshot)
                record = ModelUpdateRecord(
                    retrained=False,
                    trigger_reason=f"{trigger_reason}: training_unsuccessful",
                    round_id=round_id,
                    round_index=round_index,
                    family=family,
                    previous_model_version=prev_version,
                    new_model_version=prev_version,
                    training_sample_count=dataset.total_count,
                    false_negative_count=dataset.false_negative_count,
                    baseline_count=dataset.baseline_count,
                    fresh_legitimate_count=dataset.fresh_legitimate_count,
                    before_metrics=before_m,
                    after_metrics=after_m,
                    holdout_metrics=holdout_m,
                    details=details,
                )

        except Exception as exc:
            # Revert detector state immediately on error
            self._restore_detector_state(detector, snapshot)
            record = ModelUpdateRecord(
                retrained=False,
                trigger_reason=f"{trigger_reason}: training_exception ({exc})",
                round_id=round_id,
                round_index=round_index,
                family=family,
                previous_model_version=prev_version,
                new_model_version=prev_version,
                training_sample_count=dataset.total_count,
                false_negative_count=dataset.false_negative_count,
                baseline_count=dataset.baseline_count,
                fresh_legitimate_count=dataset.fresh_legitimate_count,
                details={"exception": str(exc)},
            )

        self._history.append(record)
        return record

    def _capture_detector_state(self, detector: Any) -> Dict[str, Any]:
        """Capture an internal state snapshot of a detector for fail-safe rollback."""
        snapshot: Dict[str, Any] = {}
        if hasattr(detector, "model_version"):
            snapshot["model_version"] = detector.model_version
        if hasattr(detector, "threshold"):
            snapshot["threshold"] = detector.threshold
        if hasattr(detector, "weights"):
            snapshot["weights"] = copy.deepcopy(getattr(detector, "weights"))
        if hasattr(detector, "_weights"):
            snapshot["_weights"] = copy.deepcopy(getattr(detector, "_weights"))
        if hasattr(detector, "_model"):
            snapshot["_model"] = getattr(detector, "_model")
        if hasattr(detector, "_explainer"):
            snapshot["_explainer"] = getattr(detector, "_explainer")
        return snapshot

    def _restore_detector_state(self, detector: Any, snapshot: Dict[str, Any]) -> None:
        """Restore a detector instance from a captured fail-safe snapshot."""
        if "model_version" in snapshot:
            detector.model_version = snapshot["model_version"]
        if "threshold" in snapshot:
            detector.threshold = snapshot["threshold"]
        if "_weights" in snapshot and hasattr(detector, "_weights"):
            detector._weights = snapshot["_weights"]
        elif "weights" in snapshot and hasattr(detector, "weights"):
            try:
                detector.weights = snapshot["weights"]
            except AttributeError:
                pass
        if "_model" in snapshot and hasattr(detector, "_model"):
            detector._model = snapshot["_model"]
        if "_explainer" in snapshot and hasattr(detector, "_explainer"):
            detector._explainer = snapshot["_explainer"]

    def retrain_all_families(
        self,
        round_id: str = "manual",
        round_index: int = 0,
        trigger_reason: str = "manual_trigger",
    ) -> List[ModelUpdateRecord]:
        """Trigger retraining for all registered families that have accumulated failures."""
        records: List[ModelUpdateRecord] = []
        for fam_key in list(self._detectors.keys()):
            rec = self.retrain_family(
                family=fam_key,
                round_id=round_id,
                round_index=round_index,
                trigger_reason=trigger_reason,
            )
            records.append(rec)
        return records

    def get_history(self) -> List[ModelUpdateRecord]:
        """Return all recorded model update events in chronological order."""
        return list(self._history)

    def get_latest_update(self, family: Optional[Union[AttackFamily, str]] = None) -> Optional[ModelUpdateRecord]:
        """Return the most recent ModelUpdateRecord, optionally filtered by family."""
        if not self._history:
            return None
        if family is None:
            return self._history[-1]
        target = self._family_key(family)
        for rec in reversed(self._history):
            if rec.family and self._family_key(rec.family) == target:
                return rec
        return None

    def clear_history(self) -> None:
        """Clear the recorded model update history."""
        self._history.clear()
