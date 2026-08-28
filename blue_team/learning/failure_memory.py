"""
blue_team/learning/failure_memory.py — Blue-Team False-Negative Failure Memory.

Provides a family-agnostic, deterministic, append-oriented memory store for recording
and preserving detector failures (false negatives: actual attack present, but detector
missed it).

This forms the foundational data layer for subsequent Blue-Team retraining and model
adaptation without modifying frozen schemas or pipeline contracts.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, Field

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.round import RoundResult


def _serialize_scenario_value(val: Any) -> Any:
    """Helper to safely serialize nested scenario items including Pydantic models."""
    if hasattr(val, "model_dump") and callable(val.model_dump):
        return val.model_dump()
    if isinstance(val, dict):
        return {k: _serialize_scenario_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_serialize_scenario_value(v) for v in val]
    return copy.deepcopy(val)


def _serialize_scenario(scenario: Any) -> Dict[str, Any]:
    """Ensure scenario data is safely serialized into a pure dict snapshot."""
    if scenario is None:
        return {}
    if hasattr(scenario, "model_dump") and callable(scenario.model_dump):
        return scenario.model_dump()
    if isinstance(scenario, dict):
        return {k: _serialize_scenario_value(v) for k, v in scenario.items()}
    return {"raw_scenario": str(scenario)}


def is_false_negative(
    event: AttackEvent,
    prediction: PredictionResult,
    feedback: Optional[BlueTeamFeedback] = None,
) -> bool:
    """
    Determine whether an interaction constitutes a Blue-Team false negative failure.

    A false negative occurs when an actual synthetic attack was present (ground_truth = True)
    but the detector failed to identify it as fraud (prediction = False).
    """
    if feedback is not None:
        return bool(feedback.false_negative)
    return bool(event.ground_truth) and not bool(prediction.prediction)


class FailureRecord(BaseModel):
    """
    Immutable representation of a Blue-Team failure (missed attack / false negative).

    Preserves all relevant context needed for offline or periodic retraining:
    - Attack family and identifiers
    - Attack genome at time of failure
    - Detector risk score and model version
    - Feature contributions / SHAP explanations
    - Scenario details and metadata
    """

    round_id: str = Field(..., min_length=1)
    attack_id: str = Field(..., min_length=1)
    attack_family: Union[AttackFamily, str]
    attack_genome: Dict[str, float] = Field(default_factory=dict)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    prediction: bool = False
    ground_truth: bool = True
    detected: bool = False
    false_negative: bool = True
    prediction_id: Optional[str] = None
    model_version: Optional[str] = None
    feature_contributions: Dict[str, float] = Field(default_factory=dict)
    important_features: Dict[str, float] = Field(default_factory=dict)
    scenario: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = None
    explanation_data: Optional[Dict[str, Any]] = None
    feedback_id: Optional[str] = None

    @classmethod
    def from_round_result(cls, round_result: RoundResult) -> FailureRecord:
        """Construct a FailureRecord from a completed RoundResult."""
        return cls.from_components(
            event=round_result.attack_event,
            prediction=round_result.prediction_result,
            feedback=round_result.feedback,
            outcome_metrics=round_result.outcome_metrics,
        )

    @classmethod
    def from_components(
        cls,
        event: AttackEvent,
        prediction: PredictionResult,
        feedback: Optional[BlueTeamFeedback] = None,
        outcome_metrics: Optional[Dict[str, Any]] = None,
    ) -> FailureRecord:
        """Construct a FailureRecord from individual round components."""
        scenario_data = _serialize_scenario(event.scenario)
        metadata_data = copy.deepcopy(event.metadata) if event.metadata else {}
        if outcome_metrics:
            metadata_data["outcome_metrics"] = copy.deepcopy(outcome_metrics)

        feat_contribs = dict(prediction.feature_contributions or {})
        imp_features = (
            dict(feedback.important_features)
            if feedback and feedback.important_features
            else dict(feat_contribs)
        )

        return cls(
            round_id=event.round_id,
            attack_id=event.attack_id,
            attack_family=event.attack_family,
            attack_genome=copy.deepcopy(event.attack_genome),
            risk_score=float(prediction.risk_score),
            prediction=bool(prediction.prediction),
            ground_truth=bool(event.ground_truth),
            detected=bool(feedback.detected) if feedback else False,
            false_negative=(
                bool(feedback.false_negative)
                if feedback
                else (bool(event.ground_truth) and not bool(prediction.prediction))
            ),
            prediction_id=prediction.prediction_id,
            model_version=prediction.model_version,
            feature_contributions=feat_contribs,
            important_features=imp_features,
            scenario=scenario_data,
            metadata=metadata_data,
            explanation=prediction.explanation,
            explanation_data=(
                copy.deepcopy(feedback.explanation_data)
                if feedback and feedback.explanation_data
                else None
            ),
            feedback_id=feedback.feedback_id if feedback else None,
        )


class FailureMemory:
    """
    In-memory store for accumulating Blue-Team false negatives (missed attacks).

    Key properties:
    - Family-agnostic: Operates uniformly on all three attack families.
    - Deterministic: Preserves strict chronological insertion order.
    - Safe: Defensive copying prevents external state mutations.
    - Append-oriented: Supports accumulating missed attacks across rounds.
    - Serialization-ready: Export and restore memory snapshots cleanly.
    """

    def __init__(self, records: Optional[Sequence[FailureRecord]] = None) -> None:
        self._records: List[FailureRecord] = []
        if records:
            for r in records:
                self.record_failure(r)

    def record_round(self, round_result: RoundResult) -> bool:
        """
        Evaluate a RoundResult and record it if it represents a false negative.

        Returns:
            True if recorded (was a false negative), False if ignored (not a failure).
        """
        if not is_false_negative(
            event=round_result.attack_event,
            prediction=round_result.prediction_result,
            feedback=round_result.feedback,
        ):
            return False

        record = FailureRecord.from_round_result(round_result)
        self._records.append(record)
        return True

    def record_event(
        self,
        event: AttackEvent,
        prediction: PredictionResult,
        feedback: Optional[BlueTeamFeedback] = None,
        outcome_metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate individual components and record if it represents a false negative.

        Returns:
            True if recorded, False if ignored.
        """
        if not is_false_negative(event=event, prediction=prediction, feedback=feedback):
            return False

        record = FailureRecord.from_components(
            event=event,
            prediction=prediction,
            feedback=feedback,
            outcome_metrics=outcome_metrics,
        )
        self._records.append(record)
        return True

    def record_failure(self, record: FailureRecord) -> bool:
        """
        Directly append a pre-constructed FailureRecord.

        Only accepts records representing true false negatives (ground_truth=True,
        prediction=False, false_negative=True).
        """
        if not (record.ground_truth and not record.prediction and record.false_negative):
            return False

        self._records.append(record.model_copy(deep=True))
        return True

    def record(self, item: Union[RoundResult, FailureRecord]) -> bool:
        """Polymorphic recording helper for RoundResult or FailureRecord."""
        if isinstance(item, RoundResult):
            return self.record_round(item)
        if isinstance(item, FailureRecord):
            return self.record_failure(item)
        raise TypeError(f"Unsupported item type: {type(item).__name__}")

    def ingest_many(self, results: Sequence[RoundResult]) -> int:
        """
        Ingest a sequence of RoundResult objects, recording only false negatives.

        Returns:
            Number of false negatives recorded.
        """
        added = 0
        for r in results:
            if self.record_round(r):
                added += 1
        return added

    def get_failures(self) -> List[FailureRecord]:
        """Return all recorded failure records in deterministic insertion order."""
        return list(self._records)

    def get_by_family(self, family: Union[AttackFamily, str]) -> List[FailureRecord]:
        """Filter failure records by attack family."""
        target = family.value if isinstance(family, AttackFamily) else str(family)
        return [
            r
            for r in self._records
            if (
                r.attack_family.value
                if isinstance(r.attack_family, AttackFamily)
                else str(r.attack_family)
            )
            == target
        ]

    def get_by_round_id(self, round_id: str) -> Optional[FailureRecord]:
        """Find the first failure record matching the given round_id."""
        for r in self._records:
            if r.round_id == round_id:
                return r
        return None

    def get_by_attack_id(self, attack_id: str) -> Optional[FailureRecord]:
        """Find the first failure record matching the given attack_id."""
        for r in self._records:
            if r.attack_id == attack_id:
                return r
        return None

    def get_genomes(self) -> List[Dict[str, float]]:
        """Return a list of attack genomes from all recorded failures."""
        return [copy.deepcopy(r.attack_genome) for r in self._records]

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """Return a list of scenario dictionaries from all recorded failures."""
        return [copy.deepcopy(r.scenario) for r in self._records]

    def clear(self) -> None:
        """Clear all stored failure records."""
        self._records.clear()

    @property
    def count(self) -> int:
        """Return total number of recorded failures."""
        return len(self._records)

    @property
    def is_empty(self) -> bool:
        """Return True if memory contains no records."""
        return len(self._records) == 0

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, index: int) -> FailureRecord:
        return self._records[index]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize memory records to a list of dicts."""
        return [r.model_dump() for r in self._records]

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize memory records to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> FailureMemory:
        """Restore FailureMemory from a list of dicts."""
        records = [FailureRecord.model_validate(item) for item in data]
        mem = cls()
        for r in records:
            mem.record_failure(r)
        return mem

    @classmethod
    def from_json(cls, json_str: str) -> FailureMemory:
        """Restore FailureMemory from a JSON string."""
        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array of records")
        return cls.from_dict(data)
