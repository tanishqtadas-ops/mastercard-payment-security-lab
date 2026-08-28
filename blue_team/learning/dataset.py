"""
blue_team/learning/dataset.py — Blue-Team Retraining Dataset Assembly Layer.

Provides a family-agnostic, deterministic, and safe dataset assembly mechanism for
Blue-Team model retraining. Assembles training datasets by combining:
1. Canonical legitimate baseline data (label=0)
2. Accumulated false-negative failure records from FailureMemory (label=1)
3. Optional fresh legitimate data (label=0)

Strictly enforces evaluation holdout separation to prevent data leakage.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from pydantic import BaseModel, Field

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.round import RoundResult
from blue_team.learning.failure_memory import FailureRecord, FailureMemory


class ProvenanceType(str, Enum):
    """Origin and provenance category for a dataset sample."""

    BASELINE_LEGITIMATE = "baseline_legitimate"
    FALSE_NEGATIVE = "false_negative"
    FRESH_LEGITIMATE = "fresh_legitimate"


class DatasetAssemblyError(Exception):
    """Base exception for dataset assembly errors."""

    pass


class HoldoutDataLeakageError(DatasetAssemblyError):
    """Raised when held-out evaluation data is detected in a training dataset."""

    pass


def _extract_sample_id(item: Any, fallback_prefix: str = "sample", index: int = 0) -> str:
    """Extract or construct a deterministic unique sample identifier."""
    if hasattr(item, "sample_id") and getattr(item, "sample_id"):
        return str(getattr(item, "sample_id"))
    if hasattr(item, "identity_id") and getattr(item, "identity_id"):
        return str(getattr(item, "identity_id"))
    if hasattr(item, "transaction_id") and getattr(item, "transaction_id"):
        return str(getattr(item, "transaction_id"))
    if hasattr(item, "event_id") and getattr(item, "event_id"):
        return str(getattr(item, "event_id"))
    if hasattr(item, "attack_id") and getattr(item, "attack_id"):
        return str(getattr(item, "attack_id"))
    if isinstance(item, dict):
        for key in ("sample_id", "identity_id", "transaction_id", "event_id", "attack_id", "id"):
            if key in item and item[key]:
                return str(item[key])
    return f"{fallback_prefix}_{index:05d}"


def _extract_family(item: Any, default_family: Optional[Union[AttackFamily, str]] = None) -> Optional[Union[AttackFamily, str]]:
    """Extract attack family tag from an item if available."""
    if hasattr(item, "attack_family") and getattr(item, "attack_family") is not None:
        return getattr(item, "attack_family")
    if hasattr(item, "family") and getattr(item, "family") is not None:
        return getattr(item, "family")
    if isinstance(item, dict):
        if "attack_family" in item and item["attack_family"] is not None:
            return item["attack_family"]
        if "family" in item and item["family"] is not None:
            return item["family"]
    return default_family


def _serialize_payload(item: Any) -> Dict[str, Any]:
    """Safely serialize an input item into a dictionary payload."""
    if item is None:
        return {}
    if hasattr(item, "model_dump") and callable(item.model_dump):
        return item.model_dump()
    if isinstance(item, dict):
        return copy.deepcopy(item)
    return {"raw_data": str(item)}


class DatasetSample(BaseModel):
    """
    A single training sample in an assembled retraining dataset.

    Attributes:
        sample_id: Unique identifier for this sample.
        label: Binary classification target (0 = legitimate / benign, 1 = attack / fraud).
        provenance: ProvenanceType indicating origin (baseline_legitimate, false_negative, fresh_legitimate).
        family: Optional attack family tag.
        data: Serialized representation / payload of the sample.
        features: Optional extracted feature values or genome dimensions.
        metadata: Optional metadata (e.g. source round, seed, timestamp).
    """

    sample_id: str = Field(..., min_length=1)
    label: int = Field(..., ge=0, le=1)
    provenance: ProvenanceType
    family: Optional[Union[AttackFamily, str]] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    features: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_legitimate(self) -> bool:
        """Return True if sample is legitimate (label == 0)."""
        return self.label == 0

    @property
    def is_attack(self) -> bool:
        """Return True if sample is an attack / false negative (label == 1)."""
        return self.label == 1


class RetrainingDataset(BaseModel):
    """
    Assembled retraining dataset containing legitimate and failure samples.
    """

    samples: List[DatasetSample] = Field(default_factory=list)
    name: str = "retraining_dataset"
    description: Optional[str] = None
    created_at: Optional[str] = None

    @property
    def total_count(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    @property
    def legitimate_count(self) -> int:
        """Return count of legitimate samples (label == 0)."""
        return sum(1 for s in self.samples if s.is_legitimate)

    @property
    def attack_count(self) -> int:
        """Return count of attack / false-negative samples (label == 1)."""
        return sum(1 for s in self.samples if s.is_attack)

    @property
    def baseline_count(self) -> int:
        """Return count of baseline legitimate samples."""
        return sum(1 for s in self.samples if s.provenance == ProvenanceType.BASELINE_LEGITIMATE)

    @property
    def false_negative_count(self) -> int:
        """Return count of false negative failure samples."""
        return sum(1 for s in self.samples if s.provenance == ProvenanceType.FALSE_NEGATIVE)

    @property
    def fresh_legitimate_count(self) -> int:
        """Return count of fresh legitimate samples."""
        return sum(1 for s in self.samples if s.provenance == ProvenanceType.FRESH_LEGITIMATE)

    @property
    def family_counts(self) -> Dict[str, int]:
        """Return distribution of sample counts across attack families."""
        counts: Dict[str, int] = {}
        for s in self.samples:
            fam_key = (
                s.family.value
                if isinstance(s.family, AttackFamily)
                else str(s.family or "unspecified")
            )
            counts[fam_key] = counts.get(fam_key, 0) + 1
        return counts

    def get_by_provenance(self, provenance: Union[ProvenanceType, str]) -> List[DatasetSample]:
        """Filter samples by provenance type."""
        prov = ProvenanceType(provenance) if isinstance(provenance, str) else provenance
        return [s for s in self.samples if s.provenance == prov]

    def get_by_family(self, family: Union[AttackFamily, str]) -> List[DatasetSample]:
        """Filter samples by attack family."""
        target = family.value if isinstance(family, AttackFamily) else str(family)
        return [
            s
            for s in self.samples
            if (
                s.family.value
                if isinstance(s.family, AttackFamily)
                else str(s.family)
            )
            == target
        ]

    def get_legitimate_samples(self) -> List[DatasetSample]:
        """Return all legitimate training samples (label == 0)."""
        return [s for s in self.samples if s.is_legitimate]

    def get_attack_samples(self) -> List[DatasetSample]:
        """Return all attack / failure training samples (label == 1)."""
        return [s for s in self.samples if s.is_attack]

    def get_sample_ids(self) -> List[str]:
        """Return list of all sample identifiers in deterministic order."""
        return [s.sample_id for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize dataset samples to a list of dictionaries."""
        return [s.model_dump() for s in self.samples]

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize dataset samples to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(
        cls,
        data: List[Dict[str, Any]],
        name: str = "retraining_dataset",
    ) -> RetrainingDataset:
        """Restore RetrainingDataset from a list of sample dicts."""
        samples = [DatasetSample.model_validate(item) for item in data]
        return cls(samples=samples, name=name)

    @classmethod
    def from_json(cls, json_str: str) -> RetrainingDataset:
        """Restore RetrainingDataset from a JSON string."""
        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array of sample dicts")
        return cls.from_dict(data)


def validate_no_holdout_leakage(
    dataset_or_samples: Union[RetrainingDataset, Sequence[DatasetSample], Sequence[Any]],
    held_out_data_or_ids: Union[Sequence[Any], Set[str]],
) -> None:
    """
    Strictly verify that zero samples from the held-out evaluation set exist in the training data.

    Raises:
        HoldoutDataLeakageError: If any sample ID or matching identifier collides.
    """
    # Extract held-out sample IDs
    held_out_ids: Set[str] = set()
    if isinstance(held_out_data_or_ids, set):
        held_out_ids = set(held_out_data_or_ids)
    else:
        for idx, item in enumerate(held_out_data_or_ids):
            held_out_ids.add(_extract_sample_id(item, fallback_prefix="heldout", index=idx))

    if not held_out_ids:
        return

    # Extract dataset sample IDs
    sample_ids: Set[str] = set()
    if isinstance(dataset_or_samples, RetrainingDataset):
        sample_ids = set(dataset_or_samples.get_sample_ids())
    else:
        for idx, item in enumerate(dataset_or_samples):
            sample_ids.add(_extract_sample_id(item, fallback_prefix="train", index=idx))

    overlap = sample_ids.intersection(held_out_ids)
    if overlap:
        raise HoldoutDataLeakageError(
            f"Holdout data leakage detected! {len(overlap)} samples from the evaluation set "
            f"were found in the training dataset: {sorted(list(overlap))[:10]}"
        )


class RetrainingDatasetAssembler:
    """
    Family-agnostic assembler that compiles retraining datasets from:
    1. Baseline legitimate data
    2. False-negative failure memory records
    3. Optional fresh legitimate data
    """

    def __init__(
        self,
        known_holdout_ids: Optional[Set[str]] = None,
    ) -> None:
        self._known_holdout_ids: Set[str] = set(known_holdout_ids or [])

    def assemble(
        self,
        baseline_data: Optional[Sequence[Any]] = None,
        failure_memory: Optional[Union[FailureMemory, Sequence[FailureRecord]]] = None,
        fresh_legitimate_data: Optional[Sequence[Any]] = None,
        family: Optional[Union[AttackFamily, str]] = None,
        held_out_data: Optional[Sequence[Any]] = None,
        deduplicate_failures: bool = False,
        name: str = "retraining_dataset",
        description: Optional[str] = None,
    ) -> RetrainingDataset:
        """
        Assemble a complete RetrainingDataset in deterministic order.

        Args:
            baseline_data: Sequence of original legitimate baseline items (label=0).
            failure_memory: FailureMemory instance or sequence of FailureRecord objects (label=1).
            fresh_legitimate_data: Optional sequence of fresh legitimate items (label=0).
            family: Optional family filter.
            held_out_data: Optional held-out dataset to explicitly assert absence against.
            deduplicate_failures: If True, deduplicates failure records by attack_id.
            name: Dataset identifier name.
            description: Optional human-readable description.

        Returns:
            RetrainingDataset containing all assembled and validated samples.
        """
        samples: List[DatasetSample] = []
        seen_sample_ids: Set[str] = set()

        # 1. Process Baseline Legitimate Data (Label 0)
        if baseline_data:
            for idx, item in enumerate(baseline_data):
                # Guard against passing a path string pointing to held_out
                if isinstance(item, (str, Path)) and "held_out" in str(item):
                    raise HoldoutDataLeakageError(
                        f"Cannot use held-out evaluation path '{item}' as baseline training data."
                    )

                s_id = _extract_sample_id(item, fallback_prefix="baseline_legit", index=idx)
                s_fam = _extract_family(item, default_family=family)
                payload = _serialize_payload(item)
                metadata = {"source_index": idx, "dataset_role": "baseline"}

                sample = DatasetSample(
                    sample_id=s_id,
                    label=0,
                    provenance=ProvenanceType.BASELINE_LEGITIMATE,
                    family=s_fam,
                    data=payload,
                    metadata=metadata,
                )
                samples.append(sample)
                seen_sample_ids.add(s_id)

        # 2. Process False Negative Failure Records (Label 1)
        if failure_memory:
            failure_records: Sequence[FailureRecord]
            if isinstance(failure_memory, FailureMemory):
                failure_records = failure_memory.get_failures()
            else:
                failure_records = failure_memory

            seen_failure_keys: Set[str] = set()

            for idx, record in enumerate(failure_records):
                if not isinstance(record, FailureRecord):
                    # Validate that item is a valid FailureRecord
                    raise DatasetAssemblyError(
                        f"Expected FailureRecord instance in failure_memory, got {type(record).__name__}"
                    )

                # Validate false negative status
                if not (record.ground_truth and not record.prediction and record.false_negative):
                    # Reject non-false-negatives
                    raise DatasetAssemblyError(
                        f"Record {record.attack_id} is not a valid false negative "
                        f"(ground_truth={record.ground_truth}, prediction={record.prediction}, "
                        f"false_negative={record.false_negative})"
                    )

                # Optional family filtering
                if family is not None:
                    target_fam = family.value if isinstance(family, AttackFamily) else str(family)
                    rec_fam = (
                        record.attack_family.value
                        if isinstance(record.attack_family, AttackFamily)
                        else str(record.attack_family)
                    )
                    if rec_fam != target_fam:
                        continue

                # Optional deduplication by attack_id / unique key
                dedup_key = record.attack_id
                if deduplicate_failures and dedup_key in seen_failure_keys:
                    continue
                seen_failure_keys.add(dedup_key)

                sample_id = record.attack_id
                sample = DatasetSample(
                    sample_id=sample_id,
                    label=1,
                    provenance=ProvenanceType.FALSE_NEGATIVE,
                    family=record.attack_family,
                    data={
                        "scenario": copy.deepcopy(record.scenario),
                        "attack_genome": copy.deepcopy(record.attack_genome),
                        "risk_score": record.risk_score,
                        "round_id": record.round_id,
                        "attack_id": record.attack_id,
                    },
                    features=copy.deepcopy(record.attack_genome),
                    metadata={
                        "round_id": record.round_id,
                        "model_version": record.model_version,
                        "feature_contributions": copy.deepcopy(record.feature_contributions),
                        "original_metadata": copy.deepcopy(record.metadata),
                    },
                )
                samples.append(sample)
                seen_sample_ids.add(sample_id)

        # 3. Process Fresh Legitimate Data (Label 0, Optional)
        if fresh_legitimate_data:
            for idx, item in enumerate(fresh_legitimate_data):
                if isinstance(item, (str, Path)) and "held_out" in str(item):
                    raise HoldoutDataLeakageError(
                        f"Cannot use held-out evaluation path '{item}' as fresh training data."
                    )

                s_id = _extract_sample_id(item, fallback_prefix="fresh_legit", index=idx)
                s_fam = _extract_family(item, default_family=family)
                payload = _serialize_payload(item)
                metadata = {"fresh_sample_index": idx, "dataset_role": "fresh_legitimate"}

                sample = DatasetSample(
                    sample_id=s_id,
                    label=0,
                    provenance=ProvenanceType.FRESH_LEGITIMATE,
                    family=s_fam,
                    data=payload,
                    metadata=metadata,
                )
                samples.append(sample)
                seen_sample_ids.add(s_id)

        dataset = RetrainingDataset(
            samples=samples,
            name=name,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # 4. Strict Holdout Leakage Verification
        holdout_check_set = set(self._known_holdout_ids)
        if held_out_data:
            for idx, h in enumerate(held_out_data):
                holdout_check_set.add(_extract_sample_id(h, fallback_prefix="heldout", index=idx))

        if holdout_check_set:
            validate_no_holdout_leakage(dataset, holdout_check_set)

        return dataset


def assemble_retraining_dataset(
    baseline_data: Optional[Sequence[Any]] = None,
    failure_memory: Optional[Union[FailureMemory, Sequence[FailureRecord]]] = None,
    fresh_legitimate_data: Optional[Sequence[Any]] = None,
    family: Optional[Union[AttackFamily, str]] = None,
    held_out_data: Optional[Sequence[Any]] = None,
    deduplicate_failures: bool = False,
    name: str = "retraining_dataset",
    description: Optional[str] = None,
) -> RetrainingDataset:
    """
    Convenience function to assemble a retraining dataset.

    Args:
        baseline_data: Sequence of original legitimate baseline items (label=0).
        failure_memory: FailureMemory instance or sequence of FailureRecord objects (label=1).
        fresh_legitimate_data: Optional sequence of fresh legitimate items (label=0).
        family: Optional family filter.
        held_out_data: Optional held-out dataset to explicitly assert absence against.
        deduplicate_failures: If True, deduplicates failure records by attack_id.
        name: Dataset name.
        description: Optional dataset description.

    Returns:
        RetrainingDataset instance.
    """
    assembler = RetrainingDatasetAssembler()
    return assembler.assemble(
        baseline_data=baseline_data,
        failure_memory=failure_memory,
        fresh_legitimate_data=fresh_legitimate_data,
        family=family,
        held_out_data=held_out_data,
        deduplicate_failures=deduplicate_failures,
        name=name,
        description=description,
    )
