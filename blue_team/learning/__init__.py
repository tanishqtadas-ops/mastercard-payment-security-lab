"""
blue_team/learning — Blue-Team failure memory, dataset assembly, and learning foundation.
"""

from .failure_memory import (
    FailureRecord,
    FailureMemory,
    is_false_negative,
)
from .dataset import (
    ProvenanceType,
    DatasetAssemblyError,
    HoldoutDataLeakageError,
    DatasetSample,
    RetrainingDataset,
    RetrainingDatasetAssembler,
    assemble_retraining_dataset,
    validate_no_holdout_leakage,
)

__all__ = [
    "FailureRecord",
    "FailureMemory",
    "is_false_negative",
    "ProvenanceType",
    "DatasetAssemblyError",
    "HoldoutDataLeakageError",
    "DatasetSample",
    "RetrainingDataset",
    "RetrainingDatasetAssembler",
    "assemble_retraining_dataset",
    "validate_no_holdout_leakage",
]
