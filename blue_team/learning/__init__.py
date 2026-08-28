"""
blue_team/learning — Blue-Team failure memory, dataset assembly, and retraining controller.
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
from .retraining import (
    EvaluationMetrics,
    ModelUpdateRecord,
    DetectorTrainer,
    Family1TransactionTrainer,
    Family2AIAgentTrainer,
    Family3SyntheticIdentityTrainer,
    RetrainingController,
    compute_binary_metrics,
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
    "EvaluationMetrics",
    "ModelUpdateRecord",
    "DetectorTrainer",
    "Family1TransactionTrainer",
    "Family2AIAgentTrainer",
    "Family3SyntheticIdentityTrainer",
    "RetrainingController",
    "compute_binary_metrics",
]
