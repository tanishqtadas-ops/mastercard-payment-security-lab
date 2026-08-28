"""
evaluation — Unified Cross-Family Evaluation Suite.

Provides family-agnostic benchmarking, metric consolidation, holdout isolation evaluation,
and learning progression analysis for the Mastercard Payment Security Lab.
"""

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
)
from .evaluator import UnifiedEvaluator

__all__ = [
    "ClassificationMetrics",
    "ConfusionMatrix",
    "RiskMetrics",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_risk_metrics",
    "BeforeAfterComparison",
    "ConsolidatedMetrics",
    "FamilyEvaluationResult",
    "HoldoutEvaluationResult",
    "RecoverySummary",
    "UnifiedEvaluationReport",
    "FamilyEvaluationAdapter",
    "Family1EvaluationAdapter",
    "Family2EvaluationAdapter",
    "Family3EvaluationAdapter",
    "UnifiedEvaluator",
]
