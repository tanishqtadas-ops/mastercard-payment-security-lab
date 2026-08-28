"""
evaluation/metrics.py — Statistical, Confusion, and Classification Metric Utilities.

Provides rigorous, mathematically sound evaluation calculations for binary classification,
risk distribution analysis, and recovery tracking in adversarial payment security scenarios.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, Field


class ConfusionMatrix(BaseModel):
    """Confusion counts for binary classification."""

    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    true_negatives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    total_samples: int = Field(default=0, ge=0)

    @property
    def total_positives(self) -> int:
        """Total actual positive (attack) cases: TP + FN."""
        return self.true_positives + self.false_negatives

    @property
    def total_negatives(self) -> int:
        """Total actual negative (legitimate) cases: TN + FP."""
        return self.true_negatives + self.false_positives

    @property
    def total_predicted_positives(self) -> int:
        """Total predicted positive cases: TP + FP."""
        return self.true_positives + self.false_positives

    @property
    def total_predicted_negatives(self) -> int:
        """Total predicted negative cases: TN + FN."""
        return self.true_negatives + self.false_negatives


class ClassificationMetrics(BaseModel):
    """
    Standard classification performance metrics.

    Where a metric is mathematically undefined (e.g. division by zero when no
    positive samples exist), Optional fields represent the unavailability
    cleanly rather than inventing values.
    """

    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    false_positive_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sample_count: int = Field(..., ge=0)


class RiskMetrics(BaseModel):
    """Statistical distribution of detector-assigned risk scores."""

    average_risk: float = Field(..., ge=0.0, le=1.0)
    min_risk: float = Field(..., ge=0.0, le=1.0)
    max_risk: float = Field(..., ge=0.0, le=1.0)
    median_risk: float = Field(..., ge=0.0, le=1.0)
    std_risk: float = Field(default=0.0, ge=0.0)


def compute_confusion_matrix(
    y_true: Sequence[Union[bool, int]],
    y_pred: Sequence[Union[bool, int]],
) -> ConfusionMatrix:
    """
    Compute confusion matrix counts from binary ground truth and predictions.

    Args:
        y_true: True labels (1/True for attack, 0/False for legitimate).
        y_pred: Predicted labels (1/True for fraud, 0/False for legitimate).

    Returns:
        ConfusionMatrix with tp, fp, tn, fn counts.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if bool(yt) and bool(yp))
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not bool(yt) and not bool(yp))
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not bool(yt) and bool(yp))
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if bool(yt) and not bool(yp))
    total = len(y_true)

    return ConfusionMatrix(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        total_samples=total,
    )


def compute_classification_metrics(
    cm: ConfusionMatrix,
) -> ClassificationMetrics:
    """
    Compute standard accuracy, precision, recall, FPR, and F1 from a ConfusionMatrix.

    Undefined metrics (zero denominator) are returned as None to prevent fabrication.
    """
    if cm.total_samples == 0:
        return ClassificationMetrics(
            accuracy=1.0,
            precision=None,
            recall=None,
            false_positive_rate=None,
            f1_score=None,
            sample_count=0,
        )

    accuracy = (cm.true_positives + cm.true_negatives) / cm.total_samples

    # Recall (Detection Rate) = TP / (TP + FN)
    recall: Optional[float] = None
    if cm.total_positives > 0:
        recall = round(cm.true_positives / cm.total_positives, 4)

    # Precision = TP / (TP + FP)
    precision: Optional[float] = None
    if cm.total_predicted_positives > 0:
        precision = round(cm.true_positives / cm.total_predicted_positives, 4)

    # False Positive Rate = FP / (FP + TN)
    fpr: Optional[float] = None
    if cm.total_negatives > 0:
        fpr = round(cm.false_positives / cm.total_negatives, 4)

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1: Optional[float] = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round((2 * precision * recall) / (precision + recall), 4)

    return ClassificationMetrics(
        accuracy=round(float(accuracy), 4),
        precision=precision,
        recall=recall,
        false_positive_rate=fpr,
        f1_score=f1,
        sample_count=cm.total_samples,
    )


def compute_risk_metrics(scores: Sequence[float]) -> Optional[RiskMetrics]:
    """
    Compute summary statistics over a sequence of risk scores in [0.0, 1.0].
    """
    if not scores:
        return None

    clean_scores = [float(min(max(s, 0.0), 1.0)) for s in scores]
    n = len(clean_scores)
    avg_r = sum(clean_scores) / n
    min_r = min(clean_scores)
    max_r = max(clean_scores)

    sorted_scores = sorted(clean_scores)
    if n % 2 == 1:
        med_r = sorted_scores[n // 2]
    else:
        med_r = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0

    if n > 1:
        variance = sum((x - avg_r) ** 2 for x in clean_scores) / (n - 1)
        std_r = math.sqrt(variance)
    else:
        std_r = 0.0

    return RiskMetrics(
        average_risk=round(avg_r, 4),
        min_risk=round(min_r, 4),
        max_risk=round(max_r, 4),
        median_risk=round(med_r, 4),
        std_risk=round(std_r, 4),
    )
