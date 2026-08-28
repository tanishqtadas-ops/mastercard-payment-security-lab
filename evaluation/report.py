"""
evaluation/report.py — Structured Data Models for Cross-Family Evaluation Reports.

Defines Pydantic models for single-family outcomes, consolidated multi-family benchmarks,
clean held-out generalization results, and before/after learning progression summaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from schemas.common import AttackFamily
from .metrics import ClassificationMetrics, ConfusionMatrix, RiskMetrics


class FamilyEvaluationResult(BaseModel):
    """Evaluation summary for a specific attack family."""

    family: str
    model_version: str
    sample_count: int = Field(default=0, ge=0)
    attack_count: int = Field(default=0, ge=0)
    legitimate_count: int = Field(default=0, ge=0)
    confusion_matrix: ConfusionMatrix = Field(default_factory=ConfusionMatrix)
    metrics: ClassificationMetrics
    risk_metrics: Optional[RiskMetrics] = None
    false_negative_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)


class ConsolidatedMetrics(BaseModel):
    """
    Consolidated performance metrics aggregated across all evaluated families.
    """

    total_samples: int = Field(default=0, ge=0)
    total_attacks: int = Field(default=0, ge=0)
    total_legitimate: int = Field(default=0, ge=0)
    confusion_matrix: ConfusionMatrix = Field(default_factory=ConfusionMatrix)
    overall_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_detection_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overall_precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overall_false_positive_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overall_f1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    average_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class HoldoutEvaluationResult(BaseModel):
    """
    Evaluation metrics on strictly isolated legitimate held-out test datasets.
    """

    family: str
    sample_count: int = Field(..., ge=0)
    true_negative_count: int = Field(default=0, ge=0)
    false_positive_count: int = Field(default=0, ge=0)
    clean_pass_rate: float = Field(..., ge=0.0, le=1.0)
    false_positive_rate: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    details: Dict[str, Any] = Field(default_factory=dict)


class BeforeAfterComparison(BaseModel):
    """
    Comparative performance analysis before and after a Blue-Team learning/retraining event.
    """

    family: str
    previous_model_version: str
    new_model_version: str
    before_metrics: Optional[ClassificationMetrics] = None
    after_metrics: Optional[ClassificationMetrics] = None
    accuracy_delta: Optional[float] = None
    detection_rate_delta: Optional[float] = None
    false_positive_rate_delta: Optional[float] = None
    false_negatives_used: int = Field(default=0, ge=0)
    details: Dict[str, Any] = Field(default_factory=dict)


class RecoverySummary(BaseModel):
    """
    Summary of adversarial evasion and subsequent Blue-Team recovery events.
    """

    total_evasions: int = Field(default=0, ge=0)
    total_recoveries: int = Field(default=0, ge=0)
    recovery_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_rounds_to_recover: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class UnifiedEvaluationReport(BaseModel):
    """
    Comprehensive, judge-readable consolidated evaluation report covering
    all three attack families, held-out validation, and learning progression.
    """

    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    per_family_results: Dict[str, FamilyEvaluationResult] = Field(default_factory=dict)
    consolidated_metrics: Optional[ConsolidatedMetrics] = None
    holdout_results: Dict[str, HoldoutEvaluationResult] = Field(default_factory=dict)
    before_after_comparisons: List[BeforeAfterComparison] = Field(default_factory=list)
    recovery_summary: Optional[RecoverySummary] = None
    model_versions: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
