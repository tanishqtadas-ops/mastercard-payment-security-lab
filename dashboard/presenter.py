"""
dashboard/presenter.py — Presentation adapter and display data extraction for the Dashboard layer.

Converts core simulation RoundResult instances into decoupled presentation models
and formatted representations without modifying core pipeline objects.
"""

from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field

from schemas import AttackFamily, RoundResult


class RoundDisplayData(BaseModel):
    """
    Presentation-ready view of a single simulation round.

    Decouples dashboard components from core simulation schema internals while
    exposing all key dimensions required for presentation.
    """
    round_id: str
    family: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    detected: bool
    missed: bool
    status: str
    genome: Dict[str, float] = Field(default_factory=dict)
    prediction: bool
    ground_truth: bool
    model_version: str = ""
    explanation: Optional[str] = None
    feature_contributions: Optional[Dict[str, float]] = None
    outcome_metrics: Dict[str, Any] = Field(default_factory=dict)


def extract_display_data(result: RoundResult) -> RoundDisplayData:
    """
    Extract presentation-ready RoundDisplayData from a RoundResult.

    Args:
        result: A completed RoundResult from RoundController or Pipeline.

    Returns:
        RoundDisplayData instance with structured presentation fields.
    """
    event = result.attack_event
    pred = result.prediction_result
    feedback = result.feedback

    # Determine family display name
    family_str = (
        event.attack_family.value
        if isinstance(event.attack_family, AttackFamily)
        else str(event.attack_family)
    )

    # Determine status string
    if feedback.detected:
        status = "DETECTED"
    elif feedback.false_negative:
        status = "MISSED"
    elif feedback.false_positive:
        status = "FALSE_POSITIVE"
    else:
        status = "APPROVED" if not pred.prediction else "FLAGGED"

    is_missed = bool(feedback.false_negative or (event.ground_truth and not feedback.detected))

    return RoundDisplayData(
        round_id=result.round_id,
        family=family_str,
        risk_score=float(pred.risk_score),
        detected=bool(feedback.detected),
        missed=is_missed,
        status=status,
        genome=dict(event.attack_genome),
        prediction=bool(pred.prediction),
        ground_truth=bool(event.ground_truth),
        model_version=str(pred.model_version),
        explanation=pred.explanation,
        feature_contributions=dict(pred.feature_contributions) if pred.feature_contributions else None,
        outcome_metrics=dict(result.outcome_metrics),
    )


def format_round_dict(data: Union[RoundResult, RoundDisplayData]) -> Dict[str, Any]:
    """
    Format round data as a JSON-serializable dictionary for API/presentation consumers.

    Args:
        data: Either a raw RoundResult or an extracted RoundDisplayData.

    Returns:
        Dictionary containing presentation data.
    """
    if isinstance(data, RoundResult):
        display_data = extract_display_data(data)
    else:
        display_data = data

    return display_data.model_dump(mode="json")


def format_round_summary(data: Union[RoundResult, RoundDisplayData]) -> str:
    """
    Format a concise human-readable text summary of a simulation round.

    Args:
        data: Either a raw RoundResult or an extracted RoundDisplayData.

    Returns:
        Formatted multi-line text summary string.
    """
    if isinstance(data, RoundResult):
        d = extract_display_data(data)
    else:
        d = data

    genome_str = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(d.genome.items()))
    lines = [
        f"--- Round {d.round_id} Summary ---",
        f"Family:     {d.family}",
        f"Status:     {d.status} (Detected: {d.detected}, Missed: {d.missed})",
        f"Risk Score: {d.risk_score:.4f}",
        f"Prediction: {d.prediction} (Ground Truth: {d.ground_truth})",
        f"Genome:     [{genome_str}]",
    ]
    if d.explanation:
        lines.append(f"Explanation: {d.explanation}")
    return "\n".join(lines)
