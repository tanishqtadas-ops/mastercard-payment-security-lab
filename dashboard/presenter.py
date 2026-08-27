"""
dashboard/presenter.py — Presentation adapter and display data extraction for the Dashboard layer.

Converts core simulation RoundResult instances into decoupled presentation models
and formatted representations without modifying core pipeline objects.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, Field

from schemas import AttackFamily, RoundResult

if TYPE_CHECKING:
    from .feed import DashboardFeed


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


class RoundResultViewer:
    """
    Presentation and replay component for completed RoundResult records.

    Consumes RoundResult objects, RoundDisplayData, or a DashboardFeed to provide
    deterministic human-readable views and replay capabilities for dashboard consumers.
    """

    def __init__(self, feed: Optional["DashboardFeed"] = None) -> None:
        if feed is None:
            from .feed import DashboardFeed
            self._feed: DashboardFeed = DashboardFeed()
        else:
            self._feed = feed

    @property
    def feed(self) -> "DashboardFeed":
        """Return the underlying DashboardFeed instance."""
        return self._feed

    def load_round(self, result: Union[RoundResult, RoundDisplayData]) -> RoundDisplayData:
        """Ingest a single round result and return the extracted display data."""
        return self._feed.ingest(result)

    def load_rounds(
        self,
        results: Sequence[Union[RoundResult, RoundDisplayData]],
    ) -> List[RoundDisplayData]:
        """Ingest multiple round results in order."""
        return self._feed.ingest_many(results)

    def display_round(self, data: Union[RoundResult, RoundDisplayData]) -> str:
        """Render a single RoundResult or RoundDisplayData in human-readable format."""
        return format_round_summary(data)

    def display_latest(self) -> Optional[str]:
        """Render the most recently ingested round summary, or None if empty."""
        latest = self._feed.get_latest_round()
        if latest is None:
            return None
        return format_round_summary(latest)

    def replay(self) -> List[str]:
        """Replay all ingested round summaries in chronological sequence."""
        return [format_round_summary(r) for r in self._feed.get_rounds()]

    def display_all(self, separator: str = "\n\n") -> str:
        """Render all ingested rounds concatenated with a separator."""
        return separator.join(self.replay())

    def get_round_dict(self, data: Union[RoundResult, RoundDisplayData]) -> Dict[str, Any]:
        """Format a round as a structured dictionary for JSON/API consumers."""
        return format_round_dict(data)
