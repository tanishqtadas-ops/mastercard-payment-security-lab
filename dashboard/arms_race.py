"""
dashboard/arms_race.py — Dashboard Arms-Race Visualization & Progression Layer.

Computes structured presentation models and progression analytics for the Red-Team / Blue-Team
arms race across simulation rounds, including:
1. Detection rate trajectory over rounds.
2. Risk score trajectory and rolling averages.
3. Detected vs missed attack distributions.
4. Attack difficulty progression (stealth / sophistication index).
5. Model update / retraining event markers.
6. Before / after recovery segments when detection is regained.
7. Comprehensive arms-race timeline points and summary reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, Field

from schemas import AttackFamily, RoundResult
from .presenter import RoundDisplayData, extract_display_data

if TYPE_CHECKING:
    from .feed import DashboardFeed


class TimelinePoint(BaseModel):
    """Structured representation of a single round in the arms-race timeline."""

    round_index: int
    round_id: str
    family: str
    detected: bool
    missed: bool
    status: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    attack_difficulty: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    is_model_update: bool = False
    is_recovery: bool = False
    genome_summary: Dict[str, float] = Field(default_factory=dict)
    explanation: Optional[str] = None


class DetectionTrendPoint(BaseModel):
    """Cumulative detection rate and outcome at a specific round."""

    round_index: int
    round_id: str
    detected: bool
    cumulative_detections: int
    cumulative_rounds: int
    detection_rate: float = Field(..., ge=0.0, le=1.0)


class RiskTrendPoint(BaseModel):
    """Risk score and rolling average risk at a specific round."""

    round_index: int
    round_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    rolling_average_risk: float = Field(..., ge=0.0, le=1.0)


class RecoverySegment(BaseModel):
    """Details of a recovery event where Blue Team regains detection after evasion."""

    evasion_round_id: str
    evasion_round_index: int
    recovery_round_id: str
    recovery_round_index: int
    rounds_to_recover: int
    pre_recovery_risk: float = Field(..., ge=0.0, le=1.0)
    post_recovery_risk: float = Field(..., ge=0.0, le=1.0)
    model_updated: bool


class ModelUpdateMarker(BaseModel):
    """Details of a Blue Team model update / retraining event."""

    round_index: int
    round_id: str
    previous_model_version: str
    new_model_version: str
    trigger_reason: str = "model_version_changed"


class ArmsRaceSummary(BaseModel):
    """High-level summary metrics for an entire arms-race session."""

    total_rounds: int = 0
    total_detected: int = 0
    total_missed: int = 0
    overall_detection_rate: float = 0.0
    average_risk_score: float = 0.0
    average_attack_difficulty: float = 0.0
    model_update_count: int = 0
    recovery_count: int = 0


class ArmsRaceReport(BaseModel):
    """Complete structured arms-race visualization report."""

    summary: ArmsRaceSummary
    timeline: List[TimelinePoint] = Field(default_factory=list)
    detection_trend: List[DetectionTrendPoint] = Field(default_factory=list)
    risk_trend: List[RiskTrendPoint] = Field(default_factory=list)
    recovery_segments: List[RecoverySegment] = Field(default_factory=list)
    model_updates: List[ModelUpdateMarker] = Field(default_factory=list)


def calculate_attack_difficulty(
    genome: Dict[str, float],
    family: Union[AttackFamily, str, None] = None,
) -> float:
    """
    Calculate an attack difficulty index in [0.0, 1.0].

    Higher difficulty represents stealthier/more evasive attack configurations that are
    harder for the Blue Team detector to classify.
    """
    if not genome:
        return 0.5

    fam_str = family.value if isinstance(family, AttackFamily) else str(family or "")

    # Family 2: AI Agent Behavior
    # High identity confidence + low deviations = stealthy/hard attack
    if "agent_identity_confidence" in genome or "AGENT_BEHAVIOR" in fam_str:
        deviations = [
            genome.get("intent_amount_deviation", 0.5),
            genome.get("intent_category_deviation", 0.5),
            genome.get("permission_scope_deviation", 0.5),
            genome.get("session_provenance_anomaly", 0.5),
            genome.get("purchase_velocity", 0.5),
        ]
        avg_deviation = sum(deviations) / len(deviations)
        identity_confidence = genome.get("agent_identity_confidence", 0.5)
        difficulty = ((1.0 - avg_deviation) + identity_confidence) / 2.0
        return round(min(max(difficulty, 0.0), 1.0), 4)

    # Family 3: Synthetic Identity
    # High consistency and coherence = plausible/stealthy synthetic profile
    if "cross_field_consistency" in genome or "SYNTHETIC_IDENTITY" in fam_str:
        consistencies = [
            genome.get("cross_field_consistency", 0.5),
            genome.get("profile_plausibility_score", 0.5),
            genome.get("contact_consistency", 0.5),
            genome.get("device_history_score", 0.5),
            genome.get("lifecycle_behavior_coherence", 0.5),
            genome.get("time_to_risky_activity", 0.5),
        ]
        difficulty = sum(consistencies) / len(consistencies)
        return round(min(max(difficulty, 0.0), 1.0), 4)

    # Generic fallback: inverted average deviation if values look like deviations
    vals = list(genome.values())
    avg_val = sum(vals) / len(vals)
    return round(min(max(avg_val, 0.0), 1.0), 4)


def _to_display_data_list(
    results: Sequence[Union[RoundResult, RoundDisplayData]],
) -> List[RoundDisplayData]:
    """Convert a mixed sequence of RoundResult/RoundDisplayData into RoundDisplayData list."""
    display_list: List[RoundDisplayData] = []
    for r in results:
        if isinstance(r, RoundResult):
            display_list.append(extract_display_data(r))
        else:
            display_list.append(r)
    return display_list


def detection_trend(
    results: Sequence[Union[RoundResult, RoundDisplayData]],
) -> List[DetectionTrendPoint]:
    """
    Generate cumulative detection rate trend points over chronological rounds.

    Args:
        results: Sequence of completed simulation round results.

    Returns:
        List of DetectionTrendPoint objects.
    """
    display_list = _to_display_data_list(results)
    trend: List[DetectionTrendPoint] = []
    cumulative_detections = 0

    for idx, r in enumerate(display_list, start=1):
        if r.detected:
            cumulative_detections += 1

        rate = cumulative_detections / idx
        trend.append(
            DetectionTrendPoint(
                round_index=idx,
                round_id=r.round_id,
                detected=r.detected,
                cumulative_detections=cumulative_detections,
                cumulative_rounds=idx,
                detection_rate=round(rate, 4),
            )
        )
    return trend


def risk_trend(
    results: Sequence[Union[RoundResult, RoundDisplayData]],
    window_size: int = 3,
) -> List[RiskTrendPoint]:
    """
    Generate risk score trajectory and rolling average risk over rounds.

    Args:
        results: Sequence of completed simulation round results.
        window_size: Number of previous rounds to include in rolling average.

    Returns:
        List of RiskTrendPoint objects.
    """
    display_list = _to_display_data_list(results)
    trend: List[RiskTrendPoint] = []
    scores: List[float] = []

    for idx, r in enumerate(display_list, start=1):
        scores.append(r.risk_score)
        recent = scores[-window_size:] if len(scores) >= window_size else scores
        rolling_avg = sum(recent) / len(recent)

        trend.append(
            RiskTrendPoint(
                round_index=idx,
                round_id=r.round_id,
                risk_score=round(r.risk_score, 4),
                rolling_average_risk=round(rolling_avg, 4),
            )
        )
    return trend


def model_update_rounds(
    results: Sequence[Union[RoundResult, RoundDisplayData]],
) -> List[ModelUpdateMarker]:
    """
    Identify and extract model update / retraining markers across rounds.

    Args:
        results: Sequence of completed simulation round results.

    Returns:
        List of ModelUpdateMarker objects.
    """
    display_list = _to_display_data_list(results)
    markers: List[ModelUpdateMarker] = []

    for idx in range(1, len(display_list)):
        prev = display_list[idx - 1]
        curr = display_list[idx]

        # Check version transition
        if curr.model_version and prev.model_version and curr.model_version != prev.model_version:
            markers.append(
                ModelUpdateMarker(
                    round_index=idx + 1,
                    round_id=curr.round_id,
                    previous_model_version=prev.model_version,
                    new_model_version=curr.model_version,
                    trigger_reason="model_version_changed",
                )
            )
        elif curr.outcome_metrics.get("model_updated") or curr.outcome_metrics.get("retrained"):
            markers.append(
                ModelUpdateMarker(
                    round_index=idx + 1,
                    round_id=curr.round_id,
                    previous_model_version=prev.model_version or "initial",
                    new_model_version=curr.model_version or "updated",
                    trigger_reason="retraining_trigger",
                )
            )

    return markers


def recovery_segments(
    results: Sequence[Union[RoundResult, RoundDisplayData]],
) -> List[RecoverySegment]:
    """
    Identify recovery segments where Blue Team regains detection after a missed attack.

    Args:
        results: Sequence of completed simulation round results.

    Returns:
        List of RecoverySegment objects.
    """
    display_list = _to_display_data_list(results)
    segments: List[RecoverySegment] = []

    last_missed_idx: Optional[int] = None
    last_missed_round: Optional[RoundDisplayData] = None

    for idx, curr in enumerate(display_list, start=1):
        if curr.missed:
            last_missed_idx = idx
            last_missed_round = curr
        elif curr.detected and last_missed_idx is not None and last_missed_round is not None:
            # Recovery occurred
            model_upd = bool(
                curr.model_version != last_missed_round.model_version
                or curr.outcome_metrics.get("model_updated")
            )
            segments.append(
                RecoverySegment(
                    evasion_round_id=last_missed_round.round_id,
                    evasion_round_index=last_missed_idx,
                    recovery_round_id=curr.round_id,
                    recovery_round_index=idx,
                    rounds_to_recover=idx - last_missed_idx,
                    pre_recovery_risk=round(last_missed_round.risk_score, 4),
                    post_recovery_risk=round(curr.risk_score, 4),
                    model_updated=model_upd,
                )
            )
            # Reset last missed once recovered
            last_missed_idx = None
            last_missed_round = None

    return segments


def build_arms_race_history(
    results: Sequence[Union[RoundResult, RoundDisplayData]],
) -> ArmsRaceReport:
    """
    Construct a complete structured arms-race visualization report.

    Args:
        results: Sequence of completed simulation round results.

    Returns:
        ArmsRaceReport containing summary, timeline, detection trend, risk trend,
        recovery segments, and model update markers.
    """
    display_list = _to_display_data_list(results)

    if not display_list:
        return ArmsRaceReport(
            summary=ArmsRaceSummary(
                total_rounds=0,
                total_detected=0,
                total_missed=0,
                overall_detection_rate=0.0,
                average_risk_score=0.0,
                average_attack_difficulty=0.0,
                model_update_count=0,
                recovery_count=0,
            ),
            timeline=[],
            detection_trend=[],
            risk_trend=[],
            recovery_segments=[],
            model_updates=[],
        )

    det_trend = detection_trend(display_list)
    r_trend = risk_trend(display_list)
    rec_segments = recovery_segments(display_list)
    mod_updates = model_update_rounds(display_list)

    update_round_indices = {m.round_index for m in mod_updates}
    recovery_round_indices = {r.recovery_round_index for r in rec_segments}

    timeline: List[TimelinePoint] = []
    total_detected = 0
    total_missed = 0
    total_risk = 0.0
    total_difficulty = 0.0

    for idx, r in enumerate(display_list, start=1):
        if r.detected:
            total_detected += 1
        if r.missed:
            total_missed += 1

        total_risk += r.risk_score
        difficulty = calculate_attack_difficulty(r.genome, r.family)
        total_difficulty += difficulty

        timeline.append(
            TimelinePoint(
                round_index=idx,
                round_id=r.round_id,
                family=r.family,
                detected=r.detected,
                missed=r.missed,
                status=r.status,
                risk_score=round(r.risk_score, 4),
                attack_difficulty=difficulty,
                model_version=r.model_version,
                is_model_update=(idx in update_round_indices),
                is_recovery=(idx in recovery_round_indices),
                genome_summary=dict(r.genome),
                explanation=r.explanation,
            )
        )

    total_rounds = len(display_list)
    summary = ArmsRaceSummary(
        total_rounds=total_rounds,
        total_detected=total_detected,
        total_missed=total_missed,
        overall_detection_rate=round(total_detected / total_rounds, 4),
        average_risk_score=round(total_risk / total_rounds, 4),
        average_attack_difficulty=round(total_difficulty / total_rounds, 4),
        model_update_count=len(mod_updates),
        recovery_count=len(rec_segments),
    )

    return ArmsRaceReport(
        summary=summary,
        timeline=timeline,
        detection_trend=det_trend,
        risk_trend=r_trend,
        recovery_segments=rec_segments,
        model_updates=mod_updates,
    )


class ArmsRacePresenter:
    """
    Presenter and query layer for arms-race visualization.

    Maintains or wraps simulation round data and exposes structured progression analytics.
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

    def ingest(self, result: Union[RoundResult, RoundDisplayData]) -> RoundDisplayData:
        """Ingest a single round result."""
        return self._feed.ingest(result)

    def ingest_many(
        self,
        results: Sequence[Union[RoundResult, RoundDisplayData]],
    ) -> List[RoundDisplayData]:
        """Ingest multiple round results in order."""
        return self._feed.ingest_many(results)

    def get_report(self) -> ArmsRaceReport:
        """Generate and return the full ArmsRaceReport for all ingested rounds."""
        return build_arms_race_history(self._feed.get_rounds())

    def get_timeline(self) -> List[TimelinePoint]:
        """Return the chronological arms-race timeline points."""
        return self.get_report().timeline

    def get_detection_trend(self) -> List[DetectionTrendPoint]:
        """Return the cumulative detection rate trend."""
        return detection_trend(self._feed.get_rounds())

    def get_risk_trend(self, window_size: int = 3) -> List[RiskTrendPoint]:
        """Return the risk score trajectory and rolling averages."""
        return risk_trend(self._feed.get_rounds(), window_size=window_size)

    def get_recovery_segments(self) -> List[RecoverySegment]:
        """Return all recovery segments."""
        return recovery_segments(self._feed.get_rounds())

    def get_model_updates(self) -> List[ModelUpdateMarker]:
        """Return all model update / retraining markers."""
        return model_update_rounds(self._feed.get_rounds())

    def get_summary(self) -> ArmsRaceSummary:
        """Return the overall summary metrics."""
        return self.get_report().summary

    def clear(self) -> None:
        """Clear all round history."""
        self._feed.clear()
