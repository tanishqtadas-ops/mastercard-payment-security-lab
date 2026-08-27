"""
dashboard — Presentation and product layer for the Mastercard Payment Security Lab.

Provides decoupled presentation data models and feed management for simulation rounds.
"""

from .presenter import (
    RoundDisplayData,
    RoundResultViewer,
    extract_display_data,
    format_round_dict,
    format_round_summary,
)
from .feed import DashboardFeed
from .arms_race import (
    ArmsRacePresenter,
    ArmsRaceReport,
    ArmsRaceSummary,
    DetectionTrendPoint,
    ModelUpdateMarker,
    RecoverySegment,
    RiskTrendPoint,
    TimelinePoint,
    build_arms_race_history,
    calculate_attack_difficulty,
    detection_trend,
    model_update_rounds,
    recovery_segments,
    risk_trend,
)

__all__ = [
    "RoundDisplayData",
    "RoundResultViewer",
    "extract_display_data",
    "format_round_dict",
    "format_round_summary",
    "DashboardFeed",
    "ArmsRacePresenter",
    "ArmsRaceReport",
    "ArmsRaceSummary",
    "DetectionTrendPoint",
    "ModelUpdateMarker",
    "RecoverySegment",
    "RiskTrendPoint",
    "TimelinePoint",
    "build_arms_race_history",
    "calculate_attack_difficulty",
    "detection_trend",
    "model_update_rounds",
    "recovery_segments",
    "risk_trend",
]
