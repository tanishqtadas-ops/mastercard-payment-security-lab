"""
dashboard/controller.py — Unified Dashboard Presentation & Integration Controller.

Provides a unified, cohesive facade and presentation controller for the Mastercard
Payment Security Lab dashboard layer, integrating:
- DashboardFeed (state and history management)
- RoundResultViewer (presentation formatting and replay)
- ArmsRacePresenter (progression, detection trends, and arms-race analytics)
- Genome progression and mutation tracking
- Evaluation metrics and comprehensive dashboard state snapshots
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, Field

from schemas import AttackFamily, RoundResult
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
)


class GenomeProgressionStep(BaseModel):
    """Snapshot of genome dimensions and mutations at a specific simulation round."""

    round_index: int
    round_id: str
    family: str
    genome: Dict[str, float] = Field(default_factory=dict)
    deltas: Dict[str, float] = Field(default_factory=dict)
    detected: bool
    status: str
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)


class DashboardEvaluationMetrics(BaseModel):
    """Aggregate evaluation metrics computed across all ingested simulation rounds."""

    total_rounds: int = 0
    total_attacks: int = 0
    total_legitimate: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    detection_rate: float = 0.0
    accuracy: float = 0.0
    average_risk_score: float = 0.0
    average_attack_difficulty: float = 0.0
    recovery_count: int = 0
    model_update_count: int = 0


class DashboardState(BaseModel):
    """Unified, presentation-ready snapshot of the entire dashboard state."""

    total_rounds: int = 0
    is_empty: bool = True
    current_family: Optional[str] = None
    latest_round: Optional[RoundDisplayData] = None
    current_risk_score: Optional[float] = None
    current_status: Optional[str] = None
    current_genome: Dict[str, float] = Field(default_factory=dict)
    current_feature_contributions: Optional[Dict[str, float]] = None
    current_explanation: Optional[str] = None
    evaluation_metrics: DashboardEvaluationMetrics = Field(default_factory=DashboardEvaluationMetrics)
    genome_progression: List[GenomeProgressionStep] = Field(default_factory=list)
    arms_race_summary: ArmsRaceSummary = Field(default_factory=ArmsRaceSummary)
    arms_race_report: Optional[ArmsRaceReport] = None


class PaymentSecurityDashboard:
    """
    Unified presentation controller for the Mastercard Payment Security Lab dashboard.

    Integrates DashboardFeed, RoundResultViewer, and ArmsRacePresenter into a single
    cohesive API for consuming, querying, and exporting simulation results for UI layers.
    """

    def __init__(self, feed: Optional[DashboardFeed] = None) -> None:
        """
        Initialize dashboard with an optional pre-existing DashboardFeed.

        Reuses the feed across viewer and arms race components to maintain synchronized state.
        """
        self._feed = feed if feed is not None else DashboardFeed()
        self._viewer = RoundResultViewer(feed=self._feed)
        self._arms_race = ArmsRacePresenter(feed=self._feed)

    @property
    def feed(self) -> DashboardFeed:
        """Access the underlying DashboardFeed."""
        return self._feed

    @property
    def viewer(self) -> RoundResultViewer:
        """Access the RoundResultViewer component."""
        return self._viewer

    @property
    def arms_race(self) -> ArmsRacePresenter:
        """Access the ArmsRacePresenter component."""
        return self._arms_race

    # ---------------------------------------------------------------------------
    # Ingestion API
    # ---------------------------------------------------------------------------

    def ingest(self, result: Union[RoundResult, RoundDisplayData]) -> RoundDisplayData:
        """
        Ingest a single simulation round into the dashboard.

        Args:
            result: RoundResult or RoundDisplayData to ingest.

        Returns:
            The ingested RoundDisplayData presentation projection.
        """
        return self._feed.ingest(result)

    def ingest_many(
        self,
        results: Sequence[Union[RoundResult, RoundDisplayData]],
    ) -> List[RoundDisplayData]:
        """
        Ingest a sequence of simulation rounds into the dashboard.

        Args:
            results: Sequence of RoundResult or RoundDisplayData objects.

        Returns:
            List of ingested RoundDisplayData instances.
        """
        return self._feed.ingest_many(results)

    def clear(self) -> None:
        """Reset and clear all round data from the dashboard."""
        self._feed.clear()

    # ---------------------------------------------------------------------------
    # State & Query Properties
    # ---------------------------------------------------------------------------

    @property
    def round_count(self) -> int:
        """Total number of simulation rounds recorded."""
        return self._feed.round_count

    @property
    def is_empty(self) -> bool:
        """True if no rounds have been ingested."""
        return self._feed.round_count == 0

    @property
    def latest_round(self) -> Optional[RoundDisplayData]:
        """Return the most recently ingested round display projection, or None if empty."""
        return self._feed.get_latest_round()

    @property
    def current_family(self) -> Optional[str]:
        """Return the attack family of the latest round, or None if empty."""
        latest = self.latest_round
        return latest.family if latest is not None else None

    @property
    def current_risk_score(self) -> Optional[float]:
        """Return the risk score of the latest round, or None if empty."""
        latest = self.latest_round
        return latest.risk_score if latest is not None else None

    @property
    def current_status(self) -> Optional[str]:
        """Return the operational status of the latest round (e.g. DETECTED/MISSED/APPROVED)."""
        latest = self.latest_round
        return latest.status if latest is not None else None

    @property
    def current_genome(self) -> Dict[str, float]:
        """Return the attack genome of the latest round, or empty dict if empty."""
        latest = self.latest_round
        return dict(latest.genome) if latest is not None else {}

    @property
    def current_feature_contributions(self) -> Optional[Dict[str, float]]:
        """Return SHAP/feature contributions of the latest round if available."""
        latest = self.latest_round
        return dict(latest.feature_contributions) if (latest is not None and latest.feature_contributions) else None

    @property
    def current_explanation(self) -> Optional[str]:
        """Return the textual explanation of the latest round if available."""
        latest = self.latest_round
        return latest.explanation if latest is not None else None

    # ---------------------------------------------------------------------------
    # Analytical & Historical Queries
    # ---------------------------------------------------------------------------

    def get_rounds(
        self,
        family: Optional[Union[AttackFamily, str]] = None,
    ) -> List[RoundDisplayData]:
        """
        Return ingested round records, optionally filtered by attack family.

        Args:
            family: Optional AttackFamily enum or matching string filter.

        Returns:
            List of RoundDisplayData instances in chronological order.
        """
        if family is not None:
            return self._feed.get_rounds_by_family(family)
        return self._feed.get_rounds()

    def get_genome_progression(
        self,
        family: Optional[Union[AttackFamily, str]] = None,
    ) -> List[GenomeProgressionStep]:
        """
        Track and compute genome mutation progression and deltas across consecutive rounds.

        Args:
            family: Optional family filter.

        Returns:
            List of GenomeProgressionStep objects capturing dimensions and deltas.
        """
        rounds = self.get_rounds(family=family)
        progression: List[GenomeProgressionStep] = []
        prev_genome: Dict[str, float] = {}

        for idx, r in enumerate(rounds, start=1):
            curr_genome = dict(r.genome)
            deltas: Dict[str, float] = {}

            if prev_genome:
                for k, v in curr_genome.items():
                    if k in prev_genome:
                        deltas[k] = round(v - prev_genome[k], 4)

            difficulty = calculate_attack_difficulty(curr_genome, r.family)

            progression.append(
                GenomeProgressionStep(
                    round_index=idx,
                    round_id=r.round_id,
                    family=r.family,
                    genome=curr_genome,
                    deltas=deltas,
                    detected=r.detected,
                    status=r.status,
                    difficulty=difficulty,
                )
            )
            prev_genome = curr_genome

        return progression

    def get_evaluation_metrics(self) -> DashboardEvaluationMetrics:
        """
        Compute holistic evaluation metrics across all ingested simulation rounds.

        Returns:
            DashboardEvaluationMetrics model.
        """
        rounds = self._feed.get_rounds()
        if not rounds:
            return DashboardEvaluationMetrics()

        total_rounds = len(rounds)
        total_attacks = 0
        total_legit = 0
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        total_risk = 0.0
        total_diff = 0.0

        for r in rounds:
            total_risk += r.risk_score
            total_diff += calculate_attack_difficulty(r.genome, r.family)

            if r.ground_truth:
                total_attacks += 1
                if r.prediction:
                    tp += 1
                else:
                    fn += 1
            else:
                total_legit += 1
                if r.prediction:
                    fp += 1
                else:
                    tn += 1

        correct = tp + tn
        accuracy = correct / total_rounds if total_rounds > 0 else 0.0
        det_rate = tp / total_attacks if total_attacks > 0 else 0.0

        arms_race_summary = self._arms_race.get_summary()

        return DashboardEvaluationMetrics(
            total_rounds=total_rounds,
            total_attacks=total_attacks,
            total_legitimate=total_legit,
            true_positives=tp,
            false_negatives=fn,
            false_positives=fp,
            true_negatives=tn,
            detection_rate=round(det_rate, 4),
            accuracy=round(accuracy, 4),
            average_risk_score=round(total_risk / total_rounds, 4),
            average_attack_difficulty=round(total_diff / total_rounds, 4),
            recovery_count=arms_race_summary.recovery_count,
            model_update_count=arms_race_summary.model_update_count,
        )

    def get_state(self) -> DashboardState:
        """
        Compile and return the complete structured DashboardState snapshot.

        Returns:
            DashboardState model suitable for direct frontend/UI consumption.
        """
        if self.is_empty:
            return DashboardState(
                total_rounds=0,
                is_empty=True,
                current_family=None,
                latest_round=None,
                current_risk_score=None,
                current_status=None,
                current_genome={},
                current_feature_contributions=None,
                current_explanation=None,
                evaluation_metrics=DashboardEvaluationMetrics(),
                genome_progression=[],
                arms_race_summary=ArmsRaceSummary(),
                arms_race_report=None,
            )

        latest = self.latest_round
        report = self._arms_race.get_report()
        eval_metrics = self.get_evaluation_metrics()
        progression = self.get_genome_progression()

        return DashboardState(
            total_rounds=self.round_count,
            is_empty=False,
            current_family=latest.family if latest else None,
            latest_round=latest,
            current_risk_score=latest.risk_score if latest else None,
            current_status=latest.status if latest else None,
            current_genome=dict(latest.genome) if latest else {},
            current_feature_contributions=dict(latest.feature_contributions) if (latest and latest.feature_contributions) else None,
            current_explanation=latest.explanation if latest else None,
            evaluation_metrics=eval_metrics,
            genome_progression=progression,
            arms_race_summary=report.summary,
            arms_race_report=report,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the full dashboard state as a JSON-serializable dictionary.

        Returns:
            Dict containing complete state projection.
        """
        return self.get_state().model_dump(mode="json")

    def render_summary(self) -> str:
        """
        Format a concise human-readable text dashboard report for CLI or logging.

        Returns:
            Formatted multi-line summary string.
        """
        state = self.get_state()
        if state.is_empty:
            return "=== Mastercard Payment Security Lab Dashboard (No rounds recorded) ==="

        latest = state.latest_round
        metrics = state.evaluation_metrics
        arms = state.arms_race_summary

        lines = [
            "============================================================",
            "   MASTERCARD PAYMENT SECURITY LAB — ADVERSARIAL DASHBOARD   ",
            "============================================================",
            f"Total Rounds:       {state.total_rounds}",
            f"Current Family:     {state.current_family}",
            f"Current Status:     {state.current_status} (Risk Score: {state.current_risk_score:.4f})",
        ]
        if latest:
            lines.append(f"Latest Round ID:    {latest.round_id} (Model: {latest.model_version})")
            if latest.explanation:
                lines.append(f"Latest Explanation: {latest.explanation}")

        lines.extend([
            "------------------------------------------------------------",
            "METRICS & ARMS-RACE PROGRESSION:",
            f"Detection Rate:     {metrics.detection_rate * 100:.1f}% (TP: {metrics.true_positives}, FN: {metrics.false_negatives})",
            f"False Alarm Rate:   {metrics.false_positives} FP, {metrics.true_negatives} TN",
            f"Average Risk Score: {metrics.average_risk_score:.4f}",
            f"Avg Difficulty:     {metrics.average_attack_difficulty:.4f}",
            f"Recoveries Made:    {arms.recovery_count}",
            f"Model Updates:      {arms.model_update_count}",
            "============================================================",
        ])
        return "\n".join(lines)


# Convenient alias for root package
Dashboard = PaymentSecurityDashboard
