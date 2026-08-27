"""
tests/test_dashboard_arms_race.py — Focused tests for Dashboard Arms-Race Visualization (Task 11).

Tests cover:
1. Chronological ordering of timeline points.
2. Detection trend generation (cumulative detection rate over rounds).
3. Risk score trend generation (raw risk scores and rolling averages).
4. Detected vs missed attack counts and summary metrics.
5. Attack difficulty calculation and progression across rounds.
6. Model update / retraining marker identification.
7. Recovery segment identification (before/after recovery when Blue Team regains detection).
8. Deterministic replay consistency across repeated executions.
9. Empty-history behavior (graceful default handling with zero rounds).
10. ArmsRacePresenter feed ingestion and state querying.
11. Integration with multi-round Family 2 (AI Agent) simulation rounds.
12. Integration with multi-round Family 3 (Synthetic Identity) simulation rounds.
13. Non-mutation of core simulation data structures.
"""

from datetime import datetime, timezone
import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    BlueTeamFeedback,
    PredictionResult,
    RoundResult,
    AIAgentPaymentEvent,
    Transaction,
)
from simulation.round_controller import RoundController

from attacks.ai_agent import AIAgentAttackGenerator, AIAgentMutationStrategy
from blue_team.ai_agent import AIAgentBlueDetector, AIAgentFeedbackEvaluator

from attacks.synthetic_identity import (
    SyntheticIdentityAttackGenerator,
    SyntheticIdentityMutationStrategy,
)
from blue_team.synthetic_identity import (
    SyntheticIdentityBlueDetector,
    SyntheticIdentityFeedbackEvaluator,
)

from dashboard import (
    ArmsRacePresenter,
    ArmsRaceReport,
    ArmsRaceSummary,
    DashboardFeed,
    DetectionTrendPoint,
    ModelUpdateMarker,
    RecoverySegment,
    RiskTrendPoint,
    RoundDisplayData,
    TimelinePoint,
    build_arms_race_history,
    calculate_attack_difficulty,
    detection_trend,
    model_update_rounds,
    recovery_segments,
    risk_trend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_round_history() -> list[RoundResult]:
    """
    Build a deterministic multi-round simulation history with:
    - Round 1: Detected attack (model v1, high risk)
    - Round 2: Missed attack (model v1, evasion / false negative)
    - Round 3: Model updated to v2, detected recovery (recovery from round 2)
    - Round 4: Detected attack (model v2, high risk)
    - Round 5: Missed attack (model v2, evasion)
    """
    rounds: list[RoundResult] = []

    # Round 1: Detected
    ev1 = AttackEvent(
        attack_id="atk-ar-001",
        round_id="round-ar-1",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.80,
            "intent_category_deviation": 0.70,
            "permission_scope_deviation": 0.75,
            "agent_identity_confidence": 0.30,
            "session_provenance_anomaly": 0.60,
            "purchase_velocity": 0.65,
        },
        scenario={"action": "unauthorized_burst_transfer"},
        ground_truth=True,
    )
    pred1 = PredictionResult(
        prediction_id="p-001",
        prediction=True,
        risk_score=0.88,
        model_version="xgb-v1",
        explanation="High intent deviation",
    )
    fb1 = BlueTeamFeedback(
        feedback_id="fb-001",
        round_reference="round-ar-1",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.88,
        important_features={"intent_amount_deviation": 0.35},
    )
    rounds.append(RoundResult(round_id="round-ar-1", attack_event=ev1, prediction_result=pred1, feedback=fb1))

    # Round 2: Missed (evasion)
    ev2 = AttackEvent(
        attack_id="atk-ar-002",
        round_id="round-ar-2",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.20,
            "intent_category_deviation": 0.15,
            "permission_scope_deviation": 0.18,
            "agent_identity_confidence": 0.90,
            "session_provenance_anomaly": 0.12,
            "purchase_velocity": 0.22,
        },
        scenario={"action": "stealthy_procurement"},
        ground_truth=True,
    )
    pred2 = PredictionResult(
        prediction_id="p-002",
        prediction=False,
        risk_score=0.22,
        model_version="xgb-v1",
        explanation="Action within permissible bounds",
    )
    fb2 = BlueTeamFeedback(
        feedback_id="fb-002",
        round_reference="round-ar-2",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.22,
        important_features={},
    )
    rounds.append(RoundResult(round_id="round-ar-2", attack_event=ev2, prediction_result=pred2, feedback=fb2))

    # Round 3: Model updated to xgb-v2 -> Recovery detection
    ev3 = AttackEvent(
        attack_id="atk-ar-003",
        round_id="round-ar-3",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.25,
            "intent_category_deviation": 0.20,
            "permission_scope_deviation": 0.22,
            "agent_identity_confidence": 0.88,
            "session_provenance_anomaly": 0.15,
            "purchase_velocity": 0.25,
        },
        scenario={"action": "adapted_stealth_transfer"},
        ground_truth=True,
    )
    pred3 = PredictionResult(
        prediction_id="p-003",
        prediction=True,
        risk_score=0.78,
        model_version="xgb-v2",
        explanation="Retrained model flagged subtle intent drift",
    )
    fb3 = BlueTeamFeedback(
        feedback_id="fb-003",
        round_reference="round-ar-3",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.78,
        important_features={"intent_amount_deviation": 0.28},
    )
    rounds.append(RoundResult(round_id="round-ar-3", attack_event=ev3, prediction_result=pred3, feedback=fb3))

    # Round 4: Detected
    ev4 = AttackEvent(
        attack_id="atk-ar-004",
        round_id="round-ar-4",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.50,
            "intent_category_deviation": 0.45,
            "permission_scope_deviation": 0.40,
            "agent_identity_confidence": 0.70,
            "session_provenance_anomaly": 0.35,
            "purchase_velocity": 0.40,
        },
        scenario={"action": "moderate_privilege_escalation"},
        ground_truth=True,
    )
    pred4 = PredictionResult(
        prediction_id="p-004",
        prediction=True,
        risk_score=0.82,
        model_version="xgb-v2",
        explanation="Scope anomaly detected",
    )
    fb4 = BlueTeamFeedback(
        feedback_id="fb-004",
        round_reference="round-ar-4",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.82,
        important_features={"permission_scope_deviation": 0.30},
    )
    rounds.append(RoundResult(round_id="round-ar-4", attack_event=ev4, prediction_result=pred4, feedback=fb4))

    # Round 5: Missed
    ev5 = AttackEvent(
        attack_id="atk-ar-005",
        round_id="round-ar-5",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.10,
            "intent_category_deviation": 0.10,
            "permission_scope_deviation": 0.10,
            "agent_identity_confidence": 0.98,
            "session_provenance_anomaly": 0.05,
            "purchase_velocity": 0.10,
        },
        scenario={"action": "ultra_stealth_micro_transfer"},
        ground_truth=True,
    )
    pred5 = PredictionResult(
        prediction_id="p-005",
        prediction=False,
        risk_score=0.15,
        model_version="xgb-v2",
        explanation="Within noise threshold",
    )
    fb5 = BlueTeamFeedback(
        feedback_id="fb-005",
        round_reference="round-ar-5",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.15,
        important_features={},
    )
    rounds.append(RoundResult(round_id="round-ar-5", attack_event=ev5, prediction_result=pred5, feedback=fb5))

    return rounds


# ---------------------------------------------------------------------------
# 1. Chronological Ordering & Timeline Tests
# ---------------------------------------------------------------------------

def test_arms_race_timeline_chronological_ordering(multi_round_history: list[RoundResult]):
    """Verify that timeline points maintain strict 1-based chronological ordering."""
    report = build_arms_race_history(multi_round_history)

    assert len(report.timeline) == 5
    for idx, point in enumerate(report.timeline, start=1):
        assert isinstance(point, TimelinePoint)
        assert point.round_index == idx
        assert point.round_id == f"round-ar-{idx}"
        assert point.family == AttackFamily.AGENT_BEHAVIOR.value
        assert 0.0 <= point.risk_score <= 1.0
        assert 0.0 <= point.attack_difficulty <= 1.0


# ---------------------------------------------------------------------------
# 2. Detection Trend Generation
# ---------------------------------------------------------------------------

def test_detection_trend_generation(multi_round_history: list[RoundResult]):
    """Verify cumulative detection rate computation over rounds."""
    trend = detection_trend(multi_round_history)

    assert len(trend) == 5

    # Round 1: 1/1 = 1.0
    assert trend[0].cumulative_detections == 1
    assert trend[0].detection_rate == 1.0

    # Round 2: 1/2 = 0.5
    assert trend[1].cumulative_detections == 1
    assert trend[1].detection_rate == 0.5

    # Round 3: 2/3 ≈ 0.6667
    assert trend[2].cumulative_detections == 2
    assert trend[2].detection_rate == round(2 / 3, 4)

    # Round 4: 3/4 = 0.75
    assert trend[3].cumulative_detections == 3
    assert trend[3].detection_rate == 0.75

    # Round 5: 3/5 = 0.6
    assert trend[4].cumulative_detections == 3
    assert trend[4].detection_rate == 0.6


# ---------------------------------------------------------------------------
# 3. Risk Trend & Rolling Averages
# ---------------------------------------------------------------------------

def test_risk_trend_and_rolling_average(multi_round_history: list[RoundResult]):
    """Verify raw risk score trajectory and windowed rolling average calculation."""
    trend = risk_trend(multi_round_history, window_size=3)

    assert len(trend) == 5
    assert trend[0].risk_score == 0.88
    assert trend[0].rolling_average_risk == 0.88

    # Round 2: (0.88 + 0.22) / 2 = 0.55
    assert trend[1].risk_score == 0.22
    assert trend[1].rolling_average_risk == 0.55

    # Round 3: (0.88 + 0.22 + 0.78) / 3 = 0.6267
    assert trend[2].risk_score == 0.78
    assert pytest.approx(trend[2].rolling_average_risk, abs=1e-3) == round((0.88 + 0.22 + 0.78) / 3, 4)

    # Round 4: (0.22 + 0.78 + 0.82) / 3 = 0.6067
    assert trend[3].risk_score == 0.82
    assert pytest.approx(trend[3].rolling_average_risk, abs=1e-3) == round((0.22 + 0.78 + 0.82) / 3, 4)


# ---------------------------------------------------------------------------
# 4. Detected vs Missed Counts & Summary Metrics
# ---------------------------------------------------------------------------

def test_detected_vs_missed_counts_and_summary(multi_round_history: list[RoundResult]):
    """Verify summary metrics aggregation (detected count, missed count, detection rate)."""
    report = build_arms_race_history(multi_round_history)
    summary = report.summary

    assert isinstance(summary, ArmsRaceSummary)
    assert summary.total_rounds == 5
    assert summary.total_detected == 3
    assert summary.total_missed == 2
    assert summary.overall_detection_rate == 0.60
    assert summary.model_update_count == 1
    assert summary.recovery_count == 1
    assert 0.0 <= summary.average_risk_score <= 1.0
    assert 0.0 <= summary.average_attack_difficulty <= 1.0


# ---------------------------------------------------------------------------
# 5. Attack Difficulty Progression
# ---------------------------------------------------------------------------

def test_attack_difficulty_progression():
    """Verify attack difficulty calculation for subtle vs obvious genomes."""
    # Obvious/noisy attack (high deviations, low identity confidence) -> low stealth/difficulty
    noisy_genome = {
        "intent_amount_deviation": 0.95,
        "intent_category_deviation": 0.90,
        "permission_scope_deviation": 0.90,
        "agent_identity_confidence": 0.10,
        "session_provenance_anomaly": 0.85,
        "purchase_velocity": 0.90,
    }
    diff_noisy = calculate_attack_difficulty(noisy_genome, AttackFamily.AGENT_BEHAVIOR)

    # Stealthy attack (low deviations, high identity confidence) -> high stealth/difficulty
    stealth_genome = {
        "intent_amount_deviation": 0.10,
        "intent_category_deviation": 0.05,
        "permission_scope_deviation": 0.08,
        "agent_identity_confidence": 0.95,
        "session_provenance_anomaly": 0.05,
        "purchase_velocity": 0.12,
    }
    diff_stealth = calculate_attack_difficulty(stealth_genome, AttackFamily.AGENT_BEHAVIOR)

    assert 0.0 <= diff_noisy <= 1.0
    assert 0.0 <= diff_stealth <= 1.0
    assert diff_stealth > diff_noisy, "Stealthy attacks must have higher difficulty than noisy attacks"


# ---------------------------------------------------------------------------
# 6. Model Update / Retraining Markers
# ---------------------------------------------------------------------------

def test_model_update_markers(multi_round_history: list[RoundResult]):
    """Verify model version change triggers a model update marker."""
    updates = model_update_rounds(multi_round_history)

    assert len(updates) == 1
    update = updates[0]
    assert isinstance(update, ModelUpdateMarker)
    assert update.round_index == 3
    assert update.round_id == "round-ar-3"
    assert update.previous_model_version == "xgb-v1"
    assert update.new_model_version == "xgb-v2"

    # Verify timeline point flag matches
    report = build_arms_race_history(multi_round_history)
    assert report.timeline[2].is_model_update is True
    assert report.timeline[0].is_model_update is False


# ---------------------------------------------------------------------------
# 7. Recovery Segment Identification
# ---------------------------------------------------------------------------

def test_recovery_segment_identification(multi_round_history: list[RoundResult]):
    """Verify that evasion in round 2 followed by detection in round 3 produces a RecoverySegment."""
    recoveries = recovery_segments(multi_round_history)

    assert len(recoveries) == 1
    rec = recoveries[0]
    assert isinstance(rec, RecoverySegment)
    assert rec.evasion_round_id == "round-ar-2"
    assert rec.evasion_round_index == 2
    assert rec.recovery_round_id == "round-ar-3"
    assert rec.recovery_round_index == 3
    assert rec.rounds_to_recover == 1
    assert rec.pre_recovery_risk == 0.22
    assert rec.post_recovery_risk == 0.78
    assert rec.model_updated is True

    # Verify timeline point flag matches
    report = build_arms_race_history(multi_round_history)
    assert report.timeline[2].is_recovery is True
    assert report.timeline[1].is_recovery is False


# ---------------------------------------------------------------------------
# 8. Deterministic Replay Consistency
# ---------------------------------------------------------------------------

def test_arms_race_deterministic_replay(multi_round_history: list[RoundResult]):
    """Verify that repeated analysis on identical inputs produces identical outputs."""
    report1 = build_arms_race_history(multi_round_history)
    report2 = build_arms_race_history(multi_round_history)

    assert report1.model_dump() == report2.model_dump()


# ---------------------------------------------------------------------------
# 9. Empty History Handling
# ---------------------------------------------------------------------------

def test_empty_history_graceful_handling():
    """Verify that empty inputs return safe zeroed models without error."""
    report = build_arms_race_history([])

    assert isinstance(report, ArmsRaceReport)
    assert report.summary.total_rounds == 0
    assert report.summary.total_detected == 0
    assert report.summary.total_missed == 0
    assert report.summary.overall_detection_rate == 0.0
    assert report.summary.average_risk_score == 0.0
    assert report.summary.average_attack_difficulty == 0.0
    assert report.timeline == []
    assert report.detection_trend == []
    assert report.risk_trend == []
    assert report.recovery_segments == []
    assert report.model_updates == []


# ---------------------------------------------------------------------------
# 10. ArmsRacePresenter Component
# ---------------------------------------------------------------------------

def test_arms_race_presenter_feed_integration(multi_round_history: list[RoundResult]):
    """Verify ArmsRacePresenter state management, ingestion, and querying."""
    presenter = ArmsRacePresenter()
    assert presenter.get_summary().total_rounds == 0

    # Ingest one by one
    for r in multi_round_history[:2]:
        presenter.ingest(r)
    assert presenter.feed.round_count == 2
    assert len(presenter.get_timeline()) == 2

    # Ingest remainder
    presenter.ingest_many(multi_round_history[2:])
    assert presenter.feed.round_count == 5

    # Query methods
    assert len(presenter.get_detection_trend()) == 5
    assert len(presenter.get_risk_trend()) == 5
    assert len(presenter.get_recovery_segments()) == 1
    assert len(presenter.get_model_updates()) == 1

    report = presenter.get_report()
    assert report.summary.total_rounds == 5
    assert report.summary.total_detected == 3

    # Clear
    presenter.clear()
    assert presenter.feed.round_count == 0
    assert presenter.get_summary().total_rounds == 0


# ---------------------------------------------------------------------------
# 11. Integration with Live Family 2 Simulation Rounds
# ---------------------------------------------------------------------------

def test_arms_race_with_live_family2_rounds():
    """Verify arms-race analytics on live Family 2 simulation round results."""
    gen = AIAgentAttackGenerator(seed=42)
    det = AIAgentBlueDetector()
    ev = AIAgentFeedbackEvaluator()
    ctrl = RoundController(gen, det, ev)

    r1 = ctrl.run_round("f2-live-1")
    r2 = ctrl.run_round("f2-live-2")

    report = build_arms_race_history([r1, r2])
    assert report.summary.total_rounds == 2
    assert len(report.timeline) == 2
    assert report.timeline[0].family == AttackFamily.AGENT_BEHAVIOR.value
    assert 0.0 <= report.timeline[0].attack_difficulty <= 1.0


# ---------------------------------------------------------------------------
# 12. Integration with Live Family 3 Simulation Rounds
# ---------------------------------------------------------------------------

def test_arms_race_with_live_family3_rounds():
    """Verify arms-race analytics on live Family 3 simulation round results."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    det = SyntheticIdentityBlueDetector()
    ev = SyntheticIdentityFeedbackEvaluator()
    ctrl = RoundController(gen, det, ev)

    r1 = ctrl.run_round("f3-live-1")
    r2 = ctrl.run_round("f3-live-2")

    report = build_arms_race_history([r1, r2])
    assert report.summary.total_rounds == 2
    assert len(report.timeline) == 2
    assert report.timeline[0].family == AttackFamily.SYNTHETIC_IDENTITY.value
    assert 0.0 <= report.timeline[0].attack_difficulty <= 1.0


# ---------------------------------------------------------------------------
# 13. Non-Mutation Safety Test
# ---------------------------------------------------------------------------

def test_arms_race_does_not_mutate_simulation_data(multi_round_history: list[RoundResult]):
    """Verify arms-race presentation extraction never mutates input RoundResult structures."""
    first_round = multi_round_history[0]
    orig_genome = dict(first_round.attack_event.attack_genome)
    orig_risk = first_round.prediction_result.risk_score
    orig_detected = first_round.feedback.detected

    _ = build_arms_race_history(multi_round_history)
    _ = detection_trend(multi_round_history)
    _ = risk_trend(multi_round_history)
    _ = recovery_segments(multi_round_history)
    _ = model_update_rounds(multi_round_history)

    assert first_round.attack_event.attack_genome == orig_genome
    assert first_round.prediction_result.risk_score == orig_risk
    assert first_round.feedback.detected == orig_detected
