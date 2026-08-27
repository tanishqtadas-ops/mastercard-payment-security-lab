"""
tests/test_dashboard_round_results.py — Focused tests for RoundResult dashboard display & replay.

Verifies:
1. A valid RoundResult can be displayed in human-readable and structured formats.
2. The display contains the round identifier.
3. The display contains the attack family name.
4. The display contains the operational status (DETECTED / MISSED / APPROVED / etc.).
5. The display contains the numeric risk score formatted accurately.
6. The display clearly distinguishes detected vs. missed states.
7. The display contains all genome dimensions and values.
8. The display contains the explanation string when available and handles omitted explanations gracefully.
9. Existing RoundDisplayData / DashboardFeed behavior remains valid and backward-compatible.
10. Core RoundResult objects and their nested data structures are never mutated by display operations.
11. Replay and multi-round sequence presentation via RoundResultViewer.
12. Integration with live Family 2 and Family 3 RoundResult instances.
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
    SyntheticIdentity,
)
from simulation.round_controller import RoundController

from attacks.ai_agent import AIAgentAttackGenerator, DEFAULT_ATTACK_GENOME
from blue_team.ai_agent import AIAgentBlueDetector, AIAgentFeedbackEvaluator

from attacks.synthetic_identity import SyntheticIdentityAttackGenerator
from blue_team.synthetic_identity import (
    SyntheticIdentityBlueDetector,
    SyntheticIdentityFeedbackEvaluator,
)

from dashboard import (
    DashboardFeed,
    RoundDisplayData,
    RoundResultViewer,
    extract_display_data,
    format_round_dict,
    format_round_summary,
)



@pytest.fixture
def sample_detected_round() -> RoundResult:
    """Deterministic detected RoundResult fixture."""
    tx = Transaction(
        transaction_id="tx-f2-disp-01",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=850.00,
        currency="USD",
        merchant_id="merch_crypto_vault",
        merchant_category="crypto_assets",
        location="US-Online",
        device_id="device_agent_unverified",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-disp-01",
        user_intent="Purchase office supplies up to $150",
        authorized_scope="Office supplies, max $150",
        agent_identity="unverified-bot-99",
        session_context="session_anom_ip",
        actual_action="Executed crypto transfer for $850.00",
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-disp-001",
        round_id="round-f2-disp-100",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome=dict(DEFAULT_ATTACK_GENOME),
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-disp-001",
        prediction=True,
        risk_score=0.9125,
        model_version="xgb-family2-v1",
        explanation="Unauthorized crypto transaction exceeding approved intent envelope.",
        feature_contributions={
            "intent_amount_deviation": 0.35,
            "intent_category_deviation": 0.25,
        },
    )
    feedback = BlueTeamFeedback(
        feedback_id="fb-disp-001",
        round_reference="round-f2-disp-100",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.9125,
        important_features={"intent_amount_deviation": 0.35},
        explanation_data={"rule": "intent_violation"},
    )
    return RoundResult(
        round_id="round-f2-disp-100",
        attack_event=event,
        prediction_result=pred,
        feedback=feedback,
        outcome_metrics={"round_index": 1, "processing_time_ms": 14.2},
    )


@pytest.fixture
def sample_missed_round() -> RoundResult:
    """Deterministic missed (false negative) RoundResult fixture."""
    event = AttackEvent(
        attack_id="atk-disp-002",
        round_id="round-f2-disp-101",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.15,
            "intent_category_deviation": 0.10,
            "permission_scope_deviation": 0.12,
            "agent_identity_confidence": 0.95,
            "session_provenance_anomaly": 0.10,
            "purchase_velocity": 0.18,
        },
        scenario={"action": "stealthy_subscription_renewal"},
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-disp-002",
        prediction=False,
        risk_score=0.2150,
        model_version="xgb-family2-v1",
        explanation=None,
        feature_contributions={"intent_amount_deviation": 0.05},
    )
    feedback = BlueTeamFeedback(
        feedback_id="fb-disp-002",
        round_reference="round-f2-disp-101",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.2150,
        important_features={},
    )
    return RoundResult(
        round_id="round-f2-disp-101",
        attack_event=event,
        prediction_result=pred,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# 1-8. Core Presentation Field Display Tests
# ---------------------------------------------------------------------------

def test_display_valid_round_result(sample_detected_round: RoundResult):
    """Requirement 1: A valid RoundResult can be displayed cleanly in human-readable form."""
    summary = format_round_summary(sample_detected_round)
    assert isinstance(summary, str)
    assert len(summary) > 0

    viewer = RoundResultViewer()
    assert viewer.display_round(sample_detected_round) == summary



def test_display_contains_round_identifier(sample_detected_round: RoundResult):
    """Requirement 2: The display contains the exact round identifier."""
    summary = format_round_summary(sample_detected_round)
    assert "round-f2-disp-100" in summary
    assert "--- Round round-f2-disp-100 Summary ---" in summary


def test_display_contains_family(sample_detected_round: RoundResult):
    """Requirement 3: The display contains the attack family name."""
    summary = format_round_summary(sample_detected_round)
    assert "Family:" in summary
    assert AttackFamily.AGENT_BEHAVIOR.value in summary


def test_display_contains_status(sample_detected_round: RoundResult, sample_missed_round: RoundResult):
    """Requirement 4: The display contains the correct operational status."""
    detected_summary = format_round_summary(sample_detected_round)
    assert "Status:     DETECTED" in detected_summary

    missed_summary = format_round_summary(sample_missed_round)
    assert "Status:     MISSED" in missed_summary


def test_display_contains_risk_score(sample_detected_round: RoundResult, sample_missed_round: RoundResult):
    """Requirement 5: The display contains the numeric risk score formatted to 4 decimals."""
    detected_summary = format_round_summary(sample_detected_round)
    assert "Risk Score: 0.9125" in detected_summary

    missed_summary = format_round_summary(sample_missed_round)
    assert "Risk Score: 0.2150" in missed_summary


def test_display_distinguishes_detected_and_missed_state(
    sample_detected_round: RoundResult,
    sample_missed_round: RoundResult,
):
    """Requirement 6: The display clearly distinguishes detected vs missed outcomes."""
    det_sum = format_round_summary(sample_detected_round)
    assert "Detected: True, Missed: False" in det_sum

    miss_sum = format_round_summary(sample_missed_round)
    assert "Detected: False, Missed: True" in miss_sum


def test_display_contains_genome_information(sample_detected_round: RoundResult):
    """Requirement 7: The display contains all canonical genome dimensions and their values."""
    summary = format_round_summary(sample_detected_round)
    assert "Genome:" in summary
    genome = sample_detected_round.attack_event.attack_genome
    for key, val in genome.items():
        assert key in summary
        assert f"{val:.3f}" in summary


def test_display_contains_explanation_when_available(
    sample_detected_round: RoundResult,
    sample_missed_round: RoundResult,
):
    """Requirement 8: The display shows explanation when available, and omits gracefully when None."""
    summary_with_exp = format_round_summary(sample_detected_round)
    assert "Explanation: Unauthorized crypto transaction exceeding approved intent envelope." in summary_with_exp

    summary_without_exp = format_round_summary(sample_missed_round)
    assert "Explanation:" not in summary_without_exp


# ---------------------------------------------------------------------------
# 9. Existing Foundation & Backward Compatibility Tests
# ---------------------------------------------------------------------------

def test_existing_display_data_and_feed_behavior(sample_detected_round: RoundResult):
    """Requirement 9: Existing RoundDisplayData and DashboardFeed work seamlessly with viewer."""
    display_data = extract_display_data(sample_detected_round)
    assert isinstance(display_data, RoundDisplayData)
    assert display_data.round_id == "round-f2-disp-100"
    assert display_data.risk_score == 0.9125
    assert display_data.detected is True
    assert display_data.missed is False

    # Format from display data produces identical summary
    assert format_round_summary(display_data) == format_round_summary(sample_detected_round)

    # Format dict
    d = format_round_dict(sample_detected_round)
    assert isinstance(d, dict)
    assert d["round_id"] == "round-f2-disp-100"
    assert d["status"] == "DETECTED"


# ---------------------------------------------------------------------------
# 10. Non-Mutation Safety Test
# ---------------------------------------------------------------------------

def test_display_does_not_mutate_core_round_result(sample_detected_round: RoundResult):
    """Requirement 10: Display and presentation extraction never mutate core simulation objects."""
    # Capture snapshots of core objects
    original_round_id = sample_detected_round.round_id
    original_genome = dict(sample_detected_round.attack_event.attack_genome)
    original_risk_score = sample_detected_round.prediction_result.risk_score
    original_feedback_detected = sample_detected_round.feedback.detected

    # Perform multiple presentation operations
    _ = extract_display_data(sample_detected_round)
    _ = format_round_summary(sample_detected_round)
    _ = format_round_dict(sample_detected_round)

    viewer = RoundResultViewer()
    _ = viewer.load_round(sample_detected_round)
    _ = viewer.display_round(sample_detected_round)
    _ = viewer.display_latest()
    _ = viewer.replay()

    # Assert immutability
    assert sample_detected_round.round_id == original_round_id
    assert sample_detected_round.attack_event.attack_genome == original_genome
    assert sample_detected_round.prediction_result.risk_score == original_risk_score
    assert sample_detected_round.feedback.detected == original_feedback_detected


# ---------------------------------------------------------------------------
# 11. RoundResultViewer Display and Replay Functionality
# ---------------------------------------------------------------------------

def test_round_result_viewer_replay_and_history(
    sample_detected_round: RoundResult,
    sample_missed_round: RoundResult,
):
    """Test deterministic replay and multi-round sequence presentation in RoundResultViewer."""
    viewer = RoundResultViewer()
    assert viewer.feed.round_count == 0
    assert viewer.display_latest() is None
    assert viewer.replay() == []

    # Ingest rounds
    viewer.load_round(sample_detected_round)
    viewer.load_round(sample_missed_round)

    assert viewer.feed.round_count == 2
    latest = viewer.display_latest()
    assert latest is not None
    assert "round-f2-disp-101" in latest

    # Replay sequence
    replayed = viewer.replay()
    assert len(replayed) == 2
    assert "round-f2-disp-100" in replayed[0]
    assert "round-f2-disp-101" in replayed[1]

    # Display all
    all_text = viewer.display_all()
    assert "--- Round round-f2-disp-100 Summary ---" in all_text
    assert "--- Round round-f2-disp-101 Summary ---" in all_text


# ---------------------------------------------------------------------------
# 12. Integration with Live Family 2 and Family 3 Round Results
# ---------------------------------------------------------------------------

def test_viewer_with_live_family2_and_family3_simulation():
    """Verify RoundResultViewer displays live outputs from Family 2 and Family 3 simulation rounds."""
    # Run a Family 2 round
    f2_gen = AIAgentAttackGenerator(seed=42)
    f2_det = AIAgentBlueDetector()
    f2_ev = AIAgentFeedbackEvaluator()
    f2_controller = RoundController(f2_gen, f2_det, f2_ev)
    f2_result = f2_controller.run_round("live-f2-round")

    # Run a Family 3 round
    f3_gen = SyntheticIdentityAttackGenerator(seed=42)
    f3_det = SyntheticIdentityBlueDetector()
    f3_ev = SyntheticIdentityFeedbackEvaluator()
    f3_controller = RoundController(f3_gen, f3_det, f3_ev)
    f3_result = f3_controller.run_round("live-f3-round")

    viewer = RoundResultViewer()
    viewer.load_rounds([f2_result, f3_result])

    assert viewer.feed.round_count == 2

    # Check Family 2 display
    f2_summary = viewer.display_round(f2_result)
    assert "live-f2-round" in f2_summary
    assert AttackFamily.AGENT_BEHAVIOR.value in f2_summary
    assert "intent_amount_deviation" in f2_summary

    # Check Family 3 display
    f3_summary = viewer.display_round(f3_result)
    assert "live-f3-round" in f3_summary
    assert AttackFamily.SYNTHETIC_IDENTITY.value in f3_summary
    assert "cross_field_consistency" in f3_summary
