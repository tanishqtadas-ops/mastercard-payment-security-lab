"""
tests/test_dashboard_foundation.py — Unit tests for the dashboard foundation layer.

Tests:
1. Extraction of required fields from RoundResult:
   - current family
   - current round
   - risk score
   - detected/missed status
   - genome
2. Integration with Family 2 (AI Agent) round results.
3. Integration with Family 3 (Synthetic Identity) round results.
4. Presentation formatting (dictionary and text summary).
5. DashboardFeed state management (ingestion, history, filtering, latest round, clearing).
6. Non-mutation of core simulation data structures.
"""

import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    BlueTeamFeedback,
    PredictionResult,
    RoundResult,
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
    DashboardFeed,
    RoundDisplayData,
    extract_display_data,
    format_round_dict,
    format_round_summary,
)


@pytest.fixture
def sample_round_result():
    """Build a deterministic RoundResult for baseline testing."""
    event = AttackEvent(
        attack_id="atk-001",
        round_id="round-test-1",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.85,
            "intent_category_deviation": 0.90,
            "permission_scope_deviation": 0.80,
            "agent_identity_confidence": 0.20,
            "session_provenance_anomaly": 0.75,
            "purchase_velocity": 0.70,
        },
        scenario={"action": "unauthorized_wire_transfer"},
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-001",
        prediction=True,
        risk_score=0.88,
        model_version="xgb-family2-v1",
        explanation="High intent deviation and anomalous provenance",
        feature_contributions={"intent_amount_deviation": 0.35, "intent_category_deviation": 0.30},
    )
    feedback = BlueTeamFeedback(
        feedback_id="fb-001",
        round_reference="round-test-1",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.88,
        important_features={"intent_amount_deviation": 0.35},
        explanation_data={"rule": "ai_agent_deviation"},
    )
    return RoundResult(
        round_id="round-test-1",
        attack_event=event,
        prediction_result=pred,
        feedback=feedback,
        outcome_metrics={"execution_time_ms": 12.5},
    )


# ---------------------------------------------------------------------------
# 1. Extraction and Presentation Model Tests
# ---------------------------------------------------------------------------

def test_extract_display_data_required_fields(sample_round_result):
    """Verify clean extraction of all required presentation fields."""
    display = extract_display_data(sample_round_result)

    assert isinstance(display, RoundDisplayData)
    # Required core accessibility dimensions:
    assert display.round_id == "round-test-1"
    assert display.family == AttackFamily.AGENT_BEHAVIOR.value
    assert display.risk_score == 0.88
    assert display.detected is True
    assert display.missed is False
    assert display.status == "DETECTED"
    assert display.genome == sample_round_result.attack_event.attack_genome

    # Additional metadata fields:
    assert display.prediction is True
    assert display.ground_truth is True
    assert display.model_version == "xgb-family2-v1"
    assert display.explanation == "High intent deviation and anomalous provenance"
    assert display.outcome_metrics == {"execution_time_ms": 12.5}


def test_extract_display_data_missed_status():
    """Verify missed attack status evaluation."""
    event = AttackEvent(
        attack_id="atk-002",
        round_id="round-test-2",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome={"amount_deviation": 0.1, "velocity_deviation": 0.2},
        scenario={},
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-002",
        prediction=False,
        risk_score=0.25,
        model_version="heuristic-v1",
    )
    feedback = BlueTeamFeedback(
        feedback_id="fb-002",
        round_reference="round-test-2",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.25,
        important_features={},
    )
    res = RoundResult(
        round_id="round-test-2",
        attack_event=event,
        prediction_result=pred,
        feedback=feedback,
    )

    display = extract_display_data(res)
    assert display.status == "MISSED"
    assert display.detected is False
    assert display.missed is True
    assert display.risk_score == 0.25


# ---------------------------------------------------------------------------
# 2. Integration with Family 2 & Family 3 Simulation Rounds
# ---------------------------------------------------------------------------

def test_dashboard_extraction_with_family2_round():
    """Test dashboard data extraction from a live Family 2 round."""
    generator = AIAgentAttackGenerator()
    detector = AIAgentBlueDetector()
    evaluator = AIAgentFeedbackEvaluator()

    controller = RoundController(generator, detector, evaluator)
    result = controller.run_round("family2-round-1")

    display = extract_display_data(result)

    assert display.round_id == "family2-round-1"
    assert display.family == AttackFamily.AGENT_BEHAVIOR.value
    assert 0.0 <= display.risk_score <= 1.0
    assert isinstance(display.detected, bool)
    assert isinstance(display.missed, bool)
    assert set(display.genome.keys()) == {
        "intent_amount_deviation",
        "intent_category_deviation",
        "permission_scope_deviation",
        "agent_identity_confidence",
        "session_provenance_anomaly",
        "purchase_velocity",
    }


def test_dashboard_extraction_with_family3_round():
    """Test dashboard data extraction from a live Family 3 round."""
    generator = SyntheticIdentityAttackGenerator()
    detector = SyntheticIdentityBlueDetector()
    evaluator = SyntheticIdentityFeedbackEvaluator()

    controller = RoundController(generator, detector, evaluator)
    result = controller.run_round("family3-round-1")

    display = extract_display_data(result)

    assert display.round_id == "family3-round-1"
    assert display.family == AttackFamily.SYNTHETIC_IDENTITY.value
    assert 0.0 <= display.risk_score <= 1.0
    assert isinstance(display.detected, bool)
    assert isinstance(display.missed, bool)
    assert set(display.genome.keys()) == {
        "cross_field_consistency",
        "profile_plausibility_score",
        "contact_consistency",
        "device_history_score",
        "lifecycle_behavior_coherence",
        "time_to_risky_activity",
    }


# ---------------------------------------------------------------------------
# 3. Formatting Utilities Tests
# ---------------------------------------------------------------------------

def test_format_round_dict(sample_round_result):
    """Verify dictionary formatting for API/JSON consumers."""
    d = format_round_dict(sample_round_result)

    assert isinstance(d, dict)
    assert d["round_id"] == "round-test-1"
    assert d["family"] == AttackFamily.AGENT_BEHAVIOR.value
    assert d["risk_score"] == 0.88
    assert d["detected"] is True
    assert d["status"] == "DETECTED"
    assert "intent_amount_deviation" in d["genome"]


def test_format_round_summary(sample_round_result):
    """Verify human-readable summary string generation."""
    summary = format_round_summary(sample_round_result)

    assert "Round round-test-1 Summary" in summary
    assert "Family:" in summary
    assert "Status:     DETECTED" in summary
    assert "Risk Score: 0.8800" in summary
    assert "intent_amount_deviation" in summary


# ---------------------------------------------------------------------------
# 4. DashboardFeed State Management Tests
# ---------------------------------------------------------------------------

def test_dashboard_feed_lifecycle(sample_round_result):
    """Verify ingestion, querying, and filtering in DashboardFeed."""
    feed = DashboardFeed()
    assert feed.round_count == 0
    assert feed.get_latest_round() is None
    assert feed.get_rounds() == []

    # Ingest single round
    ingested = feed.ingest(sample_round_result)
    assert isinstance(ingested, RoundDisplayData)
    assert feed.round_count == 1
    assert feed.get_latest_round() == ingested

    # Ingest multiple rounds
    event2 = AttackEvent(
        attack_id="atk-003",
        round_id="round-test-3",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome={"cross_field_consistency": 0.9},
        scenario={},
        ground_truth=True,
    )
    pred2 = PredictionResult(prediction_id="p-3", prediction=True, risk_score=0.9, model_version="m3")
    fb2 = BlueTeamFeedback(
        feedback_id="fb-3",
        round_reference="round-test-3",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.9,
        important_features={},
    )
    res2 = RoundResult(round_id="round-test-3", attack_event=event2, prediction_result=pred2, feedback=fb2)

    feed.ingest(res2)
    assert feed.round_count == 2
    assert feed.get_latest_round().round_id == "round-test-3"

    # Filter by family
    family2_rounds = feed.get_rounds_by_family(AttackFamily.AGENT_BEHAVIOR)
    assert len(family2_rounds) == 1
    assert family2_rounds[0].round_id == "round-test-1"

    family3_rounds = feed.get_rounds_by_family("Synthetic Identity")
    assert len(family3_rounds) == 1
    assert family3_rounds[0].round_id == "round-test-3"

    # Clear
    feed.clear()
    assert feed.round_count == 0
    assert feed.get_latest_round() is None


def test_dashboard_feed_does_not_mutate_original_result(sample_round_result):
    """Verify that dashboard ingestion does not alter core simulation objects."""
    original_genome = dict(sample_round_result.attack_event.attack_genome)
    feed = DashboardFeed()
    display = feed.ingest(sample_round_result)

    # Mutating display data or genome in presenter does not mutate original RoundResult
    display.genome["new_key"] = 999.0
    assert "new_key" not in sample_round_result.attack_event.attack_genome
    assert sample_round_result.attack_event.attack_genome == original_genome
