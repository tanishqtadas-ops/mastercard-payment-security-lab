"""
tests/test_dashboard_integration.py — Focused tests for unified Dashboard Polish / Integration (Task 12).

Tests cover:
1. Unified dashboard initialization with default or provided feed.
2. Current family, round, risk score, and detected/missed status reporting.
3. Current genome dimensions and feature contributions exposure.
4. Genome mutation progression tracking and delta calculation across rounds.
5. Arms-race report integration (detection trends, risk trends, recovery, updates).
6. Comprehensive evaluation metrics aggregation (TP, FP, FN, TN, detection rate, accuracy).
7. Full DashboardState snapshot generation and JSON dictionary export.
8. Empty-history graceful handling (zeroed/safe state without division by zero).
9. Deterministic replay and state reproducibility.
10. Live simulation integration with multi-round Family 2 and Family 3 pipelines.
11. Immutability guarantee: underlying RoundResult data is never mutated.
12. Dashboard text summary rendering.
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
from simulation.pipeline import Pipeline

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
    Dashboard,
    DashboardEvaluationMetrics,
    DashboardFeed,
    DashboardState,
    GenomeProgressionStep,
    PaymentSecurityDashboard,
    RoundDisplayData,
)


@pytest.fixture
def sample_dashboard_rounds() -> list[RoundResult]:
    """Create a multi-round scenario across Family 2 and Family 3 for dashboard validation."""
    rounds: list[RoundResult] = []

    # Round 1: Family 2 - Detected Attack
    ev1 = AttackEvent(
        attack_id="atk-dash-001",
        round_id="round-dash-1",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.85,
            "intent_category_deviation": 0.80,
            "permission_scope_deviation": 0.75,
            "agent_identity_confidence": 0.20,
            "session_provenance_anomaly": 0.70,
            "purchase_velocity": 0.65,
        },
        scenario={"action": "unauthorized_burst_transfer"},
        ground_truth=True,
    )
    pred1 = PredictionResult(
        prediction_id="pred-dash-001",
        prediction=True,
        risk_score=0.92,
        model_version="xgb-family2-v1",
        explanation="Unauthorized intent deviation detected",
        feature_contributions={"intent_amount_deviation": 0.35, "intent_category_deviation": 0.30},
    )
    fb1 = BlueTeamFeedback(
        feedback_id="fb-dash-001",
        round_reference="round-dash-1",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.92,
        important_features={"intent_amount_deviation": 0.35},
    )
    rounds.append(RoundResult(round_id="round-dash-1", attack_event=ev1, prediction_result=pred1, feedback=fb1))

    # Round 2: Family 2 - Missed Attack (Stealthy Evasion)
    ev2 = AttackEvent(
        attack_id="atk-dash-002",
        round_id="round-dash-2",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.15,
            "intent_category_deviation": 0.10,
            "permission_scope_deviation": 0.12,
            "agent_identity_confidence": 0.95,
            "session_provenance_anomaly": 0.10,
            "purchase_velocity": 0.15,
        },
        scenario={"action": "stealthy_subscription_procure"},
        ground_truth=True,
    )
    pred2 = PredictionResult(
        prediction_id="pred-dash-002",
        prediction=False,
        risk_score=0.18,
        model_version="xgb-family2-v1",
        explanation=None,
        feature_contributions={"intent_amount_deviation": 0.05},
    )
    fb2 = BlueTeamFeedback(
        feedback_id="fb-dash-002",
        round_reference="round-dash-2",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.18,
        important_features={},
    )
    rounds.append(RoundResult(round_id="round-dash-2", attack_event=ev2, prediction_result=pred2, feedback=fb2))

    # Round 3: Family 2 - Legitimate Approved Event (True Negative)
    ev3 = AttackEvent(
        attack_id="legit-dash-003",
        round_id="round-dash-3",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.02,
            "intent_category_deviation": 0.01,
            "permission_scope_deviation": 0.01,
            "agent_identity_confidence": 0.99,
            "session_provenance_anomaly": 0.01,
            "purchase_velocity": 0.05,
        },
        scenario={"action": "approved_routine_purchase"},
        ground_truth=False,
    )
    pred3 = PredictionResult(
        prediction_id="pred-dash-003",
        prediction=False,
        risk_score=0.04,
        model_version="xgb-family2-v1",
        explanation="Routine benign transaction",
    )
    fb3 = BlueTeamFeedback(
        feedback_id="fb-dash-003",
        round_reference="round-dash-3",
        detected=False,
        false_positive=False,
        false_negative=False,
        risk_score=0.04,
        important_features={},
    )
    rounds.append(RoundResult(round_id="round-dash-3", attack_event=ev3, prediction_result=pred3, feedback=fb3))

    return rounds


# ---------------------------------------------------------------------------
# 1. Initialization and Aliasing Tests
# ---------------------------------------------------------------------------

def test_dashboard_initialization():
    """Verify PaymentSecurityDashboard and Dashboard alias initialize with clean state."""
    dash1 = PaymentSecurityDashboard()
    assert dash1.is_empty is True
    assert dash1.round_count == 0
    assert dash1.latest_round is None

    dash2 = Dashboard()
    assert isinstance(dash2, PaymentSecurityDashboard)
    assert dash2.is_empty is True


def test_dashboard_with_custom_feed():
    """Verify dashboard can be initialized with an existing DashboardFeed."""
    feed = DashboardFeed()
    dash = PaymentSecurityDashboard(feed=feed)
    assert dash.feed is feed
    assert dash.viewer.feed is feed
    assert dash.arms_race.feed is feed


# ---------------------------------------------------------------------------
# 2. State and Latest Round Reporting
# ---------------------------------------------------------------------------

def test_dashboard_latest_round_properties(sample_dashboard_rounds: list[RoundResult]):
    """Verify current family, risk score, status, genome, explanation, and feature contributions."""
    dash = PaymentSecurityDashboard()
    dash.ingest_many(sample_dashboard_rounds)

    assert dash.round_count == 3
    assert dash.is_empty is False

    # Check latest round (Round 3)
    latest = dash.latest_round
    assert latest is not None
    assert latest.round_id == "round-dash-3"
    assert dash.current_family == AttackFamily.AGENT_BEHAVIOR.value
    assert dash.current_risk_score == 0.04
    assert dash.current_status == "APPROVED"
    assert dash.current_explanation == "Routine benign transaction"
    assert "intent_amount_deviation" in dash.current_genome

    # Ingesting another round updates the latest state immediately
    ev4 = AttackEvent(
        attack_id="atk-dash-004",
        round_id="round-dash-4",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome={"cross_field_consistency": 0.40},
        scenario={},
        ground_truth=True,
    )
    pred4 = PredictionResult(
        prediction_id="p-004",
        prediction=True,
        risk_score=0.89,
        model_version="synth-v1",
        explanation="Inconsistent synthetic identity profile",
        feature_contributions={"cross_field_consistency": 0.45},
    )
    fb4 = BlueTeamFeedback(
        feedback_id="fb-dash-004",
        round_reference="round-dash-4",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.89,
        important_features={"cross_field_consistency": 0.45},
    )
    r4 = RoundResult(round_id="round-dash-4", attack_event=ev4, prediction_result=pred4, feedback=fb4)
    dash.ingest(r4)

    assert dash.round_count == 4
    assert dash.current_family == AttackFamily.SYNTHETIC_IDENTITY.value
    assert dash.current_risk_score == 0.89
    assert dash.current_status == "DETECTED"
    assert dash.current_feature_contributions == {"cross_field_consistency": 0.45}


# ---------------------------------------------------------------------------
# 3. Genome Mutation Progression Tracking
# ---------------------------------------------------------------------------

def test_genome_progression_and_deltas(sample_dashboard_rounds: list[RoundResult]):
    """Verify genome mutation progression computes accurate deltas between rounds."""
    dash = PaymentSecurityDashboard()
    dash.ingest_many(sample_dashboard_rounds)

    progression = dash.get_genome_progression()
    assert len(progression) == 3

    # Step 1: Baseline round (no deltas)
    assert progression[0].round_id == "round-dash-1"
    assert progression[0].deltas == {}
    assert progression[0].detected is True

    # Step 2: Mutated round (deltas relative to round 1)
    assert progression[1].round_id == "round-dash-2"
    # Round 1 amount dev: 0.85 -> Round 2 amount dev: 0.15 => delta = -0.70
    assert pytest.approx(progression[1].deltas["intent_amount_deviation"], abs=1e-3) == -0.70
    assert progression[1].detected is False


# ---------------------------------------------------------------------------
# 4. Comprehensive Evaluation Metrics
# ---------------------------------------------------------------------------

def test_dashboard_evaluation_metrics(sample_dashboard_rounds: list[RoundResult]):
    """Verify aggregate evaluation metrics across attack and benign events."""
    dash = PaymentSecurityDashboard()
    dash.ingest_many(sample_dashboard_rounds)

    metrics = dash.get_evaluation_metrics()
    assert isinstance(metrics, DashboardEvaluationMetrics)
    assert metrics.total_rounds == 3
    assert metrics.total_attacks == 2
    assert metrics.total_legitimate == 1
    assert metrics.true_positives == 1  # Round 1
    assert metrics.false_negatives == 1  # Round 2
    assert metrics.false_positives == 0
    assert metrics.true_negatives == 1  # Round 3
    assert metrics.detection_rate == 0.50  # 1 TP / 2 Attacks
    assert metrics.accuracy == round(2 / 3, 4)  # (1 TP + 1 TN) / 3 Rounds


# ---------------------------------------------------------------------------
# 5. Full DashboardState Snapshot & JSON Serialization
# ---------------------------------------------------------------------------

def test_dashboard_state_snapshot_and_dict_export(sample_dashboard_rounds: list[RoundResult]):
    """Verify complete DashboardState model and JSON-serializable dictionary export."""
    dash = PaymentSecurityDashboard()
    dash.ingest_many(sample_dashboard_rounds)

    state = dash.get_state()
    assert isinstance(state, DashboardState)
    assert state.total_rounds == 3
    assert state.is_empty is False
    assert state.current_family == AttackFamily.AGENT_BEHAVIOR.value
    assert state.arms_race_report is not None
    assert len(state.genome_progression) == 3

    # Export to dict
    d = dash.to_dict()
    assert isinstance(d, dict)
    assert d["total_rounds"] == 3
    assert d["is_empty"] is False
    assert d["current_family"] == AttackFamily.AGENT_BEHAVIOR.value
    assert "evaluation_metrics" in d
    assert "arms_race_summary" in d


# ---------------------------------------------------------------------------
# 6. Empty History Handling
# ---------------------------------------------------------------------------

def test_empty_dashboard_state_and_summary():
    """Verify empty dashboard returns clean zeroed state without errors."""
    dash = PaymentSecurityDashboard()
    assert dash.is_empty is True
    assert dash.current_family is None
    assert dash.current_risk_score is None
    assert dash.current_genome == {}

    state = dash.get_state()
    assert state.is_empty is True
    assert state.total_rounds == 0
    assert state.evaluation_metrics.total_rounds == 0

    summary_text = dash.render_summary()
    assert "No rounds recorded" in summary_text


# ---------------------------------------------------------------------------
# 7. Text Summary Report
# ---------------------------------------------------------------------------

def test_dashboard_render_summary(sample_dashboard_rounds: list[RoundResult]):
    """Verify human-readable dashboard text summary rendering."""
    dash = PaymentSecurityDashboard()
    dash.ingest_many(sample_dashboard_rounds)

    summary_text = dash.render_summary()
    assert "MASTERCARD PAYMENT SECURITY LAB" in summary_text
    assert "Total Rounds:       3" in summary_text
    assert "Current Family:     Family 2 - Unauthorized / Malicious AI-Agent Payment Behavior" in summary_text
    assert "METRICS & ARMS-RACE PROGRESSION:" in summary_text


# ---------------------------------------------------------------------------
# 8. Determinism and Replay Reproducibility
# ---------------------------------------------------------------------------

def test_dashboard_deterministic_replay(sample_dashboard_rounds: list[RoundResult]):
    """Verify identical inputs produce identical dashboard snapshots."""
    dash1 = PaymentSecurityDashboard()
    dash1.ingest_many(sample_dashboard_rounds)

    dash2 = PaymentSecurityDashboard()
    dash2.ingest_many(sample_dashboard_rounds)

    assert dash1.to_dict() == dash2.to_dict()


# ---------------------------------------------------------------------------
# 9. Multi-Round Simulation Pipeline Integration (Family 2 & Family 3)
# ---------------------------------------------------------------------------

def test_dashboard_with_live_family2_pipeline():
    """Verify dashboard consumes multi-round Family 2 simulation pipeline results."""
    gen = AIAgentAttackGenerator(seed=777)
    det = AIAgentBlueDetector()
    ev = AIAgentFeedbackEvaluator()
    mut = AIAgentMutationStrategy()

    pipeline = Pipeline(gen, det, ev, mut, genome_updater=gen.set_genome)
    results = pipeline.run(num_rounds=4, base_round_id="f2-dash-pipeline")

    dash = PaymentSecurityDashboard()
    dash.ingest_many(results)

    assert dash.round_count == 4
    state = dash.get_state()
    assert state.total_rounds == 4
    assert state.current_family == AttackFamily.AGENT_BEHAVIOR.value
    assert len(state.genome_progression) == 4
    assert state.arms_race_report is not None


def test_dashboard_with_live_family3_pipeline():
    """Verify dashboard consumes multi-round Family 3 simulation pipeline results."""
    gen = SyntheticIdentityAttackGenerator(seed=888)
    det = SyntheticIdentityBlueDetector()
    ev = SyntheticIdentityFeedbackEvaluator()
    mut = SyntheticIdentityMutationStrategy()

    pipeline = Pipeline(gen, det, ev, mut, genome_updater=gen.set_genome)
    results = pipeline.run(num_rounds=4, base_round_id="f3-dash-pipeline")

    dash = PaymentSecurityDashboard()
    dash.ingest_many(results)

    assert dash.round_count == 4
    state = dash.get_state()
    assert state.total_rounds == 4
    assert state.current_family == AttackFamily.SYNTHETIC_IDENTITY.value
    assert len(state.genome_progression) == 4


# ---------------------------------------------------------------------------
# 10. Non-Mutation Safety Test
# ---------------------------------------------------------------------------

def test_dashboard_does_not_mutate_underlying_round_results(sample_dashboard_rounds: list[RoundResult]):
    """Verify that all dashboard aggregation operations leave core RoundResult objects unmodified."""
    r1 = sample_dashboard_rounds[0]
    orig_genome = dict(r1.attack_event.attack_genome)
    orig_risk = r1.prediction_result.risk_score
    orig_detected = r1.feedback.detected

    dash = PaymentSecurityDashboard()
    dash.ingest_many(sample_dashboard_rounds)

    _ = dash.get_state()
    _ = dash.to_dict()
    _ = dash.render_summary()
    _ = dash.get_genome_progression()
    _ = dash.get_evaluation_metrics()

    assert r1.attack_event.attack_genome == orig_genome
    assert r1.prediction_result.risk_score == orig_risk
    assert r1.feedback.detected == orig_detected
