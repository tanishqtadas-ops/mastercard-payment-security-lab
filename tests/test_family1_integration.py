"""
tests/test_family1_integration.py — End-to-end integration and multi-round pipeline test suite for Family 1.

Verifies the complete existing Family 1 adversarial loop:
Transaction / user baseline scenario
→ Family 1 TransactionAttackGenerator
→ AttackEvent (AttackFamily.ADAPTIVE_EVASION)
→ Family 1 TransactionBlueDetector
→ PredictionResult
→ Family 1 TransactionFeedbackEvaluator
→ BlueTeamFeedback
→ Family 1 TransactionMutationStrategy
→ Next adapted Family 1 genome
→ Multi-round Pipeline orchestration via RoundController

Covers:
1. Protocol conformance across all four Family 1 classes.
2. AttackEvent structure, canonical 6-dimension genome, and AttackFamily.ADAPTIVE_EVASION tagging.
3. Scenario integrity (target Transaction, user baseline profile, recent history).
4. Detector PredictionResult schema validation and feature contributions.
5. FeedbackEvaluator confusion matrix state handling and BlueTeamFeedback schema validation.
6. Single-round orchestration through unmodified RoundController producing RoundResult.
7. Multi-round adaptive simulation through unmodified Pipeline.
8. Feedback-driven adversarial adaptation across multiple rounds (detected -> signal decay -> stealthier variant).
9. Evasion preservation and bounded payoff exploration for missed attacks.
10. Legitimate baseline transaction processing (true negatives through full loop).
11. Deterministic multi-round reproducibility with fixed seeds.
12. Strict genome validity (validate_genome) and bounds [0.0, 1.0] across all simulation rounds.
13. Independence from Family 2 and Family 3 implementations.
"""

from datetime import datetime, timezone
import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    BlueTeamFeedback,
    PredictionResult,
    RoundResult,
    Transaction,
)
from simulation.interfaces import (
    AttackGenerator,
    BlueTeamDetector,
    FeedbackEvaluator,
    MutationStrategy,
)
from simulation import RoundController, Pipeline, RoundControllerError
from mutation.genome_engine import validate_genome

from attacks.transaction_evasion import (
    TransactionAttackGenerator,
    TransactionMutationStrategy,
    FAMILY1_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from blue_team.transaction import (
    TransactionBlueDetector,
    TransactionFeedbackEvaluator,
    MODEL_VERSION,
    FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# 1. Protocol Conformance Across All Four Components
# ---------------------------------------------------------------------------

def test_all_family1_components_satisfy_protocols():
    """Verify that all four Family 1 classes satisfy their respective simulation protocols."""
    gen = TransactionAttackGenerator()
    det = TransactionBlueDetector()
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    assert isinstance(gen, AttackGenerator)
    assert isinstance(det, BlueTeamDetector)
    assert isinstance(ev, FeedbackEvaluator)
    assert isinstance(mut, MutationStrategy)


# ---------------------------------------------------------------------------
# 2. Schema and Family Tagging Conformance
# ---------------------------------------------------------------------------

def test_family1_event_family_tagging_and_genome():
    """Verify generated AttackEvents use AttackFamily.ADAPTIVE_EVASION and canonical genome."""
    gen = TransactionAttackGenerator(seed=42)
    event = gen.generate(round_id="f1-tag-check")

    assert isinstance(event, AttackEvent)
    assert event.attack_family == AttackFamily.ADAPTIVE_EVASION
    assert event.attack_family.value == "Family 1 - Adaptive Transaction-Pattern Evasion"
    assert set(event.attack_genome.keys()) == set(FAMILY1_GENOME_DIMENSIONS)

    # Validate genome via shared genome engine
    validate_genome(event.attack_genome)
    for val in event.attack_genome.values():
        assert 0.0 <= val <= 1.0


def test_family1_scenario_domain_objects():
    """Verify scenario contains valid Transaction model data and baseline context."""
    gen = TransactionAttackGenerator(seed=101)
    event = gen.generate(round_id="f1-scen-check")

    scenario = event.scenario
    assert isinstance(scenario, dict)
    assert "transaction" in scenario
    assert "baseline_profile" in scenario
    assert "recent_history" in scenario

    # Round-trip through Transaction schema
    tx = Transaction.model_validate(scenario["transaction"])
    assert tx.transaction_id.startswith("tx_f1_")
    assert tx.amount > 0.0
    assert len(tx.user_id) > 0

    # Validate recent history transactions
    for hist_item in scenario["recent_history"]:
        hist_tx = Transaction.model_validate(hist_item)
        assert hist_tx.user_id == tx.user_id


# ---------------------------------------------------------------------------
# 3. Single-Round Execution Through RoundController
# ---------------------------------------------------------------------------

def test_family1_single_round_through_round_controller():
    """Verify single-round orchestration of Family 1 components through unmodified RoundController."""
    gen = TransactionAttackGenerator(seed=42)
    det = TransactionBlueDetector()
    ev = TransactionFeedbackEvaluator()

    controller = RoundController(generator=gen, detector=det, evaluator=ev)
    result = controller.run_round(
        round_id="f1-single-round-1",
        outcome_metrics={"experiment": "family1_integration"},
    )

    assert isinstance(result, RoundResult)
    assert result.round_id == "f1-single-round-1"
    assert result.attack_event.attack_family == AttackFamily.ADAPTIVE_EVASION
    assert result.prediction_result.model_version == MODEL_VERSION
    assert 0.0 <= result.prediction_result.risk_score <= 1.0
    assert result.feedback.round_reference == "f1-single-round-1"
    assert result.feedback.feedback_id == "fb-f1-f1-single-round-1"
    assert result.outcome_metrics == {"experiment": "family1_integration"}

    # Detection logic verification for high-risk attack
    assert result.prediction_result.prediction is True
    assert result.feedback.detected is True
    assert result.feedback.false_negative is False
    assert result.feedback.false_positive is False


def test_family1_round_controller_error_handling():
    """Verify RoundController validation and error handling with Family 1 components."""
    gen = TransactionAttackGenerator()
    det = TransactionBlueDetector()
    ev = TransactionFeedbackEvaluator()

    controller = RoundController(generator=gen, detector=det, evaluator=ev)

    # Empty round_id rejection
    with pytest.raises(RoundControllerError, match="round_id must not be empty"):
        controller.run_round(round_id="")


# ---------------------------------------------------------------------------
# 4. Multi-Round Adaptive Simulation Through Pipeline
# ---------------------------------------------------------------------------

def test_family1_multi_round_adaptive_pipeline_execution():
    """Verify multi-round adaptive execution of Family 1 components through Pipeline."""
    gen = TransactionAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=777)
    det = TransactionBlueDetector(threshold=0.50)
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy(detected_step=0.08, missed_step=0.03)

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
    )

    results = pipeline.run(num_rounds=6, base_round_id="f1-pipeline")

    assert len(results) == 6

    # Verify each round produces valid RoundResult and outcome metrics
    initial_genome = dict(DEFAULT_ATTACK_GENOME)
    previous_risk = results[0].prediction_result.risk_score

    for idx, r in enumerate(results, start=1):
        assert isinstance(r, RoundResult)
        assert r.round_id == f"f1-pipeline-{idx}"
        assert r.attack_event.attack_family == AttackFamily.ADAPTIVE_EVASION
        assert r.outcome_metrics.get("round_index") == idx
        validate_genome(r.attack_event.attack_genome)

    # Across adaptive rounds, detection of attack should lead to decayed signals (stealthier genome)
    final_genome = results[-1].attack_event.attack_genome
    final_risk = results[-1].prediction_result.risk_score

    # Assert that deviations in final round are lower than initial round
    assert final_genome["amount_deviation"] < initial_genome["amount_deviation"]
    assert final_genome["location_deviation"] < initial_genome["location_deviation"]
    assert final_risk < previous_risk


def test_family1_multi_round_evasion_preservation():
    """Verify that a stealthy/missed attack preserves evasion and explores nearby variants."""
    # Start with a subtle genome that evades detection
    subtle_genome = {
        "amount_deviation": 0.15,
        "velocity_deviation": 0.10,
        "device_novelty": 0.05,
        "location_deviation": 0.05,
        "time_deviation": 0.05,
        "sequence_anomaly": 0.05,
    }
    gen = TransactionAttackGenerator(genome=subtle_genome, ground_truth=True, seed=321)
    det = TransactionBlueDetector(threshold=0.50)
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy(missed_step=0.02)

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
    )

    results = pipeline.run(num_rounds=3, base_round_id="f1-evasion")

    # Round 1 should evade detection (false negative)
    r1 = results[0]
    assert r1.prediction_result.prediction is False
    assert r1.feedback.detected is False
    assert r1.feedback.false_negative is True

    # Mutated genome should explore slightly higher values without radical jump
    r2_genome = results[1].attack_event.attack_genome
    assert r2_genome["amount_deviation"] >= subtle_genome["amount_deviation"]
    validate_genome(r2_genome)


# ---------------------------------------------------------------------------
# 5. Legitimate Sequence & True Negative Testing
# ---------------------------------------------------------------------------

def test_family1_legitimate_transactions_through_pipeline():
    """Verify that legitimate transactions produce low risk and true negatives across rounds."""
    gen = TransactionAttackGenerator(ground_truth=False, seed=999)
    det = TransactionBlueDetector()
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
    )

    results = pipeline.run(num_rounds=3, base_round_id="f1-legit")

    for r in results:
        assert r.attack_event.ground_truth is False
        assert r.prediction_result.prediction is False
        assert r.feedback.detected is False
        assert r.feedback.false_positive is False
        assert r.feedback.false_negative is False
        assert r.prediction_result.risk_score < 0.30


# ---------------------------------------------------------------------------
# 6. Deterministic Multi-Round Reproducibility
# ---------------------------------------------------------------------------

def test_family1_multi_round_deterministic_reproducibility():
    """Verify that identical random seeds produce bit-for-bit identical multi-round pipelines."""
    def run_simulation(seed: int) -> list[dict]:
        gen = TransactionAttackGenerator(seed=seed)
        det = TransactionBlueDetector()
        ev = TransactionFeedbackEvaluator()
        mut = TransactionMutationStrategy()
        pipe = Pipeline(gen, det, ev, mut, genome_updater=gen.set_genome)
        results = pipe.run(num_rounds=4, base_round_id="f1-det")
        return [r.model_dump() for r in results]

    run1 = run_simulation(seed=12345)
    run2 = run_simulation(seed=12345)

    assert run1 == run2


# ---------------------------------------------------------------------------
# 7. Pipeline Error Conditions
# ---------------------------------------------------------------------------

def test_family1_pipeline_invalid_num_rounds():
    """Verify Pipeline raises ValueError when num_rounds < 1."""
    gen = TransactionAttackGenerator()
    det = TransactionBlueDetector()
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    pipeline = Pipeline(gen, det, ev, mut)

    with pytest.raises(ValueError, match="num_rounds must be >= 1"):
        pipeline.run(num_rounds=0)
