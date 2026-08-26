"""
tests/test_family3_pipeline.py — Multi-round adaptive Pipeline integration for Family 3.

Covers:
1. End-to-end single round execution of Family 3 through RoundController.
2. Multi-round adaptive Pipeline execution with real Family 3 components:
   - SyntheticIdentityAttackGenerator
   - SyntheticIdentityBlueDetector
   - SyntheticIdentityFeedbackEvaluator
   - SyntheticIdentityMutationStrategy
3. Feedback-driven genome evolution across rounds (detected -> repair signals -> next round).
4. Evasion handling and local exploration in multi-round simulation.
5. Deterministic reproducibility across repeated multi-round runs.
6. Pipeline execution with optional genome_updater=None.
7. Validation of error conditions (num_rounds < 1).
8. Strict canonical dimension invariance and bounded values across all simulation rounds.
"""

import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    BlueTeamFeedback,
    PredictionResult,
    RoundResult,
)
from simulation import RoundController, Pipeline
from mutation.genome_engine import validate_genome

from attacks.synthetic_identity import (
    SyntheticIdentityAttackGenerator,
    SyntheticIdentityMutationStrategy,
    FAMILY3_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from blue_team.synthetic_identity import (
    SyntheticIdentityBlueDetector,
    SyntheticIdentityFeedbackEvaluator,
    MODEL_VERSION,
)


# ---------------------------------------------------------------------------
# 1. Single-Round RoundController Integration
# ---------------------------------------------------------------------------

def test_family3_single_round_through_round_controller():
    """Verify single-round orchestration of Family 3 components through RoundController."""
    gen = SyntheticIdentityAttackGenerator(seed=100)
    det = SyntheticIdentityBlueDetector()
    ev = SyntheticIdentityFeedbackEvaluator()

    controller = RoundController(generator=gen, detector=det, evaluator=ev)
    result = controller.run_round(
        round_id="f3-single-round-1",
        outcome_metrics={"source": "test_family3_pipeline"},
    )

    assert isinstance(result, RoundResult)
    assert result.round_id == "f3-single-round-1"
    assert result.attack_event.attack_family == AttackFamily.SYNTHETIC_IDENTITY
    assert result.prediction_result.model_version == MODEL_VERSION
    assert 0.0 <= result.prediction_result.risk_score <= 1.0
    assert result.feedback.round_reference == "f3-single-round-1"
    assert result.feedback.feedback_id == "fb-f3-f3-single-round-1"
    assert result.outcome_metrics == {"source": "test_family3_pipeline"}
    validate_genome(result.attack_event.attack_genome)


# ---------------------------------------------------------------------------
# 2. Multi-Round Adaptive Pipeline Execution
# ---------------------------------------------------------------------------

def test_family3_multi_round_pipeline_execution():
    """Verify multi-round adaptive execution of Family 3 through Pipeline."""
    gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=777)
    det = SyntheticIdentityBlueDetector(threshold=0.50)
    ev = SyntheticIdentityFeedbackEvaluator()
    mut = SyntheticIdentityMutationStrategy(detected_step=0.08, missed_step=0.03)

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
    )

    results = pipeline.run(num_rounds=5, base_round_id="f3-pipeline")

    assert len(results) == 5

    # Check each round produces valid RoundResult and outcome metrics
    for idx, r in enumerate(results, start=1):
        assert isinstance(r, RoundResult)
        assert r.round_id == f"f3-pipeline-{idx}"
        assert r.attack_event.attack_family == AttackFamily.SYNTHETIC_IDENTITY
        assert r.outcome_metrics.get("round_index") == idx
        assert set(r.attack_event.attack_genome.keys()) == set(FAMILY3_GENOME_DIMENSIONS)
        validate_genome(r.attack_event.attack_genome)


# ---------------------------------------------------------------------------
# 3. Feedback-Driven Genome Adaptation Across Rounds
# ---------------------------------------------------------------------------

def test_family3_pipeline_feedback_drives_adaptation():
    """
    Verify that detected attacks trigger mutation adapting subsequent round genomes
    to improve consistency and reduce detection signals.
    """
    gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=42)
    det = SyntheticIdentityBlueDetector(threshold=0.50)
    ev = SyntheticIdentityFeedbackEvaluator()
    mut = SyntheticIdentityMutationStrategy(detected_step=0.10, missed_step=0.03)

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
    )

    results = pipeline.run(num_rounds=4, base_round_id="f3-adapt")

    # Round 1 should be detected (crude initial synthetic identity)
    assert results[0].feedback.detected is True

    # Round 2 genome should be adapted (consistency / plausibility / device scores increased)
    g1 = results[0].attack_event.attack_genome
    g2 = results[1].attack_event.attack_genome

    assert g2 != g1
    # Key dimensions should have increased to reduce detection signals
    assert g2["contact_consistency"] > g1["contact_consistency"]
    assert g2["device_history_score"] > g1["device_history_score"]
    assert g2["time_to_risky_activity"] > g1["time_to_risky_activity"]


# ---------------------------------------------------------------------------
# 4. Evasion Scenario and Local Exploration
# ---------------------------------------------------------------------------

def test_family3_pipeline_evasion_scenario():
    """
    Verify that an evasive / benign-mimicking attack profile that evades detection
    correctly triggers local exploration without crashing or breaking schema contracts.
    """
    # Start with high-consistency genome that evades detector threshold
    gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_LEGITIMATE_GENOME, seed=42)
    det = SyntheticIdentityBlueDetector(threshold=0.50)
    ev = SyntheticIdentityFeedbackEvaluator()
    mut = SyntheticIdentityMutationStrategy(missed_step=0.03)

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
    )

    results = pipeline.run(num_rounds=3, base_round_id="f3-evade")

    # Round 1 should be classified as non-fraud (evasion / false negative if ground_truth=True)
    assert results[0].prediction_result.prediction is False
    assert results[0].feedback.detected is False
    assert results[0].feedback.false_negative is True

    # Local perturbation applied
    g1 = results[0].attack_event.attack_genome
    g2 = results[1].attack_event.attack_genome
    assert g2 != g1
    validate_genome(g2)


# ---------------------------------------------------------------------------
# 5. Deterministic Reproducibility
# ---------------------------------------------------------------------------

def test_family3_pipeline_deterministic_execution():
    """Verify that identical starting parameters and seed produce byte-identical multi-round runs."""
    def _create_and_run():
        gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=12345)
        det = SyntheticIdentityBlueDetector(threshold=0.50)
        ev = SyntheticIdentityFeedbackEvaluator()
        mut = SyntheticIdentityMutationStrategy(detected_step=0.08, missed_step=0.03)

        pipe = Pipeline(
            generator=gen,
            detector=det,
            evaluator=ev,
            mutator=mut,
            genome_updater=gen.set_genome,
        )
        return pipe.run(num_rounds=3, base_round_id="f3-det")

    run_a = _create_and_run()
    run_b = _create_and_run()

    for ra, rb in zip(run_a, run_b):
        assert ra.round_id == rb.round_id
        assert ra.attack_event.attack_genome == rb.attack_event.attack_genome
        assert ra.prediction_result.risk_score == rb.prediction_result.risk_score
        assert ra.prediction_result.prediction == rb.prediction_result.prediction
        assert ra.feedback.detected == rb.feedback.detected
        assert ra.feedback.important_features == rb.feedback.important_features


# ---------------------------------------------------------------------------
# 6. Pipeline without Genome Updater
# ---------------------------------------------------------------------------

def test_family3_pipeline_without_genome_updater():
    """Verify Pipeline runs cleanly when genome_updater is None."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    det = SyntheticIdentityBlueDetector()
    ev = SyntheticIdentityFeedbackEvaluator()
    mut = SyntheticIdentityMutationStrategy()

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=None,
    )

    results = pipeline.run(num_rounds=2, base_round_id="f3-no-updater")
    assert len(results) == 2
    for r in results:
        assert isinstance(r, RoundResult)


# ---------------------------------------------------------------------------
# 7. Error Handling
# ---------------------------------------------------------------------------

def test_family3_pipeline_invalid_num_rounds():
    """Verify ValueError is raised for non-positive num_rounds."""
    gen = SyntheticIdentityAttackGenerator()
    det = SyntheticIdentityBlueDetector()
    ev = SyntheticIdentityFeedbackEvaluator()
    mut = SyntheticIdentityMutationStrategy()

    pipeline = Pipeline(generator=gen, detector=det, evaluator=ev, mutator=mut)

    with pytest.raises(ValueError, match="num_rounds must be >= 1"):
        pipeline.run(num_rounds=0)

    with pytest.raises(ValueError, match="num_rounds must be >= 1"):
        pipeline.run(num_rounds=-2)


# ---------------------------------------------------------------------------
# 8. Canonical Dimensions & Invariants Across Long Runs
# ---------------------------------------------------------------------------

def test_family3_pipeline_invariants_across_multiple_rounds():
    """Verify bounds [0.0, 1.0] and canonical keys remain invariant across 10 sequential rounds."""
    gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=999)
    det = SyntheticIdentityBlueDetector(threshold=0.50)
    ev = SyntheticIdentityFeedbackEvaluator()
    mut = SyntheticIdentityMutationStrategy(detected_step=0.08, missed_step=0.03)

    pipeline = Pipeline(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
    )

    results = pipeline.run(num_rounds=10, base_round_id="f3-long")

    assert len(results) == 10
    for r in results:
        genome = r.attack_event.attack_genome
        assert set(genome.keys()) == set(FAMILY3_GENOME_DIMENSIONS)
        validate_genome(genome)
        for val in genome.values():
            assert 0.0 <= val <= 1.0



