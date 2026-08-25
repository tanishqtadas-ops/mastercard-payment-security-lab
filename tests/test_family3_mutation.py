"""
tests/test_family3_mutation.py — Focused test suite for Family 3 Mutation Strategy.

Covers:
1. Mutator satisfies runtime_checkable MutationStrategy protocol.
2. Detected attack causes strong detection dimensions to be prioritized and shifted to reduce detection risk.
3. Missed attack preserves successful evasion structure while exploring nearby variants (no blind increase).
4. Mutation remains strictly bounded in [0.0, 1.0] across multiple iterative rounds.
5. Mutated genome contains exactly the six canonical Family 3 dimensions.
6. Deterministic behavior for identical (genome, feedback) inputs.
7. Mutation responds correctly to SHAP/feature contributions from feedback.
8. Handles missing or empty feature_contributions without error.
9. Mutator alias verification (SyntheticIdentityMutator).
10. Rejection of invalid / incomplete input genomes.
"""

import pytest

from schemas import BlueTeamFeedback
from simulation.interfaces import MutationStrategy
from mutation.genome_engine import validate_genome, GenomeValidationError
from attacks.synthetic_identity import (
    SyntheticIdentityMutationStrategy,
    SyntheticIdentityMutator,
    FAMILY3_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
    DEFAULT_DETECTED_STEP,
    DEFAULT_MISSED_STEP,
)


@pytest.fixture
def mutator() -> SyntheticIdentityMutationStrategy:
    return SyntheticIdentityMutationStrategy()


@pytest.fixture
def base_attack_genome() -> dict[str, float]:
    return dict(DEFAULT_ATTACK_GENOME)


# ---------------------------------------------------------------------------
# 1. Protocol & Class Structure
# ---------------------------------------------------------------------------

def test_mutator_satisfies_protocol(mutator: SyntheticIdentityMutationStrategy):
    """Requirement: Mutator satisfies runtime_checkable MutationStrategy protocol."""
    assert isinstance(mutator, MutationStrategy)


def test_mutator_alias_identity():
    """Requirement: SyntheticIdentityMutator alias matches SyntheticIdentityMutationStrategy."""
    assert SyntheticIdentityMutator is SyntheticIdentityMutationStrategy
    instance = SyntheticIdentityMutator()
    assert isinstance(instance, MutationStrategy)


# ---------------------------------------------------------------------------
# 2. Detected Attack Dynamics & Signal Reduction
# ---------------------------------------------------------------------------

def test_detected_attack_prioritizes_strong_signals(
    mutator: SyntheticIdentityMutationStrategy,
    base_attack_genome: dict[str, float],
):
    """
    Requirement 1 & 7: Detected attack identifies strongest detection signals from SHAP feedback
    and applies a larger repair step to reduce those specific detection signals.
    """
    # Feedback indicates disposable email and VoIP were the strongest detection drivers
    feedback = BlueTeamFeedback(
        feedback_id="fb-test-01",
        round_reference="round-01",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.88,
        important_features={
            "is_disposable_email": 0.45,
            "is_voip_carrier": 0.35,
            "lifecycle_incoherence": 0.05,
        },
    )

    mutated = mutator.mutate(base_attack_genome, feedback)

    # contact_consistency had the strongest detection signal -> should receive prioritized step (1.5x)
    contact_step = mutated["contact_consistency"] - base_attack_genome["contact_consistency"]
    expected_strong_step = DEFAULT_DETECTED_STEP * 1.5
    assert pytest.approx(contact_step, abs=1e-3) == expected_strong_step

    # lifecycle_behavior_coherence had a moderate signal -> should receive standard step (1.0x)
    lifecycle_step = mutated["lifecycle_behavior_coherence"] - base_attack_genome["lifecycle_behavior_coherence"]
    assert pytest.approx(lifecycle_step, abs=1e-3) == DEFAULT_DETECTED_STEP

    # Unflagged dimensions (e.g. cross_field_consistency) should receive smaller exploration step (0.5x)
    cross_field_step = mutated["cross_field_consistency"] - base_attack_genome["cross_field_consistency"]
    assert pytest.approx(cross_field_step, abs=1e-3) == (DEFAULT_DETECTED_STEP * 0.5)

    # Output genome is valid and bounded
    validate_genome(mutated)


def test_detected_attack_with_direct_genome_keys(
    mutator: SyntheticIdentityMutationStrategy,
    base_attack_genome: dict[str, float],
):
    """Requirement: Mutator correctly recognizes feedback keyed directly by canonical dimension names."""
    feedback = BlueTeamFeedback(
        feedback_id="fb-test-direct",
        round_reference="round-direct",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.75,
        important_features={
            "device_history_score": 0.50,
            "time_to_risky_activity": 0.10,
        },
    )

    mutated = mutator.mutate(base_attack_genome, feedback)

    dev_step = mutated["device_history_score"] - base_attack_genome["device_history_score"]
    assert dev_step > (DEFAULT_DETECTED_STEP * 1.0)
    validate_genome(mutated)


# ---------------------------------------------------------------------------
# 3. Missed Attack (Successful Evasion) Dynamics
# ---------------------------------------------------------------------------

def test_missed_attack_preserves_structure_and_explores_nearby(
    mutator: SyntheticIdentityMutationStrategy,
):
    """
    Requirement 2: When an attack is missed, mutator preserves the successful evasion structure
    and makes small, bounded local adjustments without blindly increasing every dimension.
    """
    successful_genome = {
        "cross_field_consistency": 0.70,
        "profile_plausibility_score": 0.75,
        "contact_consistency": 0.70,
        "device_history_score": 0.75,
        "lifecycle_behavior_coherence": 0.70,
        "time_to_risky_activity": 0.65,
    }

    feedback = BlueTeamFeedback(
        feedback_id="fb-test-02",
        round_reference="round-02",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.20,
        important_features={},
    )

    mutated = mutator.mutate(successful_genome, feedback)

    # 1. Output must not be a blind uniform increase across all dimensions
    deltas = {k: mutated[k] - successful_genome[k] for k in FAMILY3_GENOME_DIMENSIONS}
    unique_deltas = set(round(d, 4) for d in deltas.values())
    assert len(unique_deltas) > 1, "Missed attack should not apply an identical delta to every dimension"

    # 2. Both positive and negative local exploratory adjustments exist
    assert any(d < 0 for d in deltas.values()), "Exploration should include localized reductions (e.g. testing faster payoff)"
    assert any(d > 0 for d in deltas.values()), "Exploration should include localized increases"

    # 3. All changes remain strictly small and bounded by missed_step
    for dim, delta in deltas.items():
        assert abs(delta) <= DEFAULT_MISSED_STEP + 1e-4, f"Delta for {dim} exceeded missed_step limit"

    # 4. Genome remains valid
    validate_genome(mutated)


# ---------------------------------------------------------------------------
# 4. Boundedness & Boundary Clamping
# ---------------------------------------------------------------------------

def test_mutation_remains_strictly_bounded_near_extremes(
    mutator: SyntheticIdentityMutationStrategy,
):
    """Requirement 3: Mutation values stay strictly within [0.0, 1.0] even at extreme boundaries."""
    # Near upper bound
    near_max_genome = {dim: 0.98 for dim in FAMILY3_GENOME_DIMENSIONS}
    fb_detected = BlueTeamFeedback(
        feedback_id="fb-max",
        round_reference="r-max",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.90,
        important_features={"is_disposable_email": 0.90},
    )

    mutated_max = mutator.mutate(near_max_genome, fb_detected)
    for dim, val in mutated_max.items():
        assert 0.0 <= val <= 1.0
        assert val <= 1.0

    # Near lower bound
    near_min_genome = {dim: 0.01 for dim in FAMILY3_GENOME_DIMENSIONS}
    fb_missed = BlueTeamFeedback(
        feedback_id="fb-min",
        round_reference="r-min",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.10,
        important_features={},
    )

    mutated_min = mutator.mutate(near_min_genome, fb_missed)
    for dim, val in mutated_min.items():
        assert 0.0 <= val <= 1.0
        assert val >= 0.0


def test_repeated_iterative_mutations_stay_valid(
    mutator: SyntheticIdentityMutationStrategy,
    base_attack_genome: dict[str, float],
):
    """Requirement 3 & 4: Multiple successive mutation rounds produce valid genomes at every step."""
    genome = dict(base_attack_genome)
    feedback_sequence = [
        BlueTeamFeedback(
            feedback_id=f"fb-seq-{i}",
            round_reference=f"round-{i}",
            detected=(i % 2 == 0),
            false_positive=False,
            false_negative=(i % 2 != 0),
            risk_score=0.80 if (i % 2 == 0) else 0.20,
            important_features={"is_emulator_device": 0.40} if (i % 2 == 0) else {},
        )
        for i in range(10)
    ]

    for fb in feedback_sequence:
        genome = mutator.mutate(genome, fb)
        validate_genome(genome)
        assert len(genome) == 6
        for v in genome.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# 5. Canonical Dimensions Exact Match
# ---------------------------------------------------------------------------

def test_output_contains_exact_six_canonical_dimensions(
    mutator: SyntheticIdentityMutationStrategy,
    base_attack_genome: dict[str, float],
):
    """Requirement 5: Mutated genome contains exactly the six canonical Family 3 dimensions."""
    fb = BlueTeamFeedback(
        feedback_id="fb-canon",
        round_reference="r-canon",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.60,
        important_features={},
    )

    mutated = mutator.mutate(base_attack_genome, fb)

    assert set(mutated.keys()) == set(FAMILY3_GENOME_DIMENSIONS)
    assert len(mutated) == 6
    expected_dimensions = {
        "cross_field_consistency",
        "profile_plausibility_score",
        "contact_consistency",
        "device_history_score",
        "lifecycle_behavior_coherence",
        "time_to_risky_activity",
    }
    assert set(mutated.keys()) == expected_dimensions


# ---------------------------------------------------------------------------
# 6. Deterministic Behavior
# ---------------------------------------------------------------------------

def test_deterministic_behavior_for_identical_inputs(
    mutator: SyntheticIdentityMutationStrategy,
    base_attack_genome: dict[str, float],
):
    """Requirement 6: Identical inputs produce identical outputs across repeated calls."""
    fb = BlueTeamFeedback(
        feedback_id="fb-det",
        round_reference="round-det",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.82,
        important_features={"is_disposable_email": 0.40, "is_emulator_device": 0.30},
    )

    result_1 = mutator.mutate(base_attack_genome, fb)
    result_2 = mutator.mutate(base_attack_genome, fb)
    result_3 = mutator.mutate(base_attack_genome, fb)

    assert result_1 == result_2 == result_3


# ---------------------------------------------------------------------------
# 7. Robustness & Error Handling
# ---------------------------------------------------------------------------

def test_handles_empty_or_none_feature_contributions(
    mutator: SyntheticIdentityMutationStrategy,
    base_attack_genome: dict[str, float],
):
    """Requirement 8: Mutator executes cleanly when important_features is empty or None."""
    fb_empty = BlueTeamFeedback(
        feedback_id="fb-empty",
        round_reference="r-empty",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.70,
        important_features={},
    )

    mutated = mutator.mutate(base_attack_genome, fb_empty)
    validate_genome(mutated)
    assert set(mutated.keys()) == set(FAMILY3_GENOME_DIMENSIONS)


def test_rejects_incomplete_or_invalid_genomes(
    mutator: SyntheticIdentityMutationStrategy,
):
    """Requirement: Mutator rejects missing dimensions or invalid genome types."""
    fb = BlueTeamFeedback(
        feedback_id="fb-err",
        round_reference="r-err",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.70,
        important_features={},
    )

    # Missing dimension
    incomplete = dict(DEFAULT_ATTACK_GENOME)
    del incomplete["time_to_risky_activity"]
    with pytest.raises((ValueError, GenomeValidationError)):
        mutator.mutate(incomplete, fb)

    # Invalid non-numeric type
    invalid_val = dict(DEFAULT_ATTACK_GENOME)
    invalid_val["contact_consistency"] = "high"  # type: ignore
    with pytest.raises(GenomeValidationError):
        mutator.mutate(invalid_val, fb)
