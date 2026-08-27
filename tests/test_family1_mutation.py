"""
tests/test_family1_mutation.py — Unit tests for Family 1 TransactionMutationStrategy.
"""

import pytest

from schemas.feedback import BlueTeamFeedback
from simulation.interfaces import MutationStrategy
from mutation.genome_engine import validate_genome, GenomeValidationError
from attacks.transaction_evasion.mutator import (
    TransactionMutationStrategy,
    TransactionMutator,
    FAMILY1_GENOME_DIMENSIONS,
    DEFAULT_DETECTED_STEP,
    DEFAULT_MISSED_STEP,
)
from attacks.transaction_evasion.generator import (
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)


def _make_feedback(
    round_ref: str = "round-01",
    detected: bool = True,
    risk_score: float = 0.85,
    important_features: dict | None = None,
) -> BlueTeamFeedback:
    """Helper to construct BlueTeamFeedback records for testing."""
    return BlueTeamFeedback(
        feedback_id=f"fb-{round_ref}",
        round_reference=round_ref,
        detected=detected,
        false_positive=False,
        false_negative=not detected,
        risk_score=risk_score,
        important_features=important_features or {},
    )


# ---------------------------------------------------------------------------
# 1. Protocol satisfaction and initialization
# ---------------------------------------------------------------------------

def test_mutator_satisfies_mutation_strategy_protocol():
    """Verify that TransactionMutationStrategy satisfies the MutationStrategy protocol."""
    mutator = TransactionMutationStrategy()
    assert isinstance(mutator, MutationStrategy)


def test_mutator_alias_and_custom_steps():
    """Verify alias and configurable step sizes."""
    mutator = TransactionMutator(detected_step=0.12, missed_step=0.04)
    assert mutator.detected_step == 0.12
    assert mutator.missed_step == 0.04


# ---------------------------------------------------------------------------
# 2. Detected Attack Mutation (Signal Decay & Evasion)
# ---------------------------------------------------------------------------

def test_detected_attack_reduces_deviation_signals():
    """Verify that detection triggers signal decay across genome dimensions."""
    mutator = TransactionMutationStrategy()
    initial_genome = dict(DEFAULT_ATTACK_GENOME)

    feedback = _make_feedback(detected=True, risk_score=0.90)
    mutated = mutator.mutate(initial_genome, feedback)

    # All 6 dimensions must be preserved
    assert set(mutated.keys()) == set(FAMILY1_GENOME_DIMENSIONS)

    # All deviations must decrease (be stealthier)
    for dim in FAMILY1_GENOME_DIMENSIONS:
        assert mutated[dim] < initial_genome[dim]
        assert 0.0 <= mutated[dim] <= 1.0

    validate_genome(mutated)


def test_detected_attack_prioritizes_important_features():
    """Verify that dimensions with higher feature importance receive stronger decay."""
    mutator = TransactionMutationStrategy(detected_step=0.08)
    initial_genome = {
        "amount_deviation": 0.80,
        "velocity_deviation": 0.80,
        "device_novelty": 0.80,
        "location_deviation": 0.80,
        "time_deviation": 0.80,
        "sequence_anomaly": 0.80,
    }

    # Location and amount were the primary detection drivers
    important_features = {
        "amount_deviation": 0.35,
        "location_deviation": 0.25,
        "device_novelty": 0.05,
    }
    feedback = _make_feedback(detected=True, important_features=important_features)
    mutated = mutator.mutate(initial_genome, feedback)

    # Heavily flagged features should decay more than unflagged features
    decay_amount = initial_genome["amount_deviation"] - mutated["amount_deviation"]
    decay_location = initial_genome["location_deviation"] - mutated["location_deviation"]
    decay_time = initial_genome["time_deviation"] - mutated["time_deviation"]

    assert decay_amount > decay_time
    assert decay_location > decay_time


# ---------------------------------------------------------------------------
# 3. Missed Attack Mutation (Evasion Preservation & Bounded Exploration)
# ---------------------------------------------------------------------------

def test_missed_attack_explores_bounded_variations():
    """Verify that a missed attack explores bounded variants while maintaining evasion."""
    mutator = TransactionMutationStrategy()
    initial_genome = {
        "amount_deviation": 0.30,
        "velocity_deviation": 0.25,
        "device_novelty": 0.20,
        "location_deviation": 0.20,
        "time_deviation": 0.20,
        "sequence_anomaly": 0.20,
    }

    feedback = _make_feedback(detected=False, risk_score=0.25)
    mutated = mutator.mutate(initial_genome, feedback)

    assert set(mutated.keys()) == set(FAMILY1_GENOME_DIMENSIONS)

    # Values should adjust within bounded exploration limits
    for dim in FAMILY1_GENOME_DIMENSIONS:
        assert 0.0 <= mutated[dim] <= 1.0
        delta = abs(mutated[dim] - initial_genome[dim])
        assert delta <= 0.10  # Bounded step

    validate_genome(mutated)


# ---------------------------------------------------------------------------
# 4. Clamping and Boundary Value Handling
# ---------------------------------------------------------------------------

def test_mutation_strictly_clamps_at_zero_and_one():
    """Verify values never breach [0.0, 1.0] under extreme decay or boost."""
    mutator = TransactionMutationStrategy(detected_step=0.50, missed_step=0.50)

    # Test clamping at lower bound (0.0) under detection
    low_genome = {dim: 0.02 for dim in FAMILY1_GENOME_DIMENSIONS}
    feedback_detected = _make_feedback(detected=True)
    mutated_low = mutator.mutate(low_genome, feedback_detected)

    for dim, val in mutated_low.items():
        assert val == 0.0
    validate_genome(mutated_low)

    # Test clamping at upper bound (1.0) under missed exploration
    high_genome = {dim: 0.98 for dim in FAMILY1_GENOME_DIMENSIONS}
    feedback_missed = _make_feedback(detected=False)
    mutated_high = mutator.mutate(high_genome, feedback_missed)

    for dim, val in mutated_high.items():
        assert val == 1.0
    validate_genome(mutated_high)


# ---------------------------------------------------------------------------
# 5. Determinism and Input Validation
# ---------------------------------------------------------------------------

def test_mutation_is_deterministic():
    """Verify that identical inputs produce identical mutated outputs."""
    mutator = TransactionMutationStrategy()
    genome = dict(DEFAULT_ATTACK_GENOME)
    feedback = _make_feedback(detected=True, important_features={"amount_deviation": 0.20})

    mutated1 = mutator.mutate(genome, feedback)
    mutated2 = mutator.mutate(genome, feedback)

    assert mutated1 == mutated2


def test_mutation_rejects_invalid_genome():
    """Verify mutator raises appropriate exceptions for malformed genomes."""
    mutator = TransactionMutationStrategy()
    feedback = _make_feedback(detected=True)

    # Missing keys
    with pytest.raises(ValueError, match="missing canonical Family 1 dimensions"):
        mutator.mutate({"amount_deviation": 0.5}, feedback)

    # Non-numeric / boolean values (rejected by genome_engine)
    with pytest.raises(GenomeValidationError):
        mutator.mutate({"amount_deviation": True, **{d: 0.5 for d in FAMILY1_GENOME_DIMENSIONS if d != "amount_deviation"}}, feedback)
