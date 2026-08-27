"""
tests/test_family1_generator.py — Unit tests for Family 1 TransactionAttackGenerator.
"""

from datetime import datetime
import pytest

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.transaction import Transaction
from simulation.interfaces import AttackGenerator
from mutation.genome_engine import validate_genome, GenomeValidationError
from attacks.transaction_evasion.generator import (
    TransactionAttackGenerator,
    FAMILY1_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)


# ---------------------------------------------------------------------------
# 1. Protocol satisfaction and initialization
# ---------------------------------------------------------------------------

def test_generator_satisfies_attack_generator_protocol():
    """Verify that TransactionAttackGenerator satisfies the AttackGenerator protocol."""
    generator = TransactionAttackGenerator()
    assert isinstance(generator, AttackGenerator)


def test_generator_default_initialization():
    """Verify default attack genome and properties on initialization."""
    gen_attack = TransactionAttackGenerator(ground_truth=True)
    assert gen_attack.genome == DEFAULT_ATTACK_GENOME

    gen_legit = TransactionAttackGenerator(ground_truth=False)
    assert gen_legit.genome == DEFAULT_LEGITIMATE_GENOME


def test_generator_custom_genome_initialization():
    """Verify initialization with a custom valid genome."""
    custom_genome = {
        "amount_deviation": 0.40,
        "velocity_deviation": 0.35,
        "device_novelty": 0.50,
        "location_deviation": 0.60,
        "time_deviation": 0.45,
        "sequence_anomaly": 0.30,
    }
    generator = TransactionAttackGenerator(genome=custom_genome)
    assert generator.genome == custom_genome


# ---------------------------------------------------------------------------
# 2. AttackEvent structure and schema compliance
# ---------------------------------------------------------------------------

def test_generate_returns_valid_attack_event():
    """Verify generate produces a valid, schema-compliant AttackEvent."""
    generator = TransactionAttackGenerator(seed=42)
    event = generator.generate("round-001")

    assert isinstance(event, AttackEvent)
    assert event.round_id == "round-001"
    assert event.attack_id == "atk-f1-round-001"
    assert event.attack_family == AttackFamily.ADAPTIVE_EVASION
    assert event.ground_truth is True

    # Genome validation
    assert set(event.attack_genome.keys()) == set(FAMILY1_GENOME_DIMENSIONS)
    for dim, val in event.attack_genome.items():
        assert 0.0 <= val <= 1.0
    validate_genome(event.attack_genome)


def test_generate_scenario_contains_required_contexts():
    """Verify scenario contains target transaction, user baseline profile, and recent history."""
    generator = TransactionAttackGenerator(seed=123)
    event = generator.generate("round-002")

    scenario = event.scenario
    assert isinstance(scenario, dict)
    assert "transaction" in scenario
    assert "baseline_profile" in scenario
    assert "recent_history" in scenario

    # Target transaction validation
    tx_data = scenario["transaction"]
    target_tx = Transaction.model_validate(tx_data)
    assert target_tx.transaction_id == "tx_f1_round-002"
    assert len(target_tx.user_id) > 0
    assert target_tx.amount > 0.0
    assert len(target_tx.currency) == 3
    assert len(target_tx.merchant_id) > 0
    assert len(target_tx.merchant_category) > 0
    assert len(target_tx.location) > 0
    assert len(target_tx.device_id) > 0
    assert len(target_tx.payment_channel) > 0

    # User baseline profile validation
    profile = scenario["baseline_profile"]
    assert profile["user_id"] == target_tx.user_id
    assert profile["avg_amount"] > 0
    assert len(profile["frequent_locations"]) > 0
    assert len(profile["registered_devices"]) > 0
    assert len(profile["frequent_categories"]) > 0

    # Recent history validation
    history = scenario["recent_history"]
    assert isinstance(history, list)
    assert len(history) >= 3
    for hist_item in history:
        hist_tx = Transaction.model_validate(hist_item)
        assert hist_tx.user_id == target_tx.user_id
        assert hist_tx.currency == target_tx.currency


def test_legitimate_mode_generation():
    """Verify generation with ground_truth=False produces legitimate context."""
    generator = TransactionAttackGenerator(ground_truth=False, seed=99)
    event = generator.generate("round-legit-01")

    assert event.ground_truth is False
    assert event.attack_family == AttackFamily.ADAPTIVE_EVASION
    assert event.attack_genome == DEFAULT_LEGITIMATE_GENOME

    tx_data = event.scenario["transaction"]
    target_tx = Transaction.model_validate(tx_data)
    assert target_tx.amount > 0.0


# ---------------------------------------------------------------------------
# 3. Determinism and Seeding
# ---------------------------------------------------------------------------

def test_deterministic_seeded_generation():
    """Verify that identical random seeds produce identical AttackEvents."""
    gen1 = TransactionAttackGenerator(seed=777)
    gen2 = TransactionAttackGenerator(seed=777)

    event1 = gen1.generate("round-det-01")
    event2 = gen2.generate("round-det-01")

    assert event1.model_dump() == event2.model_dump()


def test_different_seeds_produce_variation():
    """Verify that different seeds produce varying scenarios."""
    gen1 = TransactionAttackGenerator(seed=101)
    gen2 = TransactionAttackGenerator(seed=202)

    event1 = gen1.generate("round-var-01")
    event2 = gen2.generate("round-var-01")

    # Round ID and attack family match, but scenario attributes vary
    assert event1.round_id == event2.round_id
    assert event1.attack_family == event2.attack_family
    assert event1.scenario["transaction"]["transaction_id"] == event2.scenario["transaction"]["transaction_id"]


# ---------------------------------------------------------------------------
# 4. Genome Setter & Validation Handling
# ---------------------------------------------------------------------------

def test_set_genome_validates_and_updates():
    """Verify set_genome updates active genome and rejects invalid inputs."""
    generator = TransactionAttackGenerator()
    new_genome = {
        "amount_deviation": 0.55,
        "velocity_deviation": 0.65,
        "device_novelty": 0.75,
        "location_deviation": 0.45,
        "time_deviation": 0.35,
        "sequence_anomaly": 0.25,
    }
    generator.set_genome(new_genome)
    assert generator.genome == new_genome

    event = generator.generate("round-updated")
    assert event.attack_genome == new_genome


def test_set_genome_rejects_missing_keys():
    """Verify set_genome raises ValueError when dimensions are missing."""
    generator = TransactionAttackGenerator()
    incomplete_genome = {
        "amount_deviation": 0.50,
        "velocity_deviation": 0.50,
    }
    with pytest.raises(ValueError, match="missing canonical Family 1 dimensions"):
        generator.set_genome(incomplete_genome)


def test_set_genome_rejects_extra_keys():
    """Verify set_genome raises ValueError when extraneous keys are provided."""
    generator = TransactionAttackGenerator()
    extra_genome = dict(DEFAULT_ATTACK_GENOME)
    extra_genome["non_existent_gene"] = 0.5
    with pytest.raises(ValueError, match="contains non-Family 1 dimensions"):
        generator.set_genome(extra_genome)


def test_set_genome_rejects_out_of_bounds_values():
    """Verify set_genome raises ValueError when values are not in [0.0, 1.0]."""
    generator = TransactionAttackGenerator()
    out_of_bounds = dict(DEFAULT_ATTACK_GENOME)
    out_of_bounds["amount_deviation"] = 1.50
    with pytest.raises(ValueError, match="out of bounds"):
        generator.set_genome(out_of_bounds)

    negative_bounds = dict(DEFAULT_ATTACK_GENOME)
    negative_bounds["velocity_deviation"] = -0.10
    with pytest.raises(ValueError, match="out of bounds"):
        generator.set_genome(negative_bounds)


# ---------------------------------------------------------------------------
# 5. Boundary Genome Values
# ---------------------------------------------------------------------------

def test_extreme_boundary_genomes():
    """Verify generator handles minimum (0.0) and maximum (1.0) boundary genomes."""
    min_genome = {dim: 0.0 for dim in FAMILY1_GENOME_DIMENSIONS}
    max_genome = {dim: 1.0 for dim in FAMILY1_GENOME_DIMENSIONS}

    gen_min = TransactionAttackGenerator(genome=min_genome, seed=1)
    event_min = gen_min.generate("round-min")
    assert event_min.attack_genome == min_genome
    Transaction.model_validate(event_min.scenario["transaction"])

    gen_max = TransactionAttackGenerator(genome=max_genome, seed=2)
    event_max = gen_max.generate("round-max")
    assert event_max.attack_genome == max_genome
    Transaction.model_validate(event_max.scenario["transaction"])
