"""
tests/test_family3_generator.py — Comprehensive test suite for Family 3 Attack Generator.

Covers:
1. Generator satisfies runtime_checkable AttackGenerator protocol.
2. Generated object is a valid AttackEvent.
3. Correct AttackFamily = AttackFamily.SYNTHETIC_IDENTITY.
4. Canonical six-dimension Family 3 genome.
5. Genome values are valid and bounded [0.0, 1.0] (passes validate_genome).
6. Deterministic behavior when seeded.
7. Scenario contains valid SyntheticIdentity with all required lifecycle information.
8. Repeated generation works cleanly across different round IDs.
9. set_genome validation (rejects missing keys, extra keys, out of bounds).
10. AttackEvent serialization round-trip.
11. Genome dimension modulation (e.g. low device score, low contact consistency, low time-to-risk).
"""

import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    SyntheticIdentity,
)
from simulation.interfaces import AttackGenerator
from mutation.genome_engine import validate_genome

from attacks.synthetic_identity import (
    SyntheticIdentityAttackGenerator,
    FAMILY3_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)


def test_generator_satisfies_attack_generator_protocol():
    """Verify SyntheticIdentityAttackGenerator satisfies the AttackGenerator protocol."""
    gen = SyntheticIdentityAttackGenerator()
    assert isinstance(gen, AttackGenerator)


def test_generated_object_is_attack_event():
    """Verify generated output is a fully validated AttackEvent."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="round-test-001")

    assert isinstance(event, AttackEvent)
    assert event.attack_id == "family3-attack-round-test-001"
    assert event.round_id == "round-test-001"
    assert event.ground_truth is True


def test_correct_attack_family():
    """Verify generated event is tagged with AttackFamily.SYNTHETIC_IDENTITY."""
    gen = SyntheticIdentityAttackGenerator()
    event = gen.generate(round_id="round-f3-01")

    assert event.attack_family == AttackFamily.SYNTHETIC_IDENTITY
    assert event.attack_family.value == "Family 3 - Synthetic Identity + AI-Generated Identity Artifacts"
    assert event.metadata["attack_family"] == AttackFamily.SYNTHETIC_IDENTITY.value


def test_canonical_six_dimension_genome():
    """Verify generated attack genome contains exactly the 6 canonical Family 3 dimensions."""
    gen = SyntheticIdentityAttackGenerator()
    event = gen.generate(round_id="round-canon-01")

    assert set(event.attack_genome.keys()) == set(FAMILY3_GENOME_DIMENSIONS)
    assert len(event.attack_genome) == 6

    expected_dimensions = {
        "cross_field_consistency",
        "profile_plausibility_score",
        "contact_consistency",
        "device_history_score",
        "lifecycle_behavior_coherence",
        "time_to_risky_activity",
    }
    assert set(event.attack_genome.keys()) == expected_dimensions


def test_genome_values_are_valid_and_bounded():
    """Verify genome values pass genome_engine.validate_genome and are in [0.0, 1.0]."""
    gen = SyntheticIdentityAttackGenerator()
    event = gen.generate(round_id="round-val-01")

    validate_genome(event.attack_genome)
    for dim, val in event.attack_genome.items():
        assert 0.0 <= val <= 1.0, f"Dimension {dim} value {val} out of bounds."


def test_deterministic_behavior_when_seeded():
    """Verify identical seeds produce identical AttackEvents and scenarios."""
    gen1 = SyntheticIdentityAttackGenerator(seed=12345)
    gen2 = SyntheticIdentityAttackGenerator(seed=12345)

    event1 = gen1.generate(round_id="round-det-01")
    event2 = gen2.generate(round_id="round-det-01")

    assert event1.attack_id == event2.attack_id
    assert event1.round_id == event2.round_id
    assert event1.attack_family == event2.attack_family
    assert event1.attack_genome == event2.attack_genome
    assert event1.scenario == event2.scenario
    assert event1.metadata == event2.metadata


def test_scenario_contains_valid_synthetic_identity_and_lifecycle():
    """Verify scenario parses into a valid SyntheticIdentity with structured lifecycle data."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="round-scenario-01")

    identity = SyntheticIdentity.model_validate(event.scenario)

    assert identity.identity_id.startswith("ident_f3_")
    assert "first_name" in identity.identity_attributes
    assert "last_name" in identity.identity_attributes
    assert "dob" in identity.identity_attributes
    assert "ssn_proxy" in identity.identity_attributes
    assert "primary_email" in identity.contact_attributes
    assert "phone_number" in identity.contact_attributes
    assert "account_id" in identity.account_metadata
    assert "account_open_date" in identity.account_metadata
    assert "primary_device_id" in identity.device_context
    assert "account_age_days" in identity.lifecycle_info
    assert "days_to_first_transaction" in identity.lifecycle_info
    assert "lifecycle_events_summary" in identity.lifecycle_info


def test_repeated_generation_for_different_round_ids():
    """Verify sequential generation produces round-specific attack IDs."""
    gen = SyntheticIdentityAttackGenerator(seed=999)

    round_ids = ["round-alpha", "round-beta", "round-gamma"]
    events = [gen.generate(rid) for rid in round_ids]

    for rid, event in zip(round_ids, events):
        assert event.round_id == rid
        assert event.attack_id == f"family3-attack-{rid}"
        assert event.attack_family == AttackFamily.SYNTHETIC_IDENTITY
        assert event.metadata["generator"] == "SyntheticIdentityAttackGenerator"


def test_set_genome_validation():
    """Verify set_genome enforces canonical keys and bounds."""
    gen = SyntheticIdentityAttackGenerator()

    # Valid replacement
    custom_genome = {
        "cross_field_consistency": 0.50,
        "profile_plausibility_score": 0.60,
        "contact_consistency": 0.40,
        "device_history_score": 0.30,
        "lifecycle_behavior_coherence": 0.70,
        "time_to_risky_activity": 0.80,
    }
    gen.set_genome(custom_genome)
    assert gen.genome == custom_genome

    # Missing dimension
    incomplete = dict(custom_genome)
    del incomplete["time_to_risky_activity"]
    with pytest.raises(ValueError, match="missing"):
        gen.set_genome(incomplete)

    # Extra non-Family 3 dimension
    extra = dict(custom_genome)
    extra["intent_amount_deviation"] = 0.5  # Family 2 dimension
    with pytest.raises(ValueError, match="non-Family 3"):
        gen.set_genome(extra)

    # Out of bounds
    out_of_bounds = dict(custom_genome)
    out_of_bounds["cross_field_consistency"] = 1.25
    with pytest.raises(ValueError, match="out of bounds"):
        gen.set_genome(out_of_bounds)


def test_attack_event_serialization_roundtrip():
    """Verify AttackEvent packaging and pydantic JSON serialization round-trip."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="round-rt-01")

    dumped = event.model_dump(mode="json")
    reconstructed = AttackEvent.model_validate(dumped)

    assert reconstructed.attack_id == event.attack_id
    assert reconstructed.round_id == event.round_id
    assert reconstructed.attack_family == AttackFamily.SYNTHETIC_IDENTITY
    assert reconstructed.attack_genome == event.attack_genome

    # Validate embedded scenario reconstructs to SyntheticIdentity
    scenario_obj = SyntheticIdentity.model_validate(reconstructed.scenario)
    assert scenario_obj.identity_id == event.scenario["identity_id"]


def test_genome_dimensions_modulate_scenario_characteristics():
    """Verify high vs low genome settings modulate generated scenario characteristics."""
    # Low score genome (high risk attack)
    gen_attack = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=42)
    event_attack = gen_attack.generate(round_id="round-atk-01")
    id_attack = SyntheticIdentity.model_validate(event_attack.scenario)

    # High score genome (legitimate/benign profile)
    gen_legit = SyntheticIdentityAttackGenerator(genome=DEFAULT_LEGITIMATE_GENOME, ground_truth=False, seed=42)
    event_legit = gen_legit.generate(round_id="round-legit-01")
    id_legit = SyntheticIdentity.model_validate(event_legit.scenario)

    # Low contact consistency -> disposable domain / VoIP
    assert any(disp in id_attack.contact_attributes["email_domain"] for disp in ["tempmail", "guerrilla", "throwaway", "burner", "10minute", "fakeinbox"])
    assert "VoIP" in id_attack.contact_attributes["phone_carrier_proxy"] or "Virtual" in id_attack.contact_attributes["phone_carrier_proxy"]

    # Low device history score -> emulator / proxy
    assert id_attack.device_context["trusted_device_count"] == 0
    assert id_attack.device_context["known_devices"][0]["is_emulator"] is True

    # Low time to risky activity -> early bust-out attempt
    assert id_attack.lifecycle_info["days_to_risky_activity"] in [1, 2, 3]
    assert id_attack.lifecycle_info["risk_event_count"] >= 1

    # High/legitimate genome -> high coherence, 0 risk events
    assert id_legit.device_context["trusted_device_count"] >= 1
    assert id_legit.lifecycle_info["risk_event_count"] == 0
    assert id_legit.lifecycle_info["days_to_risky_activity"] is None
