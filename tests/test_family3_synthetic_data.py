"""
tests/test_family3_synthetic_data.py -- Test suite for Family 3 synthetic legitimate environment.

Verifies:
- Deterministic generation with identical seeds.
- Discrepancy / uniqueness with different seeds.
- Record count adherence.
- Schema compliance with schemas.identity.SyntheticIdentity.
- Attribute coherence across demographic, contact, account, device, and lifecycle domains.
- Absence of attack variants or fraud labels in baseline training data.
- Dataset persistence (save/load round-trips).
- Integrity and presence of the baseline dataset in data/legitimate/baseline_identities.json.
"""

from datetime import date
from pathlib import Path
import pytest
from faker.providers.address.en_US import Provider

from data.generators.identity_generator import (
    LegitimateIdentityGenerator,
    calculate_age_at_date,
    generate_legitimate_baseline,
    load_dataset,
    save_dataset,
)
from schemas.identity import SyntheticIdentity
from schemas.attack import AttackEvent


def test_deterministic_generation_same_seed():
    """Two generators with the same seed produce byte-for-byte identical datasets."""
    gen1 = LegitimateIdentityGenerator(seed=42)
    gen2 = LegitimateIdentityGenerator(seed=42)

    records1 = gen1.generate_dataset(n=10)
    records2 = gen2.generate_dataset(n=10)

    assert len(records1) == len(records2) == 10
    for r1, r2 in zip(records1, records2):
        assert r1.identity_id == r2.identity_id
        assert r1.identity_attributes == r2.identity_attributes
        assert r1.contact_attributes == r2.contact_attributes
        assert r1.account_metadata == r2.account_metadata
        assert r1.device_context == r2.device_context
        assert r1.lifecycle_info == r2.lifecycle_info


def test_different_seed_produces_different_dataset():
    """Generators with different seeds produce distinct identities."""
    gen1 = LegitimateIdentityGenerator(seed=42)
    gen2 = LegitimateIdentityGenerator(seed=99)

    records1 = gen1.generate_dataset(n=5)
    records2 = gen2.generate_dataset(n=5)

    assert len(records1) == len(records2) == 5
    assert records1[0].identity_id != records2[0].identity_id
    assert records1[0].identity_attributes["first_name"] != records2[0].identity_attributes["first_name"] or \
           records1[0].identity_attributes["last_name"] != records2[0].identity_attributes["last_name"]
    assert records1[0].contact_attributes["primary_email"] != records2[0].contact_attributes["primary_email"]


def test_requested_count_is_respected():
    """Generator respects requested count N."""
    gen = LegitimateIdentityGenerator(seed=123)

    for n in [1, 5, 20]:
        records = gen.generate_dataset(n=n)
        assert len(records) == n
        for idx, rec in enumerate(records):
            assert isinstance(rec, SyntheticIdentity)
            assert rec.identity_id == f"ident_{123:04d}_{idx:05d}"


def test_invalid_requested_count_raises():
    """Generator raises ValueError on invalid dataset size."""
    gen = LegitimateIdentityGenerator(seed=123)
    with pytest.raises(ValueError):
        gen.generate_dataset(n=0)
    with pytest.raises(ValueError):
        gen.generate_dataset(n=-5)


def test_records_validate_synthetic_identity_schema():
    """Generated records validate against schemas.identity.SyntheticIdentity."""
    gen = LegitimateIdentityGenerator(seed=42)
    records = gen.generate_dataset(n=10)

    for rec in records:
        assert isinstance(rec, SyntheticIdentity)
        dump = rec.model_dump()
        # Verify schema round-trip
        reconstructed = SyntheticIdentity(**dump)
        assert reconstructed.identity_id == rec.identity_id


def test_demographic_attributes_coherence():
    """Demographic attributes are internally coherent."""
    gen = LegitimateIdentityGenerator(seed=42)
    records = gen.generate_dataset(n=25)

    for rec in records:
        attrs = rec.identity_attributes
        assert "first_name" in attrs and len(attrs["first_name"]) > 0
        assert "last_name" in attrs and len(attrs["last_name"]) > 0
        assert "full_name" in attrs and f"{attrs['first_name']} {attrs['last_name']}" == attrs["full_name"]
        assert "dob" in attrs
        assert "age" in attrs
        assert 20 <= attrs["age"] <= 75
        # Verify exact DOB-derived age
        assert calculate_age_at_date(attrs["dob"], LegitimateIdentityGenerator.REFERENCE_DATE) == attrs["age"]
        assert "ssn_proxy" in attrs and attrs["ssn_proxy"].startswith("9")  # Synthetic SSN format
        assert "employment_status" in attrs and attrs["employment_status"] in [
            "employed", "self-employed", "retired", "student"
        ]
        assert "annual_income" in attrs and attrs["annual_income"] > 0
        assert "credit_score_proxy" in attrs and 600 <= attrs["credit_score_proxy"] <= 850


def test_dob_and_age_exact_coherence_relative_to_reference_date():
    """Proves DOB-derived age exactly matches the stored age relative to the reference date across seeds."""
    for seed in [1, 42, 99, 2026]:
        gen = LegitimateIdentityGenerator(seed=seed)
        records = gen.generate_dataset(n=50)

        for rec in records:
            attrs = rec.identity_attributes
            dob_str = attrs["dob"]
            stored_age = attrs["age"]
            calculated_age = calculate_age_at_date(dob_str, LegitimateIdentityGenerator.REFERENCE_DATE)
            assert calculated_age == stored_age, (
                f"Mismatch for identity {rec.identity_id}: DOB={dob_str}, stored_age={stored_age}, "
                f"calculated_age={calculated_age} on reference date {LegitimateIdentityGenerator.REFERENCE_DATE}"
            )


def test_contact_attributes_coherence():
    """Contact attributes align with demographic profile and location."""
    gen = LegitimateIdentityGenerator(seed=42)
    records = gen.generate_dataset(n=25)

    for rec in records:
        demog = rec.identity_attributes
        contacts = rec.contact_attributes

        assert "primary_email" in contacts
        assert "email_domain" in contacts and len(contacts["email_domain"]) > 0
        first_initial = demog["first_name"][0].lower()
        assert first_initial in contacts["primary_email"]
        assert contacts["email_age_years"] >= 2.0
        assert "phone_number" in contacts and len(contacts["phone_number"]) > 0
        assert "phone_carrier_proxy" in contacts
        assert "street_address" in contacts
        assert "city" in contacts
        assert "state" in contacts
        assert "zip_code" in contacts
        assert contacts["country"] == "USA"
        # Validate postal code is within state's valid range in Faker Provider
        assert contacts["state"] in Provider.states_postcode
        min_zip, max_zip = Provider.states_postcode[contacts["state"]]
        assert min_zip <= int(contacts["zip_code"]) <= max_zip
        assert len(contacts["zip_code"]) == 5
        assert 0.8 <= contacts["contact_stability_score"] <= 1.0


def test_location_city_state_zip_coherence():
    """Proves city, state, and ZIP are mutually coherent and aligned with device context."""
    for seed in [10, 42, 100]:
        gen = LegitimateIdentityGenerator(seed=seed)
        records = gen.generate_dataset(n=30)

        for rec in records:
            contacts = rec.contact_attributes
            dev = rec.device_context

            state = contacts["state"]
            zip_code = contacts["zip_code"]

            assert state in Provider.known_usps_abbr
            assert state in Provider.states_postcode
            min_zip, max_zip = Provider.states_postcode[state]
            assert min_zip <= int(zip_code) <= max_zip
            assert len(zip_code) == 5

            # Device context IP location must match residential location
            assert dev["ip_city"] == contacts["city"]
            assert dev["ip_state"] == contacts["state"]


def test_account_metadata_coherence():
    """Account metadata contains proper lifecycle fields for legitimate accounts."""
    gen = LegitimateIdentityGenerator(seed=42)
    records = gen.generate_dataset(n=25)

    for rec in records:
        meta = rec.account_metadata
        assert "account_id" in meta and len(meta["account_id"]) > 0
        assert "account_open_date" in meta
        assert "account_age_days" in meta and meta["account_age_days"] >= 30
        assert meta["currency"] == "USD"
        assert meta["initial_deposit"] >= 100.0
        assert meta["kyc_verification_status"] == "verified"
        assert meta["kyc_level"] == "level_2_full"
        assert meta["account_status"] == "active"


def test_device_context_coherence():
    """Device context reflects legitimate device history and geographic proximity."""
    gen = LegitimateIdentityGenerator(seed=42)
    records = gen.generate_dataset(n=25)

    for rec in records:
        dev = rec.device_context
        contacts = rec.contact_attributes
        assert "primary_device_id" in dev and len(dev["primary_device_id"]) > 0
        assert "os" in dev and "browser" in dev
        assert "ip_address" in dev and len(dev["ip_address"]) > 0
        # IP city and state match residential contact location
        assert dev["ip_city"] == contacts["city"]
        assert dev["ip_state"] == contacts["state"]
        assert dev["trusted_device_count"] >= 1
        assert dev["device_consistency_ratio"] >= 0.85
        assert len(dev["known_devices"]) >= 1


def test_lifecycle_info_longitudinal_coherence():
    """Lifecycle info models normal longitudinal progression without fraud artifacts."""
    gen = LegitimateIdentityGenerator(seed=42)
    records = gen.generate_dataset(n=25)

    for rec in records:
        lifecycle = rec.lifecycle_info
        account_meta = rec.account_metadata

        assert lifecycle["account_age_days"] == account_meta["account_age_days"]
        assert 1 <= lifecycle["days_to_first_transaction"] <= 7
        assert lifecycle["days_to_first_transaction"] <= lifecycle["account_age_days"]
        assert lifecycle["login_frequency_per_month"] > 0
        assert lifecycle["transaction_count"] >= 1
        assert lifecycle["avg_transaction_amount"] > 0
        assert lifecycle["total_spend"] > 0
        # Legitimate baseline has zero risk events and high coherence
        assert lifecycle["risk_event_count"] == 0
        assert lifecycle["days_to_risky_activity"] is None
        assert lifecycle["lifecycle_coherence_score"] >= 0.85
        assert len(lifecycle["lifecycle_events_summary"]) >= 3


def test_no_attack_objects_or_fraud_labels_in_baseline():
    """Legitimate baseline dataset contains zero AttackEvent or attack-family structures."""
    gen = LegitimateIdentityGenerator(seed=42)
    records = gen.generate_dataset(n=10)

    for rec in records:
        # Must be a pure SyntheticIdentity, not an AttackEvent
        assert not isinstance(rec, AttackEvent)
        assert isinstance(rec, SyntheticIdentity)
        dump = rec.model_dump()
        assert "attack_family" not in dump
        assert "attack_genome" not in dump
        assert "ground_truth" not in dump


def test_save_and_load_dataset(tmp_path: Path):
    """Saving and loading dataset preserves all fields and SyntheticIdentity types."""
    gen = LegitimateIdentityGenerator(seed=77)
    original_dataset = gen.generate_dataset(n=15)

    file_path = tmp_path / "test_identities.json"
    save_dataset(original_dataset, file_path)

    assert file_path.exists()
    loaded_dataset = load_dataset(file_path)

    assert len(loaded_dataset) == len(original_dataset)
    for orig, loaded in zip(original_dataset, loaded_dataset):
        assert isinstance(loaded, SyntheticIdentity)
        assert orig.identity_id == loaded.identity_id
        assert orig.identity_attributes == loaded.identity_attributes
        assert orig.contact_attributes == loaded.contact_attributes
        assert orig.account_metadata == loaded.account_metadata
        assert orig.device_context == loaded.device_context
        assert orig.lifecycle_info == loaded.lifecycle_info


def test_baseline_dataset_file_integrity():
    """Verifies that the generated baseline dataset under data/legitimate/ is valid."""
    baseline_path = Path("data/legitimate/baseline_identities.json")
    assert baseline_path.exists(), "Baseline dataset file data/legitimate/baseline_identities.json does not exist."

    dataset = load_dataset(baseline_path)
    assert len(dataset) == 500, f"Expected 500 baseline records, got {len(dataset)}"

    for rec in dataset:
        assert isinstance(rec, SyntheticIdentity)
        assert rec.identity_id.startswith("ident_")
        assert rec.account_metadata["kyc_verification_status"] == "verified"
        assert rec.lifecycle_info["risk_event_count"] == 0
