"""
tests/test_retraining_dataset.py — Unit and safety tests for Retraining Dataset Assembly (Task 7.2).

Validates:
1. Baseline-only dataset assembly.
2. Baseline + False Negatives assembly.
3. Baseline + False Negatives + Fresh Legitimate data assembly.
4. False-negative filtering & invalid record rejection.
5. Critical Holdout Exclusion (zero leakage from data/held_out/).
6. Family 1 (Transaction Evasion) dataset representation.
7. Family 2 (AI Agent Behavior) dataset representation.
8. Family 3 (Synthetic Identity) dataset representation.
9. Provenance tracking across all sample sources.
10. Deterministic sample ordering and indexing.
11. Duplicate failure handling and deduplication options.
12. Empty failure memory handling.
13. Empty optional fresh sample handling.
14. Input immutability & defensive isolation.
15. Serialization and replay (to_dict/from_dict, to_json/from_json).
16. Sample count properties and composition breakdowns.
"""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.round import RoundResult
from schemas.transaction import Transaction
from schemas.agent_event import AIAgentPaymentEvent
from schemas.identity import SyntheticIdentity

from data.generators.identity_generator import load_dataset
from blue_team.learning.failure_memory import (
    FailureRecord,
    FailureMemory,
)
from blue_team.learning.dataset import (
    ProvenanceType,
    DatasetAssemblyError,
    HoldoutDataLeakageError,
    DatasetSample,
    RetrainingDataset,
    RetrainingDatasetAssembler,
    assemble_retraining_dataset,
    validate_no_holdout_leakage,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def family1_failure_memory():
    """Create a FailureMemory with Family 1 false negatives."""
    event = AttackEvent(
        attack_id="atk-f1-001",
        round_id="round-f1-1",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome={
            "amount_deviation": 0.15,
            "velocity_deviation": 0.25,
            "device_novelty": 0.05,
            "location_deviation": 0.10,
            "time_deviation": 0.20,
            "sequence_anomaly": 0.08,
        },
        scenario={
            "transaction": {
                "transaction_id": "tx-101",
                "user_id": "user-42",
                "amount": 250.0,
                "currency": "USD",
                "merchant_id": "m-99",
            }
        },
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-f1-001",
        prediction=False,
        risk_score=0.28,
        model_version="xgb-f1-v1",
        feature_contributions={"amount_deviation": 0.12},
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-f1-001",
        round_reference="round-f1-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.28,
        important_features={"amount_deviation": 0.12},
    )
    mem = FailureMemory()
    mem.record_event(event, pred, fb)
    return mem


@pytest.fixture
def family2_failure_memory():
    """Create a FailureMemory with Family 2 false negatives."""
    event = AttackEvent(
        attack_id="atk-f2-002",
        round_id="round-f2-1",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={
            "intent_amount_deviation": 0.20,
            "intent_category_deviation": 0.10,
            "permission_scope_deviation": 0.15,
            "agent_identity_confidence": 0.85,
            "session_provenance_anomaly": 0.05,
            "purchase_velocity": 0.18,
        },
        scenario={
            "event_id": "agent-evt-202",
            "agent_identity": "procurement-bot-7",
            "actual_action": "purchase_gift_cards",
        },
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-f2-002",
        prediction=False,
        risk_score=0.35,
        model_version="lgb-f2-v1",
        feature_contributions={"intent_amount_deviation": 0.15},
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-f2-002",
        round_reference="round-f2-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.35,
        important_features={"intent_amount_deviation": 0.15},
    )
    mem = FailureMemory()
    mem.record_event(event, pred, fb)
    return mem


@pytest.fixture
def family3_failure_memory():
    """Create a FailureMemory with Family 3 false negatives."""
    event = AttackEvent(
        attack_id="atk-f3-003",
        round_id="round-f3-1",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome={
            "cross_field_consistency": 0.88,
            "profile_plausibility_score": 0.82,
            "contact_consistency": 0.90,
            "device_history_score": 0.75,
            "lifecycle_behavior_coherence": 0.80,
            "time_to_risky_activity": 0.70,
        },
        scenario={
            "identity_id": "syn-id-303",
            "identity_attributes": {"name": "Alex Mercer"},
        },
        ground_truth=True,
    )
    pred = PredictionResult(
        prediction_id="pred-f3-003",
        prediction=False,
        risk_score=0.42,
        model_version="syn-det-v1",
        feature_contributions={"cross_field_consistency": 0.10},
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-f3-003",
        round_reference="round-f3-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.42,
        important_features={"cross_field_consistency": 0.10},
    )
    mem = FailureMemory()
    mem.record_event(event, pred, fb)
    return mem


# ===========================================================================
# 1. Baseline-Only Dataset Assembly
# ===========================================================================

def test_baseline_only_dataset():
    """Assembling with only baseline data yields label 0 samples exclusively."""
    baseline_samples = [
        {"identity_id": "legit-001", "name": "Alice"},
        {"identity_id": "legit-002", "name": "Bob"},
    ]
    dataset = assemble_retraining_dataset(
        baseline_data=baseline_samples,
        family=AttackFamily.SYNTHETIC_IDENTITY,
    )

    assert dataset.total_count == 2
    assert dataset.legitimate_count == 2
    assert dataset.attack_count == 0
    assert dataset.baseline_count == 2
    assert dataset.false_negative_count == 0
    assert dataset.fresh_legitimate_count == 0
    assert dataset[0].sample_id == "legit-001"
    assert dataset[0].label == 0
    assert dataset[0].provenance == ProvenanceType.BASELINE_LEGITIMATE
    assert dataset[0].is_legitimate is True
    assert dataset[0].is_attack is False


# ===========================================================================
# 2. Baseline + False Negatives Assembly
# ===========================================================================

def test_baseline_plus_false_negatives(family1_failure_memory):
    """Combining baseline legitimate and false negatives properly sets labels 0 and 1."""
    baseline_tx = [
        Transaction(
            transaction_id="tx-legit-01",
            user_id="u1",
            timestamp=datetime(2026, 8, 29, 10, 0, tzinfo=timezone.utc),
            amount=45.0,
            currency="USD",
            merchant_id="m1",
            merchant_category="groceries",
            location="NYC",
            device_id="d1",
            payment_channel="pos",
        )
    ]
    dataset = assemble_retraining_dataset(
        baseline_data=baseline_tx,
        failure_memory=family1_failure_memory,
        family=AttackFamily.ADAPTIVE_EVASION,
    )

    assert dataset.total_count == 2
    assert dataset.legitimate_count == 1
    assert dataset.attack_count == 1
    assert dataset.baseline_count == 1
    assert dataset.false_negative_count == 1

    legit = dataset.get_legitimate_samples()[0]
    assert legit.sample_id == "tx-legit-01"
    assert legit.label == 0
    assert legit.provenance == ProvenanceType.BASELINE_LEGITIMATE

    atk = dataset.get_attack_samples()[0]
    assert atk.sample_id == "atk-f1-001"
    assert atk.label == 1
    assert atk.provenance == ProvenanceType.FALSE_NEGATIVE
    assert atk.features == {
        "amount_deviation": 0.15,
        "velocity_deviation": 0.25,
        "device_novelty": 0.05,
        "location_deviation": 0.10,
        "time_deviation": 0.20,
        "sequence_anomaly": 0.08,
    }


# ===========================================================================
# 3. Baseline + False Negatives + Fresh Legitimate Data
# ===========================================================================

def test_baseline_plus_failures_plus_fresh_legitimate(family2_failure_memory):
    """Dataset includes baseline, false negatives, and fresh legitimate samples."""
    baseline = [{"event_id": "base-evt-1", "user_intent": "buy office goods"}]
    fresh = [
        {"event_id": "fresh-evt-1", "user_intent": "book flight"},
        {"event_id": "fresh-evt-2", "user_intent": "order lunch"},
    ]

    dataset = assemble_retraining_dataset(
        baseline_data=baseline,
        failure_memory=family2_failure_memory,
        fresh_legitimate_data=fresh,
        family=AttackFamily.AGENT_BEHAVIOR,
    )

    assert dataset.total_count == 4
    assert dataset.baseline_count == 1
    assert dataset.false_negative_count == 1
    assert dataset.fresh_legitimate_count == 2
    assert dataset.legitimate_count == 3
    assert dataset.attack_count == 1

    fresh_samples = dataset.get_by_provenance(ProvenanceType.FRESH_LEGITIMATE)
    assert len(fresh_samples) == 2
    assert fresh_samples[0].sample_id == "fresh-evt-1"
    assert fresh_samples[1].sample_id == "fresh-evt-2"
    assert all(s.label == 0 for s in fresh_samples)


# ===========================================================================
# 4. False-Negative Filtering & Invalid Record Rejection
# ===========================================================================

def test_reject_non_false_negative_record():
    """Assembler rejects records that do not satisfy false-negative requirements."""
    # A true positive (detected) record
    invalid_record = FailureRecord(
        round_id="r-1",
        attack_id="atk-invalid",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        risk_score=0.90,
        prediction=True,
        ground_truth=True,
        detected=True,
        false_negative=False,
    )
    with pytest.raises(DatasetAssemblyError, match="not a valid false negative"):
        assemble_retraining_dataset(
            baseline_data=[{"id": "b1"}],
            failure_memory=[invalid_record],
        )


def test_reject_invalid_object_type_in_failure_memory():
    """Assembler rejects non-FailureRecord objects in failure sequence."""
    with pytest.raises(DatasetAssemblyError, match="Expected FailureRecord instance"):
        assemble_retraining_dataset(
            baseline_data=[{"id": "b1"}],
            failure_memory=["not_a_record"],  # type: ignore
        )


# ===========================================================================
# 5. Critical Holdout Exclusion
# ===========================================================================

def test_heldout_path_rejected_as_baseline():
    """Passing a file path referencing held_out raises HoldoutDataLeakageError immediately."""
    heldout_path = Path("data/held_out/heldout_identities.json")
    with pytest.raises(HoldoutDataLeakageError, match="held-out evaluation path"):
        assemble_retraining_dataset(baseline_data=[heldout_path])


def test_heldout_identities_detected_and_blocked():
    """If samples from heldout_identities.json are in training data, leakage check triggers."""
    heldout_path = Path("data/held_out/heldout_identities.json")
    if heldout_path.exists():
        heldout_identities = load_dataset(heldout_path)
        heldout_sample = heldout_identities[0]

        # Attempt to pass a held-out identity as baseline training data
        with pytest.raises(HoldoutDataLeakageError, match="Holdout data leakage detected"):
            assemble_retraining_dataset(
                baseline_data=[heldout_sample],
                held_out_data=heldout_identities,
            )


def test_canonical_baseline_has_zero_heldout_leakage():
    """Verify that canonical baseline data passes strict holdout validation against heldout set."""
    baseline_path = Path("data/legitimate/baseline_identities.json")
    heldout_path = Path("data/held_out/heldout_identities.json")

    if baseline_path.exists() and heldout_path.exists():
        baseline_dataset = load_dataset(baseline_path)
        heldout_dataset = load_dataset(heldout_path)

        dataset = assemble_retraining_dataset(
            baseline_data=baseline_dataset[:50],
            family=AttackFamily.SYNTHETIC_IDENTITY,
            held_out_data=heldout_dataset,
        )

        assert dataset.total_count == 50
        assert dataset.baseline_count == 50
        # Explicit validation should pass cleanly
        validate_no_holdout_leakage(dataset, heldout_dataset)


# ===========================================================================
# 6. Family 1, Family 2, Family 3 Representation
# ===========================================================================

def test_family1_transaction_assembly(family1_failure_memory):
    """Family 1 transactions and failures are properly represented."""
    dataset = assemble_retraining_dataset(
        baseline_data=[{"transaction_id": "tx-legit-f1", "amount": 20.0}],
        failure_memory=family1_failure_memory,
        family=AttackFamily.ADAPTIVE_EVASION,
    )
    assert dataset.total_count == 2
    f1_samples = dataset.get_by_family(AttackFamily.ADAPTIVE_EVASION)
    assert len(f1_samples) == 2


def test_family2_agent_assembly(family2_failure_memory):
    """Family 2 agent events and failures are properly represented."""
    dataset = assemble_retraining_dataset(
        baseline_data=[{"event_id": "agent-legit-f2", "actual_action": "auth_scope"}],
        failure_memory=family2_failure_memory,
        family=AttackFamily.AGENT_BEHAVIOR,
    )
    assert dataset.total_count == 2
    f2_samples = dataset.get_by_family(AttackFamily.AGENT_BEHAVIOR)
    assert len(f2_samples) == 2


def test_family3_identity_assembly(family3_failure_memory):
    """Family 3 synthetic identities and failures are properly represented."""
    dataset = assemble_retraining_dataset(
        baseline_data=[{"identity_id": "ident-legit-f3", "kyc_status": "verified"}],
        failure_memory=family3_failure_memory,
        family=AttackFamily.SYNTHETIC_IDENTITY,
    )
    assert dataset.total_count == 2
    f3_samples = dataset.get_by_family(AttackFamily.SYNTHETIC_IDENTITY)
    assert len(f3_samples) == 2


# ===========================================================================
# 7. Deterministic Ordering & Duplicate Handling
# ===========================================================================

def test_deterministic_ordering(family1_failure_memory, family2_failure_memory):
    """Dataset assembly produces deterministic, identical sample sequence."""
    base = [{"id": "b-1"}, {"id": "b-2"}]
    mem = FailureMemory()
    mem.record_failure(family1_failure_memory[0])
    mem.record_failure(family2_failure_memory[0])

    ds1 = assemble_retraining_dataset(baseline_data=base, failure_memory=mem)
    ds2 = assemble_retraining_dataset(baseline_data=base, failure_memory=mem)

    assert ds1.get_sample_ids() == ds2.get_sample_ids()
    assert ds1.to_dict() == ds2.to_dict()


def test_deduplicate_failures_option(family1_failure_memory):
    """Deduplicating failures preserves the first occurrence deterministically."""
    rec = family1_failure_memory[0]
    mem = FailureMemory([rec, rec, rec])
    assert len(mem) == 3

    # Without deduplication
    ds_raw = assemble_retraining_dataset(failure_memory=mem, deduplicate_failures=False)
    assert ds_raw.total_count == 3

    # With deduplication
    ds_dedup = assemble_retraining_dataset(failure_memory=mem, deduplicate_failures=True)
    assert ds_dedup.total_count == 1
    assert ds_dedup[0].sample_id == rec.attack_id


# ===========================================================================
# 8. Empty Inputs & Edge Cases
# ===========================================================================

def test_empty_failure_memory_and_empty_fresh():
    """Empty failure memory and empty fresh data produces clean baseline dataset."""
    base = [{"id": "b1"}, {"id": "b2"}]
    mem = FailureMemory()

    dataset = assemble_retraining_dataset(
        baseline_data=base,
        failure_memory=mem,
        fresh_legitimate_data=[],
    )
    assert dataset.total_count == 2
    assert dataset.attack_count == 0
    assert dataset.fresh_legitimate_count == 0


def test_completely_empty_dataset():
    """Completely empty inputs yield a valid empty RetrainingDataset."""
    dataset = assemble_retraining_dataset()
    assert dataset.total_count == 0
    assert len(dataset) == 0
    assert dataset.legitimate_count == 0
    assert dataset.attack_count == 0
    assert dataset.to_dict() == []
    assert dataset.to_json() == "[]"


# ===========================================================================
# 9. Input Immutability & Defensive Isolation
# ===========================================================================

def test_input_immutability(family1_failure_memory):
    """Mutating input baseline list or failure records does not alter assembled dataset."""
    base_dict = {"identity_id": "b-safe", "val": 100}
    base_list = [base_dict]

    dataset = assemble_retraining_dataset(
        baseline_data=base_list,
        failure_memory=family1_failure_memory,
    )

    # Mutate source list and dict
    base_list.append({"identity_id": "b-new"})
    base_dict["val"] = 999
    base_dict["mutated"] = True

    # Mutate failure memory record outside
    family1_failure_memory[0].attack_genome["amount_deviation"] = 0.999

    assert dataset.total_count == 2
    assert dataset[0].data["val"] == 100
    assert "mutated" not in dataset[0].data
    assert dataset[1].features["amount_deviation"] == 0.15


# ===========================================================================
# 10. Serialization and Replay
# ===========================================================================

def test_to_dict_and_from_dict(family1_failure_memory):
    """Dataset serializes to dict and restores losslessly."""
    base = [{"identity_id": "b-100", "score": 0.1}]
    ds = assemble_retraining_dataset(
        baseline_data=base,
        failure_memory=family1_failure_memory,
        name="test_ds",
    )

    dict_data = ds.to_dict()
    assert isinstance(dict_data, list)
    assert len(dict_data) == 2

    restored = RetrainingDataset.from_dict(dict_data, name="test_ds")
    assert restored.total_count == 2
    assert restored.baseline_count == 1
    assert restored.false_negative_count == 1
    assert restored[0].sample_id == "b-100"
    assert restored[1].sample_id == "atk-f1-001"


def test_to_json_and_from_json(family2_failure_memory):
    """Dataset serializes to JSON string and restores faithfully."""
    base = [{"event_id": "agent-evt-99", "status": "ok"}]
    ds = assemble_retraining_dataset(
        baseline_data=base,
        failure_memory=family2_failure_memory,
    )

    json_str = ds.to_json(indent=2)
    assert isinstance(json_str, str)
    assert "agent-evt-99" in json_str

    restored = RetrainingDataset.from_json(json_str)
    assert restored.total_count == 2
    assert restored.get_sample_ids() == ["agent-evt-99", "atk-f2-002"]


# ===========================================================================
# 11. Family Counts Breakdown
# ===========================================================================

def test_family_counts_breakdown(
    family1_failure_memory,
    family2_failure_memory,
    family3_failure_memory,
):
    """Verify family counts distribution across all samples."""
    mem = FailureMemory()
    mem.record_failure(family1_failure_memory[0])
    mem.record_failure(family2_failure_memory[0])
    mem.record_failure(family3_failure_memory[0])

    dataset = assemble_retraining_dataset(
        baseline_data=[
            {"id": "b1", "family": AttackFamily.ADAPTIVE_EVASION},
            {"id": "b2", "family": AttackFamily.AGENT_BEHAVIOR},
        ],
        failure_memory=mem,
    )

    assert dataset.total_count == 5
    counts = dataset.family_counts
    assert counts[AttackFamily.ADAPTIVE_EVASION.value] == 2
    assert counts[AttackFamily.AGENT_BEHAVIOR.value] == 2
    assert counts[AttackFamily.SYNTHETIC_IDENTITY.value] == 1
