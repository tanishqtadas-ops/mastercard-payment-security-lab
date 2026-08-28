"""
tests/test_failure_memory.py — Unit and integration tests for Blue-Team Failure Memory (Task 7.1).

Validates:
1. Adding a false negative (ground_truth=True, prediction=False).
2. Refusing/ignoring non-false-negatives (true positive, true negative, false positive).
3. Representation across all three attack families (Family 1, Family 2, Family 3).
4. Stored genome preservation (exact float values and dimensions).
5. Feature contributions preservation (SHAP / important features).
6. Risk score preservation.
7. Round and attack identity preservation.
8. Insertion order and deterministic retrieval.
9. Multiple false negatives accumulating across rounds.
10. Defensive isolation / input objects not mutated and subsequent external changes not affecting memory.
11. Serialization and replay behavior (to_dict/from_dict, to_json/from_json).
12. Empty memory behavior and clean queries.
13. Ingesting from RoundResult objects.
"""

import copy
import json
import pytest

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.round import RoundResult
from schemas.transaction import Transaction
from schemas.agent_event import AIAgentPaymentEvent
from schemas.identity import SyntheticIdentity

from blue_team.learning.failure_memory import (
    FailureRecord,
    FailureMemory,
    is_false_negative,
)


# ===========================================================================
# Fixtures & Helpers
# ===========================================================================

@pytest.fixture
def family1_false_negative():
    """Create a Family 1 false negative (missed transaction evasion attack)."""
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
        metadata={"attack_type": "transaction_evasion", "seed": 42},
    )
    prediction = PredictionResult(
        prediction_id="pred-f1-001",
        prediction=False,
        risk_score=0.28,
        model_version="xgb-f1-v1",
        explanation="Risk score 0.28 below threshold 0.50",
        feature_contributions={"amount_deviation": 0.12, "velocity_deviation": 0.16},
    )
    feedback = BlueTeamFeedback(
        feedback_id="fb-f1-001",
        round_reference="round-f1-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.28,
        important_features={"amount_deviation": 0.12, "velocity_deviation": 0.16},
        explanation_data={"ground_truth": True, "prediction": False},
    )
    return event, prediction, feedback


@pytest.fixture
def family2_false_negative():
    """Create a Family 2 false negative (missed malicious AI agent action)."""
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
        metadata={"agent_type": "autonomous", "tier": "enterprise"},
    )
    prediction = PredictionResult(
        prediction_id="pred-f2-002",
        prediction=False,
        risk_score=0.35,
        model_version="lgb-f2-v1",
        explanation="Agent action aligned with estimated scope",
        feature_contributions={"intent_amount_deviation": 0.15, "permission_scope_deviation": 0.20},
    )
    feedback = BlueTeamFeedback(
        feedback_id="fb-f2-002",
        round_reference="round-f2-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.35,
        important_features={"intent_amount_deviation": 0.15, "permission_scope_deviation": 0.20},
        explanation_data={"ground_truth": True, "prediction": False},
    )
    return event, prediction, feedback


@pytest.fixture
def family3_false_negative():
    """Create a Family 3 false negative (missed synthetic identity fraud)."""
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
            "identity_attributes": {"name": "Alex Mercer", "ssn_match": True},
        },
        ground_truth=True,
        metadata={"fabrication_mode": "composite", "depth": 3},
    )
    prediction = PredictionResult(
        prediction_id="pred-f3-003",
        prediction=False,
        risk_score=0.42,
        model_version="syn-det-v1",
        explanation="Identity profile consistency appears normal",
        feature_contributions={"cross_field_consistency": 0.10, "profile_plausibility_score": 0.15},
    )
    feedback = BlueTeamFeedback(
        feedback_id="fb-f3-003",
        round_reference="round-f3-1",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.42,
        important_features={"cross_field_consistency": 0.10, "profile_plausibility_score": 0.15},
        explanation_data={"ground_truth": True, "prediction": False},
    )
    return event, prediction, feedback


# ===========================================================================
# 1. False Negative Detection & Adding
# ===========================================================================

def test_add_false_negative_event(family1_false_negative):
    """Test adding a false negative via record_event."""
    event, prediction, feedback = family1_false_negative
    memory = FailureMemory()

    assert is_false_negative(event, prediction, feedback) is True
    recorded = memory.record_event(event, prediction, feedback)

    assert recorded is True
    assert len(memory) == 1
    assert memory.count == 1
    assert not memory.is_empty


def test_add_false_negative_round_result(family1_false_negative):
    """Test adding a false negative via record_round with RoundResult."""
    event, prediction, feedback = family1_false_negative
    round_result = RoundResult(
        round_id="round-f1-1",
        attack_event=event,
        prediction_result=prediction,
        feedback=feedback,
        outcome_metrics={"round_index": 1, "duration_ms": 12.5},
    )
    memory = FailureMemory()
    recorded = memory.record_round(round_result)

    assert recorded is True
    assert len(memory) == 1
    record = memory[0]
    assert record.round_id == "round-f1-1"
    assert record.attack_id == "atk-f1-001"
    assert record.metadata.get("outcome_metrics") == {"round_index": 1, "duration_ms": 12.5}


def test_polymorphic_record_helper(family1_false_negative):
    """Test the polymorphic record() method."""
    event, prediction, feedback = family1_false_negative
    round_result = RoundResult(
        round_id="round-f1-1",
        attack_event=event,
        prediction_result=prediction,
        feedback=feedback,
    )
    memory = FailureMemory()

    # Record via RoundResult
    assert memory.record(round_result) is True
    assert len(memory) == 1

    # Record via FailureRecord
    rec = FailureRecord.from_components(event, prediction, feedback)
    assert memory.record(rec) is True
    assert len(memory) == 2

    # Invalid type
    with pytest.raises(TypeError):
        memory.record("invalid_type")  # type: ignore


# ===========================================================================
# 2. Refusing / Ignoring Non-False-Negatives
# ===========================================================================

def test_refuse_true_positive(family1_false_negative):
    """True Positive (detected attack) must be refused/ignored."""
    event, _, _ = family1_false_negative
    # Detector successfully detected fraud
    pred = PredictionResult(
        prediction_id="pred-tp",
        prediction=True,
        risk_score=0.88,
        model_version="xgb-v1",
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-tp",
        round_reference="round-f1-1",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.88,
        important_features={},
    )
    memory = FailureMemory()
    assert is_false_negative(event, pred, fb) is False
    assert memory.record_event(event, pred, fb) is False
    assert len(memory) == 0


def test_refuse_true_negative():
    """True Negative (legitimate transaction correctly predicted as non-fraud) must be refused."""
    event = AttackEvent(
        attack_id="legit-001",
        round_id="round-legit-1",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome={"amount_deviation": 0.02},
        scenario={},
        ground_truth=False,
    )
    pred = PredictionResult(
        prediction_id="pred-tn",
        prediction=False,
        risk_score=0.05,
        model_version="xgb-v1",
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-tn",
        round_reference="round-legit-1",
        detected=False,
        false_positive=False,
        false_negative=False,
        risk_score=0.05,
        important_features={},
    )
    memory = FailureMemory()
    assert is_false_negative(event, pred, fb) is False
    assert memory.record_event(event, pred, fb) is False
    assert len(memory) == 0


def test_refuse_false_positive():
    """False Positive (legitimate transaction flagged as fraud) must be refused."""
    event = AttackEvent(
        attack_id="legit-002",
        round_id="round-legit-2",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome={"amount_deviation": 0.05},
        scenario={},
        ground_truth=False,
    )
    pred = PredictionResult(
        prediction_id="pred-fp",
        prediction=True,
        risk_score=0.75,
        model_version="xgb-v1",
    )
    fb = BlueTeamFeedback(
        feedback_id="fb-fp",
        round_reference="round-legit-2",
        detected=False,
        false_positive=True,
        false_negative=False,
        risk_score=0.75,
        important_features={},
    )
    memory = FailureMemory()
    assert is_false_negative(event, pred, fb) is False
    assert memory.record_event(event, pred, fb) is False
    assert len(memory) == 0


# ===========================================================================
# 3. All Three Families Can Be Represented
# ===========================================================================

def test_all_three_families_representation(
    family1_false_negative,
    family2_false_negative,
    family3_false_negative,
):
    """Verify Family 1, Family 2, and Family 3 false negatives are properly preserved."""
    e1, p1, f1 = family1_false_negative
    e2, p2, f2 = family2_false_negative
    e3, p3, f3 = family3_false_negative

    memory = FailureMemory()
    assert memory.record_event(e1, p1, f1) is True
    assert memory.record_event(e2, p2, f2) is True
    assert memory.record_event(e3, p3, f3) is True

    assert len(memory) == 3

    # Query by family
    f1_records = memory.get_by_family(AttackFamily.ADAPTIVE_EVASION)
    assert len(f1_records) == 1
    assert f1_records[0].attack_id == "atk-f1-001"
    assert "amount_deviation" in f1_records[0].attack_genome

    f2_records = memory.get_by_family(AttackFamily.AGENT_BEHAVIOR)
    assert len(f2_records) == 1
    assert f2_records[0].attack_id == "atk-f2-002"
    assert "intent_amount_deviation" in f2_records[0].attack_genome

    f3_records = memory.get_by_family(AttackFamily.SYNTHETIC_IDENTITY)
    assert len(f3_records) == 1
    assert f3_records[0].attack_id == "atk-f3-003"
    assert "cross_field_consistency" in f3_records[0].attack_genome


# ===========================================================================
# 4. Genome & Feature Contributions Preservation
# ===========================================================================

def test_genome_preservation(family1_false_negative):
    """Genome dimensions and floating point values must be preserved exactly."""
    event, prediction, feedback = family1_false_negative
    memory = FailureMemory()
    memory.record_event(event, prediction, feedback)

    record = memory.get_failures()[0]
    expected_genome = {
        "amount_deviation": 0.15,
        "velocity_deviation": 0.25,
        "device_novelty": 0.05,
        "location_deviation": 0.10,
        "time_deviation": 0.20,
        "sequence_anomaly": 0.08,
    }
    assert record.attack_genome == expected_genome
    assert memory.get_genomes() == [expected_genome]


def test_feature_contributions_and_risk_score_preservation(family2_false_negative):
    """Feature contributions and risk score must be accurately preserved."""
    event, prediction, feedback = family2_false_negative
    memory = FailureMemory()
    memory.record_event(event, prediction, feedback)

    record = memory.get_failures()[0]
    assert record.risk_score == pytest.approx(0.35)
    assert record.feature_contributions == {
        "intent_amount_deviation": 0.15,
        "permission_scope_deviation": 0.20,
    }
    assert record.important_features == {
        "intent_amount_deviation": 0.15,
        "permission_scope_deviation": 0.20,
    }
    assert record.model_version == "lgb-f2-v1"


# ===========================================================================
# 5. Round & Attack Identity Preservation
# ===========================================================================

def test_round_and_attack_identity(family3_false_negative):
    """Round and attack identifiers must be queryable and preserved."""
    event, prediction, feedback = family3_false_negative
    memory = FailureMemory()
    memory.record_event(event, prediction, feedback)

    by_round = memory.get_by_round_id("round-f3-1")
    assert by_round is not None
    assert by_round.round_id == "round-f3-1"
    assert by_round.attack_id == "atk-f3-003"

    by_attack = memory.get_by_attack_id("atk-f3-003")
    assert by_attack is not None
    assert by_attack.attack_id == "atk-f3-003"

    assert memory.get_by_round_id("non-existent") is None
    assert memory.get_by_attack_id("non-existent") is None


# ===========================================================================
# 6. Insertion Order & Accumulation
# ===========================================================================

def test_insertion_order_and_accumulation(
    family1_false_negative,
    family2_false_negative,
    family3_false_negative,
):
    """Memory must accumulate records strictly preserving insertion order."""
    e1, p1, f1 = family1_false_negative
    e2, p2, f2 = family2_false_negative
    e3, p3, f3 = family3_false_negative

    memory = FailureMemory()
    memory.record_event(e3, p3, f3)
    memory.record_event(e1, p1, f1)
    memory.record_event(e2, p2, f2)

    failures = memory.get_failures()
    assert len(failures) == 3
    assert failures[0].round_id == "round-f3-1"
    assert failures[1].round_id == "round-f1-1"
    assert failures[2].round_id == "round-f2-1"

    # Iteration behaves deterministically
    round_ids = [r.round_id for r in memory]
    assert round_ids == ["round-f3-1", "round-f1-1", "round-f2-1"]


# ===========================================================================
# 7. Defensive Isolation / Immutability
# ===========================================================================

def test_input_objects_not_mutated_and_defensive_copies(family1_false_negative):
    """Mutating input event or genome after recording must not alter memory contents."""
    event, prediction, feedback = family1_false_negative
    original_genome = dict(event.attack_genome)
    original_scenario = copy.deepcopy(event.scenario)

    memory = FailureMemory()
    memory.record_event(event, prediction, feedback)

    # Mutate the source event outside
    event.attack_genome["amount_deviation"] = 0.999
    event.attack_genome["new_key"] = 1.0
    event.scenario["transaction"]["amount"] = 999999.0
    event.metadata["mutated"] = True

    # Check stored record
    stored = memory[0]
    assert stored.attack_genome["amount_deviation"] == 0.15
    assert "new_key" not in stored.attack_genome
    assert stored.attack_genome == original_genome
    assert stored.scenario["transaction"]["amount"] == 250.0
    assert "mutated" not in stored.metadata


# ===========================================================================
# 8. Serialization, Replay & Snapshot
# ===========================================================================

def test_to_dict_and_from_dict(
    family1_false_negative,
    family2_false_negative,
):
    """Memory must cleanly serialize to dicts and restore into an equivalent instance."""
    e1, p1, f1 = family1_false_negative
    e2, p2, f2 = family2_false_negative

    memory = FailureMemory()
    memory.record_event(e1, p1, f1)
    memory.record_event(e2, p2, f2)

    serialized = memory.to_dict()
    assert isinstance(serialized, list)
    assert len(serialized) == 2

    restored = FailureMemory.from_dict(serialized)
    assert len(restored) == 2
    assert restored[0].round_id == memory[0].round_id
    assert restored[0].attack_genome == memory[0].attack_genome
    assert restored[1].round_id == memory[1].round_id
    assert restored[1].attack_genome == memory[1].attack_genome


def test_json_serialization_and_restore(
    family1_false_negative,
    family3_false_negative,
):
    """Memory must serialize to JSON string and restore with fidelity."""
    e1, p1, f1 = family1_false_negative
    e3, p3, f3 = family3_false_negative

    memory = FailureMemory()
    memory.record_event(e1, p1, f1)
    memory.record_event(e3, p3, f3)

    json_str = memory.to_json(indent=2)
    assert isinstance(json_str, str)
    # Check valid json
    parsed = json.loads(json_str)
    assert len(parsed) == 2

    restored = FailureMemory.from_json(json_str)
    assert len(restored) == 2
    assert restored[0].attack_id == "atk-f1-001"
    assert restored[1].attack_id == "atk-f3-003"


# ===========================================================================
# 9. Empty Memory & Edge Cases
# ===========================================================================

def test_empty_memory_behavior():
    """Empty memory must behave cleanly without errors."""
    memory = FailureMemory()

    assert memory.is_empty is True
    assert len(memory) == 0
    assert memory.count == 0
    assert memory.get_failures() == []
    assert memory.get_genomes() == []
    assert memory.get_scenarios() == []
    assert memory.get_by_family(AttackFamily.ADAPTIVE_EVASION) == []
    assert memory.get_by_round_id("any") is None
    assert memory.get_by_attack_id("any") is None
    assert memory.to_dict() == []
    assert memory.to_json() == "[]"

    # Ingesting empty list
    added = memory.ingest_many([])
    assert added == 0
    assert len(memory) == 0


def test_clear_memory(family1_false_negative):
    """Clearing memory must remove all stored records."""
    e1, p1, f1 = family1_false_negative
    memory = FailureMemory()
    memory.record_event(e1, p1, f1)
    assert len(memory) == 1

    memory.clear()
    assert len(memory) == 0
    assert memory.is_empty is True


# ===========================================================================
# 10. Scenario with Pydantic Domain Model Objects
# ===========================================================================

def test_pydantic_domain_model_scenario_preservation():
    """Verify that domain model objects inside scenario are serialized cleanly."""
    from datetime import datetime, timezone

    tx = Transaction(
        transaction_id="tx-live-999",
        user_id="user-88",
        timestamp=datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc),
        amount=145.50,
        currency="USD",
        merchant_id="merch-12",
        merchant_category="electronics",
        location="US-NYC",
        device_id="dev-99",
        payment_channel="online",
    )
    event = AttackEvent(
        attack_id="atk-tx-099",
        round_id="round-tx-99",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome={"amount_deviation": 0.12},
        scenario={"transaction": tx},
        ground_truth=True,
    )
    prediction = PredictionResult(
        prediction_id="pred-99",
        prediction=False,
        risk_score=0.20,
        model_version="xgb-v1",
    )
    memory = FailureMemory()
    assert memory.record_event(event, prediction) is True

    record = memory[0]
    assert isinstance(record.scenario["transaction"], dict)
    assert record.scenario["transaction"]["transaction_id"] == "tx-live-999"
    assert record.scenario["transaction"]["amount"] == 145.50


# ===========================================================================
# 11. Advanced Querying & Edge Cases
# ===========================================================================

def test_query_family_string_vs_enum(
    family1_false_negative,
    family2_false_negative,
):
    """Querying by family should work seamlessly with either enum or string."""
    e1, p1, f1 = family1_false_negative
    e2, p2, f2 = family2_false_negative

    memory = FailureMemory()
    memory.record_event(e1, p1, f1)
    memory.record_event(e2, p2, f2)

    # Query with string
    res_str = memory.get_by_family("Family 1 - Adaptive Transaction-Pattern Evasion")
    assert len(res_str) == 1
    assert res_str[0].attack_id == "atk-f1-001"

    # Query with Enum
    res_enum = memory.get_by_family(AttackFamily.ADAPTIVE_EVASION)
    assert len(res_enum) == 1
    assert res_enum[0].attack_id == "atk-f1-001"


def test_reject_invalid_failure_record_in_record_failure(family1_false_negative):
    """Direct record_failure should reject records that are not false negatives."""
    event, _, _ = family1_false_negative
    # Construct a non-false-negative record
    invalid_record = FailureRecord(
        round_id="round-invalid",
        attack_id="atk-invalid",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        risk_score=0.90,
        prediction=True,
        ground_truth=True,
        detected=True,
        false_negative=False,
    )
    memory = FailureMemory()
    assert memory.record_failure(invalid_record) is False
    assert len(memory) == 0


def test_constructor_initial_records(family1_false_negative, family2_false_negative):
    """Constructor should accept an initial sequence of FailureRecords."""
    e1, p1, f1 = family1_false_negative
    e2, p2, f2 = family2_false_negative

    r1 = FailureRecord.from_components(e1, p1, f1)
    r2 = FailureRecord.from_components(e2, p2, f2)

    memory = FailureMemory([r1, r2])
    assert len(memory) == 2
    assert memory[0].attack_id == "atk-f1-001"
    assert memory[1].attack_id == "atk-f2-002"

