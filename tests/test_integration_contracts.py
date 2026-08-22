"""
tests/test_integration_contracts.py — Phase 5 contract smoke tests.

Purpose
-------
Prove that all stable shared interfaces can be imported, instantiated with
minimal fakes, and satisfy their runtime protocol checks.

These tests do NOT test family-specific logic (no Family 1/2/3 code exists
yet beyond the mock pipeline).  They verify that the contract layer itself
is intact and that future family implementations can satisfy it.

Nothing in this module modifies Phase 0-4 behaviour.
"""

import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    BlueTeamFeedback,
    PredictionResult,
    RoundResult,
    Transaction,
    AIAgentPaymentEvent,
    SyntheticIdentity,
)
from simulation import RoundController, RoundControllerError
from simulation.interfaces import (
    AttackGenerator,
    BlueTeamDetector,
    FeedbackEvaluator,
    MutationStrategy,
)
from mutation.genome_engine import (
    validate_genome,
    normalize_genome,
    compare_genomes,
    serialize_genome,
    deserialize_genome,
    GenomeValidationError,
)


# ---------------------------------------------------------------------------
# Minimal fakes — satisfy the four protocols with the smallest possible code
# ---------------------------------------------------------------------------

class _MinimalGenerator:
    def generate(self, round_id: str) -> AttackEvent:
        return AttackEvent(
            attack_id=f"smoke-{round_id}",
            round_id=round_id,
            attack_family=AttackFamily.ADAPTIVE_EVASION,
            attack_genome={"dim_a": 0.3},
            scenario={},
            ground_truth=True,
        )


class _MinimalDetector:
    def detect(self, event: AttackEvent) -> PredictionResult:
        return PredictionResult(
            prediction_id=f"pred-{event.round_id}",
            prediction=True,
            risk_score=0.8,
            model_version="smoke-v1",
        )


class _MinimalEvaluator:
    def evaluate(self, event: AttackEvent, prediction: PredictionResult) -> BlueTeamFeedback:
        return BlueTeamFeedback(
            feedback_id=f"fb-{event.round_id}",
            round_reference=event.round_id,
            detected=prediction.prediction and event.ground_truth,
            false_positive=False,
            false_negative=False,
            risk_score=prediction.risk_score,
            important_features={"dim_a": 0.3},
        )


class _MinimalMutator:
    def mutate(self, genome: dict, feedback: BlueTeamFeedback) -> dict:
        step = -0.05 if feedback.detected else 0.02
        return {k: min(max(v + step, 0.0), 1.0) for k, v in genome.items()}


# ---------------------------------------------------------------------------
# 1. Protocol isinstance checks — all four interfaces
# ---------------------------------------------------------------------------

def test_attack_generator_protocol_satisfied():
    assert isinstance(_MinimalGenerator(), AttackGenerator)


def test_blue_team_detector_protocol_satisfied():
    assert isinstance(_MinimalDetector(), BlueTeamDetector)


def test_feedback_evaluator_protocol_satisfied():
    assert isinstance(_MinimalEvaluator(), FeedbackEvaluator)


def test_mutation_strategy_protocol_satisfied():
    assert isinstance(_MinimalMutator(), MutationStrategy)


# ---------------------------------------------------------------------------
# 2. RoundController accepts the minimal fakes and returns a valid RoundResult
# ---------------------------------------------------------------------------

def test_round_controller_smoke():
    ctrl = RoundController(
        generator=_MinimalGenerator(),
        detector=_MinimalDetector(),
        evaluator=_MinimalEvaluator(),
    )
    result = ctrl.run_round(round_id="smoke-round-1")

    assert isinstance(result, RoundResult)
    assert result.round_id == "smoke-round-1"
    assert isinstance(result.attack_event, AttackEvent)
    assert isinstance(result.prediction_result, PredictionResult)
    assert isinstance(result.feedback, BlueTeamFeedback)


# ---------------------------------------------------------------------------
# 3. All three AttackFamily enum values are importable and distinct
# ---------------------------------------------------------------------------

def test_attack_family_enum_values():
    families = list(AttackFamily)
    assert len(families) == 3
    assert AttackFamily.ADAPTIVE_EVASION in families
    assert AttackFamily.AGENT_BEHAVIOR in families
    assert AttackFamily.SYNTHETIC_IDENTITY in families


# ---------------------------------------------------------------------------
# 4. RoundController works with every AttackFamily (no family branching)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", list(AttackFamily))
def test_controller_family_agnostic(family: AttackFamily):
    class _FamilyGenerator:
        def generate(self, round_id: str) -> AttackEvent:
            return AttackEvent(
                attack_id=f"ev-{round_id}",
                round_id=round_id,
                attack_family=family,
                attack_genome={"x": 0.5},
                scenario={},
                ground_truth=True,
            )

    ctrl = RoundController(
        generator=_FamilyGenerator(),
        detector=_MinimalDetector(),
        evaluator=_MinimalEvaluator(),
    )
    result = ctrl.run_round(round_id=f"round-{family.name}")
    assert result.attack_event.attack_family == family


# ---------------------------------------------------------------------------
# 5. Genome engine utilities are importable and functional
# ---------------------------------------------------------------------------

def test_genome_engine_validate_accept():
    validate_genome({"a": 0.5, "b": 0.0, "c": 1.0})  # must not raise


def test_genome_engine_validate_reject_nan():
    import math
    with pytest.raises(GenomeValidationError):
        validate_genome({"a": math.nan})


def test_genome_engine_compare():
    before = {"a": 0.5, "b": 0.3}
    after  = {"a": 0.4, "b": 0.3}
    diff = compare_genomes(before, after)
    assert "a" in diff["changed"]
    assert diff["changed"]["a"]["delta"] == pytest.approx(-0.1)


def test_genome_engine_serialize_roundtrip():
    genome = {"z": 0.9, "a": 0.1}
    serialized = serialize_genome(genome)
    restored = deserialize_genome(serialized)
    assert restored == genome


# ---------------------------------------------------------------------------
# 6. Family-specific domain schemas are importable and constructable
# ---------------------------------------------------------------------------

def test_transaction_schema_constructable():
    from datetime import datetime
    t = Transaction(
        transaction_id="tx-001",
        user_id="u-001",
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        amount=99.99,
        currency="USD",
        merchant_id="m-001",
        merchant_category="retail",
        location="New York",
        device_id="dev-001",
        payment_channel="online",
    )
    assert t.transaction_id == "tx-001"
    assert t.currency == "USD"


def test_ai_agent_payment_event_schema_constructable():
    ev = AIAgentPaymentEvent(
        event_id="ae-001",
        user_intent="Buy one coffee under $10",
        authorized_scope="food:purchase:max_10_usd",
        agent_identity="agent-barista-v2",
        session_context="sess-xyz",
        actual_action="purchase:coffee:$9.50",
    )
    assert ev.event_id == "ae-001"


def test_synthetic_identity_schema_constructable():
    sid = SyntheticIdentity(
        identity_id="sid-001",
        identity_attributes={"name": "Alice Smith", "dob": "1990-05-12"},
        contact_attributes={"email": "alice@example.com", "phone": "+1-555-0100"},
        account_metadata={"open_date": "2025-01-01", "account_type": "checking"},
        device_context={"device_fingerprint": "fp-abc123"},
        lifecycle_info={"days_to_first_transaction": 3},
    )
    assert sid.identity_id == "sid-001"


# ---------------------------------------------------------------------------
# 7. MutationStrategy produces valid genome after mutation
# ---------------------------------------------------------------------------

def test_mutator_produces_valid_genome_after_detected():
    genome = {"x": 0.9, "y": 0.8}
    feedback = BlueTeamFeedback(
        feedback_id="fb-mut",
        round_reference="r-mut",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.9,
        important_features={"x": 0.9},
    )
    mutated = _MinimalMutator().mutate(genome, feedback)
    validate_genome(mutated)  # must not raise
    assert mutated["x"] < genome["x"]  # decay on detected


def test_mutator_produces_valid_genome_after_missed():
    genome = {"x": 0.3, "y": 0.2}
    feedback = BlueTeamFeedback(
        feedback_id="fb-mut2",
        round_reference="r-mut2",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.2,
        important_features={"x": 0.3},
    )
    mutated = _MinimalMutator().mutate(genome, feedback)
    validate_genome(mutated)  # must not raise
    assert mutated["x"] > genome["x"]  # boost on missed


# ---------------------------------------------------------------------------
# 8. RoundControllerError is raised for wrong dependency types
# ---------------------------------------------------------------------------

def test_wrong_generator_type_raises_contract_error():
    with pytest.raises(RoundControllerError):
        RoundController(
            generator=object(),
            detector=_MinimalDetector(),
            evaluator=_MinimalEvaluator(),
        )


def test_wrong_detector_type_raises_contract_error():
    with pytest.raises(RoundControllerError):
        RoundController(
            generator=_MinimalGenerator(),
            detector=object(),
            evaluator=_MinimalEvaluator(),
        )


def test_wrong_evaluator_type_raises_contract_error():
    with pytest.raises(RoundControllerError):
        RoundController(
            generator=_MinimalGenerator(),
            detector=_MinimalDetector(),
            evaluator=object(),
        )
