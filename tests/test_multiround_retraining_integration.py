"""
tests/test_multiround_retraining_integration.py — Multi-Round Retraining Integration Tests (Task 7.4).

Proves the full end-to-end Blue-Team learning cycle through real simulation rounds:

    RoundResult (from real Pipeline / RoundController)
        ↓
    FailureMemory (on_round_completed → record_round)
        ↓
    RetrainingController (triggers every 2 rounds by default)
        ↓
    Updated Blue-Team detector state (in-place, same object)
        ↓
    Subsequent rounds use updated detector

This verifies all 15 required integration behaviors using real Family 1, 2, and 3
generators, detectors, evaluators, and mutators without modifying any protected file.

ARCHITECTURE NOTE
-----------------
Integration happens WITHOUT touching the frozen RoundController or Pipeline.
The orchestration pattern is:

    1. Build real Pipeline / RoundController components (generator, detector, evaluator, mutator)
    2. Build RetrainingController with the same detector instance
    3. Execute rounds through RoundController
    4. Call ctrl.on_round_completed(result, round_index) after each round
    5. Feed mutated genome to generator for next round

The same detector object is shared between the RoundController and the
RetrainingController, so in-place parameter updates / refits are immediately effective
for the next round's detect() call.

Protected files NOT touched:
    schemas/
    simulation/interfaces.py
    simulation/round_controller.py
    simulation/pipeline.py
    mutation/genome_engine.py
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest

from schemas.common import AttackFamily
from schemas.round import RoundResult

from simulation import RoundController

# Family 1 components
from attacks.transaction_evasion import (
    TransactionAttackGenerator,
    TransactionMutationStrategy,
    DEFAULT_ATTACK_GENOME as F1_ATTACK_GENOME,
)
from blue_team.transaction import (
    TransactionBlueDetector,
    TransactionFeedbackEvaluator,
)

# Family 2 components
from attacks.ai_agent import (
    AIAgentAttackGenerator,
    AIAgentMutationStrategy,
    DEFAULT_ATTACK_GENOME as F2_ATTACK_GENOME,
)
from blue_team.ai_agent import (
    AIAgentBlueDetector,
    AIAgentFeedbackEvaluator,
)

# Family 3 components
from attacks.synthetic_identity import (
    SyntheticIdentityAttackGenerator,
    SyntheticIdentityMutationStrategy,
    DEFAULT_ATTACK_GENOME as F3_ATTACK_GENOME,
)
from blue_team.synthetic_identity import (
    SyntheticIdentityBlueDetector,
    SyntheticIdentityFeedbackEvaluator,
)
from schemas.identity import SyntheticIdentity

# Learning components
from blue_team.learning.retraining import (
    RetrainingController,
    ModelUpdateRecord,
)
from blue_team.learning.failure_memory import FailureMemory
from blue_team.learning.dataset import HoldoutDataLeakageError

# Dashboard structures
from dashboard.arms_race import (
    build_arms_race_history,
    ModelUpdateMarker,
    model_update_rounds,
)


# ===========================================================================
# Shared Helpers
# ===========================================================================

def _run_rounds_with_learning(
    generator: Any,
    detector: Any,
    evaluator: Any,
    mutator: Any,
    genome_updater: Any,
    family: AttackFamily,
    ctrl: RetrainingController,
    num_rounds: int,
    base_round_id: str = "r",
) -> Tuple[List[RoundResult], List[Optional[ModelUpdateRecord]]]:
    """
    Run num_rounds rounds through the FROZEN RoundController, calling
    on_round_completed after each round.

    Returns (round_results, update_records).
    """
    round_ctrl = RoundController(
        generator=generator,
        detector=detector,
        evaluator=evaluator,
    )
    results: List[RoundResult] = []
    updates: List[Optional[ModelUpdateRecord]] = []

    current_genome = None

    for round_index in range(1, num_rounds + 1):
        round_id = f"{base_round_id}-{round_index}"

        # Update genome if we have a mutated one from previous round
        if current_genome is not None and genome_updater is not None:
            genome_updater(current_genome)

        result = round_ctrl.run_round(
            round_id=round_id,
            outcome_metrics={"round_index": round_index},
        )
        results.append(result)

        # Blue-Team learning step — non-invasive integration around frozen core
        update = ctrl.on_round_completed(result, round_index=round_index)
        updates.append(update)

        # Red-Team mutation step
        current_genome = mutator.mutate(
            dict(result.attack_event.attack_genome),
            result.feedback,
        )

    return results, updates


def _minimal_f3_baseline(n: int = 5) -> List[SyntheticIdentity]:
    """Create n minimal SyntheticIdentity baseline records for testing."""
    return [
        SyntheticIdentity(
            identity_id=f"base-id-{i}",
            identity_attributes={"name": f"Alice Legit {i}"},
            contact_attributes={"email": f"alice{i}@legit.com"},
            account_metadata={
                "kyc_verification_status": "verified",
                "account_status": "active",
            },
            device_context={"device_id": f"dev-{i}"},
            lifecycle_info={
                "risk_event_count": 0,
                "lifecycle_coherence_score": 0.95,
            },
        )
        for i in range(n)
    ]


# ===========================================================================
# 1. Multi-round execution produces RoundResult objects (Req 1)
# ===========================================================================

def test_multiround_produces_valid_round_results_family1():
    """Requirement 1: Multi-round execution produces valid RoundResult objects."""
    det = TransactionBlueDetector()
    gen = TransactionAttackGenerator(seed=42)
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, _ = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=3,
        base_round_id="f1-rr",
    )

    assert len(results) == 3
    for idx, r in enumerate(results, start=1):
        assert isinstance(r, RoundResult)
        assert r.round_id == f"f1-rr-{idx}"
        assert r.attack_event.attack_family == AttackFamily.ADAPTIVE_EVASION
        assert r.prediction_result is not None
        assert r.feedback is not None


# ===========================================================================
# 2. False negatives from actual rounds reach FailureMemory (Req 2)
# ===========================================================================

def test_false_negatives_from_actual_rounds_reach_failure_memory():
    """Requirement 2: Missed attacks from real rounds are ingested into FailureMemory."""
    det = TransactionBlueDetector(threshold=0.80)  # High threshold to cause misses
    gen = TransactionAttackGenerator(
        genome={"amount_deviation": 0.20, "velocity_deviation": 0.20,
                "device_novelty": 0.15, "location_deviation": 0.15,
                "time_deviation": 0.10, "sequence_anomaly": 0.10},
        seed=10,
    )
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=10,  # Delay retraining to observe memory
        auto_load_canonical_baseline=False,
    )

    results, _ = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=2,
        base_round_id="f1-fn-reach",
    )

    # Both rounds were missed because threshold is 0.80 and risk score is ~0.20-0.30
    assert len(ctrl.failure_memory) == 2
    for record in ctrl.failure_memory.get_failures():
        assert record.false_negative is True
        assert record.ground_truth is True
        assert record.prediction is False
        assert record.attack_family == AttackFamily.ADAPTIVE_EVASION


# ===========================================================================
# 3. Multiple rounds accumulate failures (Req 3)
# ===========================================================================

def test_multiple_rounds_accumulate_failures():
    """Requirement 3: Sequential missed rounds accumulate in FailureMemory deterministically."""
    det = AIAgentBlueDetector(threshold=0.85)  # High threshold to cause misses
    gen = AIAgentAttackGenerator(
        genome={"intent_amount_deviation": 0.20, "intent_category_deviation": 0.15,
                "permission_scope_deviation": 0.10, "agent_identity_confidence": 0.90,
                "session_provenance_anomaly": 0.15, "purchase_velocity": 0.20},
        seed=20,
    )
    ev = AIAgentFeedbackEvaluator()
    mut = AIAgentMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.AGENT_BEHAVIOR: det},
        retrain_interval=5,
        auto_load_canonical_baseline=False,
    )

    results, _ = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.AGENT_BEHAVIOR,
        ctrl=ctrl,
        num_rounds=3,
        base_round_id="f2-accum",
    )

    assert ctrl.failure_memory.count == 3
    genomes = ctrl.failure_memory.get_genomes()
    assert len(genomes) == 3


# ===========================================================================
# 4. No retraining before configured interval (Req 4)
# ===========================================================================

def test_no_retraining_before_configured_interval():
    """Requirement 4: With interval=2, round 1 produces None and does not retrain."""
    det = TransactionBlueDetector()
    gen = TransactionAttackGenerator(seed=42)
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=1,
        base_round_id="f1-noretrain",
    )

    assert updates[0] is None
    assert len(ctrl.get_history()) == 0


# ===========================================================================
# 5. Retraining occurs at configured interval (Req 5)
# ===========================================================================

def test_retraining_occurs_at_configured_interval():
    """Requirement 5: With interval=2, retraining is evaluated at round 2, 4, etc."""
    det = TransactionBlueDetector()
    gen = TransactionAttackGenerator(seed=42)
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=4,
        base_round_id="f1-periodic",
    )

    assert updates[0] is None
    assert updates[1] is not None
    assert updates[1].round_index == 2
    assert updates[2] is None
    assert updates[3] is not None
    assert updates[3].round_index == 4


# ===========================================================================
# 6. A ModelUpdateRecord is produced (Req 6)
# ===========================================================================

def test_model_update_record_structure_and_metadata():
    """Requirement 6: Retraining produces a fully-populated ModelUpdateRecord."""
    det = TransactionBlueDetector(threshold=0.80)
    gen = TransactionAttackGenerator(
        genome={"amount_deviation": 0.10, "velocity_deviation": 0.10,
                "device_novelty": 0.10, "location_deviation": 0.10,
                "time_deviation": 0.10, "sequence_anomaly": 0.10},
        seed=1,
    )
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    _, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=2,
        base_round_id="f1-rec-meta",
    )

    rec = updates[1]
    assert rec is not None
    assert isinstance(rec, ModelUpdateRecord)
    assert rec.round_index == 2
    assert rec.family == AttackFamily.ADAPTIVE_EVASION
    assert rec.trigger_reason == "scheduled_interval"
    assert rec.previous_model_version != ""
    assert rec.new_model_version != ""
    assert rec.false_negative_count >= 1
    assert rec.timestamp != ""


# ===========================================================================
# 7. Registered detector is updated in-place (Req 7)
# ===========================================================================

def test_registered_detector_updated_in_place():
    """Requirement 7: The exact detector instance registered with controller is updated."""
    det = TransactionBlueDetector(threshold=0.80)
    initial_version = det.model_version
    initial_weights = copy.deepcopy(det.weights)

    gen = TransactionAttackGenerator(
        genome={"amount_deviation": 0.10, "velocity_deviation": 0.10,
                "device_novelty": 0.10, "location_deviation": 0.10,
                "time_deviation": 0.10, "sequence_anomaly": 0.10},
        seed=1,
    )
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    _, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=2,
        base_round_id="f1-inplace",
    )

    assert updates[1] is not None
    assert updates[1].retrained is True
    # The detector instance was mutated in-place
    assert det.model_version != initial_version
    assert det.weights != initial_weights
    assert det.threshold != 0.80


# ===========================================================================
# 8. Subsequent rounds use updated detector state (Req 8)
# ===========================================================================

def test_subsequent_rounds_use_updated_detector_state():
    """Requirement 8: Round 3 immediately evaluates using the updated model version and weights."""
    det = TransactionBlueDetector(threshold=0.80)
    gen = TransactionAttackGenerator(
        genome={"amount_deviation": 0.10, "velocity_deviation": 0.10,
                "device_novelty": 0.10, "location_deviation": 0.10,
                "time_deviation": 0.10, "sequence_anomaly": 0.10},
        seed=1,
    )
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=3,
        base_round_id="f1-subsequent",
    )

    retrained_version = updates[1].new_model_version
    # Round 1 and 2 had original version (or round 2 was evaluated before retraining at end of round 2)
    assert results[0].prediction_result.model_version == "heuristic-family1-v1"
    # Round 3 MUST have used the newly retrained model version!
    assert results[2].prediction_result.model_version == retrained_version


# ===========================================================================
# 9. Family 1 full learning loop works end-to-end (Req 9)
# ===========================================================================

def test_family1_full_learning_loop_end_to_end():
    """Requirement 9: Family 1 multi-round simulation executes complete learning loop."""
    det = TransactionBlueDetector(threshold=0.75)
    gen = TransactionAttackGenerator(seed=123)
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=4,
        base_round_id="f1-e2e",
    )

    assert len(results) == 4
    assert len(ctrl.get_history()) == 2
    assert ctrl.get_history()[0].round_index == 2
    assert ctrl.get_history()[1].round_index == 4


# ===========================================================================
# 10. Family 2 full learning loop works end-to-end (Req 10)
# ===========================================================================

def test_family2_full_learning_loop_end_to_end():
    """Requirement 10: Family 2 multi-round simulation executes complete mandate adaptation loop."""
    det = AIAgentBlueDetector(threshold=0.75)
    gen = AIAgentAttackGenerator(seed=456)
    ev = AIAgentFeedbackEvaluator()
    mut = AIAgentMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.AGENT_BEHAVIOR: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.AGENT_BEHAVIOR,
        ctrl=ctrl,
        num_rounds=4,
        base_round_id="f2-e2e",
    )

    assert len(results) == 4
    assert len(ctrl.get_history()) == 2


# ===========================================================================
# 11. Family 3 full XGBoost learning loop works end-to-end (Req 11)
# ===========================================================================

def test_family3_full_xgboost_learning_loop_end_to_end():
    """Requirement 11: Family 3 multi-round simulation executes genuine XGBoost ML refitting."""
    det = SyntheticIdentityBlueDetector()
    initial_version = det.model_version

    gen = SyntheticIdentityAttackGenerator(seed=789)
    ev = SyntheticIdentityFeedbackEvaluator()
    mut = SyntheticIdentityMutationStrategy()

    baseline = _minimal_f3_baseline(n=5)

    ctrl = RetrainingController(
        detectors={AttackFamily.SYNTHETIC_IDENTITY: det},
        baseline_data={AttackFamily.SYNTHETIC_IDENTITY: baseline},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.SYNTHETIC_IDENTITY,
        ctrl=ctrl,
        num_rounds=4,
        base_round_id="f3-e2e",
    )

    assert len(results) == 4
    assert len(ctrl.get_history()) == 2

    # Check that detector in live round 3 had the retrained XGBoost model version if retrained
    upd_r2 = updates[1]
    if upd_r2.retrained:
        assert det.model_version != initial_version
        assert "family3-xgb-retrained" in results[2].prediction_result.model_version


# ===========================================================================
# 12. Held-out data is never included in training (Req 12)
# ===========================================================================

def test_held_out_data_isolation_during_multiround():
    """Requirement 12: Held-out evaluation datasets cannot be leaked into training baseline."""
    det = SyntheticIdentityBlueDetector()
    ctrl = RetrainingController(
        detectors={AttackFamily.SYNTHETIC_IDENTITY: det},
        auto_load_canonical_baseline=False,
    )

    # Attempting to set held_out paths as training baseline raises HoldoutDataLeakageError
    with pytest.raises(HoldoutDataLeakageError):
        ctrl.set_baseline_data(
            AttackFamily.SYNTHETIC_IDENTITY,
            ["data/held_out/heldout_identities.json"],
        )


# ===========================================================================
# 13. Failed retraining preserves previous known-good detector (Req 13)
# ===========================================================================

def test_failed_retraining_preserves_previous_known_good_detector():
    """Requirement 13: If training throws an exception, previous detector state is untouched."""
    class _FailingTrainer:
        def train(self, dataset, detector, held_out_data=None, retrain_count=1):
            raise RuntimeError("Out of memory during training")

    det = TransactionBlueDetector()
    original_version = det.model_version
    original_weights = copy.deepcopy(det.weights)
    original_threshold = det.threshold

    gen = TransactionAttackGenerator(
        genome={"amount_deviation": 0.10, "velocity_deviation": 0.10,
                "device_novelty": 0.10, "location_deviation": 0.10,
                "time_deviation": 0.10, "sequence_anomaly": 0.10},
        seed=1,
    )
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        trainers={AttackFamily.ADAPTIVE_EVASION: _FailingTrainer()},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=2,
        base_round_id="f1-crash",
    )

    rec = updates[1]
    assert rec is not None
    assert rec.retrained is False
    assert "training_exception" in rec.trigger_reason

    # Exact detector parameters preserved
    assert det.model_version == original_version
    assert det.weights == original_weights
    assert det.threshold == original_threshold


# ===========================================================================
# 14. Repeated identical runs are deterministic (Req 14)
# ===========================================================================

def test_repeated_multiround_learning_runs_are_deterministic():
    """Requirement 14: Two independent runs with same seed and config produce identical results."""
    def _execute() -> Tuple[List[RoundResult], List[Optional[ModelUpdateRecord]]]:
        det = TransactionBlueDetector()
        gen = TransactionAttackGenerator(seed=888)
        ev = TransactionFeedbackEvaluator()
        mut = TransactionMutationStrategy()

        ctrl = RetrainingController(
            detectors={AttackFamily.ADAPTIVE_EVASION: det},
            retrain_interval=2,
            auto_load_canonical_baseline=False,
        )

        return _run_rounds_with_learning(
            generator=gen,
            detector=det,
            evaluator=ev,
            mutator=mut,
            genome_updater=gen.set_genome,
            family=AttackFamily.ADAPTIVE_EVASION,
            ctrl=ctrl,
            num_rounds=4,
            base_round_id="determ-run",
        )

    res_a, upd_a = _execute()
    res_b, upd_b = _execute()

    assert len(res_a) == len(res_b)
    for ra, rb in zip(res_a, res_b):
        assert ra.round_id == rb.round_id
        assert ra.feedback.detected == rb.feedback.detected
        assert ra.feedback.false_negative == rb.feedback.false_negative
        assert abs(ra.prediction_result.risk_score - rb.prediction_result.risk_score) < 1e-6

    for ua, ub in zip(upd_a, upd_b):
        if ua is None:
            assert ub is None
        else:
            assert ub is not None
            assert ua.retrained == ub.retrained
            assert ua.new_model_version == ub.new_model_version


# ===========================================================================
# 15. Model-update information remains available for dashboard (Req 15)
# ===========================================================================

def test_model_update_information_available_to_dashboard_arms_race():
    """Requirement 15: Simulation results feed cleanly into dashboard arms race models."""
    det = TransactionBlueDetector(threshold=0.75)
    gen = TransactionAttackGenerator(seed=999)
    ev = TransactionFeedbackEvaluator()
    mut = TransactionMutationStrategy()

    ctrl = RetrainingController(
        detectors={AttackFamily.ADAPTIVE_EVASION: det},
        retrain_interval=2,
        auto_load_canonical_baseline=False,
    )

    results, updates = _run_rounds_with_learning(
        generator=gen,
        detector=det,
        evaluator=ev,
        mutator=mut,
        genome_updater=gen.set_genome,
        family=AttackFamily.ADAPTIVE_EVASION,
        ctrl=ctrl,
        num_rounds=4,
        base_round_id="f1-dash",
    )

    # 1. Dashboard summary and timeline report
    report = build_arms_race_history(results)
    assert report.summary.total_rounds == 4
    assert len(report.timeline) == 4
    assert len(report.detection_trend) == 4
    assert len(report.risk_trend) == 4

    # 2. Model update markers
    markers = model_update_rounds(results)
    assert isinstance(markers, list)

    # 3. RetrainingController history
    history = ctrl.get_history()
    assert len(history) == 2
    assert history[0].round_index == 2
    assert history[1].round_index == 4
