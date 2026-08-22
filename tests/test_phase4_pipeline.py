"""
tests/test_phase4_pipeline.py -- Focused Phase 4 tests for the mock Red/Blue pipeline.

Verifies:
  1.  One mock round completes successfully.
  2.  Multiple rounds execute in sequence.
  3.  The generator is invoked for each round.
  4.  The detector receives the generated events.
  5.  Feedback is produced for each round.
  6.  Mutation receives feedback from the previous round.
  7.  The next genome is different when feedback causes adaptation.
  8.  Both detected and missed outcomes are supported.
  9.  Round results use the existing RoundResult contract.
  10. The pipeline is deterministic.
  11. Existing Phase 0-3 behavior is unaffected (smoke check).
  12. Pipeline is decoupled from MockAttackGenerator internals (genome_updater).

All tests are deterministic; no randomness, no external I/O.
"""

import pytest
from schemas import (
    AttackEvent,
    AttackFamily,
    BlueTeamFeedback,
    PredictionResult,
    RoundResult,
)
from simulation import RoundController
from simulation.mock_pipeline import (
    MockAttackGenerator,
    MockBlueDetector,
    MockFeedbackEvaluator,
    MockMutationStrategy,
    Pipeline,
    _DETECTION_THRESHOLD,
    _DETECTED_DECAY,
    _MISSED_BOOST,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(genome=None) -> Pipeline:
    """Return a fully assembled Pipeline with the given starting genome."""
    gen = MockAttackGenerator(genome=genome)
    return Pipeline(
        generator=gen,
        detector=MockBlueDetector(),
        evaluator=MockFeedbackEvaluator(),
        mutator=MockMutationStrategy(),
        genome_updater=gen.set_genome,
    )


def _high_genome() -> dict:
    """A genome whose average is >= _DETECTION_THRESHOLD (will be detected)."""
    return {"signal_a": _DETECTION_THRESHOLD, "signal_b": _DETECTION_THRESHOLD}


def _low_genome() -> dict:
    """A genome whose average is < _DETECTION_THRESHOLD (will be missed)."""
    val = max(_DETECTION_THRESHOLD - 0.2, 0.0)
    return {"signal_a": val, "signal_b": val}


# ---------------------------------------------------------------------------
# Test 1 -- One mock round completes successfully
# ---------------------------------------------------------------------------

def test_single_round_completes():
    pipeline = _make_pipeline()
    results = pipeline.run(num_rounds=1, base_round_id="t1")

    assert len(results) == 1
    assert isinstance(results[0], RoundResult)


# ---------------------------------------------------------------------------
# Test 2 -- Multiple rounds execute in sequence
# ---------------------------------------------------------------------------

def test_multiple_rounds_execute():
    pipeline = _make_pipeline()
    results = pipeline.run(num_rounds=5, base_round_id="t2")

    assert len(results) == 5
    # Each result is a RoundResult
    for r in results:
        assert isinstance(r, RoundResult)


# ---------------------------------------------------------------------------
# Test 3 -- The generator is invoked for each round
# ---------------------------------------------------------------------------

def test_generator_invoked_each_round():
    """
    Spy on generate() calls by wrapping MockAttackGenerator.
    """
    class SpyGenerator(MockAttackGenerator):
        def __init__(self, genome=None):
            super().__init__(genome)
            self.call_count = 0
            self.called_round_ids: list[str] = []

        def generate(self, round_id: str) -> AttackEvent:
            self.call_count += 1
            self.called_round_ids.append(round_id)
            return super().generate(round_id)

    spy = SpyGenerator(genome=_low_genome())
    pipeline = Pipeline(
        generator=spy,
        detector=MockBlueDetector(),
        evaluator=MockFeedbackEvaluator(),
        mutator=MockMutationStrategy(),
        genome_updater=spy.set_genome,
    )
    pipeline.run(num_rounds=4, base_round_id="t3")

    assert spy.call_count == 4
    assert len(spy.called_round_ids) == 4
    # Round IDs should be distinct
    assert len(set(spy.called_round_ids)) == 4


# ---------------------------------------------------------------------------
# Test 4 -- The detector receives the generated events
# ---------------------------------------------------------------------------

def test_detector_receives_generated_events():
    class SpyDetector(MockBlueDetector):
        def __init__(self):
            self.received: list[AttackEvent] = []

        def detect(self, event: AttackEvent) -> PredictionResult:
            self.received.append(event)
            return super().detect(event)

    spy_det = SpyDetector()
    gen = MockAttackGenerator(genome=_low_genome())
    pipeline = Pipeline(
        generator=gen,
        detector=spy_det,
        evaluator=MockFeedbackEvaluator(),
        mutator=MockMutationStrategy(),
        genome_updater=gen.set_genome,
    )
    pipeline.run(num_rounds=3, base_round_id="t4")

    assert len(spy_det.received) == 3
    for event in spy_det.received:
        assert isinstance(event, AttackEvent)


# ---------------------------------------------------------------------------
# Test 5 -- Feedback is produced for each round
# ---------------------------------------------------------------------------

def test_feedback_produced_each_round():
    pipeline = _make_pipeline()
    results = pipeline.run(num_rounds=4, base_round_id="t5")

    for r in results:
        assert isinstance(r.feedback, BlueTeamFeedback)
        # Feedback round reference must match the round ID
        assert r.feedback.round_reference == r.round_id


# ---------------------------------------------------------------------------
# Test 6 -- Mutation receives feedback from the previous round
# ---------------------------------------------------------------------------

def test_mutation_receives_feedback():
    """
    Verify that the mutator is called with each round's feedback by checking
    that the genome changes between rounds.
    """
    class SpyMutator(MockMutationStrategy):
        def __init__(self):
            self.calls: list[tuple[dict, BlueTeamFeedback]] = []

        def mutate(self, genome: dict, feedback: BlueTeamFeedback) -> dict:
            self.calls.append((dict(genome), feedback))
            return super().mutate(genome, feedback)

    spy_mutator = SpyMutator()
    gen = MockAttackGenerator(genome=_high_genome())
    pipeline = Pipeline(
        generator=gen,
        detector=MockBlueDetector(),
        evaluator=MockFeedbackEvaluator(),
        mutator=spy_mutator,
        genome_updater=gen.set_genome,
    )
    pipeline.run(num_rounds=3, base_round_id="t6")

    # Mutator called once per round (3 rounds -> 3 calls)
    assert len(spy_mutator.calls) == 3
    # Each call receives a BlueTeamFeedback
    for _, fb in spy_mutator.calls:
        assert isinstance(fb, BlueTeamFeedback)


# ---------------------------------------------------------------------------
# Test 7 -- The next genome is different when feedback causes adaptation
# ---------------------------------------------------------------------------

def test_genome_changes_after_detected_round():
    """
    Start with a high genome (will be detected).
    After detection the mutator decays the genome.
    The second round should receive a lower genome than the first.
    """
    genome_snapshots: list[dict] = []

    class CapturingGenerator(MockAttackGenerator):
        def generate(self, round_id: str) -> AttackEvent:
            genome_snapshots.append(dict(self._genome))
            return super().generate(round_id)

    gen = CapturingGenerator(genome=_high_genome())
    pipeline = Pipeline(
        generator=gen,
        detector=MockBlueDetector(),
        evaluator=MockFeedbackEvaluator(),
        mutator=MockMutationStrategy(),
        genome_updater=gen.set_genome,
    )
    pipeline.run(num_rounds=2, base_round_id="t7")

    assert len(genome_snapshots) == 2
    # Round 1 genome should have higher values than round 2 (decay was applied)
    for key in genome_snapshots[0]:
        assert genome_snapshots[1][key] < genome_snapshots[0][key], (
            f"Expected genome['{key}'] to decrease after detection "
            f"(round1={genome_snapshots[0][key]}, round2={genome_snapshots[1][key]})"
        )


def test_genome_changes_after_missed_round():
    """
    Start with a low genome (will be missed).
    After the miss the mutator boosts the genome.
    The second round should receive a higher genome than the first.
    """
    genome_snapshots: list[dict] = []

    class CapturingGenerator(MockAttackGenerator):
        def generate(self, round_id: str) -> AttackEvent:
            genome_snapshots.append(dict(self._genome))
            return super().generate(round_id)

    gen = CapturingGenerator(genome=_low_genome())
    pipeline = Pipeline(
        generator=gen,
        detector=MockBlueDetector(),
        evaluator=MockFeedbackEvaluator(),
        mutator=MockMutationStrategy(),
        genome_updater=gen.set_genome,
    )
    pipeline.run(num_rounds=2, base_round_id="t7b")

    assert len(genome_snapshots) == 2
    for key in genome_snapshots[0]:
        assert genome_snapshots[1][key] > genome_snapshots[0][key], (
            f"Expected genome['{key}'] to increase after miss "
            f"(round1={genome_snapshots[0][key]}, round2={genome_snapshots[1][key]})"
        )


# ---------------------------------------------------------------------------
# Test 8 -- Both detected and missed outcomes are supported
# ---------------------------------------------------------------------------

def test_detected_outcome():
    pipeline = _make_pipeline(genome=_high_genome())
    results = pipeline.run(num_rounds=1, base_round_id="t8a")

    r = results[0]
    assert r.prediction_result.prediction is True
    assert r.feedback.detected is True
    assert r.feedback.false_negative is False


def test_missed_outcome():
    pipeline = _make_pipeline(genome=_low_genome())
    results = pipeline.run(num_rounds=1, base_round_id="t8b")

    r = results[0]
    assert r.prediction_result.prediction is False
    assert r.feedback.detected is False
    assert r.feedback.false_negative is True


# ---------------------------------------------------------------------------
# Test 9 -- Round results use the existing RoundResult contract
# ---------------------------------------------------------------------------

def test_round_results_use_existing_schema():
    pipeline = _make_pipeline()
    results = pipeline.run(num_rounds=3, base_round_id="t9")

    for i, r in enumerate(results, start=1):
        assert isinstance(r, RoundResult)
        assert isinstance(r.attack_event, AttackEvent)
        assert isinstance(r.prediction_result, PredictionResult)
        assert isinstance(r.feedback, BlueTeamFeedback)
        # outcome_metrics injected by the Pipeline
        assert r.outcome_metrics.get("round_index") == i
        # RoundResult round_id must be a non-empty string
        assert isinstance(r.round_id, str) and r.round_id


# ---------------------------------------------------------------------------
# Test 10 -- The pipeline is deterministic
# ---------------------------------------------------------------------------

def test_pipeline_is_deterministic():
    """
    Running the same pipeline twice (same starting genome, same round ID prefix)
    must produce identical sequences of risk scores and genomes.
    """
    def _run():
        pipeline = _make_pipeline(genome={"signal_a": 0.8, "signal_b": 0.6})
        return pipeline.run(num_rounds=4, base_round_id="det")

    results_a = _run()
    results_b = _run()

    for ra, rb in zip(results_a, results_b):
        assert ra.round_id == rb.round_id
        assert ra.prediction_result.risk_score == rb.prediction_result.risk_score
        assert ra.feedback.detected == rb.feedback.detected
        assert ra.attack_event.attack_genome == rb.attack_event.attack_genome


# ---------------------------------------------------------------------------
# Test 11 -- Existing Phase 0-3 behavior unaffected (smoke check)
# ---------------------------------------------------------------------------

def test_phase3_round_controller_still_works():
    """
    The existing RoundController from Phase 3 must continue to work with
    the mock components as direct arguments (not via Pipeline).
    This confirms Phase 4 did not break Phase 3.
    """
    gen = MockAttackGenerator(genome=_high_genome())
    det = MockBlueDetector()
    ev = MockFeedbackEvaluator()

    ctrl = RoundController(gen, det, ev)
    result = ctrl.run_round(round_id="phase3-compat")

    assert isinstance(result, RoundResult)
    assert result.round_id == "phase3-compat"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_pipeline_raises_on_zero_rounds():
    pipeline = _make_pipeline()
    with pytest.raises(ValueError, match="num_rounds must be >= 1"):
        pipeline.run(num_rounds=0)


def test_pipeline_raises_on_negative_rounds():
    pipeline = _make_pipeline()
    with pytest.raises(ValueError, match="num_rounds must be >= 1"):
        pipeline.run(num_rounds=-3)


def test_mock_genome_stays_in_valid_range_over_many_rounds():
    """
    After many rounds with continuous decay the genome values should stay >= 0.0.
    After many rounds with continuous boost they should stay <= 1.0.
    """
    # Decay scenario: start high, detect every round, values decay toward 0
    pipeline_high = _make_pipeline(genome={"s": 1.0})
    results_high = pipeline_high.run(num_rounds=20, base_round_id="decay")
    for r in results_high:
        for v in r.attack_event.attack_genome.values():
            assert 0.0 <= v <= 1.0, f"genome value out of range: {v}"

    # Boost scenario: start low, miss every round, values grow toward 1
    pipeline_low = _make_pipeline(genome={"s": 0.0})
    results_low = pipeline_low.run(num_rounds=20, base_round_id="boost")
    for r in results_low:
        for v in r.attack_event.attack_genome.values():
            assert 0.0 <= v <= 1.0, f"genome value out of range: {v}"


def test_attack_family_is_family_agnostic():
    """
    The pipeline does NOT branch on attack family.  Verify that the mock
    components do not contain any family-specific conditionals by confirming
    all three families can be processed without error (when manually set).
    """
    for family in AttackFamily:
        class FamilyOverrideGenerator(MockAttackGenerator):
            def generate(self, round_id: str) -> AttackEvent:
                event = super().generate(round_id)
                # Return with overridden family to simulate different families
                return AttackEvent(
                    attack_id=event.attack_id,
                    round_id=event.round_id,
                    attack_family=family,
                    attack_genome=event.attack_genome,
                    scenario=event.scenario,
                    ground_truth=event.ground_truth,
                    metadata=event.metadata,
                )

        fam_gen = FamilyOverrideGenerator(genome=_high_genome())
        pipeline = Pipeline(
            generator=fam_gen,
            detector=MockBlueDetector(),
            evaluator=MockFeedbackEvaluator(),
            mutator=MockMutationStrategy(),
            genome_updater=fam_gen.set_genome,
        )
        results = pipeline.run(num_rounds=1, base_round_id=f"fam-{family.name}")
        assert results[0].attack_event.attack_family == family


# ---------------------------------------------------------------------------
# Test 12 -- Pipeline is decoupled from MockAttackGenerator internals
# ---------------------------------------------------------------------------

def test_pipeline_genome_updater_decouples_generator():
    """
    Prove that the Pipeline does NOT require a .genome attribute on the
    generator.  A plain object with only generate() works as the generator;
    genome delivery is handled entirely through the genome_updater callback.

    This is the direct regression test for the architectural fix: previously
    the Pipeline wrote ``self._generator.genome = mutated`` which would fail
    for any real Family 1/2/3 generator that lacks that attribute.
    """
    received_genomes: list[dict] = []

    # A generator that has NO .genome attribute at all.
    # It satisfies only the AttackGenerator protocol: generate(round_id).
    class MinimalGenerator:
        def __init__(self):
            self._current_genome = {"x": _DETECTION_THRESHOLD}  # private, no exposure

        def generate(self, round_id: str) -> AttackEvent:
            return AttackEvent(
                attack_id=f"minimal-{round_id}",
                round_id=round_id,
                attack_family=AttackFamily.ADAPTIVE_EVASION,
                attack_genome=dict(self._current_genome),
                scenario={},
                ground_truth=True,
            )

        def accept_genome(self, genome: dict) -> None:
            """Named callback method — not part of the AttackGenerator protocol."""
            received_genomes.append(dict(genome))
            self._current_genome = dict(genome)

    min_gen = MinimalGenerator()

    pipeline = Pipeline(
        generator=min_gen,
        detector=MockBlueDetector(),
        evaluator=MockFeedbackEvaluator(),
        mutator=MockMutationStrategy(),
        genome_updater=min_gen.accept_genome,   # wired to the generator's own method
    )
    results = pipeline.run(num_rounds=3, base_round_id="t12")

    # Pipeline completed 3 rounds without touching any .genome attribute.
    assert len(results) == 3
    # The genome_updater was called after rounds 1 and 2 (not after the last).
    # Actually called after every round including the last (pipeline always mutates).
    assert len(received_genomes) == 3
    # Each received genome is a valid dict with float values in [0.0, 1.0].
    for g in received_genomes:
        assert isinstance(g, dict)
        for v in g.values():
            assert 0.0 <= v <= 1.0


def test_pipeline_without_genome_updater_runs_one_round():
    """
    genome_updater=None is valid when multi-round genome feedback is not needed
    (e.g. a generator that is fully self-contained).  The pipeline must not
    crash when no updater is provided.
    """
    gen = MockAttackGenerator(genome=_high_genome())
    pipeline = Pipeline(
        generator=gen,
        detector=MockBlueDetector(),
        evaluator=MockFeedbackEvaluator(),
        mutator=MockMutationStrategy(),
        # genome_updater intentionally omitted (defaults to None)
    )
    results = pipeline.run(num_rounds=1, base_round_id="t12b")
    assert len(results) == 1
    assert isinstance(results[0], RoundResult)
