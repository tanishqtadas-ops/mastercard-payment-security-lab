"""
simulation/mock_pipeline.py -- Deterministic mock Red/Blue pipeline for Phase 4.

Purpose
-------
Prove that future real Family 1/2/3 implementations can plug into the existing
architecture and that feedback from one round can influence the next round.

This is NOT a real fraud-detection system.
This is NOT a Family 1/2/3 implementation.
This is NOT an ML or RL system.

All components are intentionally simple, deterministic, and replaceable.

Components
----------
MockAttackGenerator
    Produces an AttackEvent whose genome is supplied by the caller.
    Genome is updated externally (by the Pipeline) between rounds so that
    mutation feedback can influence the next attack variant.

MockBlueDetector
    Detects attacks by comparing the average of genome values to a threshold.
    Deterministic: same genome -> same prediction every time.

MockFeedbackEvaluator
    Compares prediction.prediction against event.ground_truth to decide
    detected / false_positive / false_negative.

MockMutationStrategy
    If the attack was detected   -> reduce every genome value by DETECTED_DECAY.
    If the attack was missed     -> increase every genome value by MISSED_BOOST.
    Values are clamped to [0.0, 1.0] to keep them valid genome floats.
    Intentionally simple: the purpose is only to demonstrate
        previous feedback -> changed next genome.

Pipeline
    Assembles the four components above together with the existing
    RoundController and runs N rounds sequentially.
    After each round the genome is mutated and fed back to the generator.
"""

import uuid
from typing import Callable, Dict, List, Optional

from schemas import (
    AttackEvent,
    AttackFamily,
    BlueTeamFeedback,
    PredictionResult,
    RoundResult,
)
from .interfaces import AttackGenerator, BlueTeamDetector, FeedbackEvaluator, MutationStrategy
from .round_controller import RoundController


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

_DEFAULT_GENOME: Dict[str, float] = {
    "signal_a": 0.5,
    "signal_b": 0.5,
}

# Detection threshold: average(genome.values()) >= threshold -> predict fraud.
_DETECTION_THRESHOLD: float = 0.7

# Mutation steps
_DETECTED_DECAY: float = 0.1
_MISSED_BOOST: float = 0.05

_MOCK_FAMILY: AttackFamily = AttackFamily.ADAPTIVE_EVASION


# ---------------------------------------------------------------------------
# Mock component 1 -- Attack Generator
# ---------------------------------------------------------------------------

class MockAttackGenerator:
    """
    Produces a deterministic AttackEvent from a mutable genome dict.

    The genome is updated via ``set_genome()`` between rounds so that
    mutation feedback can influence the next attack variant.

    ``set_genome`` is the explicit, named genome-feedback port for this
    mock implementation.  It is NOT part of the ``AttackGenerator``
    protocol — real Family 1/2/3 generators carry their own internal
    genome management and do not need this method.

    Satisfies the ``AttackGenerator`` protocol declared in
    ``simulation.interfaces``.
    """

    def __init__(self, genome: Dict[str, float] | None = None) -> None:
        self._genome: Dict[str, float] = dict(genome or _DEFAULT_GENOME)

    def set_genome(self, genome: Dict[str, float]) -> None:
        """Replace the active genome for subsequent generate() calls."""
        self._genome = dict(genome)

    def generate(self, round_id: str) -> AttackEvent:
        """Return an AttackEvent whose genome reflects the current state."""
        return AttackEvent(
            attack_id=f"mock-attack-{round_id}",
            round_id=round_id,
            attack_family=_MOCK_FAMILY,
            attack_genome=dict(self._genome),
            scenario={"source": "mock_pipeline"},
            ground_truth=True,
            metadata={"generator": "MockAttackGenerator"},
        )


# ---------------------------------------------------------------------------
# Mock component 2 -- Blue-Team Detector
# ---------------------------------------------------------------------------

class MockBlueDetector:
    """
    Detects attacks using the average of genome values vs. a threshold.

    Rule:
        average(attack_genome.values()) >= _DETECTION_THRESHOLD -> detected
        average(attack_genome.values()) <  _DETECTION_THRESHOLD -> missed

    The risk score equals the average, already in [0.0, 1.0] for valid genomes.

    Satisfies the ``BlueTeamDetector`` protocol declared in
    ``simulation.interfaces``.
    """

    def detect(self, event: AttackEvent) -> PredictionResult:
        values = list(event.attack_genome.values())
        risk_score = sum(values) / max(len(values), 1)
        risk_score = min(max(risk_score, 0.0), 1.0)
        detected = risk_score >= _DETECTION_THRESHOLD

        return PredictionResult(
            prediction_id=f"mock-pred-{event.round_id}",
            prediction=detected,
            risk_score=risk_score,
            model_version="mock-v1",
            explanation=(
                "Detected: genome signal above threshold"
                if detected
                else "Missed: genome signal below threshold"
            ),
            feature_contributions=dict(event.attack_genome),
        )


# ---------------------------------------------------------------------------
# Mock component 3 -- Feedback Evaluator
# ---------------------------------------------------------------------------

class MockFeedbackEvaluator:
    """
    Compares prediction against ground truth and produces BlueTeamFeedback.

    Logic (ground_truth is always True for mock attacks):
        prediction=True,  ground_truth=True  -> detected=True,  FN=False, FP=False
        prediction=False, ground_truth=True  -> detected=False, FN=True,  FP=False

    Satisfies the ``FeedbackEvaluator`` protocol declared in
    ``simulation.interfaces``.
    """

    def evaluate(
        self,
        event: AttackEvent,
        prediction: PredictionResult,
    ) -> BlueTeamFeedback:
        detected = prediction.prediction and event.ground_truth
        false_negative = (not prediction.prediction) and event.ground_truth
        false_positive = prediction.prediction and (not event.ground_truth)

        return BlueTeamFeedback(
            feedback_id=f"mock-fb-{event.round_id}",
            round_reference=event.round_id,
            detected=detected,
            false_positive=false_positive,
            false_negative=false_negative,
            risk_score=prediction.risk_score,
            important_features=dict(prediction.feature_contributions or {}),
            explanation_data={
                "ground_truth": event.ground_truth,
                "prediction": prediction.prediction,
            },
        )


# ---------------------------------------------------------------------------
# Mock component 4 -- Mutation Strategy
# ---------------------------------------------------------------------------

class MockMutationStrategy:
    """
    Adapts a genome based on the previous round feedback.

    Rules (intentionally simple):
        detected -> decay each dimension by _DETECTED_DECAY
                    (attacker reduces signals that caused detection)
        missed   -> boost each dimension by _MISSED_BOOST
                    (attacker reinforces what worked)

    All values clamped to [0.0, 1.0].

    Satisfies the ``MutationStrategy`` protocol declared in
    ``simulation.interfaces``.
    """

    def mutate(
        self,
        genome: Dict[str, float],
        feedback: BlueTeamFeedback,
    ) -> Dict[str, float]:
        if feedback.detected:
            step = -_DETECTED_DECAY
        else:
            step = _MISSED_BOOST

        return {
            key: min(max(val + step, 0.0), 1.0)
            for key, val in genome.items()
        }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Runs multiple Red/Blue rounds sequentially, feeding mutation output back
    to the generator before each subsequent round via a caller-supplied callback.

    Flow per round:
      1. RoundController.run_round() executes the full pipeline.
      2. MutationStrategy.mutate() adapts the genome from the round result.
      3. ``genome_updater(mutated_genome)`` is called so the caller can route
         the new genome wherever the generator expects it.

    Design note — the genome-feedback handoff
    -----------------------------------------
    The ``AttackGenerator`` protocol only guarantees ``generate(round_id)``.
    It does NOT guarantee any genome-setter attribute or method.  Therefore
    the Pipeline does NOT write to generator internals directly.

    Instead, the caller provides ``genome_updater``: a small callable that
    bridges the Pipeline's mutation output to the generator's input channel.
    For the mock this is ``gen.set_genome``.  For a real Family 1/2/3
    generator it could be any appropriate method — or ``None`` if the
    generator manages its own genome state independently.

    This keeps the Pipeline typed against the shared Phase 3 protocol
    interfaces and free of any Mock-class-specific knowledge.

    Args:
        generator:      Any object satisfying the ``AttackGenerator`` protocol.
        detector:       Any object satisfying the ``BlueTeamDetector`` protocol.
        evaluator:      Any object satisfying the ``FeedbackEvaluator`` protocol.
        mutator:        Any object satisfying the ``MutationStrategy`` protocol.
        genome_updater: Optional callable ``(genome: dict) -> None`` invoked
                        after each round with the mutated genome.  Pass
                        ``None`` (default) when the generator manages its own
                        genome state and needs no external notification.
    """

    def __init__(
        self,
        generator: AttackGenerator,
        detector: BlueTeamDetector,
        evaluator: FeedbackEvaluator,
        mutator: MutationStrategy,
        genome_updater: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        self._mutator = mutator
        self._genome_updater = genome_updater
        self._controller = RoundController(generator, detector, evaluator)

    def run(self, num_rounds: int, base_round_id: str | None = None) -> List[RoundResult]:
        """
        Execute ``num_rounds`` simulation rounds in sequence.

        Args:
            num_rounds:    Number of rounds to execute (must be >= 1).
            base_round_id: Optional prefix for round IDs.

        Returns:
            List of RoundResult objects in execution order.

        Raises:
            ValueError: If num_rounds < 1.
        """
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")

        prefix = base_round_id or uuid.uuid4().hex[:8]
        results: List[RoundResult] = []

        for i in range(num_rounds):
            round_id = f"{prefix}-round-{i + 1}"

            result = self._controller.run_round(
                round_id=round_id,
                outcome_metrics={"round_index": i + 1},
            )
            results.append(result)

            # Compute the mutated genome for the next round.
            mutated_genome = self._mutator.mutate(
                genome=result.attack_event.attack_genome,
                feedback=result.feedback,
            )

            # Deliver it via the caller-supplied callback — the Pipeline
            # never touches generator internals directly.
            if self._genome_updater is not None:
                self._genome_updater(mutated_genome)

        return results
