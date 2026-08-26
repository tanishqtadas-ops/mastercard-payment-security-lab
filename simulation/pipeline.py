"""
simulation/pipeline.py

Multi-round adaptive simulation pipeline.

The Pipeline extends the single-round RoundController by adding
genome mutation between rounds.

Flow:

    AttackGenerator
        ↓
    AttackEvent
        ↓
    BlueTeamDetector
        ↓
    PredictionResult
        ↓
    FeedbackEvaluator
        ↓
    BlueTeamFeedback
        ↓
    MutationStrategy
        ↓
    updated genome
        ↓
    next round
"""

from __future__ import annotations

from typing import Callable, Optional

from schemas import RoundResult
from .interfaces import (
    AttackGenerator,
    BlueTeamDetector,
    FeedbackEvaluator,
    MutationStrategy,
)
from .round_controller import RoundController


class Pipeline:
    """
    Runs multiple adaptive simulation rounds.

    The Pipeline is responsible only for orchestration. It does not know
    how a specific attack family generates attacks, detects them, evaluates
    feedback, or mutates its genome.
    """

    def __init__(
        self,
        generator: AttackGenerator,
        detector: BlueTeamDetector,
        evaluator: FeedbackEvaluator,
        mutator: MutationStrategy,
        genome_updater: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._generator = generator
        self._detector = detector
        self._evaluator = evaluator
        self._mutator = mutator
        self._genome_updater = genome_updater

        # Reuse the existing Phase 3 controller for each individual round.
        self._round_controller = RoundController(
            generator=generator,
            detector=detector,
            evaluator=evaluator,
        )

    def run(
        self,
        num_rounds: int,
        base_round_id: str = "round",
    ) -> list[RoundResult]:
        """
        Execute multiple adaptive simulation rounds.

        Args:
            num_rounds:
                Number of rounds to execute. Must be >= 1.

            base_round_id:
                Prefix used to construct deterministic round IDs.

        Returns:
            List of RoundResult objects, one for each round.

        Raises:
            ValueError:
                If num_rounds is less than 1.
        """

        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")

        results: list[RoundResult] = []

        for round_index in range(1, num_rounds + 1):
            round_id = f"{base_round_id}-{round_index}"

            # ---------------------------------------------------------
            # 1. Run one complete Phase 3 round.
            # ---------------------------------------------------------
            result = self._round_controller.run_round(
                round_id=round_id,
                outcome_metrics={
                    "round_index": round_index,
                },
            )

            results.append(result)

            # ---------------------------------------------------------
            # 2. Mutate the genome using this round's feedback.
            # ---------------------------------------------------------
            #
            # The current genome is taken from the AttackEvent produced
            # by this round. This means the Pipeline does NOT need to
            # access generator internals such as generator.genome.
            #
            current_genome = dict(result.attack_event.attack_genome)

            mutated_genome = self._mutator.mutate(
                current_genome,
                result.feedback,
            )

            # ---------------------------------------------------------
            # 3. Give the mutated genome to the generator.
            # ---------------------------------------------------------
            #
            # This is intentionally done through the injected callback.
            # The Pipeline therefore works with generators that have no
            # .genome attribute at all.
            #
            if self._genome_updater is not None:
                self._genome_updater(dict(mutated_genome))

        return results