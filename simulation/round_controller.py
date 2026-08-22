"""
simulation/round_controller.py — Orchestrates a single simulation round.

The Round Controller owns the pipeline:

    AttackGenerator → AttackEvent
        → BlueTeamDetector → PredictionResult
            → FeedbackEvaluator → BlueTeamFeedback
                → RoundResult

It coordinates these steps through the shared interfaces defined in
``simulation.interfaces``.  It does NOT know which attack family is active,
how detection works internally, or how mutation mathematics are applied.
"""

import uuid
from typing import Dict, Any

from schemas import AttackEvent, PredictionResult, BlueTeamFeedback, RoundResult
from .interfaces import AttackGenerator, BlueTeamDetector, FeedbackEvaluator


class RoundControllerError(Exception):
    """Raised when the Round Controller encounters an unrecoverable problem."""
    pass


class RoundController:
    """
    Orchestrates one simulation round end-to-end.

    Dependencies are injected at construction time so they can be replaced
    with test doubles without modifying the controller.

    Args:
        generator:   Produces the AttackEvent for each round.
        detector:    Evaluates the AttackEvent and returns a PredictionResult.
        evaluator:   Compares prediction against ground truth and produces feedback.
    """

    def __init__(
        self,
        generator: AttackGenerator,
        detector: BlueTeamDetector,
        evaluator: FeedbackEvaluator,
    ) -> None:
        if not isinstance(generator, AttackGenerator):
            raise RoundControllerError(
                "generator must satisfy the AttackGenerator protocol"
            )
        if not isinstance(detector, BlueTeamDetector):
            raise RoundControllerError(
                "detector must satisfy the BlueTeamDetector protocol"
            )
        if not isinstance(evaluator, FeedbackEvaluator):
            raise RoundControllerError(
                "evaluator must satisfy the FeedbackEvaluator protocol"
            )

        self._generator = generator
        self._detector = detector
        self._evaluator = evaluator

    def run_round(
        self,
        round_id: str | None = None,
        outcome_metrics: Dict[str, Any] | None = None,
    ) -> RoundResult:
        """
        Execute a single simulation round.

        Steps:
          1. Generate an AttackEvent via the AttackGenerator.
          2. Pass it to the BlueTeamDetector → PredictionResult.
          3. Pass both to the FeedbackEvaluator → BlueTeamFeedback.
          4. Assemble and return a RoundResult.

        Args:
            round_id:        Optional round identifier; a UUID is generated if
                             omitted.
            outcome_metrics: Optional caller-supplied metrics to embed in the
                             RoundResult (e.g. wall-clock time, round index).

        Returns:
            A fully-populated RoundResult.

        Raises:
            RoundControllerError: If any dependency returns an object of the
                                  wrong type or raises an unexpected exception.
        """
        if round_id is None:
            round_id = str(uuid.uuid4())

        if not round_id:
            raise RoundControllerError("round_id must not be empty")

        # --- Step 1: generate attack ---
        try:
            event = self._generator.generate(round_id)
        except Exception as exc:
            raise RoundControllerError(
                f"AttackGenerator raised an error: {exc}"
            ) from exc

        if not isinstance(event, AttackEvent):
            raise RoundControllerError(
                f"AttackGenerator.generate() must return AttackEvent, "
                f"got {type(event).__name__}"
            )

        # --- Step 2: detect ---
        try:
            prediction = self._detector.detect(event)
        except Exception as exc:
            raise RoundControllerError(
                f"BlueTeamDetector raised an error: {exc}"
            ) from exc

        if not isinstance(prediction, PredictionResult):
            raise RoundControllerError(
                f"BlueTeamDetector.detect() must return PredictionResult, "
                f"got {type(prediction).__name__}"
            )

        # --- Step 3: evaluate feedback ---
        try:
            feedback = self._evaluator.evaluate(event, prediction)
        except Exception as exc:
            raise RoundControllerError(
                f"FeedbackEvaluator raised an error: {exc}"
            ) from exc

        if not isinstance(feedback, BlueTeamFeedback):
            raise RoundControllerError(
                f"FeedbackEvaluator.evaluate() must return BlueTeamFeedback, "
                f"got {type(feedback).__name__}"
            )

        # --- Step 4: assemble result ---
        return RoundResult(
            round_id=round_id,
            attack_event=event,
            prediction_result=prediction,
            feedback=feedback,
            outcome_metrics=outcome_metrics or {},
        )
