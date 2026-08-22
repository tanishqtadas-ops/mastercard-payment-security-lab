"""
simulation/interfaces.py — Shared protocol interfaces for the round lifecycle.

Each interface represents a single responsibility in the attack/defence loop.
Concrete family implementations (Family 1, 2, 3) will satisfy these protocols
without requiring the Round Controller to know which family is in use.

Design notes
------------
* Python ``typing.Protocol`` is used so implementors do NOT need to inherit
  from a base class — duck typing at the call site, checked statically.
* The MutationStrategy interface is included as a future dependency boundary.
  Phase 3 does not implement mutation behaviour, but the boundary is declared
  here so Phase 4+ can plug in without touching the controller.
"""

from typing import Protocol, runtime_checkable

from schemas import AttackEvent, PredictionResult, BlueTeamFeedback


@runtime_checkable
class AttackGenerator(Protocol):
    """
    Generates a single AttackEvent for the current round.

    The generator knows everything about a specific attack family, but the
    Round Controller only sees this interface.
    """

    def generate(self, round_id: str) -> AttackEvent:
        """
        Produce an AttackEvent for the given round_id.

        Args:
            round_id: Unique identifier for the current simulation round.

        Returns:
            A fully-populated AttackEvent ready for Blue-Team evaluation.
        """
        ...


@runtime_checkable
class BlueTeamDetector(Protocol):
    """
    Evaluates an AttackEvent and returns a PredictionResult.

    The detector embodies the Blue Team's detection capability for a
    specific attack family.  Phase 3 does not implement real ML models;
    any object satisfying this protocol (including mocks) can be used.
    """

    def detect(self, event: AttackEvent) -> PredictionResult:
        """
        Run detection on an attack event.

        Args:
            event: The AttackEvent produced by the AttackGenerator.

        Returns:
            A PredictionResult containing the prediction flag and risk score.
        """
        ...


@runtime_checkable
class FeedbackEvaluator(Protocol):
    """
    Compares a PredictionResult against an AttackEvent's ground truth and
    produces structured BlueTeamFeedback.

    Keeping evaluation separate from detection makes it straightforward to
    change how feedback is calculated without modifying the detector or
    the controller.
    """

    def evaluate(
        self,
        event: AttackEvent,
        prediction: PredictionResult,
    ) -> BlueTeamFeedback:
        """
        Produce feedback from an attack event and its detection result.

        Args:
            event:      The original AttackEvent (carries ground_truth).
            prediction: The PredictionResult from the BlueTeamDetector.

        Returns:
            BlueTeamFeedback describing detection outcome and key signals.
        """
        ...


@runtime_checkable
class MutationStrategy(Protocol):
    """
    Adapts an attack genome based on feedback from the previous round.

    Phase 3 declares this boundary but does NOT implement mutation logic.
    Future phases will provide concrete implementations per attack family.
    """

    def mutate(
        self,
        genome: dict,
        feedback: BlueTeamFeedback,
    ) -> dict:
        """
        Produce a new genome informed by the round's feedback.

        Args:
            genome:   The attack genome from the current round.
            feedback: The BlueTeamFeedback from the current round.

        Returns:
            A new genome dictionary for use in the next round.
        """
        ...
