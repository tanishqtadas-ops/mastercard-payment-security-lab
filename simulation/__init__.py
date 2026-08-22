"""
simulation package — Round lifecycle orchestration.

Exports the shared interfaces and the Round Controller.
"""

from .interfaces import AttackGenerator, BlueTeamDetector, FeedbackEvaluator, MutationStrategy
from .round_controller import RoundController, RoundControllerError

__all__ = [
    "AttackGenerator",
    "BlueTeamDetector",
    "FeedbackEvaluator",
    "MutationStrategy",
    "RoundController",
    "RoundControllerError",
]
