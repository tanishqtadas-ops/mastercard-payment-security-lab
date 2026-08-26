"""
simulation package — Round lifecycle orchestration.

Exports the shared interfaces and the Round Controller.
"""

from .interfaces import AttackGenerator, BlueTeamDetector, FeedbackEvaluator, MutationStrategy
from .round_controller import RoundController, RoundControllerError
from .pipeline import Pipeline

__all__ = [
    "AttackGenerator",
    "BlueTeamDetector",
    "FeedbackEvaluator",
    "MutationStrategy",
    "RoundController",
    "RoundControllerError",
    "Pipeline",
]

