"""
blue_team/ai_agent — Family 2 (AI-Agent Behavior) defense module.
"""

from .detector import (
    AIAgentBlueDetector,
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_DETECTION_THRESHOLD,
    MODEL_VERSION,
)
from .evaluator import AIAgentFeedbackEvaluator

__all__ = [
    "AIAgentBlueDetector",
    "AIAgentFeedbackEvaluator",
    "DEFAULT_FEATURE_WEIGHTS",
    "DEFAULT_DETECTION_THRESHOLD",
    "MODEL_VERSION",
]
