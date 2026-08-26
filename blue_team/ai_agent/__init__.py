"""
blue_team/ai_agent — Family 2 (AI-Agent Behavior) defense module.
"""

from .detector import (
    AIAgentBlueDetector,
    AgentMandate,
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_DETECTION_THRESHOLD,
    MODEL_VERSION,
    extract_mandate_features,
)
from .evaluator import AIAgentFeedbackEvaluator

__all__ = [
    "AIAgentBlueDetector",
    "AIAgentFeedbackEvaluator",
    "AgentMandate",
    "DEFAULT_FEATURE_WEIGHTS",
    "DEFAULT_DETECTION_THRESHOLD",
    "MODEL_VERSION",
    "extract_mandate_features",
]
