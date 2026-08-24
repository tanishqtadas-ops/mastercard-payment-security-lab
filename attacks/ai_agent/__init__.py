"""
attacks/ai_agent — Family 2 (AI-Agent Behavior) attack module.
"""

from .generator import (
    AIAgentAttackGenerator,
    FAMILY2_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from .mutator import (
    AIAgentMutationStrategy,
    DEFAULT_DETECTED_DECAY,
    DEFAULT_MISSED_BOOST,
)

__all__ = [
    "AIAgentAttackGenerator",
    "AIAgentMutationStrategy",
    "FAMILY2_GENOME_DIMENSIONS",
    "DEFAULT_ATTACK_GENOME",
    "DEFAULT_LEGITIMATE_GENOME",
    "DEFAULT_DETECTED_DECAY",
    "DEFAULT_MISSED_BOOST",
]
