"""
attacks/synthetic_identity — Family 3 (Synthetic Identity) attack module.
"""

from .generator import (
    SyntheticIdentityAttackGenerator,
    FAMILY3_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from .mutator import (
    SyntheticIdentityMutationStrategy,
    SyntheticIdentityMutator,
    DEFAULT_DETECTED_STEP,
    DEFAULT_MISSED_STEP,
)

__all__ = [
    "SyntheticIdentityAttackGenerator",
    "SyntheticIdentityMutationStrategy",
    "SyntheticIdentityMutator",
    "FAMILY3_GENOME_DIMENSIONS",
    "DEFAULT_ATTACK_GENOME",
    "DEFAULT_LEGITIMATE_GENOME",
    "DEFAULT_DETECTED_STEP",
    "DEFAULT_MISSED_STEP",
]

