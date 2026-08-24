"""
attacks/synthetic_identity — Family 3 (Synthetic Identity) attack module.
"""

from .generator import (
    SyntheticIdentityAttackGenerator,
    FAMILY3_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)

__all__ = [
    "SyntheticIdentityAttackGenerator",
    "FAMILY3_GENOME_DIMENSIONS",
    "DEFAULT_ATTACK_GENOME",
    "DEFAULT_LEGITIMATE_GENOME",
]
