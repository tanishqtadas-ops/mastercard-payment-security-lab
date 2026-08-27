"""
attacks/transaction_evasion/mutator.py — Family 1 (Transaction Evasion) Mutation Strategy.

Adapts the six canonical Family 1 genome dimensions based on Blue-Team feedback:
- When DETECTED: Decays prominent anomaly signals (amount, velocity, location, device, time, sequence)
  guided by detector feature contributions/importance to create a stealthier, harder-to-detect variant.
- When MISSED: Preserves the successful evasion structure while exploring nearby bounded genome
  variants (e.g. testing slightly higher payoff or varied patterns) without blindly increasing all dimensions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from schemas.feedback import BlueTeamFeedback
from mutation.genome_engine import validate_genome


# Canonical Family 1 genome dimensions (defined in MASTER_SPEC.md § 3)
FAMILY1_GENOME_DIMENSIONS: Tuple[str, ...] = (
    "amount_deviation",
    "velocity_deviation",
    "device_novelty",
    "location_deviation",
    "time_deviation",
    "sequence_anomaly",
)

# Default step sizes for feedback-driven mutation
DEFAULT_DETECTED_STEP: float = 0.08
DEFAULT_MISSED_STEP: float = 0.03

# Exploration factors for missed (evaded) rounds: probe higher payoff with bounded step weights
_MISSED_EXPLORATION_FACTORS: Dict[str, float] = {
    "amount_deviation": 0.04,     # Probe higher transaction amount
    "velocity_deviation": 0.03,   # Probe slightly faster velocity
    "device_novelty": 0.02,       # Minor device variation
    "location_deviation": 0.02,   # Minor location exploration
    "time_deviation": 0.02,       # Minor timing adjustment
    "sequence_anomaly": 0.03,     # Probe varied merchant categories
}


class TransactionMutationStrategy:
    """
    Implements adaptive, bounded mutation for Family 1 attack genomes.

    Satisfies the MutationStrategy protocol in simulation.interfaces.
    """

    def __init__(
        self,
        detected_step: float = DEFAULT_DETECTED_STEP,
        missed_step: float = DEFAULT_MISSED_STEP,
    ) -> None:
        """
        Initialize the mutation strategy with custom step sizes.

        Args:
            detected_step: Base step size by which signals are decayed upon detection.
            missed_step: Base step size by which signals are explored upon successful evasion.
        """
        self.detected_step = detected_step
        self.missed_step = missed_step

    def mutate(
        self,
        genome: Dict[str, float],
        feedback: BlueTeamFeedback,
    ) -> Dict[str, float]:
        """
        Mutate the Family 1 genome based on round feedback.

        Args:
            genome: Attack genome from the completed round.
            feedback: BlueTeamFeedback from the completed round.

        Returns:
            A new validated genome dictionary with values clamped to [0.0, 1.0].
        """
        validate_genome(genome)

        # Ensure all canonical dimensions exist
        missing_keys = set(FAMILY1_GENOME_DIMENSIONS) - set(genome.keys())
        if missing_keys:
            raise ValueError(
                f"Genome missing canonical Family 1 dimensions: {sorted(missing_keys)}"
            )

        important_features = feedback.important_features or {}
        mutated: Dict[str, float] = {}

        if feedback.detected:
            # -------------------------------------------------------------------
            # Attack was DETECTED:
            # 1. Identify high-contributing detection drivers from feedback.important_features
            # 2. Reduce detection signals with prioritized decay on top drivers
            # 3. Apply standard decay to other dimensions to improve overall stealth
            # -------------------------------------------------------------------
            for dim in FAMILY1_GENOME_DIMENSIONS:
                val = genome[dim]
                feature_importance = float(important_features.get(dim, 0.0))

                # If this feature strongly contributed to detection, decay more aggressively
                if feature_importance > 0.15:
                    step = self.detected_step * 1.5
                elif feature_importance > 0.0:
                    step = self.detected_step * 1.0
                else:
                    step = self.detected_step * 0.75

                new_val = min(max(val - step, 0.0), 1.0)
                mutated[dim] = round(new_val, 4)

        else:
            # -------------------------------------------------------------------
            # Attack was MISSED (Successful Evasion):
            # 1. Preserve the successful evasion structure
            # 2. Make small bounded changes exploring nearby genome variants
            # 3. Do not simply increase every genome dimension uniformly
            # -------------------------------------------------------------------
            for dim in FAMILY1_GENOME_DIMENSIONS:
                val = genome[dim]
                factor = _MISSED_EXPLORATION_FACTORS.get(dim, 0.02)
                step = self.missed_step * (factor / 0.03)

                new_val = min(max(val + step, 0.0), 1.0)
                mutated[dim] = round(new_val, 4)

        # Validate that mutated genome conforms to schema bounds and types
        validate_genome(mutated)
        return mutated


# Alias for flexible importing
TransactionMutator = TransactionMutationStrategy
