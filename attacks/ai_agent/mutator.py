"""
attacks/ai_agent/mutator.py — Family 2 (AI-Agent Behavior) Mutation Strategy.

Adapts the six canonical Family 2 genome dimensions based on Blue-Team feedback.
- When DETECTED: Decays detection signals and reinforces agent identity confidence
  to produce a stealthier, less detectable variant.
- When MISSED: Escalates deviation dimensions to test evasion limits and produce a
  stronger attack variant.
"""

from typing import Dict, Optional

from schemas.feedback import BlueTeamFeedback
from mutation.genome_engine import validate_genome


# Default step sizes for feedback-driven mutation
DEFAULT_DETECTED_DECAY: float = 0.10
DEFAULT_MISSED_BOOST: float = 0.05


class AIAgentMutationStrategy:
    """
    Implements adaptive mutation for Family 2 attack genomes.

    Satisfies the MutationStrategy protocol in simulation.interfaces.
    """

    def __init__(
        self,
        detected_decay: float = DEFAULT_DETECTED_DECAY,
        missed_boost: float = DEFAULT_MISSED_BOOST,
    ) -> None:
        """
        Initialize the mutation strategy with custom step sizes.

        Args:
            detected_decay: Step size by which signals are decayed upon detection.
            missed_boost: Step size by which signals are boosted upon successful evasion.
        """
        self.detected_decay = detected_decay
        self.missed_boost = missed_boost

    def mutate(
        self,
        genome: Dict[str, float],
        feedback: BlueTeamFeedback,
    ) -> Dict[str, float]:
        """
        Mutate the Family 2 genome based on round feedback.

        Args:
            genome: Attack genome from the completed round.
            feedback: BlueTeamFeedback from the completed round.

        Returns:
            A new validated genome dictionary with values clamped to [0.0, 1.0].
        """
        validate_genome(genome)

        mutated: Dict[str, float] = {}
        # feedback.important_features carries the detector's weighted feature contributions (w_i * dimension_anomaly).
        # We use these weighted contributions to apply stronger decay to the features that most contributed to detection.
        weighted_contributions = feedback.important_features or {}

        if feedback.detected:
            # Attack was detected by Blue Team:
            # 1. Reduce high-deviation signals (amount, category, scope, velocity, provenance anomaly)
            # 2. Increase agent_identity_confidence (attacker impersonates legitimate agent better)
            for key, val in genome.items():
                feature_weight = weighted_contributions.get(key, 0.0)
                decay_scale = 1.2 if feature_weight > 0.15 else 1.0
                effective_step = self.detected_decay * decay_scale

                if key == "agent_identity_confidence":
                    # Improving identity confidence reduces identity anomaly
                    new_val = min(val + effective_step, 1.0)
                else:
                    new_val = max(val - effective_step, 0.0)

                mutated[key] = round(new_val, 4)
        else:
            # Attack was missed (successful evasion):
            # 1. Attacker seeks higher payoff: increase deviations and velocity
            # 2. Keep identity confidence high to avoid triggering identity alarms
            for key, val in genome.items():
                if key == "agent_identity_confidence":
                    # Maintain high identity confidence to preserve stealth
                    new_val = min(val + (self.missed_boost * 0.5), 1.0)
                elif key == "session_provenance_anomaly":
                    # Keep session anomaly modest during expansion
                    new_val = min(val + (self.missed_boost * 0.5), 1.0)
                else:
                    new_val = min(val + self.missed_boost, 1.0)

                mutated[key] = round(new_val, 4)

        # Validate that mutated genome conforms to schema bounds
        validate_genome(mutated)
        return mutated
