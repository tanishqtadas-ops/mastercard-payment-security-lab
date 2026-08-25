"""
attacks/synthetic_identity/mutator.py — Family 3 (Synthetic Identity) Mutation Strategy.

Adapts the six canonical Family 3 genome dimensions based on Blue-Team feedback:
- When DETECTED: Identifies the strongest detection signals from feature contributions/feedback,
  reduces those signals gradually by increasing consistency/seasoning/plausibility, and shifts
  exploration toward weaker signals with bounded steps.
- When MISSED: Preserves the successful evasion structure while exploring nearby genome
  variants with small, bounded local adjustments without blindly increasing every dimension.
"""

from typing import Dict, List, Optional, Tuple

from schemas.feedback import BlueTeamFeedback
from mutation.genome_engine import validate_genome


# Canonical Family 3 genome dimensions (defined in MASTER_SPEC.md § 3)
FAMILY3_GENOME_DIMENSIONS: Tuple[str, ...] = (
    "cross_field_consistency",
    "profile_plausibility_score",
    "contact_consistency",
    "device_history_score",
    "lifecycle_behavior_coherence",
    "time_to_risky_activity",
)

# Mapping from canonical Family 3 genome dimensions to observable detector feature indicators
DIMENSION_FEATURE_MAP: Dict[str, Tuple[str, ...]] = {
    "cross_field_consistency": (
        "cross_field_consistency",
        "name_email_anomaly",
        "ssn_format_irregularity",
        "location_mismatch_anomaly",
    ),
    "profile_plausibility_score": (
        "profile_plausibility_score",
        "profile_implausibility",
    ),
    "contact_consistency": (
        "contact_consistency",
        "is_disposable_email",
        "is_voip_carrier",
        "email_phone_tenure_deficit",
    ),
    "device_history_score": (
        "device_history_score",
        "is_emulator_device",
        "device_reputation_deficit",
        "is_datacenter_proxy_ip",
    ),
    "lifecycle_behavior_coherence": (
        "lifecycle_behavior_coherence",
        "lifecycle_incoherence",
    ),
    "time_to_risky_activity": (
        "time_to_risky_activity",
        "early_bust_out_risk",
    ),
}

# Default step sizes for feedback-driven mutation
DEFAULT_DETECTED_STEP: float = 0.08
DEFAULT_MISSED_STEP: float = 0.03

# Deterministic local perturbation weights for missed (successful evasion) exploration
_MISSED_EXPLORATION_FACTORS: Dict[str, float] = {
    "cross_field_consistency": 0.25,
    "profile_plausibility_score": -0.30,
    "contact_consistency": 0.35,
    "device_history_score": 0.20,
    "lifecycle_behavior_coherence": -0.25,
    "time_to_risky_activity": -0.50,  # Probe faster monetization/payoff
}


class SyntheticIdentityMutationStrategy:
    """
    Implements adaptive, deterministic mutation for Family 3 synthetic identity attack genomes.

    Satisfies the MutationStrategy protocol in simulation.interfaces.
    """

    def __init__(
        self,
        detected_step: float = DEFAULT_DETECTED_STEP,
        missed_step: float = DEFAULT_MISSED_STEP,
    ) -> None:
        """
        Initialize the Family 3 mutation strategy with configurable step sizes.

        Args:
            detected_step: Base step size for repairing detection signals upon detection.
            missed_step: Base step size for local neighborhood exploration upon evasion.
        """
        self.detected_step = detected_step
        self.missed_step = missed_step

    def _extract_dimension_signals(
        self, important_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Maps feedback feature contributions / SHAP values to the 6 Family 3 genome dimensions.
        """
        signals: Dict[str, float] = {}
        for dim, feat_names in DIMENSION_FEATURE_MAP.items():
            dim_values: List[float] = []
            for feat in feat_names:
                if feat in important_features:
                    val = float(important_features[feat])
                    # In SHAP / importance, positive contributions indicate fraud detection triggers
                    if val > 0.0:
                        dim_values.append(val)
                    else:
                        dim_values.append(abs(val) * 0.5)

            signals[dim] = max(dim_values) if dim_values else 0.0

        return signals

    def mutate(
        self,
        genome: Dict[str, float],
        feedback: BlueTeamFeedback,
    ) -> Dict[str, float]:
        """
        Produce an adapted Family 3 attack genome based on round feedback.

        Args:
            genome: Canonical Family 3 attack genome from the completed round.
            feedback: BlueTeamFeedback from the completed round.

        Returns:
            A new validated genome dictionary containing exactly the six Family 3
            dimensions, with values bounded strictly in [0.0, 1.0].
        """
        validate_genome(genome)

        # Ensure all canonical dimensions exist
        missing_keys = set(FAMILY3_GENOME_DIMENSIONS) - set(genome.keys())
        if missing_keys:
            raise ValueError(
                f"Genome missing canonical Family 3 dimensions: {sorted(missing_keys)}"
            )

        important_features = feedback.important_features or {}
        mutated: Dict[str, float] = {}

        if feedback.detected:
            # -------------------------------------------------------------------
            # Attack was DETECTED:
            # 1. Identify strongest detection signals from feedback / SHAP weights
            # 2. Repair strong detection signals with larger steps
            # 3. Apply modest improvements to weaker signals
            # -------------------------------------------------------------------
            signals = self._extract_dimension_signals(important_features)
            max_signal = max(signals.values()) if signals else 0.0

            for dim in FAMILY3_GENOME_DIMENSIONS:
                val = genome[dim]
                dim_signal = signals.get(dim, 0.0)

                if max_signal > 0.0 and dim_signal >= (max_signal * 0.70):
                    # Primary detection driver — apply prioritized repair
                    step = self.detected_step * 1.5
                elif dim_signal > 0.0:
                    # Moderate detection driver
                    step = self.detected_step * 1.0
                else:
                    # Weak or unobserved signal — modest exploration
                    step = self.detected_step * 0.5

                # Increasing dimension value increases consistency / seasoning / stealth
                new_val = min(max(val + step, 0.0), 1.0)
                mutated[dim] = round(new_val, 4)

        else:
            # -------------------------------------------------------------------
            # Attack was MISSED (Successful evasion):
            # 1. Preserve the successful evasion structure
            # 2. Make small bounded changes exploring nearby genome variants
            # 3. Do NOT blindly increase every dimension
            # -------------------------------------------------------------------
            for dim in FAMILY3_GENOME_DIMENSIONS:
                val = genome[dim]
                factor = _MISSED_EXPLORATION_FACTORS.get(dim, 0.0)
                step = self.missed_step * factor

                new_val = min(max(val + step, 0.0), 1.0)
                mutated[dim] = round(new_val, 4)

        # Validate that mutated genome conforms to schema bounds and types
        validate_genome(mutated)
        return mutated


# Alias for flexible importing
SyntheticIdentityMutator = SyntheticIdentityMutationStrategy
