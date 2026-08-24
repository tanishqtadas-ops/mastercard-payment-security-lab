"""
blue_team/ai_agent/detector.py — Family 2 (AI-Agent Behavior) Blue-Team Detector.

Detects unauthorized or malicious AI-agent payment behavior using a transparent,
deterministic heuristic weighted across all six Family 2 genome dimensions.
"""

from typing import Dict, Optional

from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult


# Default weighted contributions for Family 2 dimensions (sum to 1.0)
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "intent_amount_deviation": 0.25,
    "intent_category_deviation": 0.20,
    "permission_scope_deviation": 0.25,
    "agent_identity_confidence": 0.10,  # Applied to (1.0 - confidence) as identity anomaly
    "session_provenance_anomaly": 0.10,
    "purchase_velocity": 0.10,
}

# Detection threshold: risk_score >= threshold triggers fraud alert
DEFAULT_DETECTION_THRESHOLD: float = 0.50

MODEL_VERSION: str = "heuristic-family2-v1"


class AIAgentBlueDetector:
    """
    Evaluates Family 2 AttackEvents and computes fraud predictions.

    Calculates an interpretable risk score from intent deviation, permission scope
    deviation, agent identity confidence, session provenance, and velocity.

    Satisfies the BlueTeamDetector protocol in simulation.interfaces.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        weights: Optional[Dict[str, float]] = None,
        model_version: str = MODEL_VERSION,
    ) -> None:
        """
        Initialize the detector.

        Args:
            threshold: Fraud decision threshold in [0.0, 1.0].
            weights: Optional dictionary of feature weights summing to 1.0.
            model_version: Identifier string for model traceability.
        """
        self.threshold = threshold
        self.weights = dict(weights or DEFAULT_FEATURE_WEIGHTS)
        self.model_version = model_version

    def detect(self, event: AttackEvent) -> PredictionResult:
        """
        Run detection on an AttackEvent and return a PredictionResult.

        Args:
            event: AttackEvent containing Family 2 attack_genome.

        Returns:
            PredictionResult containing boolean prediction, risk score, explanation,
            and feature contributions.
        """
        genome = event.attack_genome

        # Extract normalized dimensional deviations
        amt_dev = genome.get("intent_amount_deviation", 0.0)
        cat_dev = genome.get("intent_category_deviation", 0.0)
        scope_dev = genome.get("permission_scope_deviation", 0.0)
        # Identity anomaly is inverse of identity confidence
        id_conf = genome.get("agent_identity_confidence", 1.0)
        id_anom = max(0.0, min(1.0 - id_conf, 1.0))
        sess_anom = genome.get("session_provenance_anomaly", 0.0)
        vel_dev = genome.get("purchase_velocity", 0.0)

        # Compute weighted contributions
        w_amt = self.weights.get("intent_amount_deviation", 0.25)
        w_cat = self.weights.get("intent_category_deviation", 0.20)
        w_scope = self.weights.get("permission_scope_deviation", 0.25)
        w_id = self.weights.get("agent_identity_confidence", 0.10)
        w_sess = self.weights.get("session_provenance_anomaly", 0.10)
        w_vel = self.weights.get("purchase_velocity", 0.10)

        contrib_amt = w_amt * amt_dev
        contrib_cat = w_cat * cat_dev
        contrib_scope = w_scope * scope_dev
        contrib_id = w_id * id_anom
        contrib_sess = w_sess * sess_anom
        contrib_vel = w_vel * vel_dev

        feature_contributions: Dict[str, float] = {
            "intent_amount_deviation": round(contrib_amt, 4),
            "intent_category_deviation": round(contrib_cat, 4),
            "permission_scope_deviation": round(contrib_scope, 4),
            "agent_identity_confidence": round(contrib_id, 4),
            "session_provenance_anomaly": round(contrib_sess, 4),
            "purchase_velocity": round(contrib_vel, 4),
        }

        total_risk = sum(feature_contributions.values())
        risk_score = min(max(total_risk, 0.0), 1.0)
        risk_score = round(risk_score, 4)

        is_fraud = risk_score >= self.threshold

        # Generate human-readable explanation
        if is_fraud:
            top_drivers = sorted(
                feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            key_signals = [
                f"{k} (contrib: {v:.3f})" for k, v in top_drivers if v > 0.05
            ]
            signals_str = ", ".join(key_signals) if key_signals else "cumulative deviation"
            explanation = (
                f"Flagged unauthorized AI-agent behavior (risk {risk_score:.2f} >= threshold {self.threshold:.2f}). "
                f"Primary drivers: {signals_str}."
            )
        else:
            explanation = (
                f"Agent transaction approved within authorized envelope "
                f"(risk {risk_score:.2f} < threshold {self.threshold:.2f})."
            )

        return PredictionResult(
            prediction_id=f"pred-f2-{event.round_id}",
            prediction=is_fraud,
            risk_score=risk_score,
            model_version=self.model_version,
            explanation=explanation,
            feature_contributions=feature_contributions,
        )
