"""
blue_team/transaction/detector.py — Family 1 (Transaction Evasion) Blue-Team Detector.

Evaluates Family 1 AttackEvents by computing behavioral deviation signals between
the observed transaction and the user's historical baseline profile.
Produces PredictionResult containing fraud prediction, calibrated risk score,
human-readable explanation, and SHAP-compatible weighted feature contributions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from .feature_extractor import FEATURE_NAMES, extract_transaction_features


# Default weighted contributions for Family 1 dimensions (sum to 1.0)
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "amount_deviation": 0.25,
    "velocity_deviation": 0.20,
    "device_novelty": 0.15,
    "location_deviation": 0.15,
    "time_deviation": 0.10,
    "sequence_anomaly": 0.15,
}

# Detection threshold: risk_score >= threshold triggers fraud alert
DEFAULT_DETECTION_THRESHOLD: float = 0.50

MODEL_VERSION: str = "heuristic-family1-v1"


class TransactionBlueDetector:
    """
    Evaluates Family 1 AttackEvents and computes transaction evasion fraud predictions.

    Uses observable behavioral deviation signals (amount, velocity, device novelty,
    geographic location deviation, off-hours timing, and merchant sequence anomaly)
    relative to the user's historical baseline profile.

    Satisfies the BlueTeamDetector protocol in simulation.interfaces.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        weights: Optional[Dict[str, float]] = None,
        model_version: str = MODEL_VERSION,
    ) -> None:
        """
        Initialize the Family 1 detector.

        Args:
            threshold: Fraud decision threshold in [0.0, 1.0].
            weights: Optional dictionary of feature weights summing to 1.0.
            model_version: Identifier string for model versioning.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Detection threshold must be in [0.0, 1.0], got {threshold}")

        self.threshold = threshold
        self.model_version = model_version

        if weights is not None:
            # Validate weights
            for k in FEATURE_NAMES:
                if k not in weights:
                    raise ValueError(f"Missing weight for feature '{k}'")
            total_w = sum(weights.values())
            if not (0.95 <= total_w <= 1.05):
                raise ValueError(f"Feature weights must sum to approximately 1.0 (got {total_w})")
            self._weights = dict(weights)
        else:
            self._weights = dict(DEFAULT_FEATURE_WEIGHTS)

    @property
    def weights(self) -> Dict[str, float]:
        """Return a copy of the active feature weights."""
        return dict(self._weights)

    def detect(self, event: AttackEvent) -> PredictionResult:
        """
        Run detection on an AttackEvent and return a PredictionResult.

        Extracts observable signals strictly from event.scenario (never reading
        event.attack_genome or event.ground_truth).

        Args:
            event: AttackEvent containing Family 1 scenario data.

        Returns:
            PredictionResult containing boolean prediction, risk score, explanation,
            and feature contributions.
        """
        scenario = event.scenario or {}
        feature_signals = extract_transaction_features(scenario)

        # Calculate weighted feature contributions (w_i * signal_i)
        feature_contributions: Dict[str, float] = {}
        weighted_sum = 0.0

        for feat_name, weight in self._weights.items():
            signal_val = feature_signals.get(feat_name, 0.0)
            contrib = weight * signal_val
            feature_contributions[feat_name] = round(contrib, 4)
            weighted_sum += contrib

        risk_score = round(min(max(weighted_sum, 0.0), 1.0), 4)
        is_fraud = bool(risk_score >= self.threshold)

        # Generate structured, human-readable explanation from observable evidence
        if is_fraud:
            top_drivers = sorted(
                feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            key_signals = [
                f"{k} (impact: +{v:.3f})"
                for k, v in top_drivers
                if v >= 0.05 or feature_signals.get(k, 0.0) >= 0.50
            ]
            signals_str = ", ".join(key_signals[:3]) if key_signals else "composite behavioral anomalies"
            explanation = (
                f"Flagged anomalous transaction risk (risk {risk_score:.2f} >= threshold {self.threshold:.2f}). "
                f"Primary drivers: {signals_str}."
            )
        else:
            explanation = (
                f"Transaction verified as consistent with user historical baseline "
                f"(risk {risk_score:.2f} < threshold {self.threshold:.2f})."
            )

        return PredictionResult(
            prediction_id=f"pred-f1-{event.round_id}",
            prediction=is_fraud,
            risk_score=risk_score,
            model_version=self.model_version,
            explanation=explanation,
            feature_contributions=feature_contributions,
        )


# Alias for flexible importing
TransactionDetector = TransactionBlueDetector
