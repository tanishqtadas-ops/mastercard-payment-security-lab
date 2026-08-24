"""
blue_team/ai_agent/evaluator.py — Family 2 (AI-Agent Behavior) Feedback Evaluator.

Compares detector predictions against ground truth and produces structured
BlueTeamFeedback for Family 2 simulation rounds.
"""

from typing import Any, Dict

from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback


class AIAgentFeedbackEvaluator:
    """
    Evaluates Blue-Team detection results against ground-truth attack events.

    Produces BlueTeamFeedback containing accuracy flags (detected, false positive,
    false negative), risk scores, and contributing features.

    Satisfies the FeedbackEvaluator protocol in simulation.interfaces.
    """

    def evaluate(
        self,
        event: AttackEvent,
        prediction: PredictionResult,
    ) -> BlueTeamFeedback:
        """
        Evaluate detector outcome and return BlueTeamFeedback.

        Args:
            event: The AttackEvent evaluated during this round.
            prediction: The PredictionResult produced by the detector.

        Returns:
            BlueTeamFeedback structured record.
        """
        is_attack = bool(event.ground_truth)
        predicted_fraud = bool(prediction.prediction)

        detected = predicted_fraud and is_attack
        false_negative = (not predicted_fraud) and is_attack
        false_positive = predicted_fraud and (not is_attack)

        # Map the detector's weighted feature contributions (w_i * dimension_anomaly)
        # to BlueTeamFeedback.important_features so downstream mutation strategies can
        # inspect the exact contribution weights that triggered detection.
        weighted_feature_contributions: Dict[str, float] = dict(
            prediction.feature_contributions or {}
        )

        explanation_data: Dict[str, Any] = {
            "ground_truth": is_attack,
            "prediction": predicted_fraud,
            "risk_score": prediction.risk_score,
            "model_version": prediction.model_version,
            "explanation": prediction.explanation,
            "attack_family": event.attack_family.value,
        }

        return BlueTeamFeedback(
            feedback_id=f"fb-f2-{event.round_id}",
            round_reference=event.round_id,
            detected=detected,
            false_positive=false_positive,
            false_negative=false_negative,
            risk_score=prediction.risk_score,
            important_features=weighted_feature_contributions,
            explanation_data=explanation_data,
        )
