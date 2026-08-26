"""
blue_team/ai_agent/evaluator.py — Family 2 (AI-Agent Behavior) Feedback Evaluator.

Compares Blue-Team detector predictions against ground truth and produces structured
BlueTeamFeedback for Family 2 (AI-Agent Behavior) simulation rounds.
"""

from typing import Any, Dict

from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.agent_event import AIAgentPaymentEvent


class AIAgentFeedbackEvaluator:
    """
    Evaluates Blue-Team detection results against ground-truth AI-agent payment events.

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
        Evaluate detector outcome against event ground truth and return BlueTeamFeedback.

        Args:
            event: The AttackEvent evaluated during this round (carries ground_truth).
            prediction: The PredictionResult produced by the detector.

        Returns:
            BlueTeamFeedback structured record.
        """
        is_attack = bool(event.ground_truth)
        predicted_fraud = bool(prediction.prediction)

        detected = predicted_fraud and is_attack
        false_negative = (not predicted_fraud) and is_attack
        false_positive = predicted_fraud and (not is_attack)

        # Map the detector's weighted feature contributions
        # to BlueTeamFeedback.important_features so downstream mutation strategies can
        # inspect the exact contribution weights that triggered detection.
        important_features: Dict[str, float] = dict(
            prediction.feature_contributions or {}
        )

        attack_fam = (
            event.attack_family.value
            if hasattr(event.attack_family, "value")
            else str(event.attack_family)
        )

        explanation_data: Dict[str, Any] = {
            "ground_truth": is_attack,
            "prediction": predicted_fraud,
            "risk_score": prediction.risk_score,
            "model_version": prediction.model_version,
            "explanation": prediction.explanation,
            "attack_family": attack_fam,
        }

        # Include AI-agent domain context in explanation data if available in scenario
        if isinstance(event.scenario, AIAgentPaymentEvent):
            explanation_data["event_id"] = event.scenario.event_id
            explanation_data["agent_identity"] = event.scenario.agent_identity
            explanation_data["actual_action"] = event.scenario.actual_action
        elif isinstance(event.scenario, dict):
            if "event_id" in event.scenario:
                explanation_data["event_id"] = event.scenario["event_id"]
            if "agent_identity" in event.scenario:
                explanation_data["agent_identity"] = event.scenario["agent_identity"]
            if "actual_action" in event.scenario:
                explanation_data["actual_action"] = event.scenario["actual_action"]

        return BlueTeamFeedback(
            feedback_id=f"fb-f2-{event.round_id}",
            round_reference=event.round_id,
            detected=detected,
            false_positive=false_positive,
            false_negative=false_negative,
            risk_score=prediction.risk_score,
            important_features=important_features,
            explanation_data=explanation_data,
        )


# Alias for flexible importing
AIAgentEvaluator = AIAgentFeedbackEvaluator
