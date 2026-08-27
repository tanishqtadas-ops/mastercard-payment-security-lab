"""
blue_team/transaction/evaluator.py — Family 1 (Transaction Evasion) Feedback Evaluator.

Compares Blue-Team detector predictions against ground truth and produces structured
BlueTeamFeedback for Family 1 (Adaptive Transaction-Pattern Evasion) simulation rounds.
"""

from __future__ import annotations

from typing import Any, Dict

from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.feedback import BlueTeamFeedback
from schemas.transaction import Transaction


class TransactionFeedbackEvaluator:
    """
    Evaluates Blue-Team detection results against ground-truth transaction events.

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

        # Classification outcome flags
        detected = predicted_fraud and is_attack
        false_negative = (not predicted_fraud) and is_attack
        false_positive = predicted_fraud and (not is_attack)

        # Map the detector's weighted feature contributions to BlueTeamFeedback.important_features
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

        # Include transaction domain context in explanation data if available in scenario
        scenario = event.scenario or {}
        if isinstance(scenario, dict):
            tx_data = scenario.get("transaction")
            if isinstance(tx_data, Transaction):
                tx_dict = tx_data.model_dump()
            elif isinstance(tx_data, dict):
                tx_dict = tx_data
            else:
                tx_dict = {}

            if tx_dict:
                if "transaction_id" in tx_dict:
                    explanation_data["transaction_id"] = tx_dict["transaction_id"]
                if "user_id" in tx_dict:
                    explanation_data["user_id"] = tx_dict["user_id"]
                if "amount" in tx_dict:
                    explanation_data["amount"] = tx_dict["amount"]
                if "currency" in tx_dict:
                    explanation_data["currency"] = tx_dict["currency"]
                if "merchant_category" in tx_dict:
                    explanation_data["merchant_category"] = tx_dict["merchant_category"]
                if "location" in tx_dict:
                    explanation_data["location"] = tx_dict["location"]

        return BlueTeamFeedback(
            feedback_id=f"fb-f1-{event.round_id}",
            round_reference=event.round_id,
            detected=detected,
            false_positive=false_positive,
            false_negative=false_negative,
            risk_score=prediction.risk_score,
            important_features=important_features,
            explanation_data=explanation_data,
        )


# Alias for flexible importing
TransactionEvaluator = TransactionFeedbackEvaluator
