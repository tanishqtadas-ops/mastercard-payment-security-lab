from typing import Dict, Any
from pydantic import BaseModel, Field
from .attack import AttackEvent
from .prediction import PredictionResult
from .feedback import BlueTeamFeedback

class RoundResult(BaseModel):
    round_id: str = Field(..., min_length=1)
    attack_event: AttackEvent
    prediction_result: PredictionResult
    feedback: BlueTeamFeedback
    outcome_metrics: Dict[str, Any] = Field(default_factory=dict)
