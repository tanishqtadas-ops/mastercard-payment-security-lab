from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class BlueTeamFeedback(BaseModel):
    feedback_id: str = Field(..., min_length=1)
    round_reference: str = Field(..., min_length=1)
    detected: bool
    false_positive: bool
    false_negative: bool
    risk_score: float = Field(..., ge=0.0, le=1.0)
    important_features: Dict[str, float]
    explanation_data: Optional[Dict[str, Any]] = None
