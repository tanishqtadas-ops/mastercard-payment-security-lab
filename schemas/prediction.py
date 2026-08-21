from typing import Dict, Optional
from pydantic import BaseModel, Field

class PredictionResult(BaseModel):
    prediction_id: str = Field(..., min_length=1)
    prediction: bool
    risk_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(..., min_length=1)
    explanation: Optional[str] = None
    feature_contributions: Optional[Dict[str, float]] = None
