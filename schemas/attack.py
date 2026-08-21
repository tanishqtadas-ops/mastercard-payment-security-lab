from typing import Dict, Any
from pydantic import BaseModel, Field
from .common import AttackFamily

class AttackEvent(BaseModel):
    attack_id: str = Field(..., min_length=1)
    round_id: str = Field(..., min_length=1)
    attack_family: AttackFamily
    attack_genome: Dict[str, float]
    scenario: Dict[str, Any]
    ground_truth: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
