from typing import Dict, Any
from pydantic import BaseModel, Field

class SyntheticIdentity(BaseModel):
    identity_id: str = Field(..., min_length=1)
    identity_attributes: Dict[str, Any]
    contact_attributes: Dict[str, Any]
    account_metadata: Dict[str, Any]
    device_context: Dict[str, Any]
    lifecycle_info: Dict[str, Any]
