from typing import Optional
from pydantic import BaseModel, Field
from .transaction import Transaction

class AIAgentPaymentEvent(BaseModel):
    event_id: str = Field(..., min_length=1)
    user_intent: str
    authorized_scope: str
    agent_identity: str
    session_context: str
    actual_action: str
    transaction: Optional[Transaction] = None
