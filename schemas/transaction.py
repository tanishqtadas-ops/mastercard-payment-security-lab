from datetime import datetime
from pydantic import BaseModel, Field

class Transaction(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    timestamp: datetime
    amount: float = Field(..., ge=0.0)
    currency: str = Field(..., min_length=3, max_length=3)
    merchant_id: str = Field(..., min_length=1)
    merchant_category: str
    location: str
    device_id: str
    payment_channel: str
