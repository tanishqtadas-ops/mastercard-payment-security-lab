import pytest
from datetime import datetime
from pydantic import ValidationError
from schemas import (
    AttackFamily,
    Transaction,
    AIAgentPaymentEvent,
    SyntheticIdentity,
    AttackEvent,
    PredictionResult,
    BlueTeamFeedback,
    RoundResult
)

def test_transaction_valid():
    tx = Transaction(
        transaction_id="tx_123",
        user_id="u_1",
        timestamp=datetime.now(),
        amount=100.50,
        currency="USD",
        merchant_id="m_1",
        merchant_category="electronics",
        location="NY",
        device_id="dev_1",
        payment_channel="online"
    )
    assert tx.transaction_id == "tx_123"
    assert tx.amount == 100.50
    # serialization
    data = tx.model_dump()
    assert data["currency"] == "USD"

def test_transaction_invalid_amount():
    with pytest.raises(ValidationError):
        Transaction(
            transaction_id="tx_123",
            user_id="u_1",
            timestamp=datetime.now(),
            amount=-10.0,  # invalid
            currency="USD",
            merchant_id="m_1",
            merchant_category="electronics",
            location="NY",
            device_id="dev_1",
            payment_channel="online"
        )

def test_transaction_missing_fields():
    with pytest.raises(ValidationError):
        Transaction(
            transaction_id="tx_123"
            # missing user_id, etc.
        )

def test_transaction_invalid_currency():
    with pytest.raises(ValidationError):
        Transaction(
            transaction_id="tx_123",
            user_id="u_1",
            timestamp=datetime.now(),
            amount=10.0,
            currency="US", # too short
            merchant_id="m_1",
            merchant_category="electronics",
            location="NY",
            device_id="dev_1",
            payment_channel="online"
        )

def test_agent_event_valid():
    tx = Transaction(
        transaction_id="tx_123",
        user_id="u_1",
        timestamp=datetime.now(),
        amount=100.50,
        currency="USD",
        merchant_id="m_1",
        merchant_category="electronics",
        location="NY",
        device_id="dev_1",
        payment_channel="online"
    )
    event = AIAgentPaymentEvent(
        event_id="e_1",
        user_intent="buy phone",
        authorized_scope="up to 500",
        agent_identity="agent_x",
        session_context="web",
        actual_action="bought phone",
        transaction=tx
    )
    assert event.event_id == "e_1"

def test_agent_event_missing_field():
    with pytest.raises(ValidationError):
        AIAgentPaymentEvent(event_id="e_1")

def test_synthetic_identity_valid():
    identity = SyntheticIdentity(
        identity_id="id_1",
        identity_attributes={"name": "John"},
        contact_attributes={"email": "john@example.com"},
        account_metadata={"status": "active"},
        device_context={"ip": "127.0.0.1"},
        lifecycle_info={"created_at": "2023-01-01"}
    )
    assert identity.identity_id == "id_1"
    assert identity.identity_attributes["name"] == "John"

def test_synthetic_identity_invalid_type():
    with pytest.raises(ValidationError):
        SyntheticIdentity(
            identity_id="id_1",
            identity_attributes="Not a dict", # invalid type
            contact_attributes={},
            account_metadata={},
            device_context={},
            lifecycle_info={}
        )

def test_attack_event_valid():
    attack = AttackEvent(
        attack_id="a_1",
        round_id="r_1",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome={"amount_deviation": 0.5},
        scenario={"time": "night"},
        ground_truth=True
    )
    assert attack.attack_family == AttackFamily.ADAPTIVE_EVASION

def test_attack_event_invalid_enum():
    with pytest.raises(ValidationError):
        AttackEvent(
            attack_id="a_1",
            round_id="r_1",
            attack_family="Invalid Family",
            attack_genome={},
            scenario={},
            ground_truth=True
        )

def test_prediction_result_valid():
    pred = PredictionResult(
        prediction_id="p_1",
        prediction=True,
        risk_score=0.85,
        model_version="v1.0"
    )
    assert pred.risk_score == 0.85

def test_prediction_result_invalid_risk_score():
    with pytest.raises(ValidationError):
        PredictionResult(
            prediction_id="p_1",
            prediction=True,
            risk_score=1.5, # Out of range
            model_version="v1.0"
        )

def test_blue_team_feedback_valid():
    fb = BlueTeamFeedback(
        feedback_id="fb_1",
        round_reference="r_1",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.9,
        important_features={"f1": 0.5}
    )
    assert fb.detected is True

def test_round_result_valid():
    attack = AttackEvent(
        attack_id="a_1",
        round_id="r_1",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={"intent_deviation": 0.9},
        scenario={},
        ground_truth=True
    )
    pred = PredictionResult(
        prediction_id="p_1",
        prediction=True,
        risk_score=0.95,
        model_version="v1.1"
    )
    fb = BlueTeamFeedback(
        feedback_id="fb_1",
        round_reference="r_1",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.95,
        important_features={"intent_deviation": 0.8}
    )
    rnd = RoundResult(
        round_id="r_1",
        attack_event=attack,
        prediction_result=pred,
        feedback=fb,
        outcome_metrics={"accuracy": 1.0}
    )
    assert rnd.round_id == "r_1"

def test_round_result_serialization():
    attack = AttackEvent(
        attack_id="a_1",
        round_id="r_1",
        attack_family=AttackFamily.SYNTHETIC_IDENTITY,
        attack_genome={},
        scenario={},
        ground_truth=True
    )
    pred = PredictionResult(
        prediction_id="p_1",
        prediction=True,
        risk_score=0.95,
        model_version="v1.1"
    )
    fb = BlueTeamFeedback(
        feedback_id="fb_1",
        round_reference="r_1",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.95,
        important_features={"intent_deviation": 0.8}
    )
    rnd = RoundResult(
        round_id="r_1",
        attack_event=attack,
        prediction_result=pred,
        feedback=fb,
        outcome_metrics={"accuracy": 1.0}
    )
    data = rnd.model_dump()
    assert data["attack_event"]["attack_family"] == AttackFamily.SYNTHETIC_IDENTITY.value
    # Ensure it's valid to deserialize back
    rnd2 = RoundResult(**data)
    assert rnd2.attack_event.attack_family == AttackFamily.SYNTHETIC_IDENTITY
