"""
tests/test_family2_detector.py — Focused test suite for Family 2 Blue-Team Detector.

Covers:
1. Detector satisfies runtime_checkable BlueTeamDetector protocol.
2. Clearly authorized agent action is approved with low risk score.
3. Amount exceeding the mandate increases amount deviation and overall risk.
4. Disallowed category increases category deviation and overall risk.
5. Disallowed merchant increases permission scope deviation and overall risk.
6. Combined violations trigger fraud detection with high risk score.
7. Risk score is strictly normalized to [0.0, 1.0] across boundary cases.
8. PredictionResult compatibility and schema round-trip integrity.
9. Detector produces zero ground_truth leakage (prediction is purely feature-based).
10. Decision threshold consistency (prediction boolean matches risk_score >= threshold).
11. Explicit AgentMandate override correctly enforces customized authorization bounds.
12. Independence Requirement 1: Valid scenario + low-risk genome -> evaluates scenario.
13. Independence Requirement 2: Valid scenario + extreme malicious genome -> does NOT copy genome.
14. Independence Requirement 3: Malicious scenario + benign genome -> detects scenario violation.
"""

from datetime import datetime, timezone
import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    AIAgentPaymentEvent,
    Transaction,
    PredictionResult,
)
from simulation.interfaces import BlueTeamDetector
from blue_team.ai_agent import (
    AIAgentBlueDetector,
    AgentMandate,
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_DETECTION_THRESHOLD,
    MODEL_VERSION,
    extract_mandate_features,
)


@pytest.fixture
def detector() -> AIAgentBlueDetector:
    """Instantiate a default Family 2 Blue-Team Detector."""
    return AIAgentBlueDetector()


@pytest.fixture
def standard_mandate() -> AgentMandate:
    """Instantiate a standard test AgentMandate for office supplies procurement."""
    return AgentMandate(
        max_amount=150.0,
        allowed_categories=["office_supplies"],
        allowed_merchants=["Staples Direct", "OfficeDepot Corp", "Quill Supply"],
        currency="USD",
        require_verified_identity=True,
        allow_burst=False,
    )


# ---------------------------------------------------------------------------
# Protocol Compliance
# ---------------------------------------------------------------------------

def test_detector_satisfies_protocol(detector: AIAgentBlueDetector):
    """Verify AIAgentBlueDetector satisfies runtime_checkable BlueTeamDetector protocol."""
    assert isinstance(detector, BlueTeamDetector)


# ---------------------------------------------------------------------------
# 1. Clearly Authorized Action
# ---------------------------------------------------------------------------

def test_clearly_authorized_action_is_approved(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Test 1: Clearly authorized action stays within mandate.
    - Amount $120.00 <= $150.00 limit
    - Category 'office_supplies' authorized
    - Merchant 'Staples Direct' authorized
    - Agent identity verified
    """
    tx = Transaction(
        transaction_id="tx-auth-001",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=120.00,
        currency="USD",
        merchant_id="merch_staples_01",
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_agent_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-auth-01",
        user_intent="Purchase office supplies and stationery up to $150",
        authorized_scope="Office supplies, single purchase up to $150, approved merchants only",
        agent_identity="procure-bot-v2-valid",
        session_context="session_auth_mfa_valid_001",
        actual_action="Purchased office supplies from Staples Direct for $120.00. Agent adhered strictly to authorized operational bounds.",
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-f2-auth-01",
        round_id="round-auth-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=False,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    assert isinstance(pred, PredictionResult)
    assert pred.prediction is False
    assert pred.risk_score < 0.50
    assert pred.explanation is not None
    assert "approved" in pred.explanation.lower() or "within authorized" in pred.explanation.lower()
    assert pred.feature_contributions is not None
    assert pred.feature_contributions["intent_amount_deviation"] < 0.05
    assert pred.feature_contributions["intent_category_deviation"] < 0.05
    assert pred.feature_contributions["permission_scope_deviation"] < 0.05


# ---------------------------------------------------------------------------
# 2. Amount Exceeding the Mandate
# ---------------------------------------------------------------------------

def test_amount_exceeding_mandate_increases_risk(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Test 2: Observed transaction amount ($750.00) exceeds permitted maximum ($150.00).
    - Amount exceeds limit significantly
    - Category & merchant remain authorized
    - Verifies intent_amount_deviation is primary risk driver
    """
    tx = Transaction(
        transaction_id="tx-overlimit-001",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=750.00,  # 5x authorized maximum
        currency="USD",
        merchant_id="merch_staples_01",
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_agent_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-overlimit-01",
        user_intent="Purchase office supplies and stationery up to $150",
        authorized_scope="Office supplies, single purchase up to $150, approved merchants only",
        agent_identity="procure-bot-v2-valid",
        session_context="session_auth_mfa_valid_001",
        actual_action="Purchased bulk office furniture from Staples Direct for $750.00. Exceeded authorized spending limit.",
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-f2-overlimit-01",
        round_id="round-overlimit-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    assert isinstance(pred, PredictionResult)
    assert pred.feature_contributions is not None
    assert pred.feature_contributions["intent_amount_deviation"] >= 0.15
    assert pred.risk_score > 0.20


# ---------------------------------------------------------------------------
# 3. Disallowed Category
# ---------------------------------------------------------------------------

def test_disallowed_category_increases_risk(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Test 3: Observed merchant category ('crypto_assets') is not authorized in mandate ('office_supplies').
    - Amount is within limit ($140.00 <= $150.00)
    - Category violation triggers high category deviation contribution
    """
    tx = Transaction(
        transaction_id="tx-cat-violation-001",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=140.00,
        currency="USD",
        merchant_id="merch_crypto_hub",
        merchant_category="crypto_assets",  # Disallowed category
        location="US-Online",
        device_id="device_agent_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-cat-01",
        user_intent="Purchase office supplies and stationery up to $150",
        authorized_scope="Office supplies, single purchase up to $150, approved merchants only",
        agent_identity="procure-bot-v2-valid",
        session_context="session_auth_mfa_valid_001",
        actual_action="Purchased cryptocurrency vouchers for $140.00. Unauthorized category deviation.",
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-f2-cat-01",
        round_id="round-cat-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    assert isinstance(pred, PredictionResult)
    assert pred.feature_contributions is not None
    assert pred.feature_contributions["intent_category_deviation"] >= 0.15
    assert pred.risk_score > 0.15


# ---------------------------------------------------------------------------
# 4. Disallowed Merchant
# ---------------------------------------------------------------------------

def test_disallowed_merchant_increases_risk(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Test 4: Observed merchant ('Apex Electronics' / 'CryptoVault Exchange') is not in the approved merchant list.
    - Category and amount within limit
    - Disallowed merchant triggers permission_scope_deviation contribution
    """
    tx = Transaction(
        transaction_id="tx-merch-violation-001",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=135.00,
        currency="USD",
        merchant_id="merch_apex_electronics_unapproved",  # Unapproved merchant
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_agent_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-merch-01",
        user_intent="Purchase office supplies and stationery up to $150",
        authorized_scope="Office supplies, single purchase up to $150, approved merchants only",
        agent_identity="procure-bot-v2-valid",
        session_context="session_auth_mfa_valid_001",
        actual_action="Purchased supplies from Apex Electronics (unapproved merchant) for $135.00.",
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-f2-merch-01",
        round_id="round-merch-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    assert isinstance(pred, PredictionResult)
    assert pred.feature_contributions is not None
    assert pred.feature_contributions["permission_scope_deviation"] >= 0.15
    assert pred.risk_score > 0.15


# ---------------------------------------------------------------------------
# 5. Combined Violations
# ---------------------------------------------------------------------------

def test_combined_violations_triggers_fraud_detection(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Test 5: Combined violations simultaneously:
    - Amount exceeds limit ($950.00 vs $150.00 limit)
    - Disallowed category ('crypto_assets' vs 'office_supplies')
    - Disallowed merchant ('CryptoVault Exchange')
    - Unverified / spoofed agent identity
    - Anomaly session provenance
    - Verifies cumulative risk triggers fraud prediction (True)
    """
    tx = Transaction(
        transaction_id="tx-combined-violation-001",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=950.00,
        currency="USD",
        merchant_id="merch_cryptovault_exchange",
        merchant_category="crypto_assets",
        location="US-Online",
        device_id="device_agent_unknown",
        payment_channel="ai_agent_batch_burst_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-combined-01",
        user_intent="Purchase office supplies and stationery up to $150",
        authorized_scope="Office supplies, single purchase up to $150, approved merchants only",
        agent_identity="unverified-agent-spoof-123",
        session_context="session_hijack_anom_remote_ip_flagged_burst_rate_n5",
        actual_action=(
            "Executed unauthorized transaction: Transferred funds to external cryptocurrency voucher service "
            "at CryptoVault Exchange (Amount: $950.00, Category: crypto_assets) (Burst of 5 rapid transactions in session). "
            "Scope violation: Agent unilaterally elevated execution privileges to 'unrestricted_admin'."
        ),
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-f2-combined-01",
        round_id="round-combined-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    assert isinstance(pred, PredictionResult)
    assert pred.prediction is True
    assert pred.risk_score >= 0.50
    assert pred.explanation is not None
    assert "flagged" in pred.explanation.lower()
    assert pred.feature_contributions is not None
    assert pred.feature_contributions["intent_amount_deviation"] > 0.15
    assert pred.feature_contributions["intent_category_deviation"] > 0.15
    assert pred.feature_contributions["permission_scope_deviation"] > 0.15


# ---------------------------------------------------------------------------
# 6. Risk Score Normalization within [0, 1]
# ---------------------------------------------------------------------------

def test_risk_score_always_within_bounds(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Test 6: Risk score is strictly normalized to [0.0, 1.0] across edge cases:
    - Zero amount
    - Massive extreme amount ($10,000,000)
    - Empty strings / missing optional fields
    - Multi-burst anomalies
    """
    test_amounts = [0.0, 1.0, 149.99, 150.0, 150.01, 1000.0, 1_000_000.0]

    for amt in test_amounts:
        tx = Transaction(
            transaction_id=f"tx-bound-{amt}",
            user_id="user-test",
            timestamp=datetime.now(timezone.utc),
            amount=amt,
            currency="USD",
            merchant_id="merch_test",
            merchant_category="office_supplies",
            location="US-Online",
            device_id="device_test",
            payment_channel="ai_agent_api",
        )
        scenario = AIAgentPaymentEvent(
            event_id=f"evt-bound-{amt}",
            user_intent="Buy supplies up to $150",
            authorized_scope="Office supplies up to $150",
            agent_identity="procure-bot",
            session_context="session_valid",
            actual_action=f"Purchased supplies for ${amt:.2f}",
            transaction=tx,
        )
        event = AttackEvent(
            attack_id=f"atk-bound-{amt}",
            round_id=f"round-bound-{amt}",
            attack_family=AttackFamily.AGENT_BEHAVIOR,
            attack_genome={},
            scenario=scenario.model_dump(mode="json"),
            ground_truth=amt > 150.0,
        )

        pred = detector.detect(event, mandate=standard_mandate)
        assert 0.0 <= pred.risk_score <= 1.0


# ---------------------------------------------------------------------------
# 7. PredictionResult Compatibility
# ---------------------------------------------------------------------------

def test_prediction_result_structure_compatibility(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Test 7: PredictionResult satisfies schema contracts and field specifications:
    - prediction_id non-empty and traceable
    - prediction boolean
    - risk_score float in [0.0, 1.0]
    - model_version non-empty string
    - explanation non-empty string
    - feature_contributions contains exact Family 2 keys
    """
    tx = Transaction(
        transaction_id="tx-compat-001",
        user_id="user-compat",
        timestamp=datetime.now(timezone.utc),
        amount=250.0,
        currency="USD",
        merchant_id="merch_compat",
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_compat",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="evt-compat-001",
        user_intent="Office supplies under $150",
        authorized_scope="Office supplies under $150",
        agent_identity="procure-bot",
        session_context="session_valid",
        actual_action="Purchased office items for $250.00",
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-compat-001",
        round_id="round-compat-001",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    assert isinstance(pred, PredictionResult)
    assert pred.prediction_id == "pred-f2-round-compat-001"
    assert isinstance(pred.prediction, bool)
    assert isinstance(pred.risk_score, float)
    assert 0.0 <= pred.risk_score <= 1.0
    assert pred.model_version == MODEL_VERSION
    assert isinstance(pred.explanation, str)
    assert len(pred.explanation) > 10
    assert isinstance(pred.feature_contributions, dict)
    for dim in DEFAULT_FEATURE_WEIGHTS.keys():
        assert dim in pred.feature_contributions
        assert isinstance(pred.feature_contributions[dim], float)

    # Verify Pydantic serialization round-trip
    dumped = pred.model_dump(mode="json")
    reconstructed = PredictionResult.model_validate(dumped)
    assert reconstructed.prediction_id == pred.prediction_id
    assert reconstructed.risk_score == pred.risk_score


# ---------------------------------------------------------------------------
# 8. Zero Ground-Truth Leakage & Decision Threshold Consistency
# ---------------------------------------------------------------------------

def test_detector_zero_ground_truth_leakage(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """Verify inverting the ground_truth flag has zero effect on risk score or prediction."""
    tx = Transaction(
        transaction_id="tx-leak-001",
        user_id="user-leak",
        timestamp=datetime.now(timezone.utc),
        amount=145.0,
        currency="USD",
        merchant_id="merch_staples",
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="evt-leak-001",
        user_intent="Office supplies under $150",
        authorized_scope="Office supplies under $150",
        agent_identity="procure-bot",
        session_context="session_valid",
        actual_action="Purchased supplies for $145.00",
        transaction=tx,
    )

    event_attack = AttackEvent(
        attack_id="atk-leak-true",
        round_id="round-leak-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )
    event_legit = AttackEvent(
        attack_id="atk-leak-false",
        round_id="round-leak-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=False,
    )

    pred_attack = detector.detect(event_attack, mandate=standard_mandate)
    pred_legit = detector.detect(event_legit, mandate=standard_mandate)

    assert pred_attack.risk_score == pred_legit.risk_score
    assert pred_attack.prediction == pred_legit.prediction
    assert pred_attack.feature_contributions == pred_legit.feature_contributions


def test_decision_threshold_consistency():
    """Verify decision threshold parameter strictly governs the boolean prediction output."""
    det_sensitive = AIAgentBlueDetector(threshold=0.10)
    det_strict = AIAgentBlueDetector(threshold=0.90)

    # Moderate violation event
    tx = Transaction(
        transaction_id="tx-thresh-001",
        user_id="user-thresh",
        timestamp=datetime.now(timezone.utc),
        amount=250.0,  # Moderately over limit
        currency="USD",
        merchant_id="merch_staples",
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="evt-thresh-001",
        user_intent="Office supplies under $150",
        authorized_scope="Office supplies under $150",
        agent_identity="procure-bot",
        session_context="session_valid",
        actual_action="Purchased office items for $250.00",
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-thresh-01",
        round_id="round-thresh-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )

    pred_sens = det_sensitive.detect(event)
    pred_strict = det_strict.detect(event)

    assert pred_sens.risk_score == pred_strict.risk_score
    assert pred_sens.prediction is True  # risk >= 0.10
    assert pred_strict.prediction is False  # risk < 0.90


def test_explicit_agent_mandate_override(detector: AIAgentBlueDetector):
    """Verify providing an explicit custom AgentMandate overrides automatic inference."""
    custom_mandate = AgentMandate(
        max_amount=50.0,  # Very strict limit
        allowed_categories=["office_supplies"],
        allowed_merchants=["Quill Supply"],
    )

    tx = Transaction(
        transaction_id="tx-custom-001",
        user_id="user-custom",
        timestamp=datetime.now(timezone.utc),
        amount=75.0,  # Exceeds custom $50 limit, but within generic $150 limit
        currency="USD",
        merchant_id="merch_staples",  # Not Quill Supply
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="evt-custom-001",
        user_intent="Office supplies under $150",
        authorized_scope="Office supplies under $150",
        agent_identity="procure-bot",
        session_context="session_valid",
        actual_action="Purchased stationery for $75.00 from Staples Direct",
        transaction=tx,
    )
    event = AttackEvent(
        attack_id="atk-custom-01",
        round_id="round-custom-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome={},
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )

    # Evaluate with default inferred mandate ($150 limit, Staples allowed) -> approved
    pred_default = detector.detect(event)
    assert pred_default.prediction is False

    # Evaluate with strict custom mandate ($50 limit, only Quill Supply) -> flagged
    pred_custom = detector.detect(event, mandate=custom_mandate)
    assert pred_custom.risk_score > pred_default.risk_score
    assert pred_custom.feature_contributions is not None
    assert pred_custom.feature_contributions["intent_amount_deviation"] > 0.05
    assert pred_custom.feature_contributions["permission_scope_deviation"] > 0.15


# ---------------------------------------------------------------------------
# 9-11. Detector Independence from attack_genome
# ---------------------------------------------------------------------------

def test_independence_valid_scenario_with_low_risk_genome(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Independence Test 1: Valid authorized scenario + low-risk genome.
    The detector correctly evaluates the observable scenario features.
    """
    tx = Transaction(
        transaction_id="tx-indep-001",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=110.00,
        currency="USD",
        merchant_id="merch_staples_01",
        merchant_category="office_supplies",
        location="US-Online",
        device_id="device_agent_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-indep-01",
        user_intent="Purchase office supplies and stationery up to $150",
        authorized_scope="Office supplies, single purchase up to $150, approved merchants only",
        agent_identity="procure-bot-v2-valid",
        session_context="session_auth_mfa_valid_001",
        actual_action="Purchased office supplies from Staples Direct for $110.00. Agent adhered strictly to authorized operational bounds.",
        transaction=tx,
    )
    low_genome = {
        "intent_amount_deviation": 0.05,
        "intent_category_deviation": 0.02,
        "permission_scope_deviation": 0.01,
        "agent_identity_confidence": 0.98,
        "session_provenance_anomaly": 0.03,
        "purchase_velocity": 0.05,
    }
    event = AttackEvent(
        attack_id="atk-f2-indep-01",
        round_id="round-indep-01",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome=low_genome,
        scenario=scenario.model_dump(mode="json"),
        ground_truth=False,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    assert pred.prediction is False
    assert pred.risk_score < 0.50
    assert pred.explanation is not None
    assert "approved" in pred.explanation.lower() or "within authorized" in pred.explanation.lower()


def test_independence_valid_scenario_with_extreme_malicious_genome(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Independence Test 2: Valid authorized scenario + extreme malicious genome (all 1.0s).
    The detector MUST NOT blindly copy the 1.0 genome; it must evaluate the observable
    valid scenario and correctly approve the transaction.
    """
    tx = Transaction(
        transaction_id="tx-indep-002",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=115.00,  # Valid ($115 <= $150)
        currency="USD",
        merchant_id="merch_staples_01",  # Valid approved merchant
        merchant_category="office_supplies",  # Valid authorized category
        location="US-Online",
        device_id="device_agent_valid",
        payment_channel="ai_agent_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-indep-02",
        user_intent="Purchase office supplies and stationery up to $150",
        authorized_scope="Office supplies, single purchase up to $150, approved merchants only",
        agent_identity="procure-bot-v2-valid",
        session_context="session_auth_mfa_valid_001",
        actual_action="Purchased office supplies from Staples Direct for $115.00. Agent adhered strictly to authorized operational bounds.",
        transaction=tx,
    )
    extreme_malicious_genome = {
        "intent_amount_deviation": 1.0,
        "intent_category_deviation": 1.0,
        "permission_scope_deviation": 1.0,
        "agent_identity_confidence": 0.0,
        "session_provenance_anomaly": 1.0,
        "purchase_velocity": 1.0,
    }
    event = AttackEvent(
        attack_id="atk-f2-indep-02",
        round_id="round-indep-02",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome=extreme_malicious_genome,  # Intentionally extreme
        scenario=scenario.model_dump(mode="json"),
        ground_truth=False,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    # Detector must independently judge the valid scenario, not copy the attack genome
    assert pred.prediction is False
    assert pred.risk_score < 0.50
    assert pred.feature_contributions is not None
    assert pred.feature_contributions["intent_amount_deviation"] < 0.05
    assert pred.feature_contributions["intent_category_deviation"] < 0.05
    assert pred.feature_contributions["permission_scope_deviation"] < 0.05


def test_independence_malicious_scenario_with_benign_genome(
    detector: AIAgentBlueDetector,
    standard_mandate: AgentMandate,
):
    """
    Independence Test 3: Malicious scenario + benign low-risk genome (all 0.05s).
    The detector MUST NOT blindly trust the benign genome; it must evaluate the observable
    malicious scenario and flag the fraud.
    """
    tx = Transaction(
        transaction_id="tx-indep-003",
        user_id="user-corp-101",
        timestamp=datetime.now(timezone.utc),
        amount=990.00,  # Blatant amount violation ($990 vs $150 limit)
        currency="USD",
        merchant_id="merch_crypto_vault_unauthorized",  # Disallowed merchant
        merchant_category="crypto_assets",  # Disallowed category
        location="US-Online",
        device_id="device_agent_unknown",
        payment_channel="ai_agent_batch_burst_api",
    )
    scenario = AIAgentPaymentEvent(
        event_id="agent-evt-indep-03",
        user_intent="Purchase office supplies and stationery up to $150",
        authorized_scope="Office supplies, single purchase up to $150, approved merchants only",
        agent_identity="unverified-agent-spoof",
        session_context="session_hijack_anom_remote_ip_flagged",
        actual_action="Executed unauthorized transaction: Purchased crypto assets at CryptoVault. Scope violation: elevated privileges.",
        transaction=tx,
    )
    benign_genome = {
        "intent_amount_deviation": 0.01,
        "intent_category_deviation": 0.01,
        "permission_scope_deviation": 0.01,
        "agent_identity_confidence": 0.99,
        "session_provenance_anomaly": 0.01,
        "purchase_velocity": 0.01,
    }
    event = AttackEvent(
        attack_id="atk-f2-indep-03",
        round_id="round-indep-03",
        attack_family=AttackFamily.AGENT_BEHAVIOR,
        attack_genome=benign_genome,  # Artificially benign genome
        scenario=scenario.model_dump(mode="json"),
        ground_truth=True,
    )

    pred = detector.detect(event, mandate=standard_mandate)

    # Detector must independently detect the violation in scenario, not be misled by genome
    assert pred.prediction is True
    assert pred.risk_score >= 0.50
    assert pred.feature_contributions is not None
    assert pred.feature_contributions["intent_amount_deviation"] > 0.15
    assert pred.feature_contributions["intent_category_deviation"] > 0.15
    assert pred.feature_contributions["permission_scope_deviation"] > 0.15
