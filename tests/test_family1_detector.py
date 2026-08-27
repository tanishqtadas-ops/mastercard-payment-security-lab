"""
tests/test_family1_detector.py — Unit tests for Family 1 TransactionBlueDetector.
"""

from datetime import datetime, timezone, timedelta
import pytest

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.transaction import Transaction
from simulation.interfaces import BlueTeamDetector
from attacks.transaction_evasion.generator import (
    TransactionAttackGenerator,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
)
from blue_team.transaction.detector import (
    TransactionBlueDetector,
    TransactionDetector,
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_DETECTION_THRESHOLD,
    MODEL_VERSION,
)
from blue_team.transaction.feature_extractor import (
    FEATURE_NAMES,
    extract_transaction_features,
)


# ---------------------------------------------------------------------------
# 1. Protocol satisfaction and initialization
# ---------------------------------------------------------------------------

def test_detector_satisfies_blue_team_detector_protocol():
    """Verify that TransactionBlueDetector satisfies the BlueTeamDetector protocol."""
    detector = TransactionBlueDetector()
    assert isinstance(detector, BlueTeamDetector)


def test_detector_default_initialization():
    """Verify default threshold, model version, and feature weights."""
    detector = TransactionBlueDetector()
    assert detector.threshold == DEFAULT_DETECTION_THRESHOLD
    assert detector.model_version == MODEL_VERSION
    assert detector.weights == DEFAULT_FEATURE_WEIGHTS


def test_detector_alias_and_custom_weights():
    """Verify alias and custom threshold/weights initialization."""
    custom_weights = {
        "amount_deviation": 0.30,
        "velocity_deviation": 0.20,
        "device_novelty": 0.15,
        "location_deviation": 0.15,
        "time_deviation": 0.10,
        "sequence_anomaly": 0.10,
    }
    detector = TransactionDetector(threshold=0.60, weights=custom_weights, model_version="custom-f1-v2")
    assert detector.threshold == 0.60
    assert detector.model_version == "custom-f1-v2"
    assert detector.weights == custom_weights


# ---------------------------------------------------------------------------
# 2. PredictionResult structure and schema compliance
# ---------------------------------------------------------------------------

def test_detect_returns_valid_prediction_result():
    """Verify detect produces a schema-compliant PredictionResult."""
    generator = TransactionAttackGenerator(seed=42)
    event = generator.generate("round-det-01")

    detector = TransactionBlueDetector()
    result = detector.detect(event)

    assert isinstance(result, PredictionResult)
    assert result.prediction_id == "pred-f1-round-det-01"
    assert isinstance(result.prediction, bool)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.model_version == MODEL_VERSION
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0

    # Feature contributions validation
    assert isinstance(result.feature_contributions, dict)
    assert set(result.feature_contributions.keys()) == set(FEATURE_NAMES)
    for feat, contrib in result.feature_contributions.items():
        assert 0.0 <= contrib <= 1.0


# ---------------------------------------------------------------------------
# 3. Legitimate vs Attack Detection Accuracy
# ---------------------------------------------------------------------------

def test_legitimate_transaction_produces_low_risk():
    """Verify that a legitimate transaction scenario produces low risk and prediction=False."""
    gen_legit = TransactionAttackGenerator(ground_truth=False, seed=123)
    event_legit = gen_legit.generate("round-legit-01")

    detector = TransactionBlueDetector()
    result = detector.detect(event_legit)

    assert result.risk_score < detector.threshold
    assert result.prediction is False
    assert "consistent with user historical baseline" in result.explanation.lower()


def test_high_deviation_attack_produces_high_risk():
    """Verify that an anomalous attack scenario produces high risk and prediction=True."""
    gen_attack = TransactionAttackGenerator(ground_truth=True, seed=123)
    event_attack = gen_attack.generate("round-attack-01")

    detector = TransactionBlueDetector()
    result = detector.detect(event_attack)

    assert result.risk_score >= detector.threshold
    assert result.prediction is True
    assert "flagged anomalous transaction risk" in result.explanation.lower()


# ---------------------------------------------------------------------------
# 4. Dimension-Specific Anomaly Detection
# ---------------------------------------------------------------------------

_FIXED_HISTORY_TS = datetime(2026, 8, 27, 14, 0, 0, tzinfo=timezone.utc)

def _build_scenario(
    amount: float = 40.0,
    timestamp: datetime | None = None,
    device_id: str = "dev_ios_iphone15_a1",
    location: str = "New York, US",
    merchant_category: str = "grocery",
    payment_channel: str = "pos_contactless",
) -> dict:
    """Helper to construct targeted scenarios with specific anomalous traits."""
    ts = timestamp or datetime(2026, 8, 28, 14, 0, 0, tzinfo=timezone.utc)
    target_tx = Transaction(
        transaction_id="tx_test_01",
        user_id="usr_retail_101",
        timestamp=ts,
        amount=amount,
        currency="USD",
        merchant_id="merch_test_01",
        merchant_category=merchant_category,
        location=location,
        device_id=device_id,
        payment_channel=payment_channel,
    )
    baseline = {
        "user_id": "usr_retail_101",
        "home_location": "New York, US",
        "frequent_locations": ["New York, US", "Jersey City, US", "Brooklyn, US"],
        "registered_devices": ["dev_ios_iphone15_a1", "dev_macbook_air_m2"],
        "typical_channels": ["pos_contactless", "online_card_on_file"],
        "currency": "USD",
        "avg_amount": 42.50,
        "std_amount": 15.00,
        "max_historical_amount": 210.00,
        "frequent_categories": ["grocery", "coffee_shop", "restaurants"],
        "active_hours": [8, 22],
        "typical_interval_hours": 24.0,
    }
    history = [
        Transaction(
            transaction_id="tx_hist_01",
            user_id="usr_retail_101",
            timestamp=_FIXED_HISTORY_TS,
            amount=38.00,
            currency="USD",
            merchant_id="merch_hist_01",
            merchant_category="grocery",
            location="New York, US",
            device_id="dev_ios_iphone15_a1",
            payment_channel="pos_contactless",
        ).model_dump()
    ]
    return {
        "transaction": target_tx.model_dump(),
        "baseline_profile": baseline,
        "recent_history": history,
    }


def test_amount_anomaly_detection():
    """Verify amount anomaly detection when transaction amount is massive."""
    normal_scen = _build_scenario(amount=45.0)
    anom_scen = _build_scenario(amount=1850.0)

    norm_features = extract_transaction_features(normal_scen)
    anom_features = extract_transaction_features(anom_scen)

    assert norm_features["amount_deviation"] < 0.20
    assert anom_features["amount_deviation"] > 0.70


def test_velocity_anomaly_detection():
    """Verify velocity anomaly detection when transaction occurs immediately after previous."""
    # Normal spacing: 24h after history timestamp
    normal_ts = _FIXED_HISTORY_TS + timedelta(hours=24)
    normal_scen = _build_scenario(timestamp=normal_ts)

    # Rapid burst: 2 minutes after history timestamp
    burst_ts = _FIXED_HISTORY_TS + timedelta(minutes=2)
    burst_scen = _build_scenario(timestamp=burst_ts)

    norm_features = extract_transaction_features(normal_scen)
    burst_features = extract_transaction_features(burst_scen)

    assert norm_features["velocity_deviation"] <= 0.10
    assert burst_features["velocity_deviation"] >= 0.85


def test_device_novelty_detection():
    """Verify device novelty detection for unfamiliar / emulator devices."""
    reg_scen = _build_scenario(device_id="dev_ios_iphone15_a1")
    emulator_scen = _build_scenario(device_id="dev_android_emulator_vbox_99")

    reg_features = extract_transaction_features(reg_scen)
    emul_features = extract_transaction_features(emulator_scen)

    assert reg_features["device_novelty"] <= 0.05
    assert emul_features["device_novelty"] >= 0.90


def test_location_anomaly_detection():
    """Verify location anomaly detection for distant foreign locations."""
    home_scen = _build_scenario(location="New York, US")
    foreign_scen = _build_scenario(location="Lagos, NG")

    home_features = extract_transaction_features(home_scen)
    foreign_features = extract_transaction_features(foreign_scen)

    assert home_features["location_deviation"] <= 0.05
    assert foreign_features["location_deviation"] >= 0.90


def test_time_anomaly_detection():
    """Verify off-hours time anomaly detection for middle-of-the-night transactions."""
    daytime_ts = datetime(2026, 8, 28, 14, 30, 0, tzinfo=timezone.utc)  # 2:30 PM
    nocturnal_ts = datetime(2026, 8, 28, 3, 15, 0, tzinfo=timezone.utc)  # 3:15 AM

    day_scen = _build_scenario(timestamp=daytime_ts)
    night_scen = _build_scenario(timestamp=nocturnal_ts)

    day_features = extract_transaction_features(day_scen)
    night_features = extract_transaction_features(night_scen)

    assert day_features["time_deviation"] <= 0.05
    assert night_features["time_deviation"] >= 0.85


def test_sequence_and_category_anomaly_detection():
    """Verify sequence anomaly detection for high-risk merchant categories."""
    normal_cat_scen = _build_scenario(merchant_category="grocery")
    crypto_cat_scen = _build_scenario(merchant_category="cryptocurrency_onramp", payment_channel="online_card_not_present")

    norm_features = extract_transaction_features(normal_cat_scen)
    crypto_features = extract_transaction_features(crypto_cat_scen)

    assert norm_features["sequence_anomaly"] <= 0.05
    assert crypto_features["sequence_anomaly"] >= 0.90


# ---------------------------------------------------------------------------
# 5. Determinism, Boundaries, and Edge Handling
# ---------------------------------------------------------------------------

def test_detector_is_strictly_deterministic():
    """Verify that identical AttackEvents produce identical PredictionResults."""
    generator = TransactionAttackGenerator(seed=888)
    event = generator.generate("round-det-888")

    detector = TransactionBlueDetector()
    res1 = detector.detect(event)
    res2 = detector.detect(event)

    assert res1.model_dump() == res2.model_dump()


def test_detector_threshold_boundary_decisions():
    """Verify decision boundary at exact threshold."""
    generator = TransactionAttackGenerator(seed=42)
    event = generator.generate("round-thresh")

    det_zero = TransactionBlueDetector(threshold=0.0)
    det_one = TransactionBlueDetector(threshold=1.0)

    res_zero = det_zero.detect(event)
    res_one = det_one.detect(event)

    assert res_zero.prediction is True  # Any non-negative risk triggers fraud
    assert res_one.prediction is False  # Extreme threshold


def test_detector_rejects_invalid_threshold():
    """Verify detector raises ValueError on out-of-range thresholds."""
    with pytest.raises(ValueError, match="threshold must be in"):
        TransactionBlueDetector(threshold=-0.1)

    with pytest.raises(ValueError, match="threshold must be in"):
        TransactionBlueDetector(threshold=1.5)


def test_detector_rejects_invalid_weights():
    """Verify detector raises ValueError when weights are incomplete or don't sum to 1.0."""
    with pytest.raises(ValueError, match="Missing weight"):
        TransactionBlueDetector(weights={"amount_deviation": 1.0})

    incomplete_sum = {feat: 0.10 for feat in FEATURE_NAMES}  # Sum = 0.60
    with pytest.raises(ValueError, match="must sum to approximately 1.0"):
        TransactionBlueDetector(weights=incomplete_sum)


def test_detector_handles_empty_or_malformed_scenario():
    """Verify detector gracefully handles empty scenario without crashing."""
    malformed_event = AttackEvent(
        attack_id="atk-malformed",
        round_id="round-malformed",
        attack_family=AttackFamily.ADAPTIVE_EVASION,
        attack_genome=DEFAULT_ATTACK_GENOME,
        scenario={},
        ground_truth=True,
    )
    detector = TransactionBlueDetector()
    result = detector.detect(malformed_event)

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.risk_score <= 1.0
