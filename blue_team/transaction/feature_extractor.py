"""
blue_team/transaction/feature_extractor.py — Feature extraction for Family 1 detector.

Extracts structured observable behavioral deviation signals from Family 1 scenario data
(target transaction, baseline profile, and recent transaction history).
Does NOT access event.ground_truth or rely on hidden event.attack_genome.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas.transaction import Transaction


FEATURE_NAMES: List[str] = [
    "amount_deviation",
    "velocity_deviation",
    "device_novelty",
    "location_deviation",
    "time_deviation",
    "sequence_anomaly",
]

_HIGH_RISK_CATEGORIES = {
    "cryptocurrency_onramp",
    "prepaid_giftcards",
    "luxury_jewelry",
    "anonymous_hosting",
    "money_transfer",
    "crypto_exchange",
    "wire_remittance",
    "virtual_goods",
}

_SUSPICIOUS_DEVICE_INDICATORS = [
    "emulator",
    "headless",
    "virtualbox",
    "tor",
    "puppeteer",
    "unknown",
    "vdi",
    "proxy",
]


def _parse_timestamp(val: Any) -> Optional[datetime]:
    """Helper to parse datetime from string or datetime object."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    return None


def extract_transaction_features(scenario: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts numerical deviation features in [0.0, 1.0] from a Family 1 scenario dictionary.

    Args:
        scenario: Dictionary containing 'transaction', 'baseline_profile', and optional 'recent_history'.

    Returns:
        Dictionary mapping the six Family 1 feature names to float signals in [0.0, 1.0].
    """
    if not isinstance(scenario, dict):
        return {feat: 0.5 for feat in FEATURE_NAMES}

    # 1. Resolve Target Transaction
    tx_data = scenario.get("transaction", {})
    if isinstance(tx_data, Transaction):
        tx_dict = tx_data.model_dump()
    elif isinstance(tx_data, dict):
        tx_dict = tx_data
    else:
        tx_dict = {}

    target_amount = float(tx_dict.get("amount", 0.0))
    target_time = _parse_timestamp(tx_dict.get("timestamp"))
    target_device = str(tx_dict.get("device_id", "")).lower()
    target_location = str(tx_dict.get("location", ""))
    target_category = str(tx_dict.get("merchant_category", "")).lower()
    target_channel = str(tx_dict.get("payment_channel", "")).lower()

    # 2. Resolve User Baseline Profile
    baseline = scenario.get("baseline_profile", {})
    if not isinstance(baseline, dict):
        baseline = {}

    avg_amt = float(baseline.get("avg_amount", 50.0))
    std_amt = float(baseline.get("std_amount", 20.0))
    max_hist_amt = float(baseline.get("max_historical_amount", avg_amt * 4.0))
    frequent_locations = [str(loc).lower() for loc in baseline.get("frequent_locations", [])]
    home_location = str(baseline.get("home_location", "")).lower()
    registered_devices = [str(dev).lower() for dev in baseline.get("registered_devices", [])]
    frequent_categories = [str(cat).lower() for cat in baseline.get("frequent_categories", [])]
    active_hours = baseline.get("active_hours", [8, 22])
    typical_interval_hours = float(baseline.get("typical_interval_hours", 24.0))

    # 3. Resolve Recent History
    recent_history = scenario.get("recent_history", [])
    if not isinstance(recent_history, list):
        recent_history = []

    # -------------------------------------------------------------------------
    # Feature 1: Amount Deviation
    # -------------------------------------------------------------------------
    if avg_amt > 0:
        if target_amount <= (avg_amt + std_amt):
            # Within 1 std dev of normal mean
            amount_signal = max(0.0, (target_amount - avg_amt) / (3.0 * std_amt)) if std_amt > 0 else 0.0
            amount_signal = min(max(amount_signal, 0.0), 0.20)
        else:
            # Over 1 std dev: compute deviation scale relative to mean and max historical
            excess_ratio = (target_amount - avg_amt) / (avg_amt * 8.0)
            if target_amount > max_hist_amt:
                excess_ratio += 0.25
            amount_signal = min(max(excess_ratio, 0.20), 1.0)
    else:
        amount_signal = 0.5
    amount_deviation = round(min(max(amount_signal, 0.0), 1.0), 4)

    # -------------------------------------------------------------------------
    # Feature 2: Velocity Deviation
    # -------------------------------------------------------------------------
    if target_time and recent_history:
        # Find the most recent transaction timestamp
        last_hist_tx = recent_history[-1]
        last_hist_time_val = last_hist_tx.get("timestamp") if isinstance(last_hist_tx, dict) else getattr(last_hist_tx, "timestamp", None)
        last_time = _parse_timestamp(last_hist_time_val)

        if last_time:
            delta_seconds = abs((target_time - last_time).total_seconds())
            delta_minutes = delta_seconds / 60.0
            delta_hours = delta_minutes / 60.0

            if delta_minutes <= 10.0:
                # Immediate burst: extreme velocity anomaly
                vel_signal = 1.0 - (delta_minutes / 20.0)
            elif delta_minutes <= 60.0:
                # Rapid succession under an hour
                vel_signal = 0.80 - (delta_minutes / 120.0)
            elif delta_hours < (typical_interval_hours * 0.4):
                # Faster than usual
                vel_signal = 0.40
            else:
                # Normal spacing
                vel_signal = 0.05
        else:
            vel_signal = 0.10
    else:
        vel_signal = 0.10
    velocity_deviation = round(min(max(vel_signal, 0.0), 1.0), 4)

    # -------------------------------------------------------------------------
    # Feature 3: Device Novelty
    # -------------------------------------------------------------------------
    if target_device:
        is_registered = any(target_device == reg for reg in registered_devices)
        is_suspicious_device = any(ind in target_device for ind in _SUSPICIOUS_DEVICE_INDICATORS)

        if is_registered:
            dev_signal = 0.02
        elif is_suspicious_device:
            dev_signal = 0.95
        else:
            # Unfamiliar but standard device ID
            dev_signal = 0.70
    else:
        dev_signal = 0.50
    device_novelty = round(min(max(dev_signal, 0.0), 1.0), 4)

    # -------------------------------------------------------------------------
    # Feature 4: Location Deviation
    # -------------------------------------------------------------------------
    target_loc_clean = target_location.lower().strip()
    if target_loc_clean:
        is_frequent_loc = any(target_loc_clean in freq or freq in target_loc_clean for freq in frequent_locations)
        is_home_loc = home_location and (target_loc_clean in home_location or home_location in target_loc_clean)

        if is_frequent_loc or is_home_loc:
            loc_signal = 0.02
        else:
            # Check country mismatch or distant international geo
            target_country = target_loc_clean.split(",")[-1].strip() if "," in target_loc_clean else ""
            home_country = home_location.split(",")[-1].strip() if "," in home_location else ""

            if target_country and home_country and target_country != home_country:
                loc_signal = 0.95  # Foreign country location
            else:
                loc_signal = 0.65  # Unfamiliar domestic location
    else:
        loc_signal = 0.50
    location_deviation = round(min(max(loc_signal, 0.0), 1.0), 4)

    # -------------------------------------------------------------------------
    # Feature 5: Time Deviation (Off-Hours Activity)
    # -------------------------------------------------------------------------
    if target_time and len(active_hours) >= 2:
        hour = target_time.hour
        start_h, end_h = active_hours[0], active_hours[1]

        if start_h <= hour <= end_h:
            # Normal waking / transacting hours
            time_signal = 0.04
        else:
            # Calculate distance in hours from active window
            if hour < start_h:
                dist = start_h - hour
            else:
                dist = (hour - end_h) % 24

            # Peak nocturnal anomaly (02:00 - 05:00) gives highest score
            if 1 <= hour <= 5:
                time_signal = 0.85 + (min(dist, 4) * 0.03)
            else:
                time_signal = 0.40 + (min(dist, 6) * 0.07)
    else:
        time_signal = 0.10
    time_deviation = round(min(max(time_signal, 0.0), 1.0), 4)

    # -------------------------------------------------------------------------
    # Feature 6: Sequence & Category Anomaly
    # -------------------------------------------------------------------------
    is_high_risk_cat = any(hr_cat in target_category for hr_cat in _HIGH_RISK_CATEGORIES)
    is_frequent_cat = any(target_category == freq for freq in frequent_categories)

    if is_high_risk_cat:
        seq_signal = 0.90
    elif is_frequent_cat:
        seq_signal = 0.03
    else:
        # Novel category for user, but benign standard retail
        seq_signal = 0.45

    if "card_not_present" in target_channel and seq_signal > 0.3:
        seq_signal = min(seq_signal + 0.10, 1.0)

    sequence_anomaly = round(min(max(seq_signal, 0.0), 1.0), 4)

    return {
        "amount_deviation": amount_deviation,
        "velocity_deviation": velocity_deviation,
        "device_novelty": device_novelty,
        "location_deviation": location_deviation,
        "time_deviation": time_deviation,
        "sequence_anomaly": sequence_anomaly,
    }
