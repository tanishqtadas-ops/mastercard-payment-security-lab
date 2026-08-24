"""
blue_team/synthetic_identity/feature_extractor.py — Feature extraction for Family 3 detector.

Extracts structured observable risk indicators from SyntheticIdentity scenarios.
Does NOT access ground_truth or rely purely on attack_genome.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from schemas.identity import SyntheticIdentity


FEATURE_NAMES: List[str] = [
    "name_email_anomaly",
    "ssn_format_irregularity",
    "location_mismatch_anomaly",
    "profile_implausibility",
    "is_disposable_email",
    "is_voip_carrier",
    "email_phone_tenure_deficit",
    "is_emulator_device",
    "device_reputation_deficit",
    "is_datacenter_proxy_ip",
    "lifecycle_incoherence",
    "early_bust_out_risk",
]

_DISPOSABLE_DOMAINS = {
    "tempmail.xyz",
    "guerrillamail.net",
    "throwawaymail.org",
    "burner-inbox.com",
    "10minutemail.co",
    "fakeinbox.io",
    "tempmail.com",
    "guerrillamail.com",
}


def _sanitize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def extract_identity_features(scenario: Union[SyntheticIdentity, Dict[str, Any]]) -> Dict[str, float]:
    """
    Extracts numerical feature vector from a SyntheticIdentity scenario.

    Args:
        scenario: SyntheticIdentity model or dictionary dump.

    Returns:
        Dictionary mapping feature names to float values.
    """
    if isinstance(scenario, dict):
        identity_data = scenario
    else:
        identity_data = scenario.model_dump()

    demog = identity_data.get("identity_attributes", {})
    contact = identity_data.get("contact_attributes", {})
    meta = identity_data.get("account_metadata", {})
    device = identity_data.get("device_context", {})
    lifecycle = identity_data.get("lifecycle_info", {})

    # 1. Name vs Email Anomaly
    first_name = _sanitize(demog.get("first_name", ""))
    last_name = _sanitize(demog.get("last_name", ""))
    email = contact.get("primary_email", "").lower()
    email_local = email.split("@")[0] if "@" in email else email

    if first_name and last_name:
        matches_name = (first_name in email_local) or (last_name in email_local)
        name_email_anomaly = 0.0 if matches_name else 1.0
    else:
        name_email_anomaly = 0.5

    # 2. SSN Format Irregularity
    ssn_proxy = str(demog.get("ssn_proxy", ""))
    if ssn_proxy.startswith("999-00-") or ssn_proxy.startswith("000-") or not ssn_proxy:
        ssn_format_irregularity = 1.0
    else:
        ssn_format_irregularity = 0.0

    # 3. Location Mismatch Anomaly
    contact_city = _sanitize(contact.get("city", ""))
    contact_state = str(contact.get("state", "")).upper()
    ip_city = _sanitize(device.get("ip_city", ""))
    ip_state = str(device.get("ip_state", "")).upper()

    if "datacenter" in ip_city or "proxy" in ip_city or ip_state == "XX":
        location_mismatch_anomaly = 1.0
    elif contact_state and ip_state and contact_state != ip_state:
        location_mismatch_anomaly = 0.8
    elif contact_city and ip_city and contact_city != ip_city:
        location_mismatch_anomaly = 0.4
    else:
        location_mismatch_anomaly = 0.0

    # 4. Profile Implausibility
    age = demog.get("age", 35)
    income = demog.get("annual_income", 50000.0)
    emp_status = demog.get("employment_status", "employed")
    credit_score = demog.get("credit_score_proxy", 700)
    account_tier = meta.get("account_tier", "standard")

    implausibility = 0.0
    if age < 24 and income > 200000.0:
        implausibility += 0.5
    if emp_status in ("student", "unemployed") and income > 150000.0:
        implausibility += 0.5
    if credit_score < 500 and account_tier in ("premium", "preferred"):
        implausibility += 0.4
    profile_implausibility = min(1.0, implausibility)

    # 5. Disposable Email Domain
    email_domain = contact.get("email_domain", "").lower()
    if email_domain in _DISPOSABLE_DOMAINS or any(burner in email_domain for burner in ["temp", "burner", "throwaway", "fake"]):
        is_disposable_email = 1.0
    else:
        is_disposable_email = 0.0

    # 6. VoIP Carrier
    carrier = str(contact.get("phone_carrier_proxy", "")).lower()
    if "voip" in carrier or "virtual" in carrier or "bandwidth" in carrier or "twilio" in carrier:
        is_voip_carrier = 1.0
    else:
        is_voip_carrier = 0.0

    # 7. Email and Phone Tenure Deficit
    email_age = float(contact.get("email_age_years", 5.0))
    phone_tenure = float(contact.get("phone_tenure_months", 24))
    tenure_deficit = 0.0
    if email_age < 0.5:
        tenure_deficit += 0.5
    if phone_tenure <= 2:
        tenure_deficit += 0.5
    email_phone_tenure_deficit = min(1.0, tenure_deficit)

    # 8. Emulator / Headless Device
    dev_type = str(device.get("device_type", "")).lower()
    os_name = str(device.get("os", "")).lower()
    known_devices = device.get("known_devices", [])
    is_emulator = False
    if "emulator" in dev_type or "headless" in dev_type or "virtual" in dev_type or "vm" in os_name:
        is_emulator = True
    for kd in known_devices:
        if kd.get("is_emulator", False) or "emulator" in str(kd.get("device_type", "")).lower():
            is_emulator = True
            break
    is_emulator_device = 1.0 if is_emulator else 0.0

    # 9. Device Reputation Deficit
    trusted_count = int(device.get("trusted_device_count", 1))
    dev_consistency = float(device.get("device_consistency_ratio", 0.9))
    if trusted_count == 0 and dev_consistency < 0.50:
        device_reputation_deficit = 1.0
    elif trusted_count == 0 or dev_consistency < 0.70:
        device_reputation_deficit = 0.5
    else:
        device_reputation_deficit = 0.0

    # 10. Datacenter / Proxy IP
    ip_addr = str(device.get("ip_address", ""))
    if ip_addr.startswith("10.") or ip_addr.startswith("172.16.") or ip_addr.startswith("192.168.") or location_mismatch_anomaly == 1.0:
        is_datacenter_proxy_ip = 1.0
    else:
        is_datacenter_proxy_ip = 0.0

    # 11. Lifecycle Incoherence
    coherence_score = float(lifecycle.get("lifecycle_coherence_score", 0.95))
    lifecycle_incoherence = max(0.0, min(1.0, 1.0 - coherence_score))

    # 12. Early Bust-out / Risky Activity
    risk_event_count = int(lifecycle.get("risk_event_count", 0))
    days_to_risk = lifecycle.get("days_to_risky_activity", None)
    if days_to_risk is not None and days_to_risk <= 5:
        early_bust_out_risk = 1.0
    elif risk_event_count >= 2:
        early_bust_out_risk = 0.9
    elif risk_event_count == 1:
        early_bust_out_risk = 0.5
    else:
        early_bust_out_risk = 0.0

    return {
        "name_email_anomaly": round(name_email_anomaly, 4),
        "ssn_format_irregularity": round(ssn_format_irregularity, 4),
        "location_mismatch_anomaly": round(location_mismatch_anomaly, 4),
        "profile_implausibility": round(profile_implausibility, 4),
        "is_disposable_email": round(is_disposable_email, 4),
        "is_voip_carrier": round(is_voip_carrier, 4),
        "email_phone_tenure_deficit": round(email_phone_tenure_deficit, 4),
        "is_emulator_device": round(is_emulator_device, 4),
        "device_reputation_deficit": round(device_reputation_deficit, 4),
        "is_datacenter_proxy_ip": round(is_datacenter_proxy_ip, 4),
        "lifecycle_incoherence": round(lifecycle_incoherence, 4),
        "early_bust_out_risk": round(early_bust_out_risk, 4),
    }
