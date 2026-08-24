"""
data/generators/identity_generator.py -- Deterministic Synthetic Identity & Lifecycle Generator.

Generates coherent legitimate synthetic identities and longitudinal account lifecycle data
conforming to the schemas.identity.SyntheticIdentity contract.

Key characteristics:
- Deterministic and fully reproducible via seeded random state (Faker + random.Random).
- 100% synthetic fictional data (no real personal data).
- Internally coherent cross-field relationships:
    - Demographic: DOB matches age, realistic income/employment/credit proxies.
    - Contact: email corresponds to name, address has consistent city/state/zip, phone carrier proxy.
    - Account: open date valid for adult age, verified KYC status, active state.
    - Device: realistic OS/browser pairing, geographic alignment with contact address, device tenure.
    - Lifecycle: normal onboarding progression, longitudinal activity history, 0 risk events.
- Structured so downstream Family 3 feature extraction and model training can derive:
    - cross_field_consistency
    - profile_plausibility_score
    - contact_consistency
    - device_history_score
    - lifecycle_behavior_coherence
    - time_to_risky_activity
"""

from __future__ import annotations

import json
import random
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from faker import Faker

from schemas.identity import SyntheticIdentity


# Common plausible email providers for legitimate identities
_LEGITIMATE_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "icloud.com",
    "protonmail.com",
    "comcast.net",
    "att.net",
    "verizon.net",
]

_PHONE_CARRIERS = ["Verizon", "AT&T", "T-Mobile", "Mint Mobile", "Cricket"]

_OS_BROWSER_PAIRS = [
    ("iOS 17.4", "Mobile Safari", "mobile"),
    ("iOS 16.6", "Mobile Safari", "mobile"),
    ("Android 14", "Chrome Mobile", "mobile"),
    ("Android 13", "Samsung Internet", "mobile"),
    ("Windows 11", "Chrome", "desktop"),
    ("Windows 11", "Edge", "desktop"),
    ("macOS 14.2", "Safari", "desktop"),
    ("macOS 14.1", "Chrome", "desktop"),
    ("iPadOS 17.3", "Mobile Safari", "tablet"),
]

_ACCOUNT_TYPES = ["checking", "savings", "credit"]
_ACCOUNT_TIERS = ["standard", "preferred", "premium"]
_EMPLOYMENT_STATUSES = ["employed", "self-employed", "employed", "retired", "student"]


def _sanitize_name_for_email(name: str) -> str:
    """Sanitizes names for standard email handle construction."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
    return cleaned if cleaned else "user"


def calculate_age_at_date(
    dob: Union[date, str], reference_date: Union[date, str] = date(2026, 1, 1)
) -> int:
    """
    Calculates exact age in completed years at reference_date.

    Args:
        dob: Date of birth (date object or ISO string YYYY-MM-DD).
        reference_date: Reference anchor date (date object or ISO string YYYY-MM-DD).

    Returns:
        Exact age as integer.
    """
    if isinstance(dob, str):
        dob = date.fromisoformat(dob)
    if isinstance(reference_date, str):
        reference_date = date.fromisoformat(reference_date)

    return reference_date.year - dob.year - ((reference_date.month, reference_date.day) < (dob.month, dob.day))


class LegitimateIdentityGenerator:
    """
    Deterministic generator for synthetic legitimate identities and accounts.
    """

    REFERENCE_DATE: date = date(2026, 1, 1)

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the generator with an isolated deterministic seed.

        Args:
            seed: Integer seed for reproducible random generation.
        """
        self.seed = seed
        self._rng = random.Random(seed)
        self._faker = Faker("en_US")
        self._faker.seed_instance(seed)

        # Anchor reference date for simulation consistency (2026-01-01)
        self._reference_date = self.REFERENCE_DATE

    def _generate_demographics(self, index: int) -> Dict[str, Any]:
        """Generates coherent demographic attributes with exact DOB-to-age consistency."""
        gender = self._rng.choice(["M", "F"])
        first_name = self._faker.first_name_male() if gender == "M" else self._faker.first_name_female()
        last_name = self._faker.last_name()

        # Age distribution: 20 to 75 years old
        age = self._rng.randint(20, 75)
        birth_month = self._rng.randint(1, 12)
        birth_day = self._rng.randint(1, 28)

        # Make DOB precisely consistent with stored age on reference date
        if (birth_month, birth_day) <= (self._reference_date.month, self._reference_date.day):
            birth_year = self._reference_date.year - age
        else:
            birth_year = self._reference_date.year - age - 1

        dob = date(birth_year, birth_month, birth_day).isoformat()

        # Coherent employment and income
        if age >= 65 and self._rng.random() < 0.7:
            employment_status = "retired"
            annual_income = round(self._rng.uniform(35000.0, 90000.0), 2)
        elif age < 24 and self._rng.random() < 0.5:
            employment_status = "student"
            annual_income = round(self._rng.uniform(15000.0, 35000.0), 2)
        else:
            employment_status = self._rng.choice(["employed", "employed", "self-employed"])
            annual_income = round(self._rng.uniform(45000.0, 165000.0), 2)

        # Credit score proxy: realistic distribution for legitimate accounts (640-830)
        credit_score_proxy = int(self._rng.gauss(725, 45))
        credit_score_proxy = min(max(credit_score_proxy, 600), 850)

        # Synthetic SSN proxy formatted clearly as synthetic
        area = self._rng.randint(900, 999)  # 900-series SSNs are invalid/synthetic in real life
        group = self._rng.randint(10, 99)
        serial = self._rng.randint(1000, 9999)
        ssn_proxy = f"{area}-{group:02d}-{serial:04d}"

        return {
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"{first_name} {last_name}",
            "gender": gender,
            "dob": dob,
            "age": age,
            "nationality": "US",
            "ssn_proxy": ssn_proxy,
            "employment_status": employment_status,
            "annual_income": annual_income,
            "credit_score_proxy": credit_score_proxy,
        }

    def _generate_contacts(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        """Generates coherent contact attributes matching the demographic profile and location."""
        fn = _sanitize_name_for_email(demographics["first_name"])
        ln = _sanitize_name_for_email(demographics["last_name"])
        domain = self._rng.choice(_LEGITIMATE_EMAIL_DOMAINS)

        email_pattern = self._rng.choice([
            f"{fn}.{ln}@{domain}",
            f"{fn}{ln}@{domain}",
            f"{fn[0]}{ln}@{domain}",
            f"{fn}.{ln}{self._rng.randint(10, 99)}@{domain}",
        ])
        email = email_pattern.lower()

        # Email age: established email history for adult identities
        max_email_age = min(demographics["age"] - 16, 18)
        email_age_years = round(self._rng.uniform(2.0, max(2.5, float(max_email_age))), 1)

        # Phone and carrier
        phone_number = self._faker.phone_number()
        phone_carrier = self._rng.choice(_PHONE_CARRIERS)
        phone_tenure_months = self._rng.randint(12, int(email_age_years * 12))

        # Mutually coherent residential address (state and valid state-bound ZIP code)
        street_address = self._faker.street_address()
        state = self._faker.state_abbr()
        city = self._faker.city()
        zip_code = self._faker.zipcode_in_state(state)

        # Contact stability: high for legitimate users
        contact_stability_score = round(self._rng.uniform(0.88, 1.0), 3)

        return {
            "primary_email": email,
            "email_domain": domain,
            "email_age_years": email_age_years,
            "phone_number": phone_number,
            "phone_carrier_proxy": phone_carrier,
            "phone_tenure_months": phone_tenure_months,
            "street_address": street_address,
            "city": city,
            "state": state,
            "zip_code": zip_code,
            "country": "USA",
            "contact_stability_score": contact_stability_score,
        }

    def _generate_account_metadata(
        self, index: int, demographics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates coherent account lifecycle metadata."""
        # Account opened 30 to 1200 days ago, must be after turning 18
        max_account_age_days = min((demographics["age"] - 18) * 365, 1200)
        account_age_days = self._rng.randint(30, max(35, int(max_account_age_days)))
        open_date = self._reference_date - timedelta(days=account_age_days)

        account_type = self._rng.choice(_ACCOUNT_TYPES)
        account_tier = self._rng.choice(_ACCOUNT_TIERS)
        initial_deposit = round(self._rng.uniform(100.0, 5000.0), 2)

        return {
            "account_id": f"acc_{self.seed:04d}_{index:05d}",
            "account_open_date": open_date.isoformat(),
            "account_age_days": account_age_days,
            "account_type": account_type,
            "account_tier": account_tier,
            "currency": "USD",
            "initial_deposit": initial_deposit,
            "kyc_verification_status": "verified",
            "kyc_level": "level_2_full",
            "account_status": "active",
        }

    def _generate_device_context(
        self, contacts: Dict[str, Any], account_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates coherent device context matching user profile and geography."""
        os_name, browser_name, device_type = self._rng.choice(_OS_BROWSER_PAIRS)
        primary_device_id = f"dev_{self._rng.randint(100000, 999999)}_{self._rng.randint(1000, 9999)}"

        # Synthetic IP address
        ip_parts = [
            str(self._rng.randint(24, 220)),
            str(self._rng.randint(10, 250)),
            str(self._rng.randint(1, 254)),
            str(self._rng.randint(1, 254)),
        ]
        ip_address = ".".join(ip_parts)

        # Device consistency and history
        trusted_device_count = self._rng.randint(1, 3)
        device_first_seen_days_ago = account_meta["account_age_days"] + self._rng.randint(0, 15)
        device_consistency_ratio = round(self._rng.uniform(0.88, 1.0), 3)

        known_devices = [
            {
                "device_id": primary_device_id,
                "device_type": device_type,
                "os": os_name,
                "browser": browser_name,
                "is_primary": True,
                "first_seen_days_ago": device_first_seen_days_ago,
            }
        ]

        return {
            "primary_device_id": primary_device_id,
            "device_type": device_type,
            "os": os_name,
            "browser": browser_name,
            "ip_address": ip_address,
            "ip_city": contacts["city"],
            "ip_state": contacts["state"],
            "trusted_device_count": trusted_device_count,
            "device_first_seen_days_ago": device_first_seen_days_ago,
            "device_consistency_ratio": device_consistency_ratio,
            "known_devices": known_devices,
        }

    def _generate_lifecycle_info(
        self, account_meta: Dict[str, Any], demographics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates longitudinal behavioral lifecycle information."""
        account_age_days = account_meta["account_age_days"]
        days_to_first_transaction = self._rng.randint(1, 7)
        days_to_first_kyc_update = self._rng.randint(0, 2)

        # Longitudinal activity rates
        months_active = max(1.0, account_age_days / 30.0)
        login_freq_per_month = round(self._rng.uniform(8.0, 35.0), 1)
        tx_per_month = self._rng.randint(5, 40)
        transaction_count = max(1, int(tx_per_month * months_active))
        avg_transaction_amount = round(self._rng.uniform(25.0, 180.0), 2)
        total_spend = round(transaction_count * avg_transaction_amount, 2)

        # Legitimate accounts have zero risk events and high lifecycle coherence
        risk_event_count = 0
        days_to_risky_activity = None
        lifecycle_coherence_score = round(self._rng.uniform(0.90, 1.0), 3)

        # Chronological milestone events
        lifecycle_events_summary = [
            {"event": "account_created", "relative_day": 0},
            {"event": "kyc_verified", "relative_day": days_to_first_kyc_update},
            {"event": "initial_deposit_completed", "relative_day": min(1, days_to_first_transaction)},
            {"event": "first_transaction", "relative_day": days_to_first_transaction},
            {"event": "routine_activity_established", "relative_day": min(30, account_age_days)},
        ]

        return {
            "account_age_days": account_age_days,
            "days_to_first_transaction": days_to_first_transaction,
            "days_to_first_kyc_update": days_to_first_kyc_update,
            "login_frequency_per_month": login_freq_per_month,
            "transaction_count": transaction_count,
            "avg_transaction_amount": avg_transaction_amount,
            "total_spend": total_spend,
            "risk_event_count": risk_event_count,
            "days_to_risky_activity": days_to_risky_activity,
            "lifecycle_coherence_score": lifecycle_coherence_score,
            "lifecycle_events_summary": lifecycle_events_summary,
        }

    def generate_identity(self, index: int = 0) -> SyntheticIdentity:
        """
        Generates a single coherent legitimate synthetic identity.

        Args:
            index: Identifier index for reproducible sequence numbering.

        Returns:
            A validated SyntheticIdentity instance.
        """
        demographics = self._generate_demographics(index)
        contacts = self._generate_contacts(demographics)
        account_meta = self._generate_account_metadata(index, demographics)
        device_ctx = self._generate_device_context(contacts, account_meta)
        lifecycle = self._generate_lifecycle_info(account_meta, demographics)

        identity_id = f"ident_{self.seed:04d}_{index:05d}"

        return SyntheticIdentity(
            identity_id=identity_id,
            identity_attributes=demographics,
            contact_attributes=contacts,
            account_metadata=account_meta,
            device_context=device_ctx,
            lifecycle_info=lifecycle,
        )

    def generate_dataset(self, n: int = 100) -> List[SyntheticIdentity]:
        """
        Generates a sequence of n legitimate synthetic identities.

        Args:
            n: Number of records to generate (must be >= 1).

        Returns:
            List of validated SyntheticIdentity instances.
        """
        if n < 1:
            raise ValueError(f"Requested dataset size n must be >= 1, got {n}")

        return [self.generate_identity(index=i) for i in range(n)]


def save_dataset(identities: List[SyntheticIdentity], file_path: Union[str, Path]) -> None:
    """
    Saves a list of SyntheticIdentity instances to a JSON file.

    Args:
        identities: List of SyntheticIdentity objects.
        file_path: Destination file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [ident.model_dump() for ident in identities]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_dataset(file_path: Union[str, Path]) -> List[SyntheticIdentity]:
    """
    Loads a list of SyntheticIdentity instances from a JSON file.

    Args:
        file_path: Source file path.

    Returns:
        List of SyntheticIdentity instances.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [SyntheticIdentity(**item) for item in data]


def generate_legitimate_baseline(
    output_path: Optional[Union[str, Path]] = None,
    n: int = 500,
    seed: int = 42,
) -> List[SyntheticIdentity]:
    """
    Generates and optionally persists the canonical baseline legitimate training dataset.

    Args:
        output_path: Optional target file path. If provided, dataset is saved to disk.
        n: Number of baseline records (default 500).
        seed: Random seed for deterministic generation (default 42).

    Returns:
        List of generated SyntheticIdentity instances.
    """
    generator = LegitimateIdentityGenerator(seed=seed)
    dataset = generator.generate_dataset(n=n)

    if output_path is not None:
        save_dataset(dataset, output_path)

    return dataset
