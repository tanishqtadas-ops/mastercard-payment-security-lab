"""
attacks/synthetic_identity/generator.py — Family 3 (Synthetic Identity) Attack Generator.

Generates AttackEvent objects containing SyntheticIdentity scenarios modeling
synthetic identity creation and account lifecycle behavior.
"""

from __future__ import annotations

from datetime import date, timedelta
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from faker import Faker

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.identity import SyntheticIdentity
from mutation.genome_engine import validate_genome


# Canonical Family 3 genome dimensions (defined in MASTER_SPEC.md § 3)
FAMILY3_GENOME_DIMENSIONS: Tuple[str, ...] = (
    "cross_field_consistency",
    "profile_plausibility_score",
    "contact_consistency",
    "device_history_score",
    "lifecycle_behavior_coherence",
    "time_to_risky_activity",
)

# Baseline high-risk attack genome (crude synthetic identity / early bust-out)
DEFAULT_ATTACK_GENOME: Dict[str, float] = {
    "cross_field_consistency": 0.25,
    "profile_plausibility_score": 0.30,
    "contact_consistency": 0.20,
    "device_history_score": 0.25,
    "lifecycle_behavior_coherence": 0.20,
    "time_to_risky_activity": 0.15,
}

# Baseline legitimate / authentic identity genome (high consistency / benign lifecycle)
DEFAULT_LEGITIMATE_GENOME: Dict[str, float] = {
    "cross_field_consistency": 0.95,
    "profile_plausibility_score": 0.92,
    "contact_consistency": 0.90,
    "device_history_score": 0.95,
    "lifecycle_behavior_coherence": 0.95,
    "time_to_risky_activity": 0.90,
}

_DISPOSABLE_EMAIL_DOMAINS = [
    "tempmail.xyz",
    "guerrillamail.net",
    "throwawaymail.org",
    "burner-inbox.com",
    "10minutemail.co",
    "fakeinbox.io",
]

_LEGITIMATE_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "icloud.com",
    "protonmail.com",
]

_VOIP_CARRIERS = [
    "Twilio VoIP Virtual",
    "Bandwidth.com VoIP",
    "TextNow Virtual Carrier",
    "Google Voice VoIP",
    "Vonage Digital Line",
]

_LEGITIMATE_CARRIERS = [
    "Verizon Wireless",
    "AT&T Mobility",
    "T-Mobile USA",
    "Mint Mobile",
]

_SUSPICIOUS_EMULATOR_DEVICES = [
    ("Android Emulator API 33", "Headless Chrome Mobile", "mobile_emulator"),
    ("Linux x86_64", "Headless Chrome / Puppeteer", "headless_browser"),
    ("Windows 10 VM", "Tor Browser / Firefox", "virtual_machine"),
    ("Android 11 NoxPlayer", "Chromium WebView", "mobile_emulator"),
]

_LEGITIMATE_DEVICES = [
    ("iOS 17.4", "Mobile Safari", "mobile"),
    ("Android 14", "Chrome Mobile", "mobile"),
    ("Windows 11", "Chrome", "desktop"),
    ("macOS 14.2", "Safari", "desktop"),
]


def _sanitize_name(name: str) -> str:
    """Sanitizes names for email handles."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
    return cleaned if cleaned else "user"


class SyntheticIdentityAttackGenerator:
    """
    Produces deterministic, schema-compliant Family 3 AttackEvents.

    Models synthetic identity creation, fictitious demographic compilation,
    disposable contact artifacts, emulator device fingerprints, and abnormal
    lifecycle / bust-out behavior across the six canonical Family 3 genome dimensions.

    Satisfies the AttackGenerator protocol in simulation.interfaces.
    """

    REFERENCE_DATE: date = date(2026, 1, 1)

    def __init__(
        self,
        genome: Optional[Dict[str, float]] = None,
        ground_truth: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Family 3 Attack Generator.

        Args:
            genome: Initial 6-dimension attack genome. Defaults to DEFAULT_ATTACK_GENOME
                    if ground_truth is True, else DEFAULT_LEGITIMATE_GENOME.
            ground_truth: True if generated events represent synthetic identity attacks.
            seed: Optional random seed for reproducible scenario generation.
        """
        self._ground_truth = ground_truth
        self.seed = seed
        self._rng = random.Random(seed)
        self._faker = Faker("en_US")
        if seed is not None:
            self._faker.seed_instance(seed)

        if genome is not None:
            self.set_genome(genome)
        else:
            default_g = DEFAULT_ATTACK_GENOME if ground_truth else DEFAULT_LEGITIMATE_GENOME
            self._genome = dict(default_g)
            validate_genome(self._genome)

    @property
    def genome(self) -> Dict[str, float]:
        """Return a copy of the currently active attack genome."""
        return dict(self._genome)

    def set_genome(self, genome: Dict[str, float]) -> None:
        """
        Replace the active attack genome.

        Validates that all six Family 3 dimensions are present, numeric, and in [0.0, 1.0].
        """
        validate_genome(genome)
        missing_keys = set(FAMILY3_GENOME_DIMENSIONS) - set(genome.keys())
        if missing_keys:
            raise ValueError(
                f"Genome missing canonical Family 3 dimensions: {sorted(missing_keys)}"
            )
        extra_keys = set(genome.keys()) - set(FAMILY3_GENOME_DIMENSIONS)
        if extra_keys:
            raise ValueError(
                f"Genome contains non-Family 3 dimensions: {sorted(extra_keys)}"
            )

        for key, val in genome.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Family 3 genome dimension '{key}' value {val} out of bounds [0.0, 1.0]"
                )

        self._genome = dict(genome)

    def _generate_synthetic_identity_scenario(
        self, round_id: str, genome: Dict[str, float]
    ) -> SyntheticIdentity:
        """
        Synthesizes a SyntheticIdentity modulated by the 6 genome dimensions.
        """
        cross_field = genome["cross_field_consistency"]
        profile_plaus = genome["profile_plausibility_score"]
        contact_cons = genome["contact_consistency"]
        dev_hist = genome["device_history_score"]
        lifecycle_coh = genome["lifecycle_behavior_coherence"]
        time_to_risk = genome["time_to_risky_activity"]

        # 1. Demographic Attributes
        gender = self._rng.choice(["M", "F"])
        first_name = self._faker.first_name_male() if gender == "M" else self._faker.first_name_female()
        last_name = self._faker.last_name()
        full_name = f"{first_name} {last_name}"

        # Plausibility modulation
        if profile_plaus >= 0.70:
            age = self._rng.randint(22, 68)
            if age >= 65:
                employment_status = "retired"
                annual_income = round(self._rng.uniform(38000.0, 95000.0), 2)
            else:
                employment_status = self._rng.choice(["employed", "self-employed"])
                annual_income = round(self._rng.uniform(48000.0, 160000.0), 2)
            credit_score_proxy = int(self._rng.gauss(720, 40))
            credit_score_proxy = min(max(credit_score_proxy, 620), 840)
        else:
            # Implausible demographic compilation (e.g. young student with astronomical income, or extreme credit)
            age = self._rng.choice([19, 21, 23, 78])
            employment_status = self._rng.choice(["student", "unemployed", "retired"])
            annual_income = round(self._rng.uniform(220000.0, 450000.0), 2)
            credit_score_proxy = self._rng.choice([430, 480, 850])

        birth_month = self._rng.randint(1, 12)
        birth_day = self._rng.randint(1, 28)
        if (birth_month, birth_day) <= (self.REFERENCE_DATE.month, self.REFERENCE_DATE.day):
            birth_year = self.REFERENCE_DATE.year - age
        else:
            birth_year = self.REFERENCE_DATE.year - age - 1
        dob = date(birth_year, birth_month, birth_day).isoformat()

        # Synthetic SSN proxy
        if cross_field >= 0.70:
            ssn_proxy = f"9{self._rng.randint(10, 99):02d}-{self._rng.randint(10, 99):02d}-{self._rng.randint(1000, 9999):04d}"
        else:
            # Irregular / synthetic pattern
            ssn_proxy = f"999-00-{self._rng.randint(1000, 9999):04d}"

        demographics = {
            "first_name": first_name,
            "last_name": last_name,
            "full_name": full_name,
            "gender": gender,
            "dob": dob,
            "age": age,
            "nationality": "US",
            "ssn_proxy": ssn_proxy,
            "employment_status": employment_status,
            "annual_income": annual_income,
            "credit_score_proxy": credit_score_proxy,
        }

        # 2. Contact Attributes
        state = self._faker.state_abbr()
        city = self._faker.city()
        zip_code = self._faker.zipcode_in_state(state)
        street_address = self._faker.street_address()

        # Cross-field & contact consistency modulation
        if cross_field >= 0.70:
            fn_clean = _sanitize_name(first_name)
            ln_clean = _sanitize_name(last_name)
            email_user = f"{fn_clean}.{ln_clean}"
        else:
            # Mismatched name handle in email (cross-field anomaly)
            unrelated_name = self._faker.first_name().lower()
            email_user = f"{unrelated_name}.anon{self._rng.randint(100, 999)}"

        if contact_cons >= 0.70:
            domain = self._rng.choice(_LEGITIMATE_EMAIL_DOMAINS)
            email_age_years = round(self._rng.uniform(2.0, 12.0), 1)
            phone_carrier = self._rng.choice(_LEGITIMATE_CARRIERS)
            phone_tenure_months = self._rng.randint(18, 96)
            contact_stability_score = round(self._rng.uniform(0.85, 1.0), 3)
        else:
            domain = self._rng.choice(_DISPOSABLE_EMAIL_DOMAINS)
            email_age_years = round(self._rng.uniform(0.01, 0.20), 2)
            phone_carrier = self._rng.choice(_VOIP_CARRIERS)
            phone_tenure_months = self._rng.randint(0, 2)
            contact_stability_score = round(self._rng.uniform(0.15, 0.45), 3)

        email = f"{email_user}@{domain}".lower()
        phone_number = self._faker.phone_number()

        contacts = {
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

        # 3. Account Metadata
        # Lifecycle / time to risk modulation
        if time_to_risk >= 0.70:
            account_age_days = self._rng.randint(90, 800)
        else:
            # Brand new synthetic account for rapid bust-out
            account_age_days = self._rng.randint(5, 30)

        open_date = (self.REFERENCE_DATE - timedelta(days=account_age_days)).isoformat()
        account_type = self._rng.choice(["checking", "credit", "savings"])
        account_tier = "premium" if profile_plaus < 0.70 else "standard"
        initial_deposit = round(self._rng.uniform(50.0, 2500.0), 2)

        account_meta = {
            "account_id": f"acc_f3_{round_id.replace('-', '_')[:8]}_{self._rng.randint(1000, 9999)}",
            "account_open_date": open_date,
            "account_age_days": account_age_days,
            "account_type": account_type,
            "account_tier": account_tier,
            "currency": "USD",
            "initial_deposit": initial_deposit,
            "kyc_verification_status": "verified" if cross_field >= 0.50 else "synthetic_bypass_flagged",
            "kyc_level": "level_2_full" if cross_field >= 0.50 else "level_1_minimal",
            "account_status": "active",
        }

        # 4. Device Context
        if dev_hist >= 0.70:
            os_name, browser_name, dev_type = self._rng.choice(_LEGITIMATE_DEVICES)
            ip_address = self._faker.ipv4()
            ip_city = city
            ip_state = state
            trusted_device_count = self._rng.randint(1, 3)
            device_first_seen_days_ago = account_age_days + self._rng.randint(0, 10)
            device_consistency_ratio = round(self._rng.uniform(0.85, 1.0), 3)
            is_emulator = False
        else:
            os_name, browser_name, dev_type = self._rng.choice(_SUSPICIOUS_EMULATOR_DEVICES)
            ip_address = f"10.{self._rng.randint(1, 254)}.{self._rng.randint(1, 254)}.{self._rng.randint(1, 254)}"  # Proxy / datacenter IP
            ip_city = "Datacenter Hub / Proxy Exit"
            ip_state = "XX"
            trusted_device_count = 0
            device_first_seen_days_ago = 0
            device_consistency_ratio = round(self._rng.uniform(0.10, 0.40), 3)
            is_emulator = True

        primary_device_id = f"dev_f3_{self._rng.randint(100000, 999999)}"
        known_devices = [
            {
                "device_id": primary_device_id,
                "device_type": dev_type,
                "os": os_name,
                "browser": browser_name,
                "is_primary": True,
                "is_emulator": is_emulator,
                "first_seen_days_ago": device_first_seen_days_ago,
            }
        ]

        device_ctx = {
            "primary_device_id": primary_device_id,
            "device_type": dev_type,
            "os": os_name,
            "browser": browser_name,
            "ip_address": ip_address,
            "ip_city": ip_city,
            "ip_state": ip_state,
            "trusted_device_count": trusted_device_count,
            "device_first_seen_days_ago": device_first_seen_days_ago,
            "device_consistency_ratio": device_consistency_ratio,
            "known_devices": known_devices,
        }

        # 5. Lifecycle Information
        if time_to_risk >= 0.70:
            days_to_first_transaction = self._rng.randint(2, 6)
            risk_event_count = 0 if not self._ground_truth else 1
            days_to_risky_activity = None if not self._ground_truth else account_age_days - 2
            login_freq = round(self._rng.uniform(10.0, 30.0), 1)
            tx_count = max(1, int(account_age_days * 0.8))
            avg_tx_amount = round(self._rng.uniform(35.0, 150.0), 2)
            total_spend = round(tx_count * avg_tx_amount, 2)
            lifecycle_coherence_score = round(self._rng.uniform(0.85, 0.98), 3) if lifecycle_coh >= 0.70 else round(self._rng.uniform(0.30, 0.55), 3)
        else:
            # Rapid bust-out or attack execution
            days_to_first_transaction = 1
            risk_event_count = 3
            days_to_risky_activity = self._rng.randint(1, 3)
            login_freq = round(self._rng.uniform(1.0, 4.0), 1)
            tx_count = self._rng.randint(2, 5)
            avg_tx_amount = round(self._rng.uniform(2500.0, 8500.0), 2)  # High burst drain
            total_spend = round(tx_count * avg_tx_amount, 2)
            lifecycle_coherence_score = round(self._rng.uniform(0.12, 0.38), 3)

        lifecycle_events_summary = [
            {"event": "account_created", "relative_day": 0},
            {"event": "kyc_processed", "relative_day": 0},
            {"event": "first_transaction", "relative_day": days_to_first_transaction},
        ]
        if days_to_risky_activity is not None:
            lifecycle_events_summary.append({
                "event": "high_value_rapid_bust_out_attempt",
                "relative_day": days_to_risky_activity,
            })

        lifecycle = {
            "account_age_days": account_age_days,
            "days_to_first_transaction": days_to_first_transaction,
            "days_to_first_kyc_update": 0,
            "login_frequency_per_month": login_freq,
            "transaction_count": tx_count,
            "avg_transaction_amount": avg_tx_amount,
            "total_spend": total_spend,
            "risk_event_count": risk_event_count,
            "days_to_risky_activity": days_to_risky_activity,
            "lifecycle_coherence_score": lifecycle_coherence_score,
            "lifecycle_events_summary": lifecycle_events_summary,
        }

        identity_id = f"ident_f3_{round_id.replace('-', '_')[:8]}_{self._rng.randint(1000, 9999)}"

        return SyntheticIdentity(
            identity_id=identity_id,
            identity_attributes=demographics,
            contact_attributes=contacts,
            account_metadata=account_meta,
            device_context=device_ctx,
            lifecycle_info=lifecycle,
        )

    def generate(self, round_id: str) -> AttackEvent:
        """
        Generate a fully-populated AttackEvent containing a SyntheticIdentity scenario.

        Args:
            round_id: Unique identifier for the simulation round.

        Returns:
            AttackEvent configured with AttackFamily.SYNTHETIC_IDENTITY and scenario data.
        """
        genome = self._genome
        synthetic_identity = self._generate_synthetic_identity_scenario(round_id, genome)

        attack_id = f"family3-attack-{round_id}"
        scenario_data = synthetic_identity.model_dump(mode="json")

        return AttackEvent(
            attack_id=attack_id,
            round_id=round_id,
            attack_family=AttackFamily.SYNTHETIC_IDENTITY,
            attack_genome=dict(genome),
            scenario=scenario_data,
            ground_truth=self._ground_truth,
            metadata={
                "generator": "SyntheticIdentityAttackGenerator",
                "attack_family": AttackFamily.SYNTHETIC_IDENTITY.value,
                "identity_id": synthetic_identity.identity_id,
                "cross_field_consistency": genome["cross_field_consistency"],
                "profile_plausibility_score": genome["profile_plausibility_score"],
                "contact_consistency": genome["contact_consistency"],
                "device_history_score": genome["device_history_score"],
                "lifecycle_behavior_coherence": genome["lifecycle_behavior_coherence"],
                "time_to_risky_activity": genome["time_to_risky_activity"],
                "attack_type": "synthetic_identity_lifecycle",
                "risk_event_count": synthetic_identity.lifecycle_info["risk_event_count"],
            },
        )
