"""
attacks/transaction_evasion/generator.py — Family 1 (Transaction Evasion) Attack Generator.

Generates AttackEvent objects containing a target Transaction evaluated against a user's
historical baseline behavior and recent transaction sequence.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import random
from typing import Any, Dict, List, Optional, Tuple

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.transaction import Transaction
from mutation.genome_engine import validate_genome


# Canonical Family 1 genome dimensions (defined in MASTER_SPEC.md § 3)
FAMILY1_GENOME_DIMENSIONS: Tuple[str, ...] = (
    "amount_deviation",
    "velocity_deviation",
    "device_novelty",
    "location_deviation",
    "time_deviation",
    "sequence_anomaly",
)

# Baseline high-risk attack genome (clearly anomalous compared with normal behavior)
DEFAULT_ATTACK_GENOME: Dict[str, float] = {
    "amount_deviation": 0.85,
    "velocity_deviation": 0.80,
    "device_novelty": 0.90,
    "location_deviation": 0.85,
    "time_deviation": 0.75,
    "sequence_anomaly": 0.80,
}

# Baseline legitimate / normal behavior genome (closely matching user baseline)
DEFAULT_LEGITIMATE_GENOME: Dict[str, float] = {
    "amount_deviation": 0.05,
    "velocity_deviation": 0.05,
    "device_novelty": 0.02,
    "location_deviation": 0.02,
    "time_deviation": 0.05,
    "sequence_anomaly": 0.03,
}

# User profile personas representing typical, legitimate spending baselines
_USER_PERSONAS: List[Dict[str, Any]] = [
    {
        "user_id": "usr_retail_101",
        "home_location": "New York, US",
        "frequent_locations": ["New York, US", "Jersey City, US", "Brooklyn, US"],
        "registered_devices": ["dev_ios_iphone15_a1", "dev_macbook_air_m2"],
        "typical_channels": ["pos_contactless", "online_card_on_file", "apple_pay"],
        "currency": "USD",
        "avg_amount": 42.50,
        "std_amount": 15.00,
        "max_historical_amount": 210.00,
        "frequent_categories": ["grocery", "coffee_shop", "restaurants", "streaming_media"],
        "frequent_merchants": [
            ("Whole Foods Market", "grocery", "merch_gro_01"),
            ("Starbucks Coffee", "coffee_shop", "merch_cof_01"),
            ("Chipotle Mexican Grill", "restaurants", "merch_rst_01"),
            ("Netflix Streaming", "streaming_media", "merch_str_01"),
        ],
        "active_hours": (8, 22),
        "typical_interval_hours": 24.0,
    },
    {
        "user_id": "usr_tech_202",
        "home_location": "San Francisco, US",
        "frequent_locations": ["San Francisco, US", "Oakland, US", "San Jose, US"],
        "registered_devices": ["dev_android_pixel8_b2", "dev_thinkpad_linux_01"],
        "typical_channels": ["online_card_on_file", "pos_chip_pin", "google_pay"],
        "currency": "USD",
        "avg_amount": 78.00,
        "std_amount": 28.00,
        "max_historical_amount": 380.00,
        "frequent_categories": ["cloud_saas", "ride_sharing", "food_delivery", "bookstore"],
        "frequent_merchants": [
            ("GitHub Enterprise", "cloud_saas", "merch_cld_01"),
            ("Uber Technologies", "ride_sharing", "merch_rde_01"),
            ("DoorDash Food", "food_delivery", "merch_fd_01"),
            ("Kinokuniya Books", "bookstore", "merch_bks_01"),
        ],
        "active_hours": (9, 23),
        "typical_interval_hours": 18.0,
    },
    {
        "user_id": "usr_traveler_303",
        "home_location": "London, UK",
        "frequent_locations": ["London, UK", "Manchester, UK", "Edinburgh, UK"],
        "registered_devices": ["dev_ios_iphone14_c3", "dev_ipad_pro_02"],
        "typical_channels": ["pos_contactless", "apple_pay", "pos_chip_pin"],
        "currency": "GBP",
        "avg_amount": 55.00,
        "std_amount": 22.00,
        "max_historical_amount": 290.00,
        "frequent_categories": ["rail_transit", "supermarket", "hospitality", "fuel_station"],
        "frequent_merchants": [
            ("Transport for London", "rail_transit", "merch_tfl_01"),
            ("Sainsbury's Supermarket", "supermarket", "merch_snb_01"),
            ("Premier Inn Hotels", "hospitality", "merch_hsp_01"),
            ("Shell Petroleum", "fuel_station", "merch_shl_01"),
        ],
        "active_hours": (7, 21),
        "typical_interval_hours": 14.0,
    },
]

# Anomalous options used when deviation dimensions are high
_ANOMALOUS_LOCATIONS = [
    "Lagos, NG",
    "St. Petersburg, RU",
    "Hong Kong, HK",
    "Kyiv, UA",
    "São Paulo, BR",
    "Reykjavik, IS",
]

_ANOMALOUS_DEVICES = [
    "dev_unrecognized_android_emulator_99",
    "dev_headless_puppeteer_node_01",
    "dev_tor_browser_virtualbox_77",
    "dev_unknown_windows_vdi_88",
]

_ANOMALOUS_MERCHANTS = [
    ("CryptoExchange Instant", "cryptocurrency_onramp", "merch_anom_crypto_01"),
    ("Prepaid Visa Gift Cards", "prepaid_giftcards", "merch_anom_gift_01"),
    ("LuxeJewel Diamond Vault", "luxury_jewelry", "merch_anom_lux_01"),
    ("DarkNode Virtual Compute", "anonymous_hosting", "merch_anom_vps_01"),
    ("GlobalWire Remittance", "money_transfer", "merch_anom_wire_01"),
]


class TransactionAttackGenerator:
    """
    Produces deterministic, schema-compliant Family 1 AttackEvents.

    Models adaptive transaction-pattern evasion by modifying transaction attributes
    (amount, velocity, device, location, time-of-day, and sequence anomaly)
    relative to a user's established historical baseline profile.

    Satisfies the AttackGenerator protocol in simulation.interfaces.
    """

    def __init__(
        self,
        genome: Optional[Dict[str, float]] = None,
        ground_truth: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Family 1 Attack Generator.

        Args:
            genome: Initial 6-dimension attack genome. Defaults to DEFAULT_ATTACK_GENOME
                    if ground_truth is True, else DEFAULT_LEGITIMATE_GENOME.
            ground_truth: True if generated events represent actual evasive/fraudulent attacks,
                          False for normal/legitimate transactions.
            seed: Optional random seed for reproducible scenario generation.
        """
        self._ground_truth = ground_truth
        self._rng = random.Random(seed)

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

        Validates that all six Family 1 dimensions are present, numeric, and in [0.0, 1.0].
        """
        validate_genome(genome)
        missing_keys = set(FAMILY1_GENOME_DIMENSIONS) - set(genome.keys())
        if missing_keys:
            raise ValueError(
                f"Genome missing canonical Family 1 dimensions: {sorted(missing_keys)}"
            )
        extra_keys = set(genome.keys()) - set(FAMILY1_GENOME_DIMENSIONS)
        if extra_keys:
            raise ValueError(
                f"Genome contains non-Family 1 dimensions: {sorted(extra_keys)}"
            )

        for key, val in genome.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Family 1 genome dimension '{key}' value {val} out of bounds [0.0, 1.0]"
                )

        self._genome = dict(genome)

    def _generate_recent_history(
        self, persona: Dict[str, Any], anchor_time: datetime
    ) -> List[Transaction]:
        """
        Generate 3-5 realistic historical transactions representing the user's normal behavior
        preceding the target transaction.
        """
        history_len = self._rng.randint(3, 5)
        history: List[Transaction] = []
        user_id = persona["user_id"]
        currency = persona["currency"]

        current_time = anchor_time
        for i in range(history_len):
            # Step back by typical interval
            interval_hours = persona["typical_interval_hours"] * self._rng.uniform(0.7, 1.3)
            current_time = current_time - timedelta(hours=interval_hours)

            # Legitimate amount within normal distribution
            amt = max(5.0, round(self._rng.gauss(persona["avg_amount"], persona["std_amount"]), 2))
            merch_name, category, merch_id = self._rng.choice(persona["frequent_merchants"])
            loc = self._rng.choice(persona["frequent_locations"])
            dev = self._rng.choice(persona["registered_devices"])
            channel = self._rng.choice(persona["typical_channels"])

            tx = Transaction(
                transaction_id=f"hist_{user_id}_{i}_{int(current_time.timestamp())}",
                user_id=user_id,
                timestamp=current_time,
                amount=amt,
                currency=currency,
                merchant_id=merch_id,
                merchant_category=category,
                location=loc,
                device_id=dev,
                payment_channel=channel,
            )
            history.append(tx)

        # Return in chronological order (oldest to newest)
        history.reverse()
        return history

    def generate(self, round_id: str) -> AttackEvent:
        """
        Generate a fully-populated AttackEvent containing a target Transaction and context.

        Args:
            round_id: Unique identifier for the simulation round.

        Returns:
            AttackEvent configured with AttackFamily.ADAPTIVE_EVASION and scenario data.
        """
        persona = self._rng.choice(_USER_PERSONAS)
        genome = self._genome

        amt_dev = genome["amount_deviation"]
        vel_dev = genome["velocity_deviation"]
        dev_nov = genome["device_novelty"]
        loc_dev = genome["location_deviation"]
        time_dev = genome["time_deviation"]
        seq_anom = genome["sequence_anomaly"]

        # Base reference time: anchor to a recent UTC date
        base_date = datetime(2026, 8, 28, 14, 0, 0, tzinfo=timezone.utc)
        recent_history = self._generate_recent_history(persona, base_date)
        last_tx_time = recent_history[-1].timestamp

        # 1. Velocity Deviation -> Time delta since last transaction
        if vel_dev <= 0.20:
            # Normal spacing between transactions (12 - 36 hours)
            delta_hours = persona["typical_interval_hours"] * (0.8 + 0.4 * (1.0 - vel_dev))
            target_timestamp = last_tx_time + timedelta(hours=delta_hours)
        else:
            # High velocity: burst activity soon after last transaction (1 to 45 minutes)
            burst_minutes = max(1, int(45.0 * (1.0 - vel_dev) + 1.0))
            target_timestamp = last_tx_time + timedelta(minutes=burst_minutes)

        # 2. Time-of-day Deviation -> Adjust hour to normal or off-hours
        start_hour, end_hour = persona["active_hours"]
        if time_dev > 0.40:
            # Shift into night/off-hours (e.g. 01:00 - 05:00)
            off_hour = (2 + int(time_dev * 3)) % 24
            target_timestamp = target_timestamp.replace(hour=off_hour, minute=self._rng.randint(0, 59))
        else:
            # Keep within active hours
            normal_hour = self._rng.randint(start_hour, min(end_hour, 23))
            target_timestamp = target_timestamp.replace(hour=normal_hour)

        # 3. Amount Deviation -> Calculate target transaction amount
        avg_amt = persona["avg_amount"]
        std_amt = persona["std_amount"]
        if amt_dev <= 0.20:
            # Normal range: mean +/- 1 standard deviation
            target_amount = max(5.0, round(avg_amt + (amt_dev * std_amt) + self._rng.uniform(-5.0, 5.0), 2))
        else:
            # Deviation: scaling up from 2x to 12x normal mean
            multiplier = 1.5 + (amt_dev * 10.5)
            target_amount = round(avg_amt * multiplier + self._rng.uniform(10.0, 50.0), 2)

        # 4. Device Novelty -> Familiar vs. novel device
        if dev_nov <= 0.30:
            device_id = self._rng.choice(persona["registered_devices"])
        else:
            device_id = self._rng.choice(_ANOMALOUS_DEVICES)

        # 5. Location Deviation -> Familiar vs. distant/anomalous location
        if loc_dev <= 0.30:
            location = self._rng.choice(persona["frequent_locations"])
        else:
            location = self._rng.choice(_ANOMALOUS_LOCATIONS)

        # 6. Sequence Anomaly -> Normal category vs. high-risk unusual merchant
        if seq_anom <= 0.30:
            merch_name, category, merch_id = self._rng.choice(persona["frequent_merchants"])
            payment_channel = self._rng.choice(persona["typical_channels"])
        else:
            merch_name, category, merch_id = self._rng.choice(_ANOMALOUS_MERCHANTS)
            payment_channel = "online_card_not_present"

        # Build target transaction object
        target_transaction = Transaction(
            transaction_id=f"tx_f1_{round_id}",
            user_id=persona["user_id"],
            timestamp=target_timestamp,
            amount=target_amount,
            currency=persona["currency"],
            merchant_id=merch_id,
            merchant_category=category,
            location=location,
            device_id=device_id,
            payment_channel=payment_channel,
        )

        # Assemble user baseline profile context for Blue Team feature extraction
        baseline_profile: Dict[str, Any] = {
            "user_id": persona["user_id"],
            "home_location": persona["home_location"],
            "frequent_locations": list(persona["frequent_locations"]),
            "registered_devices": list(persona["registered_devices"]),
            "typical_channels": list(persona["typical_channels"]),
            "currency": persona["currency"],
            "avg_amount": persona["avg_amount"],
            "std_amount": persona["std_amount"],
            "max_historical_amount": persona["max_historical_amount"],
            "frequent_categories": list(persona["frequent_categories"]),
            "active_hours": list(persona["active_hours"]),
            "typical_interval_hours": persona["typical_interval_hours"],
        }

        scenario_data: Dict[str, Any] = {
            "transaction": target_transaction.model_dump(),
            "baseline_profile": baseline_profile,
            "recent_history": [tx.model_dump() for tx in recent_history],
        }

        metadata: Dict[str, Any] = {
            "user_id": persona["user_id"],
            "merchant_name": merch_name,
            "target_amount": target_amount,
            "currency": persona["currency"],
            "ground_truth": self._ground_truth,
        }

        return AttackEvent(
            attack_id=f"atk-f1-{round_id}",
            round_id=round_id,
            attack_family=AttackFamily.ADAPTIVE_EVASION,
            attack_genome=dict(self._genome),
            scenario=scenario_data,
            ground_truth=self._ground_truth,
            metadata=metadata,
        )
