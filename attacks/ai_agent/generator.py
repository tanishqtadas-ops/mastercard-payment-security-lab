"""
attacks/ai_agent/generator.py — Family 2 (AI-Agent Behavior) Attack Generator.

Generates AttackEvent objects containing AIAgentPaymentEvent scenarios modeling
unauthorized or malicious AI-agent payment behavior (authorization abuse).
"""

from datetime import datetime, timezone
import random
from typing import Any, Dict, List, Optional

from schemas.common import AttackFamily
from schemas.attack import AttackEvent
from schemas.agent_event import AIAgentPaymentEvent
from schemas.transaction import Transaction
from mutation.genome_engine import validate_genome


# Canonical Family 2 genome dimensions (defined in MASTER_SPEC.md § 3)
FAMILY2_GENOME_DIMENSIONS = (
    "intent_amount_deviation",
    "intent_category_deviation",
    "permission_scope_deviation",
    "agent_identity_confidence",
    "session_provenance_anomaly",
    "purchase_velocity",
)

# Baseline high-risk attack genome (clearly malicious / unauthorized)
DEFAULT_ATTACK_GENOME: Dict[str, float] = {
    "intent_amount_deviation": 0.85,
    "intent_category_deviation": 0.80,
    "permission_scope_deviation": 0.75,
    "agent_identity_confidence": 0.30,
    "session_provenance_anomaly": 0.70,
    "purchase_velocity": 0.65,
}

# Baseline legitimate / authorized genome (low deviation / benign)
DEFAULT_LEGITIMATE_GENOME: Dict[str, float] = {
    "intent_amount_deviation": 0.05,
    "intent_category_deviation": 0.02,
    "permission_scope_deviation": 0.01,
    "agent_identity_confidence": 0.98,
    "session_provenance_anomaly": 0.03,
    "purchase_velocity": 0.05,
}

# Scenario templates for realistic domain coherence
_SCENARIO_TEMPLATES: List[Dict[str, Any]] = [
    {
        "intent": "Purchase office supplies and stationery up to $150",
        "auth_scope": "Office supplies, single purchase up to $150, approved merchants only",
        "authorized_category": "office_supplies",
        "authorized_limit": 150.0,
        "authorized_merchants": ["Staples Direct", "OfficeDepot Corp", "Quill Supply"],
        "divergent_categories": ["consumer_electronics", "crypto_assets", "luxury_goods"],
        "divergent_merchants": ["Apex Electronics", "CryptoVault Exchange", "Luxe Boutique Global"],
        "divergent_actions": [
            "Purchased high-end graphics card and requested expedited international delivery",
            "Transferred funds to external cryptocurrency voucher service",
            "Purchased designer luxury watches exceeding authorized credit envelope",
        ],
    },
    {
        "intent": "Book standard flight to Chicago for business conference under $500",
        "auth_scope": "Economy airfare, domestic US, max $500, corporate card",
        "authorized_category": "travel_airline",
        "authorized_limit": 500.0,
        "authorized_merchants": ["United Airlines", "Delta Air Lines", "American Airlines"],
        "divergent_categories": ["luxury_hospitality", "prepaid_gift_cards", "gaming_digital"],
        "divergent_merchants": ["UltraLuxe VIP Resorts", "QuickCard Digital Hub", "GameZone Enterprise"],
        "divergent_actions": [
            "Booked 5-star presidential penthouse suite and chartered limousine service",
            "Purchased $3,500 in non-refundable prepaid virtual debit cards",
            "Executed microtransactions across international online gaming portal",
        ],
    },
    {
        "intent": "Renew monthly cloud database hosting subscription up to $80",
        "auth_scope": "Software / SaaS hosting, recurring single month, max $80",
        "authorized_category": "cloud_infrastructure",
        "authorized_limit": 80.0,
        "authorized_merchants": ["CloudHost Provider", "AWS Cloud Services", "Azure Host"],
        "divergent_categories": ["crypto_mining", "unauthorized_wire", "hardware_servers"],
        "divergent_merchants": ["HashPower Mining Pool", "SwiftWire Settlement", "ServerDirect Pro"],
        "divergent_actions": [
            "Provisioned high-tier GPU compute cluster for cryptocurrency mining",
            "Initiated automated wire transfer to unverified third-party account",
            "Ordered dedicated physical rack server hardware with corporate billing",
        ],
    },
]


class AIAgentAttackGenerator:
    """
    Produces deterministic, schema-compliant Family 2 AttackEvents.

    Models authorization abuse by AI agents where the agent deviates from the
    user's explicit intent and authorized scope across the six canonical Family 2
    genome dimensions.

    Satisfies the AttackGenerator protocol in simulation.interfaces.
    """

    def __init__(
        self,
        genome: Optional[Dict[str, float]] = None,
        ground_truth: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Family 2 Attack Generator.

        Args:
            genome: Initial 6-dimension attack genome. Defaults to DEFAULT_ATTACK_GENOME
                    if ground_truth is True, else DEFAULT_LEGITIMATE_GENOME.
            ground_truth: True if generated events represent actual malicious/unauthorized attacks.
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

        Validates that all six Family 2 dimensions are present, numeric, and in [0.0, 1.0].
        """
        validate_genome(genome)
        missing_keys = set(FAMILY2_GENOME_DIMENSIONS) - set(genome.keys())
        if missing_keys:
            raise ValueError(
                f"Genome missing canonical Family 2 dimensions: {sorted(missing_keys)}"
            )
        extra_keys = set(genome.keys()) - set(FAMILY2_GENOME_DIMENSIONS)
        if extra_keys:
            raise ValueError(
                f"Genome contains non-Family 2 dimensions: {sorted(extra_keys)}"
            )

        for key, val in genome.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Family 2 genome dimension '{key}' value {val} out of bounds [0.0, 1.0]"
                )

        self._genome = dict(genome)

    def generate(self, round_id: str) -> AttackEvent:
        """
        Generate a fully-populated AttackEvent containing an AIAgentPaymentEvent.

        Args:
            round_id: Unique identifier for the simulation round.

        Returns:
            AttackEvent configured with AttackFamily.AGENT_BEHAVIOR and scenario data.
        """
        template = self._rng.choice(_SCENARIO_TEMPLATES)
        genome = self._genome

        amt_dev = genome["intent_amount_deviation"]
        cat_dev = genome["intent_category_deviation"]
        scope_dev = genome["permission_scope_deviation"]
        id_conf = genome["agent_identity_confidence"]
        sess_anom = genome["session_provenance_anomaly"]
        vel_dev = genome["purchase_velocity"]

        auth_limit = template["authorized_limit"]

        # Calculate actual transaction amount based on amount deviation
        if amt_dev <= 0.15:
            # Within or close to authorized limit
            actual_amount = round(auth_limit * (0.80 + 0.15 * amt_dev), 2)
        else:
            # Over authorized limit proportional to deviation
            multiplier = 1.0 + (amt_dev * 6.0)
            actual_amount = round(auth_limit * multiplier, 2)

        # Determine category and merchant based on category deviation
        if cat_dev <= 0.25:
            category = template["authorized_category"]
            merchant = self._rng.choice(template["authorized_merchants"])
        else:
            category = self._rng.choice(template["divergent_categories"])
            merchant = self._rng.choice(template["divergent_merchants"])

        # Compute velocity burst parameters
        if vel_dev > 0.30:
            burst_count = max(2, int(2 + vel_dev * 10))
            velocity_note = f" (Burst of {burst_count} rapid transactions in session)"
            payment_channel = "ai_agent_batch_burst_api" if vel_dev > 0.50 else "ai_agent_api"
        else:
            burst_count = 1
            velocity_note = ""
            payment_channel = "ai_agent_api"

        # Determine permission scope deviation modifier
        if scope_dev > 0.60:
            scope_violation = (
                "Agent unilaterally elevated execution privileges to 'unrestricted_admin', "
                "bypassed user confirmation gates, and authorized recurring secondary debit"
            )
        elif scope_dev > 0.20:
            scope_violation = (
                "Agent exceeded single-transaction operational mandate and modified authorization parameters"
            )
        else:
            scope_violation = "Agent adhered strictly to authorized operational bounds"

        # Determine actual action text based on all deviations
        if amt_dev <= 0.15 and cat_dev <= 0.25 and scope_dev <= 0.20 and vel_dev <= 0.30:
            actual_action = (
                f"Purchased {category.replace('_', ' ')} from {merchant} for ${actual_amount:.2f}. "
                f"{scope_violation}."
            )
        else:
            divergent_desc = self._rng.choice(template["divergent_actions"])
            actual_action = (
                f"Executed unauthorized transaction: {divergent_desc} at {merchant} "
                f"(Amount: ${actual_amount:.2f}, Category: {category}){velocity_note}. "
                f"Scope violation: {scope_violation}."
            )

        # Synthesize agent identity and session context
        agent_suffix = round_id.replace("-", "_")[:8]
        if id_conf >= 0.70:
            agent_id = f"procure-bot-v2-{agent_suffix}"
        else:
            agent_id = f"unverified-agent-{agent_suffix}-spoof"

        if sess_anom <= 0.30:
            session_ctx = f"session_auth_mfa_valid_{agent_suffix}"
        else:
            session_ctx = f"session_hijack_anom_remote_{agent_suffix}_ip_flagged"

        if burst_count > 1:
            session_ctx = f"{session_ctx}_burst_rate_n{burst_count}"

        tx_id = f"tx_agent_{round_id}_{agent_suffix}"
        transaction = Transaction(
            transaction_id=tx_id,
            user_id=f"user_{agent_suffix[:6]}",
            timestamp=datetime.now(timezone.utc),
            amount=actual_amount,
            currency="USD",
            merchant_id=f"merch_{merchant.lower().replace(' ', '_')[:10]}",
            merchant_category=category,
            location="US-Online",
            device_id=f"device_agent_{agent_suffix}",
            payment_channel=payment_channel,
        )

        agent_event = AIAgentPaymentEvent(
            event_id=f"agent-evt-{round_id}",
            user_intent=template["intent"],
            authorized_scope=template["auth_scope"],
            agent_identity=agent_id,
            session_context=session_ctx,
            actual_action=actual_action,
            transaction=transaction,
        )

        attack_id = f"family2-attack-{round_id}"
        scenario_data = agent_event.model_dump(mode="json")

        return AttackEvent(
            attack_id=attack_id,
            round_id=round_id,
            attack_family=AttackFamily.AGENT_BEHAVIOR,
            attack_genome=dict(genome),
            scenario=scenario_data,
            ground_truth=self._ground_truth,
            metadata={
                "generator": "AIAgentAttackGenerator",
                "attack_family": AttackFamily.AGENT_BEHAVIOR.value,
                "authorized_limit": auth_limit,
                "actual_amount": actual_amount,
                "intent_category": template["authorized_category"],
                "executed_category": category,
                "velocity_level": vel_dev,
                "burst_count": burst_count,
                "scope_deviation_level": scope_dev,
            },
        )
