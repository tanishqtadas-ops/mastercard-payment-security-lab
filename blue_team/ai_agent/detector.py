"""
blue_team/ai_agent/detector.py — Family 2 (AI-Agent Behavior) Blue-Team Detector.

Detects unauthorized or malicious AI-agent payment behavior by evaluating whether
an observed agent action stays within its AgentMandate (authorized amount limit,
category authorization, approved merchant list, permission scope, identity confidence,
session provenance, and velocity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Union

from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from schemas.agent_event import AIAgentPaymentEvent
from schemas.transaction import Transaction


# Default weighted contributions for Family 2 dimensions (sum to 1.0)
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "intent_amount_deviation": 0.25,
    "intent_category_deviation": 0.20,
    "permission_scope_deviation": 0.25,
    "agent_identity_confidence": 0.10,  # Applied to (1.0 - confidence) as identity anomaly
    "session_provenance_anomaly": 0.10,
    "purchase_velocity": 0.10,
}

# Detection threshold: risk_score >= threshold triggers fraud alert
DEFAULT_DETECTION_THRESHOLD: float = 0.50

MODEL_VERSION: str = "heuristic-family2-v1"

# Standard known category-to-approved-merchants mapping for automatic mandate resolution
KNOWN_CATEGORY_MERCHANTS: Dict[str, List[str]] = {
    "office_supplies": ["Staples Direct", "OfficeDepot Corp", "Quill Supply", "staples", "officedepot", "quill"],
    "travel_airline": ["United Airlines", "Delta Air Lines", "American Airlines", "united", "delta", "american"],
    "cloud_infrastructure": ["CloudHost Provider", "AWS Cloud Services", "Azure Host", "cloudhost", "aws", "azure"],
}


@dataclass
class AgentMandate:
    """
    Defines the authorized operational mandate / envelope for an AI agent.

    Specifies the maximum permissible amount, allowed merchant categories,
    approved merchant list, currency, and authorization parameters.
    """

    max_amount: float
    allowed_categories: List[str] = field(default_factory=list)
    allowed_merchants: List[str] = field(default_factory=list)
    currency: str = "USD"
    require_verified_identity: bool = True
    allow_burst: bool = False

    @classmethod
    def from_event(cls, event: AttackEvent) -> AgentMandate:
        """
        Infer and construct an AgentMandate from AttackEvent metadata or scenario.
        """
        meta = event.metadata or {}
        scenario = event.scenario or {}

        if isinstance(scenario, AIAgentPaymentEvent):
            user_intent = scenario.user_intent
            auth_scope = scenario.authorized_scope
        elif isinstance(scenario, dict):
            user_intent = scenario.get("user_intent", "")
            auth_scope = scenario.get("authorized_scope", "")
        else:
            user_intent = ""
            auth_scope = ""

        # 1. Resolve max_amount
        max_amount: Optional[float] = None
        if "authorized_limit" in meta:
            try:
                max_amount = float(meta["authorized_limit"])
            except (ValueError, TypeError):
                pass
        elif "max_amount" in meta:
            try:
                max_amount = float(meta["max_amount"])
            except (ValueError, TypeError):
                pass

        if max_amount is None:
            # Parse from authorized_scope or user_intent (e.g. "$150", "max $200", "under $500", "$1,500")
            combined_text = f"{auth_scope} {user_intent}"
            matches = re.findall(r"\$([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", combined_text)
            if matches:
                try:
                    max_amount = float(matches[0].replace(",", ""))
                except ValueError:
                    max_amount = 150.0
            else:
                max_amount = 150.0

        # 2. Resolve allowed_categories
        allowed_categories: List[str] = []
        if "intent_category" in meta and meta["intent_category"]:
            allowed_categories.append(str(meta["intent_category"]))
        elif "authorized_category" in meta and meta["authorized_category"]:
            allowed_categories.append(str(meta["authorized_category"]))
        elif "allowed_categories" in meta and isinstance(meta["allowed_categories"], list):
            allowed_categories.extend(str(c) for c in meta["allowed_categories"])

        if not allowed_categories:
            combined_lower = f"{auth_scope} {user_intent}".lower()
            if "office" in combined_lower or "stationery" in combined_lower:
                allowed_categories.append("office_supplies")
            elif "flight" in combined_lower or "airfare" in combined_lower or "travel" in combined_lower:
                allowed_categories.append("travel_airline")
            elif (
                "cloud" in combined_lower
                or "hosting" in combined_lower
                or "database" in combined_lower
                or "saas" in combined_lower
            ):
                allowed_categories.append("cloud_infrastructure")

        # 3. Resolve allowed_merchants
        allowed_merchants: List[str] = []
        if "authorized_merchants" in meta and isinstance(meta["authorized_merchants"], list):
            allowed_merchants.extend(str(m) for m in meta["authorized_merchants"])
        elif "allowed_merchants" in meta and isinstance(meta["allowed_merchants"], list):
            allowed_merchants.extend(str(m) for m in meta["allowed_merchants"])

        if not allowed_merchants:
            for cat in allowed_categories:
                if cat in KNOWN_CATEGORY_MERCHANTS:
                    allowed_merchants.extend(KNOWN_CATEGORY_MERCHANTS[cat])

        return cls(
            max_amount=max_amount,
            allowed_categories=allowed_categories,
            allowed_merchants=allowed_merchants,
        )


def _extract_transaction_data(scenario: Any) -> Dict[str, Any]:
    """Helper to extract transaction and action fields from scenario."""
    if isinstance(scenario, AIAgentPaymentEvent):
        tx = scenario.transaction
        return {
            "actual_action": scenario.actual_action,
            "agent_identity": scenario.agent_identity,
            "session_context": scenario.session_context,
            "amount": tx.amount if tx else None,
            "merchant_category": tx.merchant_category if tx else None,
            "merchant_id": tx.merchant_id if tx else None,
            "payment_channel": tx.payment_channel if tx else None,
        }
    elif isinstance(scenario, dict):
        tx_data = scenario.get("transaction")
        if isinstance(tx_data, dict):
            amt = tx_data.get("amount")
            cat = tx_data.get("merchant_category")
            m_id = tx_data.get("merchant_id")
            ch = tx_data.get("payment_channel")
        elif isinstance(tx_data, Transaction):
            amt = tx_data.amount
            cat = tx_data.merchant_category
            m_id = tx_data.merchant_id
            ch = tx_data.payment_channel
        else:
            amt = None
            cat = None
            m_id = None
            ch = None

        return {
            "actual_action": scenario.get("actual_action", ""),
            "agent_identity": scenario.get("agent_identity", ""),
            "session_context": scenario.get("session_context", ""),
            "amount": amt,
            "merchant_category": cat,
            "merchant_id": m_id,
            "payment_channel": ch,
        }
    return {
        "actual_action": "",
        "agent_identity": "",
        "session_context": "",
        "amount": None,
        "merchant_category": None,
        "merchant_id": None,
        "payment_channel": None,
    }


def extract_mandate_features(
    event: AttackEvent,
    mandate: Optional[AgentMandate] = None,
) -> Dict[str, float]:
    """
    Extract normalized Family 2 behavioral deviation features [0.0, 1.0]
    by evaluating the observed agent action against its AgentMandate.

    When scenario/payment data is present, features are derived directly from the
    observable event and its authorization context without being overridden by attack_genome.
    The attack_genome is only used as a fallback when scenario data is absent.
    """
    genome = event.attack_genome or {}
    scenario = event.scenario or {}
    meta = event.metadata or {}

    has_scenario_data = bool(scenario)

    # Fallback to attack_genome ONLY when scenario data is completely absent
    if not has_scenario_data:
        amt_dev = genome.get("intent_amount_deviation", 0.0)
        cat_dev = genome.get("intent_category_deviation", 0.0)
        scope_dev = genome.get("permission_scope_deviation", 0.0)
        id_conf = genome.get("agent_identity_confidence", 0.98)
        sess_anom = genome.get("session_provenance_anomaly", 0.03)
        vel_dev = genome.get("purchase_velocity", 0.05)

        return {
            "intent_amount_deviation": min(max(float(amt_dev), 0.0), 1.0),
            "intent_category_deviation": min(max(float(cat_dev), 0.0), 1.0),
            "permission_scope_deviation": min(max(float(scope_dev), 0.0), 1.0),
            "agent_identity_confidence": min(max(float(id_conf), 0.0), 1.0),
            "session_provenance_anomaly": min(max(float(sess_anom), 0.0), 1.0),
            "purchase_velocity": min(max(float(vel_dev), 0.0), 1.0),
        }

    # Evaluate observable scenario against AgentMandate
    effective_mandate = mandate or AgentMandate.from_event(event)
    tx_data = _extract_transaction_data(scenario)

    amount = tx_data["amount"] if tx_data["amount"] is not None else meta.get("actual_amount")
    category = tx_data["merchant_category"] or meta.get("executed_category")
    merchant = tx_data["merchant_id"] or ""
    action_text = tx_data["actual_action"] or ""
    agent_id = tx_data["agent_identity"] or ""
    session_ctx = tx_data["session_context"] or ""
    channel = tx_data["payment_channel"] or ""

    # 1. Intent Amount Deviation (Amount relative to permitted maximum)
    if amount is not None and effective_mandate.max_amount > 0:
        if amount <= effective_mandate.max_amount:
            # Within authorized limit
            amt_dev = max(0.0, min(0.15, (amount / effective_mandate.max_amount) * 0.10))
        else:
            # Exceeding authorized limit
            excess = amount - effective_mandate.max_amount
            amt_dev = min(1.0, 0.35 + 0.65 * min(excess / (effective_mandate.max_amount * 2.0), 1.0))
    else:
        amt_dev = 0.0

    # 2. Intent Category Deviation (Category authorization)
    if category and effective_mandate.allowed_categories:
        cat_clean = category.lower().replace("_", " ")
        cat_match = any(
            c.lower().replace("_", " ") in cat_clean or cat_clean in c.lower().replace("_", " ")
            for c in effective_mandate.allowed_categories
        )
        if cat_match:
            cat_dev = 0.02
        else:
            cat_dev = 0.85
    else:
        cat_dev = 0.0

    # 3. Permission Scope Deviation (Merchant-list authorization & operational bounds)
    scope_signals: List[float] = []

    # Merchant authorization check
    if effective_mandate.allowed_merchants and (merchant or action_text):
        check_target = f"{merchant} {action_text}".lower().replace("_", " ")
        merch_match = any(
            m.lower().replace("_", " ") in check_target
            for m in effective_mandate.allowed_merchants
        )
        if not merch_match:
            scope_signals.append(0.85)

    # Operational scope violation check
    action_lower = action_text.lower()
    if any(
        term in action_lower
        for term in [
            "unauthorized",
            "elevated",
            "privilege",
            "violation",
            "bypassed",
            "unrestricted",
            "scope violation",
        ]
    ):
        scope_signals.append(0.80)
    elif "adhered strictly" in action_lower:
        scope_signals.append(0.01)

    scope_dev = max(scope_signals) if scope_signals else 0.01

    # 4. Agent Identity Confidence
    if "unverified" in agent_id.lower() or "spoof" in agent_id.lower():
        id_conf = 0.20
    elif agent_id:
        id_conf = 0.98
    else:
        id_conf = 0.98

    # 5. Session Provenance Anomaly
    if any(term in session_ctx.lower() for term in ["hijack", "flagged", "anom"]):
        sess_anom = 0.75
    elif session_ctx:
        sess_anom = 0.03
    else:
        sess_anom = 0.03

    # 6. Purchase Velocity
    if "burst" in session_ctx.lower() or "burst" in action_lower or "burst" in channel.lower():
        vel_dev = 0.75
    else:
        vel_dev = 0.05

    return {
        "intent_amount_deviation": min(max(float(amt_dev), 0.0), 1.0),
        "intent_category_deviation": min(max(float(cat_dev), 0.0), 1.0),
        "permission_scope_deviation": min(max(float(scope_dev), 0.0), 1.0),
        "agent_identity_confidence": min(max(float(id_conf), 0.0), 1.0),
        "session_provenance_anomaly": min(max(float(sess_anom), 0.0), 1.0),
        "purchase_velocity": min(max(float(vel_dev), 0.0), 1.0),
    }


class AIAgentBlueDetector:
    """
    Evaluates Family 2 AttackEvents and computes fraud predictions.

    Determines whether an observed agent action stays within its AgentMandate
    (amount limits, category authorizations, approved merchant lists, permission
    scopes, identity confidence, session provenance, and velocity).

    Satisfies the BlueTeamDetector protocol in simulation.interfaces.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        weights: Optional[Dict[str, float]] = None,
        model_version: str = MODEL_VERSION,
        mandate: Optional[AgentMandate] = None,
    ) -> None:
        """
        Initialize the detector.

        Args:
            threshold: Fraud decision threshold in [0.0, 1.0].
            weights: Optional dictionary of feature weights summing to 1.0.
            model_version: Identifier string for model traceability.
            mandate: Optional default AgentMandate to enforce on evaluated events.
        """
        self.threshold = threshold
        self.weights = dict(weights or DEFAULT_FEATURE_WEIGHTS)
        self.model_version = model_version
        self.mandate = mandate

    def detect(
        self,
        event: AttackEvent,
        mandate: Optional[AgentMandate] = None,
    ) -> PredictionResult:
        """
        Run detection on an AttackEvent and return a PredictionResult.

        Args:
            event: AttackEvent containing Family 2 scenario and/or attack_genome.
            mandate: Optional AgentMandate override for evaluating this specific event.

        Returns:
            PredictionResult containing boolean prediction, normalized risk score,
            human-readable explanation, and feature contributions.
        """
        effective_mandate = mandate or self.mandate
        features = extract_mandate_features(event, mandate=effective_mandate)

        amt_dev = features["intent_amount_deviation"]
        cat_dev = features["intent_category_deviation"]
        scope_dev = features["permission_scope_deviation"]
        id_conf = features["agent_identity_confidence"]
        id_anom = max(0.0, min(1.0 - id_conf, 1.0))
        sess_anom = features["session_provenance_anomaly"]
        vel_dev = features["purchase_velocity"]

        w_amt = self.weights.get("intent_amount_deviation", 0.25)
        w_cat = self.weights.get("intent_category_deviation", 0.20)
        w_scope = self.weights.get("permission_scope_deviation", 0.25)
        w_id = self.weights.get("agent_identity_confidence", 0.10)
        w_sess = self.weights.get("session_provenance_anomaly", 0.10)
        w_vel = self.weights.get("purchase_velocity", 0.10)

        contrib_amt = w_amt * amt_dev
        contrib_cat = w_cat * cat_dev
        contrib_scope = w_scope * scope_dev
        contrib_id = w_id * id_anom
        contrib_sess = w_sess * sess_anom
        contrib_vel = w_vel * vel_dev

        feature_contributions: Dict[str, float] = {
            "intent_amount_deviation": round(contrib_amt, 4),
            "intent_category_deviation": round(contrib_cat, 4),
            "permission_scope_deviation": round(contrib_scope, 4),
            "agent_identity_confidence": round(contrib_id, 4),
            "session_provenance_anomaly": round(contrib_sess, 4),
            "purchase_velocity": round(contrib_vel, 4),
        }

        total_risk = sum(feature_contributions.values())
        risk_score = min(max(total_risk, 0.0), 1.0)
        risk_score = round(risk_score, 4)

        is_fraud = risk_score >= self.threshold

        # Generate human-readable explanation
        if is_fraud:
            top_drivers = sorted(
                feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            key_signals = [
                f"{k} (contrib: {v:.3f})" for k, v in top_drivers if v > 0.05
            ]
            signals_str = ", ".join(key_signals) if key_signals else "cumulative deviation"
            explanation = (
                f"Flagged unauthorized AI-agent behavior (risk {risk_score:.2f} >= threshold {self.threshold:.2f}). "
                f"Primary drivers: {signals_str}."
            )
        else:
            explanation = (
                f"Agent transaction approved within authorized envelope "
                f"(risk {risk_score:.2f} < threshold {self.threshold:.2f})."
            )

        return PredictionResult(
            prediction_id=f"pred-f2-{event.round_id}",
            prediction=is_fraud,
            risk_score=risk_score,
            model_version=self.model_version,
            explanation=explanation,
            feature_contributions=feature_contributions,
        )
