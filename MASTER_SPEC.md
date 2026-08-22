# Mastercard Payment Security Lab
## Master Specification

---

# 1. PROJECT OBJECTIVE

Build a controlled adversarial payment-security laboratory for the Mastercard Innovation Challenge 2026.

The system follows:

IDENTIFY → GENERATE → DEFEND

The prototype simulates emerging GenAI-powered payment-fraud scenarios and evaluates an ML-based defense through a controlled Red-Team / Blue-Team loop.

---

# 2. CORE LOOP

Normal Synthetic Environment
        ↓
Select Attack Family
        ↓
Attack Genome
        ↓
Python Mutation Engine
        ↓
Synthetic Attack Variant
        ↓
Blue Team
        ↓
Risk Score + Explanation
        ↓
Detected / Missed
        ↓
Red Team Adapts
+
Blue Team Stores Failure
        ↓
Every 2 Rounds
Blue Team Updates
        ↓
Next Round

---

# 3. ATTACK FAMILIES

## Family 1 — Adaptive Transaction-Pattern Evasion

Unit of analysis:
Transaction / short transaction sequence

Core question:
Does this payment behavior look suspicious compared with this user's normal behavior?

Genome:

- amount_deviation
- velocity_deviation
- device_novelty
- location_deviation
- time_deviation
- sequence_anomaly


## Family 2 — Unauthorized / Malicious AI-Agent Payment Behavior

Unit of analysis:
Authorization / payment event

Core question:
Did the AI agent actually do what the user authorized it to do?

Genome:

- intent_amount_deviation
- intent_category_deviation
- permission_scope_deviation
- agent_identity_confidence
- session_provenance_anomaly
- purchase_velocity


## Family 3 — Synthetic Identity + AI-Generated Identity Artifacts

Unit of analysis:
Identity / account lifecycle

Core question:
Does this account remain consistent with a plausible legitimate identity over time?

Genome:

- cross_field_consistency
- profile_plausibility_score
- contact_consistency
- device_history_score
- lifecycle_behavior_coherence
- time_to_risky_activity

---

# 4. BLUE TEAM

Primary model:

XGBoost or LightGBM

Explanation:

SHAP / feature importance

Outputs:

- prediction
- risk_score
- explanation
- important features

Each attack family should have its own lightweight detector/model where practical.

---

# 5. BLUE-TEAM LEARNING

When Blue misses an attack:

- store the false negative
- store its genome
- store feature values
- store prediction
- store risk score
- store explanation/SHAP values

Every 2 rounds:

Retrain/update the corresponding Blue-Team model using:

- original training data
- accumulated missed attacks
- fresh legitimate samples where appropriate

Always preserve a separate held-out legitimate evaluation set.

---

# 6. RED-TEAM MUTATION

Python controls mutation.

LLM does not control the attack loop.

If an attack is detected:

- reduce strong detection signals gradually
- explore lower-contribution dimensions
- keep mutation steps bounded

If an attack is missed:

- preserve the successful evasion pattern
- make a small local mutation
- explore nearby variants

No reinforcement-learning system is required for the core prototype.

---

# 7. DATA GENERATION

Numeric / structured data:

- Python
- NumPy
- Faker
- deterministic/probabilistic simulation

LLM use:

- qualitative synthetic content where useful

Do not use an LLM to generate thousands of numeric transaction rows.

---

# 8. DEMO PRINCIPLE

The primary demo should be precomputed/reproducible.

Live LLM generation is optional.

The main demonstration must not depend on live API availability.

---

# 9. DEVELOPMENT PRINCIPLE

Build one verified phase at a time.

Workflow:

PLAN
↓
IMPLEMENT
↓
RUN
↓
TEST
↓
DEBUG
↓
APPROVE
↓
COMMIT
↓
PUSH
↓
MERGE
↓
NEXT PHASE

---

# 10. ARCHITECTURE RULE

Do not introduce:

- large multi-agent frameworks
- LLM-to-LLM loops
- unnecessary neural networks
- unnecessary mathematical complexity
- production-scale infrastructure
- unnecessary dependencies

Prefer:

LLM = selective generation/reasoning

Python = simulation/control

ML = detection

SHAP = explanation

---

# 11. CURRENT STATUS

Phase 0 — Project setup: COMPLETE
Phase 1 — Common data contracts (schemas): COMPLETE
Phase 2 — Generic attack genome engine: COMPLETE
Phase 3 — Round Controller + shared interfaces: COMPLETE
Phase 4 — Generic Red/Blue mock pipeline: COMPLETE
Phase 5 — Integration contracts + team handoff: COMPLETE

See docs/INTEGRATION_CONTRACTS.md and docs/TEAM_HANDOFF.md for the
stable interface boundaries and team ownership assignments.