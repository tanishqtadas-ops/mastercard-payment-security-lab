# Mastercard Payment Security Lab

A controlled adversarial payment-security laboratory for the Mastercard Innovation Challenge 2026.
Simulates GenAI-era fraud scenarios across three attack families and evaluates adaptive ML-based
defenses through a Red-Team / Blue-Team adversarial loop.

---

## 1. Project Overview

### The Problem

Modern payment fraud is not static. Attackers adapt — they observe detection signals, mutate their
tactics, and probe for blind spots. A Blue Team model trained once on historical data falls behind
almost immediately.

The three emerging threat vectors this lab focuses on are:

1. **Adversarial transaction-pattern evasion** — subtle genome-driven manipulation of behavioral
   signals to stay below fraud-detection thresholds.
2. **Malicious AI-agent payment behavior** — AI agents acting outside their authorized mandate,
   exploiting ambiguity in scope, amount, and session provenance.
3. **Synthetic identity fraud** — fabricated identities with AI-generated supporting artifacts that
   pass point-in-time plausibility checks but fail over an account lifecycle.

### The Core Idea

This lab builds a controlled simulation where:

- A **Red Team** generates adversarial attack variants, adapts after every round, and probes for
  evasion gaps.
- A **Blue Team** detects attacks, records failures, and periodically updates its detection
  parameters or refits its model.
- The loop runs deterministically and fully offline — no runtime LLM or API calls are required.

---

## 2. What Makes This Different

Most fraud-detection demos show a static classifier on a fixed dataset.
This lab demonstrates an **adaptive adversarial loop**:

```
Red Attack Genome
    ↓
Synthetic Attack Event generated
    ↓
Blue Detector scores risk, explains prediction
    ↓
Feedback: detected / missed
    ↓  (if missed)
FailureMemory records false-negative genome + features
    ↓
MutationStrategy adapts genome for next round
    ↓
Every 2 rounds → RetrainingController triggers Blue update
    ↓
UnifiedEvaluator benchmarks updated detector
    ↓
Dashboard presents arms-race progression
    ↓
Next round (Red now probes updated Blue)
```

The Red Team adapts in **every round**. The Blue Team adapts every **two rounds**.
This asymmetry produces a measurable arms-race dynamic visible in the demo output.

---

## 3. Three Attack Families

| Family | Unit of Analysis | Core Question | Defense Approach |
|--------|-----------------|---------------|-----------------|
| **1 — Adaptive Transaction-Pattern Evasion** | Single transaction / short sequence | Does this payment behavior look suspicious compared to the user's normal baseline? | Heuristic weighted-deviation scorer with adaptive parameter updates |
| **2 — AI-Agent Payment Behavior** | Authorization / payment event | Did the AI agent actually do what the user authorized it to do? | Mandate-envelope heuristic that enforces authorized scope, amount limits, category, and session provenance |
| **3 — Synthetic Identity & AI-Generated Artifacts** | Identity / account lifecycle | Does this account remain consistent with a plausible legitimate identity over time? | Supervised XGBoost classifier with SHAP explainability and full model refitting |

---

## 4. Architecture

```
┌─────────────────────────────────────────────────┐
│              Synthetic Environment               │
│  (structured data — NumPy, Faker, Pydantic)     │
└───────────────────┬─────────────────────────────┘
                    │
            Attack Genome (Dict[str, float])
                    │
        ┌───────────▼──────────────┐
        │     AttackGenerator      │  ← family-specific
        │  (generate → AttackEvent)│
        └───────────┬──────────────┘
                    │  AttackEvent (shared schema)
        ┌───────────▼──────────────┐
        │    BlueTeamDetector      │  ← family-specific
        │ (detect → PredictionResult) │
        └───────────┬──────────────┘
                    │  risk_score + explanation
        ┌───────────▼──────────────┐
        │    FeedbackEvaluator     │  ← family-specific
        │ (evaluate → BlueTeamFeedback) │
        └───────────┬──────────────┘
                    │  detected / missed / FP / FN
        ┌───────────▼──────────────┐
        │      RoundResult         │  ← shared schema
        │  (aggregates all above)  │
        └──────┬──────────┬────────┘
               │          │
  ┌────────────▼──┐  ┌────▼──────────────────┐
  │ MutationStrategy│  │    FailureMemory      │
  │ (genome → next) │  │ (records missed attacks│
  └────────────────┘  │  + genomes + features) │
                       └────────┬──────────────┘
                                │ every 2 rounds
                    ┌───────────▼──────────────┐
                    │  RetrainingController    │
                    │  on_round_completed()    │
                    │  → ModelUpdateRecord     │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │    UnifiedEvaluator      │
                    │ per-family + consolidated│
                    │ + clean baseline eval    │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   Dashboard / Arms Race  │
                    │ (PaymentSecurityDashboard│
                    │  + ArmsRacePresenter)    │
                    └──────────────────────────┘
```

The shared `RoundController` orchestrates each round generically — it is
**completely family-agnostic**. Family-specific logic lives exclusively in the
family modules under `attacks/` and `blue_team/`.

All schemas (`AttackEvent`, `PredictionResult`, `BlueTeamFeedback`, `RoundResult`) are
frozen Pydantic v2 models.

---

## 5. Family 1 — Adaptive Transaction-Pattern Evasion

**Unit of analysis:** A single transaction or short transaction sequence from a user.

**Core question:** Does this payment behavior look suspicious relative to the user's normal
behavioral baseline?

**Attack genome** (six dimensions, each in `[0.0, 1.0]`):

| Dimension | What it controls |
|-----------|-----------------|
| `amount_deviation` | How much the transaction amount deviates from the user's typical amount |
| `velocity_deviation` | Unusual transaction frequency / rate |
| `device_novelty` | Whether the device is new or unusual for this user |
| `location_deviation` | Geographic distance from typical locations |
| `time_deviation` | Transaction at an unusual time of day / week |
| `sequence_anomaly` | Unusual sequence pattern in recent transaction history |

**Blue Team detector:** A weighted-deviation heuristic scorer. Each genome dimension is
mapped to an observable transaction feature. A weighted sum of deviations produces a
calibrated `risk_score` in `[0.0, 1.0]`. If `risk_score >= threshold` (default: 0.50),
the transaction is flagged as fraud.

**Red Team mutation:** If detected, high-contributing dimensions are gradually reduced. If
missed, the successful evasion pattern is preserved with a small local mutation.

**Retraining (heuristic parameter adaptation):** Every 2 rounds, the `Family1TransactionTrainer`
adapts the detector's feature weights toward the missed evasion dimensions (increasing
sensitivity where attacks evaded) and recalibrates the detection threshold. This is
**heuristic parameter adaptation, not supervised ML model fitting**.

---

## 6. Family 2 — AI-Agent Payment Behavior

**Unit of analysis:** An authorization / payment event produced by an AI agent acting on
a user's behalf.

**Core question:** Did the AI agent actually do what the user authorized it to do?

**Attack genome** (six dimensions):

| Dimension | What it controls |
|-----------|-----------------|
| `intent_amount_deviation` | How much the actual transaction amount exceeds the user's intended amount |
| `intent_category_deviation` | Whether the category/merchant deviates from the user's intent |
| `permission_scope_deviation` | How far the agent's actual action exceeds its authorized scope |
| `agent_identity_confidence` | How well the agent authenticates itself vs. authorized agent |
| `session_provenance_anomaly` | Anomalies in the session context / origin |
| `purchase_velocity` | Unusual transaction velocity within the session |

**Blue Team detector:** A mandate-envelope heuristic. The detector infers an `AgentMandate`
(authorized amount limit, category, merchant list, scope) from the event and checks whether
the agent's actual action stays within its authorized envelope. Violations across
multiple dimensions produce an aggregated `risk_score`.

**Retraining (heuristic parameter adaptation):** Every 2 rounds, the `Family2AIAgentTrainer`
adapts mandate weights toward missed evasion dimensions — shifting sensitivity to the
specific scope or amount dimensions the Red Team exploited. This is **heuristic parameter
adaptation, not supervised ML model fitting**.

---

## 7. Family 3 — Synthetic Identity & AI-Generated Artifacts

**Unit of analysis:** An identity / account lifecycle record.

**Core question:** Does this account remain consistent with a plausible legitimate identity
over time, or do inconsistencies reveal a synthetic construction?

**Attack genome** (six dimensions):

| Dimension | What it controls |
|-----------|-----------------|
| `cross_field_consistency` | Internal consistency across name, DOB, SSN-proxy, contact fields |
| `profile_plausibility_score` | Statistical plausibility of the overall profile |
| `contact_consistency` | Consistency of contact information over time |
| `device_history_score` | Device fingerprint history plausibility |
| `lifecycle_behavior_coherence` | Account lifecycle coherence (account open → first transaction timing) |
| `time_to_risky_activity` | How quickly the account moves to high-risk activity |

**Blue Team detector:** A supervised `XGBoost` classifier (`family3-xgb-v1`).
Trained on legitimate baseline identities (`data/legitimate/baseline_identities.json`, 500
records) with SHAP-compatible feature importance for explainability.

**Retraining (genuine supervised ML refitting):** The `Family3SyntheticIdentityTrainer`
assembles a labelled dataset from the baseline identities (label=0) and missed synthetic
identities (label=1) and refits the XGBoost classifier from scratch.

**Held-out evaluation:** `data/held_out/heldout_identities.json` (500 records, completely
partitioned from the training baseline) is used exclusively for false-positive rate
validation. It never enters the training pathway.

---

## 8. Adaptive Learning Loop

### False-Negative Memory

When the Blue Team misses an attack (`FeedbackEvaluator` returns `false_negative=True`),
`FailureMemory.record_round()` captures:

- The attack event and its genome
- The risk score that failed to trigger detection
- The feature contributions that masked the attack
- The round ID and family

### Retraining Dataset Assembly

`RetrainingDatasetAssembler` builds the training dataset for each periodic update:

- **Baseline legitimate data** — family-specific legitimate samples (pre-seeded)
- **Missed attacks** — false negatives from `FailureMemory`
- **Fresh legitimate data** — optional additional clean samples

`HoldoutDataLeakageError` is raised at assembly time if any path matching `held_out/`
is attempted as a training source. This guard is always active.

### Periodic Update Trigger

`RetrainingController.on_round_completed()` is called after every round. Every
**2 rounds**, it triggers `retrain_family()` for the active family:

| Family | Retraining type | Output |
|--------|----------------|--------|
| Family 1 | Heuristic weight + threshold adaptation | `heuristic-family1-retrained-vN` |
| Family 2 | Heuristic mandate weight adaptation | `heuristic-family2-retrained-vN` |
| Family 3 | Supervised XGBoost refit | `family3-xgb-retrained-vN` |

Each update returns a `ModelUpdateRecord` with training sample counts, false-negative
counts, baseline counts, and model version strings.

### Fail-Safe Rollback

If retraining raises an exception, `RetrainingController` rolls back to the previous
detector state and returns a `ModelUpdateRecord` with `retrained=False`.

---

## 9. Evaluation

The `UnifiedEvaluator` provides family-agnostic benchmarking across all three families.

### Per-Family Metrics

For each family, the evaluator computes a `FamilyEvaluationResult` with:

| Metric | Description |
|--------|-------------|
| `ConfusionMatrix` | TP / TN / FP / FN counts |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) — the primary detection rate |
| `false_positive_rate` | FP / (FP + TN) |
| `f1_score` | Harmonic mean of precision and recall |
| `accuracy` | (TP + TN) / total |
| `RiskMetrics` | Average, min, max, std of risk scores |

Undefined metrics (e.g., precision with zero positives) are returned as `None`,
never fabricated.

### Consolidated Metrics

`ConsolidatedMetrics` aggregates across all families:
- Total rounds (attacks + legitimate)
- Overall accuracy and detection rate
- Mean risk score

### Clean Evaluation Separation

| Family | Evaluation Data | Provenance |
|--------|----------------|-----------|
| Family 1 | **Clean Baseline Generalization** — 10 deterministic clean transaction profiles generated at a separate seed | Synthetic, not held-out |
| Family 2 | **Clean Baseline Generalization** — 5 deterministic clean authorized events generated at a separate seed | Synthetic, not held-out |
| Family 3 | **Isolated Held-Out Generalization** — 500 pre-partitioned clean identities from `data/held_out/heldout_identities.json` | True held-out, disk-resident |

Families 1 and 2 clean evaluation samples are generated at different seeds from
the training data (seed 142 and 242 respectively, vs training seed 42), ensuring
no object-level overlap. They are correctly labeled as **Clean Baseline Generalization**,
not held-out.

### Before/After Learning Comparison

`UnifiedEvaluator.compare_learning()` accepts pre- and post-update evaluation results
and produces a `BeforeAfterComparison` showing metric deltas per family.

---

## 10. Dashboard

The `PaymentSecurityDashboard` is a read-only presentation and analytics layer
that consumes `RoundResult` objects. It does not modify simulation behavior.

Components:

| Module | Purpose |
|--------|---------|
| `dashboard/feed.py` — `DashboardFeed` | Ingests and stores `RoundResult` objects |
| `dashboard/arms_race.py` — `build_arms_race_history()` | Computes `ArmsRaceReport` with timeline, detection trend, risk trend, model update markers, and recovery segments |
| `dashboard/arms_race.py` — `ArmsRacePresenter` | Query layer for structured analytics |
| `dashboard/controller.py` — `PaymentSecurityDashboard` | Orchestrates feed + presenter, exposes `DashboardState` |
| `dashboard/presenter.py` — `DashboardPresenter` | Formats round results for display |

Key outputs:
- Per-round timeline with `detected` / `missed` / `risk_score` / `model_version`
- Cumulative detection trend and rolling average risk
- Model update markers — identified **per-family** (cross-family version transitions are excluded)
- Recovery segments — rounds where Blue Team regains detection after a missed attack
- `ArmsRaceSummary` with aggregate detection rate, attack difficulty, and update count

---

## 11. Deterministic Demo

The demo runs fully offline. No API keys, no internet access, no LLM runtime is required.

### Run the demo

```bash
python -m demo.run_demo
```

Or directly:

```bash
python demo/run_demo.py
```

### Web Dashboard Prototype

The web prototype is local/offline and uses the existing deterministic simulation/evaluation pipeline to visualize results.

Run the web server:

```bash
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Then open your browser to:
http://127.0.0.1:8000

### JSON structured output

```bash
python demo/run_demo.py --json
```

### Options

```
--seed INT            Deterministic random seed (default: 42)
--f1-rounds INT       Family 1 rounds (default: 4)
--f2-rounds INT       Family 2 rounds (default: 2)
--f3-rounds INT       Family 3 rounds (default: 2)
--retrain-interval INT  Rounds between Blue Team updates (default: 2)
--json                Output structured result as JSON
--quiet               Suppress formatted console output
```

### Expected output structure

```
========================================================================
 MASTERCARD PAYMENT SECURITY LAB -- ADAPTIVE ARMS RACE DEMO
========================================================================
Deterministic seed: 42 | Retrain interval: 2 rounds

------------------------------------------------------------
 1. FAMILY 1: ADAPTIVE TRANSACTION-PATTERN EVASION
------------------------------------------------------------
 Round 1: ...  Risk Score: 0.2416 | Outcome: MISSED (FALSE NEGATIVE)
 Round 2: ...  Risk Score: 0.6038 | Outcome: DETECTED [FRAUD]

   >>> [BLUE TEAM MODEL RETRAINING UPDATE] <<<
   Previous Model: heuristic-family1-v1 -> Updated Model: heuristic-family1-retrained-v1
   False Negatives Used: 1 | Legitimate Baseline Samples: 10
   Total Training Samples: 11

 Round 3: ...  Outcome: MISSED (FALSE NEGATIVE)
 Round 4: ...  Outcome: MISSED (FALSE NEGATIVE)

   >>> [BLUE TEAM MODEL RETRAINING UPDATE] <<<
   Previous Model: heuristic-family1-retrained-v1 -> Updated Model: heuristic-family1-retrained-v2
   False Negatives Used: 3 | Legitimate Baseline Samples: 10

 Post-Learning Detection Recovery: NOT OBSERVED (Red-Team evolved stealthier evasion variant)

------------------------------------------------------------
 2. FAMILY 2: AI-AGENT PAYMENT BEHAVIOR (MANDATE ADHERENCE)
------------------------------------------------------------
 Round 1: ...  Risk Score: 0.8625 | Outcome: DETECTED [SCOPE BREACH]
 Round 2: ...  Risk Score: 0.8625 | Outcome: DETECTED [SCOPE BREACH]

------------------------------------------------------------
 3. FAMILY 3: SYNTHETIC IDENTITY & XGBOOST MODEL
------------------------------------------------------------
 Round 1: ...  Risk Score: 0.9836 | Outcome: DETECTED [SYNTHETIC IDENTITY]
 Round 2: ...  Risk Score: 0.9836 | Outcome: DETECTED [SYNTHETIC IDENTITY]

========================================================================
 LIVE SIMULATION ROUNDS EVALUATION (ATTACK-FOCUSED)
========================================================================
 Family: ADAPTIVE_EVASION  →  Detection Rate: 25.0%  |  TP=1 FN=3
 Family: AGENT_BEHAVIOR    →  Detection Rate: 100.0% |  TP=2 FN=0
 Family: SYNTHETIC_IDENTITY → Detection Rate: 100.0% |  TP=2 FN=0

 Consolidated: Total Rounds: 8 | Overall Detection Rate: 62.5%

========================================================================
 CLEAN GENERALIZATION & HELD-OUT EVALUATION
========================================================================
 ADAPTIVE_EVASION    Provenance: Clean Baseline Generalization   FPR: 0.0%
 AGENT_BEHAVIOR      Provenance: Clean Baseline Generalization   FPR: 0.0%
 SYNTHETIC_IDENTITY  Provenance: Isolated Held-Out Generalization FPR: 0.0%

========================================================================
 DASHBOARD & ARMS-RACE SUMMARY
========================================================================
 Model Retraining Updates Recorded: 2
 Post-Learning Recovery: NOT OBSERVED
========================================================================
```

### Reading the output

| Section | What it shows |
|---------|--------------|
| **Family rounds** | Per-round risk score, detection outcome, and model version |
| **Model Retraining Update blocks** | When Blue updates, what was used, how many samples |
| **Post-Learning Detection Recovery** | Whether the updated detector regained detection in subsequent rounds |
| **Live Simulation Rounds Evaluation** | Attack-only confusion matrix and detection rate per family |
| **Clean Generalization & Held-Out Evaluation** | False-positive control: does the detector misfire on legitimate data? |
| **Dashboard Summary** | Aggregate arms-race metrics including model update count |

---

## 12. Installation

### Requirements

- Python 3.11 or later
- No external API keys required
- No network access at runtime

### Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Core dependencies

| Package | Role |
|---------|------|
| `xgboost` | Family 3 supervised identity classifier |
| `shap` | Feature importance / explainability |
| `scikit-learn` | Supporting ML utilities |
| `numpy` | Numerical simulation |
| `pandas` | Data handling |
| `faker` | Synthetic identity and profile generation |
| `pydantic` | Schema validation and serialization |
| `pytest` | Test runner |

---

## 13. Running Tests

```bash
# Full regression suite
python -m pytest

# Verbose output
python -m pytest -v

# Focused by area
python -m pytest tests/test_family1_*.py -v
python -m pytest tests/test_family2_*.py -v
python -m pytest tests/test_family3_*.py -v
python -m pytest tests/test_demo.py -v
python -m pytest tests/test_dashboard_*.py -v
python -m pytest tests/test_retraining_controller.py tests/test_retraining_dataset.py tests/test_failure_memory.py -v
python -m pytest tests/test_unified_evaluation.py -v
```

**Current test count: 432 tests passing.**

Test coverage areas:

| Area | Tests |
|------|-------|
| Shared schemas and contracts | `test_schemas.py`, `test_integration_contracts.py` |
| Genome engine | `test_genome_engine.py` |
| Round controller | `test_round_controller.py` |
| Family 1 | `test_family1_*.py` (5 files) |
| Family 2 | `test_family2_*.py` (3 files) |
| Family 3 | `test_family3_*.py` (6 files) |
| Dashboard | `test_dashboard_*.py` (4 files) |
| Blue-Team learning | `test_failure_memory.py`, `test_retraining_dataset.py`, `test_retraining_controller.py`, `test_multiround_retraining_integration.py` |
| Evaluation | `test_unified_evaluation.py` |
| Demo | `test_demo.py` |
| Pipeline smoke test | `test_smoke.py` |

---

## 14. Current Limitations

The following limitations are acknowledged honestly:

1. **Family 1 and 2 detectors are heuristic, not deep-learning models.** They use
   weighted deviation scoring and mandate-envelope checking. Their "retraining" is
   parameter adaptation (weight shifts, threshold recalibration), not gradient-based
   optimization.

2. **Post-Learning Detection Recovery is not observed** in the default 4-round Family 1
   demo scenario. The Red Team's mutation strategy successfully evolves stealthier genomes
   that evade the updated heuristic parameters. This is the expected and honest behavior —
   the arms race is genuinely competitive.

3. **No reinforcement learning** is used for Red Team mutation. Mutation is Python-controlled
   and feedback-driven: detected attacks reduce strong detection signals, while missed attacks
   preserve successful evasion patterns and explore nearby bounded genome variants within
   `[0.0, 1.0]`.

4. **Clean Baseline Generalization (Families 1 and 2)** uses deterministically generated
   synthetic samples at held-out seeds, not a pre-partitioned static evaluation file. Only
   Family 3 has a genuinely pre-partitioned on-disk held-out set
   (`data/held_out/heldout_identities.json`).

5. **No runtime LLM dependency.** The demo and all tests run completely offline. The original
   project specification allowed for optional LLM-generated qualitative content; this is not
   implemented in the current prototype.

6. **No production infrastructure** (no API servers, no streaming, no database persistence).
   This is a research prototype designed for controlled simulation and evaluation.

---

## 15. Repository Structure

```
mastercard-payment-security-lab/
├── schemas/                    # Frozen shared Pydantic schemas (AttackEvent, RoundResult, ...)
├── simulation/                 # Frozen core: RoundController, Protocol interfaces, Pipeline
├── mutation/                   # Frozen genome engine: validate, normalize, compare, serialize
├── attacks/
│   ├── transaction_evasion/    # Family 1 generator + mutator
│   ├── ai_agent/               # Family 2 generator + mutator
│   └── synthetic_identity/     # Family 3 generator + mutator
├── blue_team/
│   ├── transaction/            # Family 1 heuristic detector + evaluator
│   ├── ai_agent/               # Family 2 mandate-envelope detector + evaluator
│   ├── synthetic_identity/     # Family 3 XGBoost detector + evaluator
│   └── learning/               # FailureMemory, RetrainingDataset, RetrainingController
├── evaluation/                 # UnifiedEvaluator, per-family adapters, metrics, reports
├── dashboard/                  # PaymentSecurityDashboard, ArmsRacePresenter, DashboardFeed
├── demo/                       # Deterministic demo runner (run_demo.py)
├── data/
│   ├── legitimate/             # baseline_identities.json (Family 3 training baseline)
│   ├── held_out/               # heldout_identities.json (Family 3 isolated eval set)
│   └── generators/             # Synthetic data generation utilities
├── tests/                      # Full regression suite (432 tests)
├── docs/
│   ├── INTEGRATION_CONTRACTS.md # Protocol + schema contracts (frozen Phase 5)
│   └── TEAM_HANDOFF.md          # Module ownership and working agreements
├── MASTER_SPEC.md              # Project objective and specification
├── requirements.txt            # All dependencies (pinned)
└── README.md                   # This document
```

---

## 16. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Frozen shared schemas (Pydantic v2) | Prevent family implementations from breaking the core pipeline contract |
| Family-agnostic `RoundController` | Any `AttackGenerator` + `BlueTeamDetector` + `FeedbackEvaluator` combo runs without code changes |
| Heuristic detectors for F1/F2 | Appropriate for prototype scope; genuine XGBoost deployed for F3 where lifecycle feature complexity justifies it |
| Genome-driven Red Team | Deterministic, reproducible, interpretable attack variants without LLM dependency |
| `HoldoutDataLeakageError` guard | Hard fail at dataset assembly time prevents accidental training on evaluation data |
| Deterministic seeded demo | Any judge can reproduce the exact output without installation of additional tools |
| Per-family `ModelUpdateMarker` tracking | Prevents cross-family version transitions from being miscounted as retraining events |

---

*Mastercard Innovation Challenge 2026 — Mastercard Payment Security Lab*