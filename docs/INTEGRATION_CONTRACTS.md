# Integration Contracts
## Mastercard Payment Security Lab — Phase 5 Handoff Reference

> **Status:** Frozen after Phase 5.  
> **Audience:** Members 2 and 3 (and any future contributor).  
> **Authority:** This document describes the *actual implemented code* in
> Phases 0–4. It is not aspirational architecture.

---

## 1. Shared Lifecycle

Every simulation round follows this fixed pipeline, orchestrated by
`RoundController`:

```
AttackGenerator.generate(round_id)
        │
        ▼
    AttackEvent          ← shared schema, family-tagged
        │
        ▼
BlueTeamDetector.detect(event)
        │
        ▼
    PredictionResult     ← shared schema, model-agnostic
        │
        ▼
FeedbackEvaluator.evaluate(event, prediction)
        │
        ▼
    BlueTeamFeedback     ← shared schema
        │
        ▼
    RoundResult          ← aggregate of the above three objects
        │
        ▼  (between rounds, not inside the controller)
MutationStrategy.mutate(genome, feedback)
        │
        ▼
 updated genome → fed back to AttackGenerator for next round
```

**Key design rules already enforced by the core:**

- The `RoundController` knows nothing about which attack family is active.
- The `RoundController` does not perform mutation. Mutation is called by
  the Pipeline runner (`simulation/mock_pipeline.py::Pipeline`) *between*
  rounds, after `RoundController.run_round()` returns.
- The controller validates that each component returns the correct schema
  type; a wrong return type raises `RoundControllerError` immediately.

---

## 2. Interface Contracts

All interfaces live in `simulation/interfaces.py`.
They are `typing.Protocol` classes decorated with `@runtime_checkable`,
which means `isinstance()` checks work at runtime without inheritance.

---

### 2.1 `AttackGenerator`

```python
class AttackGenerator(Protocol):
    def generate(self, round_id: str) -> AttackEvent: ...
```

| Item | Detail |
|---|---|
| **Input** | `round_id: str` — unique string for the current round (non-empty) |
| **Return** | `AttackEvent` — fully-populated pydantic model |
| **Responsibility** | Produce one `AttackEvent` per call. Owns all family-specific logic for constructing the event (genome selection, scenario fields, ground truth). |
| **Family-specific behaviour** | Genome key names, scenario dictionary contents, and ground_truth logic all belong in the family implementation, not in the controller. |
| **Genome feedback** | The `AttackGenerator` protocol does NOT define a `set_genome()` method. Genome feedback is delivered via a caller-supplied callback (`genome_updater`) in the `Pipeline` runner, not by the controller reaching into the generator directly. A family generator may manage its own genome state internally without any external setter. |

---

### 2.2 `BlueTeamDetector`

```python
class BlueTeamDetector(Protocol):
    def detect(self, event: AttackEvent) -> PredictionResult: ...
```

| Item | Detail |
|---|---|
| **Input** | `event: AttackEvent` — the event produced by the generator this round |
| **Return** | `PredictionResult` — pydantic model |
| **Responsibility** | Run the detection model and return a scored prediction. Must not mutate the input event. |
| **Family-specific behaviour** | Feature extraction from `event.scenario` and `event.attack_genome`, model choice (XGBoost / LightGBM / rule-based), and SHAP/explanation generation all belong inside the family detector. The core only sees `PredictionResult`. |
| **Model agnosticism** | The core does not care whether the implementation uses XGBoost, LightGBM, a heuristic, or a stub. Only the return type is enforced. |

---

### 2.3 `FeedbackEvaluator`

```python
class FeedbackEvaluator(Protocol):
    def evaluate(
        self,
        event: AttackEvent,
        prediction: PredictionResult,
    ) -> BlueTeamFeedback: ...
```

| Item | Detail |
|---|---|
| **Inputs** | `event: AttackEvent` (carries `ground_truth`), `prediction: PredictionResult` |
| **Return** | `BlueTeamFeedback` — pydantic model |
| **Responsibility** | Compare prediction against ground truth. Set `detected`, `false_positive`, `false_negative` flags. Populate `important_features` from prediction contributions. |
| **Family-specific behaviour** | Logic for determining what counts as a false positive or false negative for a family's scenario may differ. The evaluator implementation owns that logic. |

---

### 2.4 `MutationStrategy`

```python
class MutationStrategy(Protocol):
    def mutate(
        self,
        genome: dict,
        feedback: BlueTeamFeedback,
    ) -> dict: ...
```

| Item | Detail |
|---|---|
| **Inputs** | `genome: dict` — the attack genome from the completed round; `feedback: BlueTeamFeedback` from the same round |
| **Return** | `dict` — a new genome for the next round |
| **Responsibility** | Apply family-specific mutation logic. Return a genome of the same key structure unless the strategy intentionally adds/removes dimensions. |
| **Core boundary** | The generic core contains no family-specific mutation rules. Each family provides its own `MutationStrategy` implementation. |
| **When called** | Between rounds, outside `RoundController`, by the `Pipeline` runner. |

---

## 3. Schema Contracts

All schemas live in the `schemas/` package and are
**pydantic v2 `BaseModel`** classes. They are frozen for the core
pipeline. Family implementations must accept and return these types
unchanged.

---

### 3.1 `AttackFamily` (enum)

```python
# schemas/common.py
class AttackFamily(str, Enum):
    ADAPTIVE_EVASION   = "Family 1 - Adaptive Transaction-Pattern Evasion"
    AGENT_BEHAVIOR     = "Family 2 - Unauthorized / Malicious AI-Agent Payment Behavior"
    SYNTHETIC_IDENTITY = "Family 3 - Synthetic Identity + AI-Generated Identity Artifacts"
```

Every `AttackEvent` must declare exactly one `AttackFamily`. This is the
only family-discriminator the core pipeline reads.

---

### 3.2 `AttackEvent`

```python
# schemas/attack.py
class AttackEvent(BaseModel):
    attack_id:      str              # non-empty; unique per event
    round_id:       str              # non-empty; matches controller's round_id
    attack_family:  AttackFamily     # family tag
    attack_genome:  Dict[str, float] # numeric genome; keys are family-defined
    scenario:       Dict[str, Any]   # family-specific scenario data
    ground_truth:   bool             # True = this is a real attack
    metadata:       Dict[str, Any]   # optional; defaults to {}
```

**Shared fields** — used by the core pipeline for every family:
`attack_id`, `round_id`, `attack_family`, `attack_genome`, `ground_truth`.

**Family-specific fields** — belong inside `scenario` and `metadata`:
Family implementations may put any serialisable data there.

---

### 3.3 `PredictionResult`

```python
# schemas/prediction.py
class PredictionResult(BaseModel):
    prediction_id:         str              # non-empty; unique per prediction
    prediction:            bool             # True = fraud detected
    risk_score:            float            # [0.0, 1.0]
    model_version:         str              # non-empty; identifies model/version
    explanation:           Optional[str]    # human-readable explanation string
    feature_contributions: Optional[Dict[str, float]]  # SHAP-style feature weights
```

`prediction` and `risk_score` are consumed by the core pipeline and
`FeedbackEvaluator`. `explanation` and `feature_contributions` are
optional enrichment that the dashboard and failure-log layers may use.

---

### 3.4 `BlueTeamFeedback`

```python
# schemas/feedback.py
class BlueTeamFeedback(BaseModel):
    feedback_id:        str               # non-empty; unique per feedback
    round_reference:    str               # non-empty; matches round_id
    detected:           bool              # True = correctly flagged as fraud
    false_positive:     bool              # True = flagged as fraud but was legitimate
    false_negative:     bool              # True = missed actual fraud
    risk_score:         float             # [0.0, 1.0]; copied from PredictionResult
    important_features: Dict[str, float]  # genome dimensions that mattered
    explanation_data:   Optional[Dict[str, Any]]  # free-form explanation context
```

The three boolean flags must be mutually consistent. For a pure-attack
scenario (ground_truth=True):
- `detected=True, false_negative=False, false_positive=False` — correct detection
- `detected=False, false_negative=True, false_positive=False` — missed attack

`important_features` feeds into the Red Team mutation strategy.

---

### 3.5 `RoundResult`

```python
# schemas/round.py
class RoundResult(BaseModel):
    round_id:          str               # matches the round that produced it
    attack_event:      AttackEvent       # the event generated this round
    prediction_result: PredictionResult  # the detector's output
    feedback:          BlueTeamFeedback  # the evaluator's output
    outcome_metrics:   Dict[str, Any]    # caller-supplied metrics; defaults to {}
```

`RoundResult` is the complete record of one round. It is the output of
`RoundController.run_round()` and the input to downstream logging,
retraining triggers, and demo display.

---

### 3.6 Family-specific schemas (domain context, not pipeline types)

These schemas carry domain data that family implementations populate into
`AttackEvent.scenario` or use internally. They are **not** consumed by
the core pipeline directly.

#### `Transaction` — Family 1 context (`schemas/transaction.py`)

| Field | Type | Notes |
|---|---|---|
| `transaction_id` | `str` | non-empty |
| `user_id` | `str` | non-empty |
| `timestamp` | `datetime` | |
| `amount` | `float` | >= 0.0 |
| `currency` | `str` | 3-char ISO code |
| `merchant_id` | `str` | non-empty |
| `merchant_category` | `str` | |
| `location` | `str` | |
| `device_id` | `str` | |
| `payment_channel` | `str` | |

#### `AIAgentPaymentEvent` — Family 2 context (`schemas/agent_event.py`)

| Field | Type | Notes |
|---|---|---|
| `event_id` | `str` | non-empty |
| `user_intent` | `str` | what the user authorized in natural language |
| `authorized_scope` | `str` | scope of the agent's permission |
| `agent_identity` | `str` | identifier of the acting agent |
| `session_context` | `str` | session / provenance string |
| `actual_action` | `str` | what the agent actually did |
| `transaction` | `Optional[Transaction]` | linked transaction if applicable |

#### `SyntheticIdentity` — Family 3 context (`schemas/identity.py`)

| Field | Type | Notes |
|---|---|---|
| `identity_id` | `str` | non-empty |
| `identity_attributes` | `Dict[str, Any]` | name, DOB, SSN-proxy, etc. |
| `contact_attributes` | `Dict[str, Any]` | email, phone, address |
| `account_metadata` | `Dict[str, Any]` | account open date, type, tier |
| `device_context` | `Dict[str, Any]` | device fingerprint history |
| `lifecycle_info` | `Dict[str, Any]` | timeline of account activity |

---

## 4. Family Integration Rules

### How a family implementation plugs into the shared system

A family module provides concrete classes that satisfy the four protocols:
`AttackGenerator`, `BlueTeamDetector`, `FeedbackEvaluator`, and
`MutationStrategy`. These are injected into a `RoundController` (and
`Pipeline`) at construction time. The core never imports family modules
directly.

Suggested file layout per family:

```
attacks/<family>/generator.py     →  implements AttackGenerator
attacks/<family>/mutator.py       →  implements MutationStrategy
blue_team/<family>/detector.py    →  implements BlueTeamDetector
blue_team/<family>/evaluator.py   →  implements FeedbackEvaluator
```

---

### Family 1 — Adaptive Transaction-Pattern Evasion

- **Unit of analysis:** Transaction / short transaction sequence.
- **Core question:** Does this payment behavior look suspicious compared
  with this user's normal behavior?
- **Genome dimensions (defined in MASTER_SPEC.md):**
  `amount_deviation`, `velocity_deviation`, `device_novelty`,
  `location_deviation`, `time_deviation`, `sequence_anomaly`.
- **Family-specific schema:** `Transaction` (place in `scenario`).
- **Ownership:** `attacks/transaction_evasion/`, `blue_team/transaction/`.
- **Responsible member:** Member 1 (Technical Lead).

---

### Family 2 — Unauthorized / Malicious AI-Agent Payment Behavior

- **Unit of analysis:** Authorization / payment event.
- **Core question:** Did the AI agent actually do what the user authorized?
- **Genome dimensions (defined in MASTER_SPEC.md):**
  `intent_amount_deviation`, `intent_category_deviation`,
  `permission_scope_deviation`, `agent_identity_confidence`,
  `session_provenance_anomaly`, `purchase_velocity`.
- **Family-specific schema:** `AIAgentPaymentEvent` (place in `scenario`).
- **Ownership:** `attacks/ai_agent/`, `blue_team/ai_agent/`.
- **Responsible member:** Member 3.

---

### Family 3 — Synthetic Identity + AI-Generated Identity Artifacts

- **Unit of analysis:** Identity / account lifecycle.
- **Core question:** Does this account remain consistent with a plausible
  legitimate identity over time?
- **Genome dimensions (defined in MASTER_SPEC.md):**
  `cross_field_consistency`, `profile_plausibility_score`,
  `contact_consistency`, `device_history_score`,
  `lifecycle_behavior_coherence`, `time_to_risky_activity`.
- **Family-specific schema:** `SyntheticIdentity` (place in `scenario`).
- **Ownership:** `attacks/synthetic_identity/`,
  `blue_team/synthetic_identity/`, `data/`.
- **Responsible member:** Member 2.

---

### What must NOT cross family boundaries

- Family 2 must not populate Family 1 genome keys in its `AttackEvent`.
- Family 3 must not reference `AIAgentPaymentEvent` in its scenario data.
- No family implementation must add branching to `RoundController` or any
  shared schema.
- Family-specific configuration (detection thresholds, mutation step
  sizes) belongs in `config/` or inside the family module, not in the
  shared core.

---

## 5. Model Boundary

Blue-Team family implementations return `PredictionResult`. The core
pipeline is **completely agnostic** to which model produces it.

> An implementation may use XGBoost, LightGBM, a hand-crafted heuristic,
> or any other justified detector — as long as it returns a valid
> `PredictionResult` with `risk_score` in `[0.0, 1.0]`.

`model_version` in `PredictionResult` should be set to a meaningful
string (e.g., `"xgb-family2-v1"`) so that failure logs and retraining
records remain traceable.

`feature_contributions` (optional) should be populated with SHAP values
or equivalent feature importance weights when available. The core does
not require it, but the Red Team mutation strategy and the dashboard
layer benefit from it.

---

## 6. Mutation Boundary

The `MutationStrategy` protocol is the only mutation interface the
pipeline calls. The generic core — `RoundController`, `Pipeline`, all
schemas — contains **no family-specific mutation rules**.

Rules for implementors:

1. Return a genome with the same key structure unless the strategy
   intentionally adds or removes a dimension.
2. Clamp all values to `[0.0, 1.0]` unless the family's genome
   configuration legitimately uses a different range (document this).
3. Use `mutation/genome_engine.py` utilities (`validate_genome`,
   `normalize_genome`, `compare_genomes`) to validate and compare genomes
   before and after mutation.
4. Do not import family-specific mutation logic into the generic
   `mutation/genome_engine.py`. The engine is shared; family rules stay
   in family modules.

---

## 7. Testing Contract

Each family module must ship tests alongside its implementation.
The minimum bar is:

| Test area | Minimum requirement |
|---|---|
| Schema round-trip | Construct the family's `AttackEvent` with its canonical genome; assert all fields round-trip through pydantic validation. |
| Generator protocol | `assert isinstance(gen, AttackGenerator)` (runtime protocol check). |
| Detector protocol | `assert isinstance(det, BlueTeamDetector)`. |
| Evaluator protocol | `assert isinstance(ev, FeedbackEvaluator)`. |
| Mutator protocol | `assert isinstance(m, MutationStrategy)`. |
| Full round | Inject all four into `RoundController`, run one round, assert a valid `RoundResult`. |
| Detection cases | At least one test for a detected attack and one for a missed attack. |
| Mutation response | At least one test showing genome shifts in the correct direction after detected vs. missed feedback. |
| Genome validity | Assert that mutated genomes pass `validate_genome()` from `mutation/genome_engine.py`. |

Family modules must NOT break any existing test in `tests/`.
Run `pytest` from the project root before opening a pull request.

---

## 8. Architectural Freeze — Stable Core After Phase 5

The following files are **stable** and must not be modified without
coordination with the technical lead (Member 1):

| File / directory | Status | Notes |
|---|---|---|
| `schemas/` (all files) | **Frozen** | Adding new top-level schema fields requires a contract review. Family-specific data goes in `scenario` / `metadata`. |
| `mutation/genome_engine.py` | **Frozen** | Generic utilities only. No family rules. |
| `simulation/interfaces.py` | **Frozen** | All four protocol definitions. Do not add family-specific methods. |
| `simulation/round_controller.py` | **Frozen** | Orchestration logic only. No family branching. |
| `simulation/mock_pipeline.py` | **Reference, not frozen** | Demonstrates the generic pipeline pattern. May be extended for demo purposes but must not become family-specific. |

Family-specific implementation contracts (genome key names, scenario
field structure, detector internals, mutation step sizes) are NOT frozen
— they are under the owning member's discretion within the rules above.

---

*Document version: Phase 5 (2026-08-23). Update this document only when
a concrete change to the shared contracts is merged to `main`.*
