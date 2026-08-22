# Team Handoff Document
## Mastercard Payment Security Lab — Phase 5

> **Purpose:** Define module ownership and working agreements for parallel
> implementation of Family 2, Family 3, the synthetic environment, and
> the dashboard layer.  
> **Audience:** All team members.  
> **Read first:** `MASTER_SPEC.md`, then `docs/INTEGRATION_CONTRACTS.md`,
> then this document.

---

## 1. Ownership Map

### Member 1 — Technical Lead

**Owns and is responsible for:**

| Area | Files / Directories |
|---|---|
| Core architecture | `simulation/interfaces.py`, `simulation/round_controller.py`, `simulation/mock_pipeline.py`, `simulation/__init__.py` |
| Shared schemas | `schemas/` (all files) |
| Genome engine | `mutation/genome_engine.py` |
| Family 1 implementation | `attacks/transaction_evasion/`, `blue_team/transaction/` |
| Blue-Team learning / retraining | to be determined in a later phase |
| Unified evaluation | to be determined in a later phase |
| Final integration | Merging Family 2 and 3 into the shared pipeline |
| Regression test suite | `tests/` (all existing tests) |
| CI / branch hygiene | Reviewing and merging pull requests |

**Must be consulted before any change to:**
- `schemas/` (any file)
- `simulation/interfaces.py`
- `simulation/round_controller.py`
- `mutation/genome_engine.py`

---

### Member 2

**Owns and is responsible for:**

| Area | Files / Directories |
|---|---|
| Synthetic environment / data | `data/` |
| Legitimate baseline data | `data/legitimate/` (create as needed) |
| Held-out legitimate evaluation set | `data/held_out/` (create as needed) |
| Family 3 attack generator | `attacks/synthetic_identity/` |
| Family 3 blue-team detector | `blue_team/synthetic_identity/` |
| Family 3 tests | `tests/test_family3_*.py` |

**Preferred working areas:**

```
data/
attacks/synthetic_identity/
blue_team/synthetic_identity/
```

**Key deliverables:**

1. A `SyntheticIdentity`-aware `AttackGenerator` implementation that
   tags events with `AttackFamily.SYNTHETIC_IDENTITY` and populates
   the Family 3 genome dimensions.
2. A `BlueTeamDetector` for Family 3 (XGBoost / LightGBM or justified
   alternative).
3. A `FeedbackEvaluator` for Family 3 (or shared evaluator if logic is
   equivalent to the generic pattern).
4. A `MutationStrategy` for Family 3.
5. Legitimate baseline transaction / identity data for training and
   held-out evaluation.
6. All minimum tests defined in `docs/INTEGRATION_CONTRACTS.md § 7`.

---

### Member 3

**Owns and is responsible for:**

| Area | Files / Directories |
|---|---|
| Family 2 attack generator | `attacks/ai_agent/` |
| Family 2 blue-team detector | `blue_team/ai_agent/` |
| Family 2 tests | `tests/test_family2_*.py` |
| Dashboard / product layer | `dashboard/` |
| Dashboard tests | `tests/test_dashboard_*.py` |

**Preferred working areas:**

```
attacks/ai_agent/
blue_team/ai_agent/
dashboard/
```

**Key deliverables:**

1. An `AIAgentPaymentEvent`-aware `AttackGenerator` implementation that
   tags events with `AttackFamily.AGENT_BEHAVIOR` and populates the
   Family 2 genome dimensions.
2. A `BlueTeamDetector` for Family 2 (XGBoost / LightGBM or justified
   alternative).
3. A `FeedbackEvaluator` for Family 2.
4. A `MutationStrategy` for Family 2.
5. A dashboard / visualisation layer that consumes `RoundResult` objects
   (read-only; must not modify core pipeline behaviour).
6. All minimum tests defined in `docs/INTEGRATION_CONTRACTS.md § 7`.

---

## 2. Protection Rules

> These rules protect the shared core from accidental breakage during
> parallel development. They apply to every member without exception.

1. **Do not modify `schemas/` casually.** Adding, removing, or renaming
   a top-level schema field is a breaking change. Open a discussion issue
   first. Family-specific data always goes into `scenario` or `metadata`.

2. **Do not modify `simulation/round_controller.py` without the technical
   lead's approval.** The controller is frozen. If you find a concrete
   integration issue (e.g., a missing parameter), report it before
   redesigning.

3. **Do not change shared protocols in `simulation/interfaces.py` without
   coordination.** The four protocols are the contracts all families
   implement. Changes affect every family at once.

4. **Do not duplicate common contracts.** The shared schemas, interfaces,
   and genome engine already exist. Implement your family module *on top*
   of them; do not create parallel versions.

5. **Do not add family-specific branching to the core controller.** The
   controller must remain family-agnostic. Family logic belongs in the
   family implementation files.

6. **Each member owns their assigned files while implementing their
   module.** Other members should not open PRs that modify another
   member's ownership area without prior discussion.

7. **Use feature branches.** Work on a branch named with the format
   `phase/<short-description>` or `feat/<short-description>`. Never
   commit directly to `main`.

8. **Tests must accompany meaningful implementations.** Do not open a PR
   for a generator or detector without accompanying tests.

9. **Run the full regression suite before pushing.**
   ```
   .venv\Scripts\python.exe -m pytest
   ```
   All tests must pass before a PR is opened.

10. **Merge only through the established Git workflow.** Push your branch,
    open a pull request, and wait for technical integration review before
    merging.

---

## 3. Handoff Checklist

Complete these steps **before** opening a pull request for your module:

### Reading / orientation

- [ ] Read `MASTER_SPEC.md` from top to bottom.
- [ ] Read `docs/INTEGRATION_CONTRACTS.md` — understand all interface and
      schema contracts.
- [ ] Inspect all files under `schemas/` directly.
- [ ] Read `simulation/interfaces.py` — understand the four protocols.
- [ ] Read `simulation/round_controller.py` — understand the pipeline.
- [ ] Skim `simulation/mock_pipeline.py` — understand the reference
      implementation of `Pipeline`, `MockAttackGenerator`, etc.
- [ ] Read all existing tests under `tests/` to understand the test
      patterns already established.

### Implementation

- [ ] Create a feature branch: `git switch -c feat/<your-feature>`.
- [ ] Implement only your assigned ownership area (see § 1 above).
- [ ] Populate `AttackEvent.scenario` and `AttackEvent.metadata` with
      your family-specific data — do not add new top-level fields to
      `AttackEvent`.
- [ ] Set `attack_family` to the correct `AttackFamily` enum value for
      your family.
- [ ] Use `mutation/genome_engine.validate_genome()` in your mutator to
      validate the genome before returning it.

### Testing

- [ ] Write all minimum tests listed in
      `docs/INTEGRATION_CONTRACTS.md § 7`.
- [ ] Verify that each protocol isinstance check passes for your classes.
- [ ] Cover the detected and missed attack scenarios explicitly.
- [ ] Run your focused tests: `.venv\Scripts\python.exe -m pytest tests/test_family<N>_*.py -v`
- [ ] Run the full regression suite: `.venv\Scripts\python.exe -m pytest`
- [ ] Confirm all 69+ existing tests still pass (zero regressions).

### Handoff

- [ ] Write brief integration notes in your PR description:
      - What genome dimensions does your generator produce?
      - What does your detector use as features?
      - What mutation direction does your strategy apply?
      - Any assumptions the core integration should be aware of?
- [ ] Push your branch: `git push origin feat/<your-feature>`.
- [ ] Open a pull request targeting `main`.
- [ ] Wait for technical integration review from Member 1 before merging.

---

## 4. Git Workflow Summary

```
main
 └── phase/integration-contracts   ← Phase 5 (this branch)
      └── feat/family2-agent       ← Member 3 works here
      └── feat/family3-identity    ← Member 2 works here
      └── feat/dashboard           ← Member 3 works here
```

- Branches off `main` after Phase 5 is merged.
- PRs target `main`.
- Member 1 reviews and merges family PRs into `main`.
- Do not merge your own PR without review.

---

## 5. Stable Core Reference

The following is the stable shared core after Phase 5. All family
implementations depend on these files and must not modify them:

```
schemas/
├── __init__.py          ← public exports
├── common.py            ← AttackFamily enum
├── attack.py            ← AttackEvent
├── prediction.py        ← PredictionResult
├── feedback.py          ← BlueTeamFeedback
├── round.py             ← RoundResult
├── transaction.py       ← Transaction (Family 1 domain schema)
├── agent_event.py       ← AIAgentPaymentEvent (Family 2 domain schema)
└── identity.py          ← SyntheticIdentity (Family 3 domain schema)

mutation/
└── genome_engine.py     ← validate_genome, normalize_genome,
                            calculate_aggregate_intensity, compare_genomes,
                            serialize_genome, deserialize_genome

simulation/
├── interfaces.py        ← AttackGenerator, BlueTeamDetector,
│                           FeedbackEvaluator, MutationStrategy protocols
├── round_controller.py  ← RoundController, RoundControllerError
└── mock_pipeline.py     ← reference Pipeline + Mock components
```

---

*Document version: Phase 5 (2026-08-23).*
