"""
tests/test_family3_integration.py — Focused integration test suite for Family 3 (Synthetic Identity).

Verifies the complete existing Family 3 flow:
Synthetic identity/account lifecycle
→ Family 3 AttackGenerator
→ AttackEvent
→ Family 3 BlueTeamDetector
→ PredictionResult
→ Family 3 FeedbackEvaluator
→ BlueTeamFeedback
→ Family 3 MutationStrategy
→ next valid Family 3 genome

Covers the 16 Task 8 integration requirements:
1. Family 3 generator satisfies AttackGenerator protocol.
2. Generated Family 3 events use AttackFamily.SYNTHETIC_IDENTITY.
3. Generated AttackEvent contains the canonical 6-dimension Family 3 genome.
4. Generated events contain required synthetic identity/account-lifecycle data.
5. Family 3 detector satisfies BlueTeamDetector protocol.
6. Detector returns a valid PredictionResult (prediction, risk_score, model_version, explanation, feature_contributions).
7. Family 3 feedback evaluator satisfies FeedbackEvaluator protocol and produces valid BlueTeamFeedback.
8. A detected Family 3 attack produces correct feedback behavior (detected=True, FP=False, FN=False).
9. A missed Family 3 attack produces correct false-negative feedback behavior (detected=False, FP=False, FN=True).
10. Family 3 mutation produces a valid next genome (validate_genome).
11. Mutation keeps every Family 3 gene strictly within [0.0, 1.0].
12. Mutation is deterministic for identical (genome, feedback) inputs.
13. Mutation does not blindly increase/decrease every genome dimension uniformly.
14. Family 3 components execute through unmodified RoundController producing valid RoundResult.
15. Baseline training data and held-out evaluation data remain strictly separated.
16. Family 3 tests do not depend on Family 1 or Family 2 implementations.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from schemas import (
    AttackEvent,
    AttackFamily,
    BlueTeamFeedback,
    PredictionResult,
    RoundResult,
    SyntheticIdentity,
)
from simulation.interfaces import (
    AttackGenerator,
    BlueTeamDetector,
    FeedbackEvaluator,
    MutationStrategy,
)
from simulation import RoundController
from mutation.genome_engine import validate_genome
from data.generators.identity_generator import LegitimateIdentityGenerator, load_dataset

from attacks.synthetic_identity import (
    SyntheticIdentityAttackGenerator,
    SyntheticIdentityMutationStrategy,
    FAMILY3_GENOME_DIMENSIONS,
    DEFAULT_ATTACK_GENOME,
    DEFAULT_LEGITIMATE_GENOME,
    DEFAULT_DETECTED_STEP,
    DEFAULT_MISSED_STEP,
)
from blue_team.synthetic_identity import (
    SyntheticIdentityBlueDetector,
    SyntheticIdentityFeedbackEvaluator,
    MODEL_VERSION,
    FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Requirement 1: Generator Protocol Conformance
# ---------------------------------------------------------------------------

def test_family3_generator_satisfies_protocol():
    """Requirement 1: Family 3 generator satisfies runtime_checkable AttackGenerator protocol."""
    gen = SyntheticIdentityAttackGenerator()
    assert isinstance(gen, AttackGenerator)


# ---------------------------------------------------------------------------
# Requirement 2: Correct AttackFamily Tagging
# ---------------------------------------------------------------------------

def test_family3_event_attack_family_value():
    """Requirement 2: Generated Family 3 events use AttackFamily.SYNTHETIC_IDENTITY."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="f3-fam-check")

    assert isinstance(event, AttackEvent)
    assert event.attack_family == AttackFamily.SYNTHETIC_IDENTITY
    assert event.attack_family.value == "Family 3 - Synthetic Identity + AI-Generated Identity Artifacts"
    assert event.metadata.get("attack_family") == AttackFamily.SYNTHETIC_IDENTITY.value


# ---------------------------------------------------------------------------
# Requirement 3: Canonical 6-Dimension Family 3 Genome
# ---------------------------------------------------------------------------

def test_family3_canonical_genome_dimensions():
    """Requirement 3: Generated AttackEvent contains the exact six canonical Family 3 dimensions."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="f3-genome-check")

    expected_dimensions = {
        "cross_field_consistency",
        "profile_plausibility_score",
        "contact_consistency",
        "device_history_score",
        "lifecycle_behavior_coherence",
        "time_to_risky_activity",
    }
    assert set(event.attack_genome.keys()) == expected_dimensions
    assert set(FAMILY3_GENOME_DIMENSIONS) == expected_dimensions
    assert len(event.attack_genome) == 6
    validate_genome(event.attack_genome)


# ---------------------------------------------------------------------------
# Requirement 4: Synthetic Identity & Lifecycle Scenario Structure
# ---------------------------------------------------------------------------

def test_family3_scenario_and_metadata_structure():
    """Requirement 4: Event scenario contains complete SyntheticIdentity lifecycle information."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    event = gen.generate(round_id="f3-scenario-check")

    # Validate against schemas.identity.SyntheticIdentity
    identity = SyntheticIdentity.model_validate(event.scenario)

    assert identity.identity_id.startswith("ident_f3_")
    # Demographics
    assert "first_name" in identity.identity_attributes
    assert "last_name" in identity.identity_attributes
    assert "dob" in identity.identity_attributes
    assert "ssn_proxy" in identity.identity_attributes
    # Contacts
    assert "primary_email" in identity.contact_attributes
    assert "phone_number" in identity.contact_attributes
    assert "city" in identity.contact_attributes
    assert "state" in identity.contact_attributes
    # Account Metadata
    assert "account_id" in identity.account_metadata
    assert "account_open_date" in identity.account_metadata
    assert "account_age_days" in identity.account_metadata
    # Device Context
    assert "primary_device_id" in identity.device_context
    assert "known_devices" in identity.device_context
    # Lifecycle Info
    assert "days_to_first_transaction" in identity.lifecycle_info
    assert "lifecycle_events_summary" in identity.lifecycle_info


# ---------------------------------------------------------------------------
# Requirement 5: Detector Protocol Conformance
# ---------------------------------------------------------------------------

def test_family3_detector_satisfies_protocol():
    """Requirement 5: Family 3 detector satisfies runtime_checkable BlueTeamDetector protocol."""
    det = SyntheticIdentityBlueDetector()
    assert isinstance(det, BlueTeamDetector)


# ---------------------------------------------------------------------------
# Requirement 6: PredictionResult Structure and Fields
# ---------------------------------------------------------------------------

def test_family3_detector_prediction_result_fields():
    """Requirement 6: Detector returns a valid PredictionResult with all required fields."""
    gen = SyntheticIdentityAttackGenerator(seed=42)
    det = SyntheticIdentityBlueDetector()

    event = gen.generate(round_id="f3-pred-check")
    pred = det.detect(event)

    assert isinstance(pred, PredictionResult)
    assert pred.prediction_id == "pred-f3-f3-pred-check"
    assert isinstance(pred.prediction, bool)
    assert isinstance(pred.risk_score, float)
    assert 0.0 <= pred.risk_score <= 1.0
    assert pred.model_version == MODEL_VERSION
    assert isinstance(pred.explanation, str) and len(pred.explanation) > 0
    assert isinstance(pred.feature_contributions, dict)
    for feat in FEATURE_NAMES:
        assert feat in pred.feature_contributions


# ---------------------------------------------------------------------------
# Requirement 7: Feedback Evaluator Protocol and Schema
# ---------------------------------------------------------------------------

def test_family3_feedback_evaluator_satisfies_protocol_and_contract():
    """Requirement 7: Family 3 evaluator satisfies FeedbackEvaluator protocol and produces BlueTeamFeedback."""
    ev = SyntheticIdentityFeedbackEvaluator()
    assert isinstance(ev, FeedbackEvaluator)

    gen = SyntheticIdentityAttackGenerator(seed=42)
    det = SyntheticIdentityBlueDetector()
    event = gen.generate(round_id="f3-fb-check")
    pred = det.detect(event)

    fb = ev.evaluate(event, pred)
    assert isinstance(fb, BlueTeamFeedback)
    assert fb.feedback_id == "fb-f3-f3-fb-check"
    assert fb.round_reference == "f3-fb-check"
    assert fb.risk_score == pred.risk_score
    assert isinstance(fb.important_features, dict)
    assert isinstance(fb.explanation_data, dict)


# ---------------------------------------------------------------------------
# Requirement 8: Detected Attack Feedback Behavior
# ---------------------------------------------------------------------------

def test_family3_detected_attack_feedback_behavior():
    """Requirement 8: Detected attack produces detected=True, false_positive=False, false_negative=False."""
    gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=42)
    det = SyntheticIdentityBlueDetector(threshold=0.50)
    ev = SyntheticIdentityFeedbackEvaluator()

    event = gen.generate(round_id="f3-det-case")
    pred = det.detect(event)
    fb = ev.evaluate(event, pred)

    assert pred.prediction is True
    assert fb.detected is True
    assert fb.false_positive is False
    assert fb.false_negative is False


# ---------------------------------------------------------------------------
# Requirement 9: Missed Attack Feedback Behavior
# ---------------------------------------------------------------------------

def test_family3_missed_attack_false_negative_behavior():
    """Requirement 9: Missed attack produces detected=False, false_positive=False, false_negative=True."""
    gen = SyntheticIdentityAttackGenerator(genome=DEFAULT_ATTACK_GENOME, seed=42)
    # Set threshold high so detector misses the attack
    det = SyntheticIdentityBlueDetector(threshold=0.9999)
    ev = SyntheticIdentityFeedbackEvaluator()

    event = gen.generate(round_id="f3-miss-case")
    pred = det.detect(event)
    fb = ev.evaluate(event, pred)

    assert pred.prediction is False
    assert fb.detected is False
    assert fb.false_positive is False
    assert fb.false_negative is True


# ---------------------------------------------------------------------------
# Requirement 10: Mutation Strategy Protocol & Valid Next Genome
# ---------------------------------------------------------------------------

def test_family3_mutation_produces_valid_genome():
    """Requirement 10: MutationStrategy satisfies protocol and produces a schema-valid next genome."""
    mut = SyntheticIdentityMutationStrategy()
    assert isinstance(mut, MutationStrategy)

    fb = BlueTeamFeedback(
        feedback_id="fb-mut-test",
        round_reference="r-mut-test",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.85,
        important_features={"is_disposable_email": 0.40},
    )

    next_genome = mut.mutate(DEFAULT_ATTACK_GENOME, fb)
    assert isinstance(next_genome, dict)
    assert set(next_genome.keys()) == set(FAMILY3_GENOME_DIMENSIONS)
    validate_genome(next_genome)


# ---------------------------------------------------------------------------
# Requirement 11: Mutation Gene Bound Invariance [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_family3_mutation_keeps_genes_within_bounds():
    """Requirement 11: Mutation strictly preserves [0.0, 1.0] bounds even across extreme states."""
    mut = SyntheticIdentityMutationStrategy(detected_step=0.25, missed_step=0.25)

    # Near upper extreme
    high_genome = {dim: 0.95 for dim in FAMILY3_GENOME_DIMENSIONS}
    fb_detected = BlueTeamFeedback(
        feedback_id="fb-high",
        round_reference="r-high",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.95,
        important_features={"is_disposable_email": 0.90},
    )
    mutated_high = mut.mutate(high_genome, fb_detected)
    for dim, val in mutated_high.items():
        assert 0.0 <= val <= 1.0

    # Near lower extreme
    low_genome = {dim: 0.05 for dim in FAMILY3_GENOME_DIMENSIONS}
    fb_missed = BlueTeamFeedback(
        feedback_id="fb-low",
        round_reference="r-low",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.10,
        important_features={},
    )
    mutated_low = mut.mutate(low_genome, fb_missed)
    for dim, val in mutated_low.items():
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Requirement 12: Mutation Determinism
# ---------------------------------------------------------------------------

def test_family3_mutation_is_deterministic():
    """Requirement 12: Mutation produces identical output genomes for identical inputs."""
    mut = SyntheticIdentityMutationStrategy()
    fb = BlueTeamFeedback(
        feedback_id="fb-det-check",
        round_reference="r-det-check",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.80,
        important_features={"is_emulator_device": 0.50, "early_bust_out_risk": 0.30},
    )

    out1 = mut.mutate(DEFAULT_ATTACK_GENOME, fb)
    out2 = mut.mutate(DEFAULT_ATTACK_GENOME, fb)
    out3 = mut.mutate(DEFAULT_ATTACK_GENOME, fb)

    assert out1 == out2 == out3


# ---------------------------------------------------------------------------
# Requirement 13: Mutation Non-Blind Signal-Specific Adaptation
# ---------------------------------------------------------------------------

def test_family3_mutation_is_non_blind_and_feature_responsive():
    """
    Requirement 13: Mutation does NOT blindly increase all dimensions uniformly.
    Prioritizes highest SHAP signals on detection and uses directional exploration on missed rounds.
    """
    mut = SyntheticIdentityMutationStrategy(detected_step=0.08, missed_step=0.03)

    # 1. Detected round with isolated contact detection signal
    fb_contact = BlueTeamFeedback(
        feedback_id="fb-feat",
        round_reference="r-feat",
        detected=True,
        false_positive=False,
        false_negative=False,
        risk_score=0.85,
        important_features={"is_disposable_email": 0.60},
    )
    mut_detected = mut.mutate(DEFAULT_ATTACK_GENOME, fb_contact)

    # Contact consistency should have received larger step than unflagged dimensions
    contact_delta = mut_detected["contact_consistency"] - DEFAULT_ATTACK_GENOME["contact_consistency"]
    cross_field_delta = mut_detected["cross_field_consistency"] - DEFAULT_ATTACK_GENOME["cross_field_consistency"]
    assert contact_delta > cross_field_delta
    assert pytest.approx(contact_delta, abs=1e-3) == (DEFAULT_DETECTED_STEP * 1.5)
    assert pytest.approx(cross_field_delta, abs=1e-3) == (DEFAULT_DETECTED_STEP * 0.5)

    # 2. Missed round with non-uniform directional exploration
    fb_missed = BlueTeamFeedback(
        feedback_id="fb-miss",
        round_reference="r-miss",
        detected=False,
        false_positive=False,
        false_negative=True,
        risk_score=0.15,
        important_features={},
    )
    mut_missed = mut.mutate(DEFAULT_LEGITIMATE_GENOME, fb_missed)
    deltas = {k: mut_missed[k] - DEFAULT_LEGITIMATE_GENOME[k] for k in FAMILY3_GENOME_DIMENSIONS}
    unique_deltas = set(round(d, 4) for d in deltas.values())
    assert len(unique_deltas) > 1, "Missed mutation must not apply a uniform delta to all dimensions"


# ---------------------------------------------------------------------------
# Requirement 14: Unmodified RoundController Integration
# ---------------------------------------------------------------------------

def test_family3_executes_through_unmodified_round_controller():
    """Requirement 14: Family 3 components execute through unmodified RoundController."""
    gen = SyntheticIdentityAttackGenerator(seed=123)
    det = SyntheticIdentityBlueDetector()
    ev = SyntheticIdentityFeedbackEvaluator()

    controller = RoundController(generator=gen, detector=det, evaluator=ev)
    result = controller.run_round(
        round_id="f3-rc-test-1",
        outcome_metrics={"round_index": 1, "phase": "integration_test"},
    )

    assert isinstance(result, RoundResult)
    assert result.round_id == "f3-rc-test-1"
    assert result.attack_event.attack_family == AttackFamily.SYNTHETIC_IDENTITY
    assert result.prediction_result.model_version == MODEL_VERSION
    assert result.feedback.round_reference == "f3-rc-test-1"
    assert result.outcome_metrics == {"round_index": 1, "phase": "integration_test"}


# ---------------------------------------------------------------------------
# Requirement 15: Baseline Training vs Held-Out Evaluation Dataset Separation
# ---------------------------------------------------------------------------

def test_family3_training_and_heldout_dataset_strict_separation():
    """
    Requirement 15: Baseline training data (data/legitimate/) and held-out evaluation
    data (data/held_out/) have strictly zero identity overlap and detector training
    never reads the held-out dataset.
    """
    baseline_path = Path("data/legitimate/baseline_identities.json")
    heldout_path = Path("data/held_out/heldout_identities.json")

    assert baseline_path.exists(), "Baseline dataset missing"
    assert heldout_path.exists(), "Held-out evaluation dataset missing"

    baseline_data = load_dataset(baseline_path)
    heldout_data = load_dataset(heldout_path)

    baseline_ids = {rec.identity_id for rec in baseline_data}
    heldout_ids = {rec.identity_id for rec in heldout_data}

    # Primary invariant: strictly zero identity overlap
    overlap = baseline_ids.intersection(heldout_ids)
    assert len(overlap) == 0, f"Identity overlap detected: {overlap}"

    # Detector training isolation check: ensure training never accesses heldout file
    opened_files = []
    real_open = open

    def tracking_open(file, *args, **kwargs):
        opened_files.append(str(file))
        return real_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=tracking_open):
        _ = SyntheticIdentityBlueDetector()

    heldout_accessed = [f for f in opened_files if "heldout_identities.json" in f]
    assert len(heldout_accessed) == 0, f"Held-out dataset accessed during training: {heldout_accessed}"


# ---------------------------------------------------------------------------
# Requirement 16: Family 3 Independence from Family 1 and Family 2
# ---------------------------------------------------------------------------

def test_family3_independence_from_other_families():
    """Requirement 16: Family 3 modules do not import or depend on Family 1 or Family 2 implementations."""
    import sys

    # Inspect imported modules in Family 3 packages
    f3_modules = [
        "attacks.synthetic_identity.generator",
        "attacks.synthetic_identity.mutator",
        "blue_team.synthetic_identity.detector",
        "blue_team.synthetic_identity.evaluator",
        "blue_team.synthetic_identity.feature_extractor",
        "data.generators.identity_generator",
    ]

    for mod_name in f3_modules:
        assert mod_name in sys.modules, f"{mod_name} should be loaded"
        mod = sys.modules[mod_name]
        mod_file = getattr(mod, "__file__", "")
        if mod_file and Path(mod_file).exists():
            content = Path(mod_file).read_text(encoding="utf-8")
            assert "ai_agent" not in content, f"{mod_name} references Family 2 (ai_agent)"
            assert "transaction_evasion" not in content, f"{mod_name} references Family 1 (transaction_evasion)"
