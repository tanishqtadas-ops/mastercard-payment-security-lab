"""
tests/test_demo.py — Test Suite for Deterministic Demo Runner (Task 8.2 Final Verification).

Covers all required demo behaviors:
1. Dashboard counts actual model update records (exactly 2 updates, matching RetrainingController).
2. Cross-family model version transitions (F1 -> F2 -> F3) do not create spurious update markers.
3. Update markers correctly identify family and triggering round.
4. Family 1 evaluation provenance is accurately labeled as Clean Baseline Generalization.
5. Family 2 evaluation provenance is accurately labeled as Clean Baseline Generalization.
6. Family 3 evaluation is properly isolated against data/held_out/heldout_identities.json.
7. Held-out/evaluation data never enters training pathways.
8. Deterministic repeated demo execution with fixed seed.
9. Existing three-family demo execution remains fully functional.
10. Truthful recovery reporting (post-learning recovery: NOT OBSERVED).
"""

import json
from typing import Any, Dict, List
import pytest
from pydantic import ValidationError

from demo import (
    DemoConfig,
    DemoRunResult,
    DemoRunner,
    run_demo,
)
from schemas.common import AttackFamily
from schemas.round import RoundResult
from evaluation import UnifiedEvaluationReport, HoldoutEvaluationResult
from dashboard.controller import DashboardState
from dashboard.arms_race import model_update_rounds
from blue_team.learning.dataset import HoldoutDataLeakageError, assemble_retraining_dataset


# ===========================================================================
# 1, 2, & 3. Dashboard Update Count & Cross-Family Boundary Isolation (Req 1, 2, 3)
# ===========================================================================

def test_dashboard_counts_exact_model_updates_without_cross_family_artifacts():
    """Requirement 1, 2, 3: Dashboard reports exactly 2 updates without cross-family false markers."""
    result = run_demo(seed=42, quiet=True)

    # 1. Exact update records from RetrainingController
    assert len(result.update_records) == 2

    # 2. Dashboard summary matches exact count
    assert result.dashboard_state.arms_race_summary.model_update_count == 2

    # 3. Model update markers match exact rounds (Round 2 and Round 4 of Family 1)
    markers = model_update_rounds(result.all_results)
    assert len(markers) == 2

    # First update: Round 2 (Family 1)
    assert markers[0].round_index == 2
    assert markers[0].previous_model_version == "heuristic-family1-v1"
    assert markers[0].new_model_version == "heuristic-family1-retrained-v1"

    # Second update: Round 4 (Family 1)
    assert markers[1].round_index == 4
    assert markers[1].previous_model_version == "heuristic-family1-retrained-v1"
    assert markers[1].new_model_version == "heuristic-family1-retrained-v2"

    # Cross-family boundary transitions (F1 -> F2 in Round 5, F2 -> F3 in Round 7) MUST NOT be in markers
    marker_indices = [m.round_index for m in markers]
    assert 5 not in marker_indices, "Round 5 (Family 2 start) must not be counted as a model update"
    assert 7 not in marker_indices, "Round 7 (Family 3 start) must not be counted as a model update"


# ===========================================================================
# 4, 5, & 6. Evaluation Provenance & True Held-Out Distinction (Req 4, 5, 6)
# ===========================================================================

def test_evaluation_provenance_and_true_held_out_distinction():
    """Requirement 4, 5, 6: Distinguish Clean Baseline Generalization from Isolated Held-Out."""
    result = run_demo(seed=42, quiet=True)

    # Check clean evaluation output and results
    assert len(result.clean_eval_results) == 3

    # Family 1: Clean Baseline Generalization
    assert AttackFamily.ADAPTIVE_EVASION.name in result.clean_eval_results
    assert "Clean Baseline Generalization (Generated Baseline Profiles)" in result.rendered_output

    # Family 2: Clean Baseline Generalization
    assert AttackFamily.AGENT_BEHAVIOR.name in result.clean_eval_results
    assert "Clean Baseline Generalization (Generated Authorized Events)" in result.rendered_output

    # Family 3: True Isolated Held-Out
    assert AttackFamily.SYNTHETIC_IDENTITY.name in result.clean_eval_results
    assert "Isolated Held-Out Generalization (data/held_out/heldout_identities.json)" in result.rendered_output
    assert result.clean_eval_results[AttackFamily.SYNTHETIC_IDENTITY.name].sample_count == 500


# ===========================================================================
# 7. Held-Out Data Never Enters Training (Req 7)
# ===========================================================================

def test_held_out_data_never_enters_training():
    """Requirement 7: Passing held-out paths to dataset assembly raises HoldoutDataLeakageError."""
    with pytest.raises(HoldoutDataLeakageError):
        assemble_retraining_dataset(
            baseline_data=["data/held_out/heldout_identities.json"],
        )


# ===========================================================================
# 8. Deterministic Repeated Demo (Req 8)
# ===========================================================================

def test_deterministic_repeated_demo():
    """Requirement 8: Multiple runs with identical seed produce identical structured results."""
    run_a = run_demo(seed=42, quiet=True)
    run_b = run_demo(seed=42, quiet=True)

    assert run_a.rendered_output == run_b.rendered_output
    assert run_a.dashboard_state.arms_race_summary.model_update_count == run_b.dashboard_state.arms_race_summary.model_update_count
    assert run_a.post_learning_recovery_observed == run_b.post_learning_recovery_observed


# ===========================================================================
# 9 & 10. Multi-Family Execution & Truthful Recovery (Req 9, 10)
# ===========================================================================

def test_multi_family_execution_and_truthful_recovery():
    """Requirement 9, 10: Runs across 3 families and truthfully reports no post-learning recovery."""
    result = run_demo(seed=42, family1_rounds=4, family2_rounds=2, family3_rounds=2, quiet=True)

    assert len(result.family1_results) == 4
    assert len(result.family2_results) == 2
    assert len(result.family3_results) == 2
    assert len(result.all_results) == 8

    # Truthful recovery reporting
    assert result.post_learning_recovery_observed is False
    assert "Post-Learning Recovery: NOT OBSERVED" in result.rendered_output
