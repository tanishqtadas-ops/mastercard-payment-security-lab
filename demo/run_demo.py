"""
demo/run_demo.py — Deterministic Arms-Race Demo Runner & Dashboard CLI.

Executes a precomputed, deterministic, and fully reproducible demonstration of the
Mastercard Payment Security Lab's adversarial arms race across all three attack families:
1. Family 1: Adaptive Transaction-Pattern Evasion (multi-round evasion, detection, learning, update)
2. Family 2: AI-Agent Authorization & Payment Behavior
3. Family 3: Synthetic Identity Lifecycle & Supervised XGBoost Retraining

Reuses existing core components (RoundController, RetrainingController, UnifiedEvaluator,
PaymentSecurityDashboard) without duplication or network/LLM dependencies.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure repository root is on sys.path so both `python demo/run_demo.py`
# and `python -m demo.run_demo` work seamlessly.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pydantic import BaseModel, Field

from schemas.common import AttackFamily
from schemas.round import RoundResult
from schemas.transaction import Transaction
from schemas.identity import SyntheticIdentity

from simulation import RoundController

# Family 1 components
from attacks.transaction_evasion import (
    TransactionAttackGenerator,
    TransactionMutationStrategy,
)
from blue_team.transaction import (
    TransactionBlueDetector,
    TransactionFeedbackEvaluator,
)

# Family 2 components
from attacks.ai_agent import (
    AIAgentAttackGenerator,
    AIAgentMutationStrategy,
)
from blue_team.ai_agent import (
    AIAgentBlueDetector,
    AIAgentFeedbackEvaluator,
)

# Family 3 components
from attacks.synthetic_identity import (
    SyntheticIdentityAttackGenerator,
    SyntheticIdentityMutationStrategy,
)
from blue_team.synthetic_identity import (
    SyntheticIdentityBlueDetector,
    SyntheticIdentityFeedbackEvaluator,
)
from data.generators.identity_generator import load_dataset

# Learning components
from blue_team.learning.retraining import (
    RetrainingController,
    ModelUpdateRecord,
)

# Evaluation components
from evaluation import (
    UnifiedEvaluator,
    UnifiedEvaluationReport,
    HoldoutEvaluationResult,
)

# Dashboard components
from dashboard.controller import (
    PaymentSecurityDashboard,
    DashboardState,
)


class DemoConfig(BaseModel):
    """Configuration options for deterministic demo execution."""

    seed: int = 42
    family1_rounds: int = Field(default=4, ge=1)
    family2_rounds: int = Field(default=2, ge=1)
    family3_rounds: int = Field(default=2, ge=1)
    retrain_interval: int = Field(default=2, ge=1)
    auto_load_canonical_baseline: bool = True
    quiet: bool = False


class DemoRunResult(BaseModel):
    """Complete structured result produced by the demo run."""

    config: DemoConfig
    family1_results: List[RoundResult] = Field(default_factory=list)
    family2_results: List[RoundResult] = Field(default_factory=list)
    family3_results: List[RoundResult] = Field(default_factory=list)
    all_results: List[RoundResult] = Field(default_factory=list)
    update_records: List[ModelUpdateRecord] = Field(default_factory=list)
    evaluation_report: UnifiedEvaluationReport
    clean_eval_results: Dict[str, HoldoutEvaluationResult] = Field(default_factory=dict)
    dashboard_state: DashboardState
    post_learning_recovery_observed: bool = False
    rendered_output: str = ""


def _generate_deterministic_f1_baseline(n: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a deterministic set of legitimate baseline transactions for Family 1 training."""
    gen = TransactionAttackGenerator(ground_truth=False, seed=seed)
    baseline_transactions: List[Dict[str, Any]] = []
    for i in range(n):
        event = gen.generate(round_id=f"base-f1-{i}")
        tx_data = event.scenario.get("transaction", event.scenario)
        baseline_transactions.append({"transaction": tx_data, "user_id": tx_data.get("user_id", f"usr_{i}")})
    return baseline_transactions


def _generate_deterministic_f2_clean(n: int = 5, seed: int = 142) -> List[Dict[str, Any]]:
    """Generate a deterministic set of clean authorized AI-agent payment events for evaluation."""
    gen = AIAgentAttackGenerator(ground_truth=False, seed=seed)
    clean_events: List[Dict[str, Any]] = []
    for i in range(n):
        event = gen.generate(round_id=f"clean-f2-{i}")
        clean_events.append(event.scenario)
    return clean_events


class DemoRunner:
    """
    Orchestrates the deterministic adversarial arms-race demonstration.
    """

    def __init__(self, config: Optional[DemoConfig] = None) -> None:
        self.config = config or DemoConfig()

    def run(self) -> DemoRunResult:
        """
        Execute the full multi-family demonstration workflow deterministically.
        """
        lines: List[str] = []

        def log(msg: str = "") -> None:
            lines.append(msg)
            if not self.config.quiet:
                try:
                    print(msg)
                except UnicodeEncodeError:
                    print(msg.encode("ascii", "replace").decode("ascii"))

        log("=" * 72)
        log(" MASTERCARD PAYMENT SECURITY LAB -- ADAPTIVE ARMS RACE DEMO")
        log("=" * 72)
        log(f"Deterministic seed: {self.config.seed} | Retrain interval: {self.config.retrain_interval} rounds")
        log()

        # Generate legitimate baseline data for Family 1 retraining
        f1_baseline_data = _generate_deterministic_f1_baseline(n=10, seed=self.config.seed)

        # Initialize shared Blue-Team Detectors
        det_f1 = TransactionBlueDetector()
        det_f2 = AIAgentBlueDetector()
        det_f3 = SyntheticIdentityBlueDetector()

        # Initialize RetrainingController with live detector instances and baseline data
        retrain_ctrl = RetrainingController(
            detectors={
                AttackFamily.ADAPTIVE_EVASION: det_f1,
                AttackFamily.AGENT_BEHAVIOR: det_f2,
                AttackFamily.SYNTHETIC_IDENTITY: det_f3,
            },
            baseline_data={
                AttackFamily.ADAPTIVE_EVASION: f1_baseline_data,
            },
            retrain_interval=self.config.retrain_interval,
            auto_load_canonical_baseline=self.config.auto_load_canonical_baseline,
        )

        all_results: List[RoundResult] = []
        update_records: List[ModelUpdateRecord] = []

        # ===================================================================
        # SECTION 1: Family 1 Adaptive Arms Race
        # ===================================================================
        log("------------------------------------------------------------")
        log(" 1. FAMILY 1: ADAPTIVE TRANSACTION-PATTERN EVASION")
        log("------------------------------------------------------------")

        gen_f1 = TransactionAttackGenerator(
            genome={
                "amount_deviation": 0.35,
                "velocity_deviation": 0.40,
                "device_novelty": 0.20,
                "location_deviation": 0.30,
                "time_deviation": 0.25,
                "sequence_anomaly": 0.30,
            },
            seed=self.config.seed,
        )
        ev_f1 = TransactionFeedbackEvaluator()
        mut_f1 = TransactionMutationStrategy()
        round_ctrl_f1 = RoundController(generator=gen_f1, detector=det_f1, evaluator=ev_f1)

        f1_results: List[RoundResult] = []
        current_genome_f1 = None
        f1_updates: List[ModelUpdateRecord] = []

        for r_idx in range(1, self.config.family1_rounds + 1):
            round_id = f"f1-round-{r_idx}"
            if current_genome_f1 is not None:
                gen_f1.set_genome(current_genome_f1)

            res = round_ctrl_f1.run_round(
                round_id=round_id,
                outcome_metrics={"round_index": r_idx},
            )
            f1_results.append(res)
            all_results.append(res)

            status_str = "DETECTED [FRAUD]" if res.feedback.detected else "MISSED (FALSE NEGATIVE)"
            log(f" Round {r_idx}: {res.attack_event.attack_id}")
            log(f"   Risk Score: {res.prediction_result.risk_score:.4f} | Outcome: {status_str}")
            log(f"   Model Version: {res.prediction_result.model_version}")

            # Blue-Team learning hook
            update = retrain_ctrl.on_round_completed(res, round_index=r_idx)
            if update is not None and update.retrained:
                update_records.append(update)
                f1_updates.append(update)
                res.outcome_metrics["model_updated"] = True
                res.outcome_metrics["new_model_version"] = update.new_model_version
                res.outcome_metrics["previous_model_version"] = update.previous_model_version

                fam_str = update.family.value if isinstance(update.family, AttackFamily) else str(update.family)
                log()
                log("   >>> [BLUE TEAM MODEL RETRAINING UPDATE] <<<")
                log(f"   Family: {fam_str}")
                log(f"   Previous Model: {update.previous_model_version} -> Updated Model: {update.new_model_version}")
                log(f"   False Negatives Used: {update.false_negative_count} | Legitimate Baseline Samples: {update.baseline_count}")
                log(f"   Total Training Samples: {update.training_sample_count}")
                log()

            # Red-Team mutation for next round
            current_genome_f1 = mut_f1.mutate(
                dict(res.attack_event.attack_genome),
                res.feedback,
            )

        # Truthful Recovery Analysis for Family 1:
        f1_post_learning_recovery = False
        if len(f1_updates) > 0 and len(f1_results) > 2:
            first_upd_idx = f1_updates[0].round_index
            post_upd_rounds = f1_results[first_upd_idx:]
            if any(r.feedback.detected for r in post_upd_rounds):
                f1_post_learning_recovery = True

        log(f" Post-Learning Detection Recovery: {'OBSERVED' if f1_post_learning_recovery else 'NOT OBSERVED (Red-Team evolved stealthier evasion variant)'}")
        log()

        # ===================================================================
        # SECTION 2: Family 2 AI-Agent Payment Behavior
        # ===================================================================
        log("------------------------------------------------------------")
        log(" 2. FAMILY 2: AI-AGENT PAYMENT BEHAVIOR (MANDATE ADHERENCE)")
        log("------------------------------------------------------------")

        gen_f2 = AIAgentAttackGenerator(seed=self.config.seed + 1)
        ev_f2 = AIAgentFeedbackEvaluator()
        mut_f2 = AIAgentMutationStrategy()
        round_ctrl_f2 = RoundController(generator=gen_f2, detector=det_f2, evaluator=ev_f2)

        f2_results: List[RoundResult] = []
        current_genome_f2 = None

        for r_idx in range(1, self.config.family2_rounds + 1):
            round_id = f"f2-round-{r_idx}"
            if current_genome_f2 is not None:
                gen_f2.set_genome(current_genome_f2)

            res = round_ctrl_f2.run_round(
                round_id=round_id,
                outcome_metrics={"round_index": r_idx},
            )
            f2_results.append(res)
            all_results.append(res)

            status_str = "DETECTED [SCOPE BREACH]" if res.feedback.detected else "MISSED"
            log(f" Round {r_idx}: {res.attack_event.attack_id}")
            log(f"   Risk Score: {res.prediction_result.risk_score:.4f} | Outcome: {status_str}")
            log(f"   Model Version: {res.prediction_result.model_version}")

            current_genome_f2 = mut_f2.mutate(
                dict(res.attack_event.attack_genome),
                res.feedback,
            )

        log()

        # ===================================================================
        # SECTION 3: Family 3 Synthetic Identity & XGBoost
        # ===================================================================
        log("------------------------------------------------------------")
        log(" 3. FAMILY 3: SYNTHETIC IDENTITY & XGBOOST MODEL")
        log("------------------------------------------------------------")

        gen_f3 = SyntheticIdentityAttackGenerator(seed=self.config.seed + 2)
        ev_f3 = SyntheticIdentityFeedbackEvaluator()
        mut_f3 = SyntheticIdentityMutationStrategy()
        round_ctrl_f3 = RoundController(generator=gen_f3, detector=det_f3, evaluator=ev_f3)

        f3_results: List[RoundResult] = []
        current_genome_f3 = None

        for r_idx in range(1, self.config.family3_rounds + 1):
            round_id = f"f3-round-{r_idx}"
            if current_genome_f3 is not None:
                gen_f3.set_genome(current_genome_f3)

            res = round_ctrl_f3.run_round(
                round_id=round_id,
                outcome_metrics={"round_index": r_idx},
            )
            f3_results.append(res)
            all_results.append(res)

            status_str = "DETECTED [SYNTHETIC IDENTITY]" if res.feedback.detected else "MISSED"
            log(f" Round {r_idx}: {res.attack_event.attack_id}")
            log(f"   Risk Score: {res.prediction_result.risk_score:.4f} | Outcome: {status_str}")
            log(f"   Model Version: {res.prediction_result.model_version}")

            update_f3 = retrain_ctrl.on_round_completed(res, round_index=r_idx)
            if update_f3 is not None and update_f3.retrained:
                update_records.append(update_f3)
                res.outcome_metrics["model_updated"] = True
                res.outcome_metrics["new_model_version"] = update_f3.new_model_version
                res.outcome_metrics["previous_model_version"] = update_f3.previous_model_version
                log(f"   >>> Blue Team Refitted XGBoost: {update_f3.new_model_version} <<<")

            current_genome_f3 = mut_f3.mutate(
                dict(res.attack_event.attack_genome),
                res.feedback,
            )

        log()

        # ===================================================================
        # SECTION 4: Live Simulation Round Evaluation
        # ===================================================================
        log("=" * 72)
        log(" LIVE SIMULATION ROUNDS EVALUATION (ATTACK-FOCUSED)")
        log("=" * 72)
        log(" Note: Live simulation rounds evaluate adversarial attack adaptation.")
        log(" Clean legitimate pass-rate & false-positive control are evaluated separately below.")
        log()

        evaluator = UnifiedEvaluator()
        eval_report = evaluator.evaluate_round_results(
            results=all_results,
            evaluation_id=f"demo-eval-seed-{self.config.seed}",
            update_records=update_records,
            metadata={"seed": self.config.seed, "demo_mode": "precomputed_deterministic"},
        )

        for fam_name, fam_res in eval_report.per_family_results.items():
            log(f" Family: {fam_name}")
            log(f"   Evaluated Rounds: {fam_res.sample_count}")
            log(f"   Confusion Matrix: TP={fam_res.confusion_matrix.true_positives} | FN={fam_res.confusion_matrix.false_negatives} | TN={fam_res.confusion_matrix.true_negatives} | FP={fam_res.confusion_matrix.false_positives}")
            det_rate_str = f"{fam_res.metrics.recall * 100:.1f}%" if fam_res.metrics.recall is not None else "N/A"
            acc_str = f"{fam_res.metrics.accuracy * 100:.1f}%"
            log(f"   Detection Rate: {det_rate_str} | Accuracy: {acc_str}")
            if fam_res.risk_metrics:
                log(f"   Average Risk Score: {fam_res.risk_metrics.average_risk:.4f}")
            log()

        if eval_report.consolidated_metrics:
            cm = eval_report.consolidated_metrics
            log(" Consolidated Live Attack Round Metrics:")
            log(f"   Total Rounds: {cm.total_samples} (Attacks: {cm.total_attacks}, Legit: {cm.total_legitimate})")
            log(f"   Overall Accuracy: {cm.overall_accuracy * 100:.1f}%")
            if cm.overall_detection_rate is not None:
                log(f"   Overall Detection Rate: {cm.overall_detection_rate * 100:.1f}%")
            if cm.average_risk is not None:
                log(f"   Mean Risk Score: {cm.average_risk:.4f}")

        log()

        # ===================================================================
        # SECTION 5: Clean Generalization & Held-Out Evaluation
        # ===================================================================
        log("=" * 72)
        log(" CLEAN GENERALIZATION & HELD-OUT EVALUATION")
        log("=" * 72)
        log(" Strict evaluation safety: Zero evaluation samples entered the training set.")
        log()

        # Prepare clean evaluation data with clear provenance
        clean_f1 = _generate_deterministic_f1_baseline(n=10, seed=self.config.seed + 100)
        clean_f2 = _generate_deterministic_f2_clean(n=5, seed=self.config.seed + 200)

        heldout_identities_path = Path(_ROOT) / "data" / "held_out" / "heldout_identities.json"
        if heldout_identities_path.exists():
            held_out_f3 = load_dataset(str(heldout_identities_path))
        else:
            held_out_f3 = []

        eval_data_map = {
            AttackFamily.ADAPTIVE_EVASION: clean_f1,
            AttackFamily.AGENT_BEHAVIOR: clean_f2,
            AttackFamily.SYNTHETIC_IDENTITY: held_out_f3,
        }

        clean_eval_report = evaluator.evaluate_held_out(
            detectors={
                AttackFamily.ADAPTIVE_EVASION: det_f1,
                AttackFamily.AGENT_BEHAVIOR: det_f2,
                AttackFamily.SYNTHETIC_IDENTITY: det_f3,
            },
            held_out_data=eval_data_map,
        )

        provenance_labels = {
            AttackFamily.ADAPTIVE_EVASION.name: "Clean Baseline Generalization (Generated Baseline Profiles)",
            AttackFamily.AGENT_BEHAVIOR.name: "Clean Baseline Generalization (Generated Authorized Events)",
            AttackFamily.SYNTHETIC_IDENTITY.name: "Isolated Held-Out Generalization (data/held_out/heldout_identities.json)",
        }

        for fam_k, h_res in clean_eval_report.items():
            prov_label = provenance_labels.get(fam_k, "Clean Generalization")
            log(f" Family: {fam_k}")
            log(f"   Provenance: {prov_label}")
            log(f"   Clean Evaluation Samples: {h_res.sample_count}")
            log(f"   Clean Pass Rate: {h_res.clean_pass_rate * 100:.1f}% | False Alarms (FP): {h_res.false_positive_count}")
            log(f"   False-Positive Rate: {h_res.false_positive_rate * 100:.1f}%")
            log()

        # ===================================================================
        # SECTION 6: Dashboard Presentation & Arms Race Summary
        # ===================================================================
        log("=" * 72)
        log(" DASHBOARD & ARMS-RACE SUMMARY")
        log("=" * 72)

        dashboard = PaymentSecurityDashboard()
        dashboard.ingest_many(all_results)
        dash_state = dashboard.get_state()

        log(f" Total Simulation Rounds Tracked: {dash_state.total_rounds}")
        log(f" Active Family: {dash_state.current_family}")
        if dash_state.latest_round:
            log(f" Latest Risk Score: {dash_state.latest_round.risk_score:.4f} ({dash_state.latest_round.status})")

        arms_summary = dash_state.arms_race_summary
        log(f" Live Attack Detection Rate: {arms_summary.overall_detection_rate * 100:.1f}%")
        log(f" Average Attack Sophistication / Difficulty: {arms_summary.average_attack_difficulty:.4f}")
        log(f" Model Retraining Updates Recorded: {arms_summary.model_update_count}")
        log(f" Post-Learning Recovery: {'OBSERVED' if f1_post_learning_recovery else 'NOT OBSERVED'}")

        log("=" * 72)
        log(" DEMO RUN COMPLETE (All operations deterministic and offline)")
        log("=" * 72)

        rendered_text = "\n".join(lines)

        return DemoRunResult(
            config=self.config,
            family1_results=f1_results,
            family2_results=f2_results,
            family3_results=f3_results,
            all_results=all_results,
            update_records=update_records,
            evaluation_report=eval_report,
            clean_eval_results=clean_eval_report,
            dashboard_state=dash_state,
            post_learning_recovery_observed=f1_post_learning_recovery,
            rendered_output=rendered_text,
        )


def run_demo(
    seed: int = 42,
    family1_rounds: int = 4,
    family2_rounds: int = 2,
    family3_rounds: int = 2,
    retrain_interval: int = 2,
    quiet: bool = False,
) -> DemoRunResult:
    """Convenience functional interface to execute the deterministic demo."""
    config = DemoConfig(
        seed=seed,
        family1_rounds=family1_rounds,
        family2_rounds=family2_rounds,
        family3_rounds=family3_rounds,
        retrain_interval=retrain_interval,
        quiet=quiet,
    )
    runner = DemoRunner(config=config)
    return runner.run()


def serialize_deterministic_demo_json(result: DemoRunResult) -> str:
    """Serialize the demo result to JSON, stripping wall-clock timestamps to guarantee byte-for-byte determinism."""
    payload = {
        "config": result.config.model_dump(),
        "total_rounds": len(result.all_results),
        "model_updates": len(result.update_records),
        "post_learning_recovery_observed": result.post_learning_recovery_observed,
        "live_evaluation": result.evaluation_report.model_dump(),
        "clean_evaluation": {k: v.model_dump() for k, v in result.clean_eval_results.items()},
        "dashboard_summary": result.dashboard_state.arms_race_summary.model_dump(),
        "family1_results": [r.model_dump() for r in result.family1_results],
        "family2_results": [r.model_dump() for r in result.family2_results],
        "family3_results": [r.model_dump() for r in result.family3_results],
        "update_records": [u.model_dump() for u in result.update_records],
    }

    def _strip_timestamps(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _strip_timestamps(v) for k, v in obj.items() if k != "timestamp"}
        elif isinstance(obj, list):
            return [_strip_timestamps(v) for v in obj]
        return obj

    sanitized_payload = _strip_timestamps(payload)
    return json.dumps(sanitized_payload, indent=2, default=str)


def main() -> int:
    """CLI Entrypoint for judges and evaluators."""
    parser = argparse.ArgumentParser(
        description="Mastercard Payment Security Lab — Precomputed Deterministic Arms-Race Demo"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--f1-rounds", type=int, default=4, help="Family 1 rounds (default: 4)")
    parser.add_argument("--f2-rounds", type=int, default=2, help="Family 2 rounds (default: 2)")
    parser.add_argument("--f3-rounds", type=int, default=2, help="Family 3 rounds (default: 2)")
    parser.add_argument("--retrain-interval", type=int, default=2, help="Rounds between retraining (default: 2)")
    parser.add_argument("--json", action="store_true", help="Output raw structured result as JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress formatted console output")

    args = parser.parse_args()

    config = DemoConfig(
        seed=args.seed,
        family1_rounds=args.f1_rounds,
        family2_rounds=args.f2_rounds,
        family3_rounds=args.f3_rounds,
        retrain_interval=args.retrain_interval,
        quiet=args.quiet or args.json,
    )

    runner = DemoRunner(config=config)
    result = runner.run()

    if args.json:
        print(serialize_deterministic_demo_json(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
