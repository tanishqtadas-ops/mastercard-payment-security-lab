"""
blue_team/synthetic_identity/detector.py — Family 3 (Synthetic Identity) Blue-Team Detector.

Detects synthetic identities and account lifecycle fraud using an XGBoost model
trained on legitimate identity baseline data and synthetic identity attack patterns,
accompanied by SHAP feature explanations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import xgboost as xgb
import shap

from schemas.attack import AttackEvent
from schemas.prediction import PredictionResult
from data.generators.identity_generator import load_dataset, LegitimateIdentityGenerator
from attacks.synthetic_identity.generator import SyntheticIdentityAttackGenerator
from .feature_extractor import FEATURE_NAMES, extract_identity_features


DEFAULT_DETECTION_THRESHOLD: float = 0.50
MODEL_VERSION: str = "family3-xgb-v1"
DEFAULT_BASELINE_PATH = Path("data/legitimate/baseline_identities.json")


def _train_default_model(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    n_attack_samples: int = 500,
    training_seed: int = 101,
) -> xgb.XGBClassifier:
    """
    Trains an XGBoost detector on legitimate baseline training data (label=0)
    and synthetic attack variants (label=1).

    NEVER accesses the held-out evaluation dataset (data/held_out/).
    """
    feature_rows: List[List[float]] = []
    labels: List[int] = []

    # 1. Legitimate Baseline Training Samples (Label 0)
    if baseline_path.exists():
        legit_identities = load_dataset(baseline_path)
    else:
        # Fallback to generating legitimate training baseline with seed 42
        gen_legit = LegitimateIdentityGenerator(seed=42)
        legit_identities = gen_legit.generate_dataset(n=n_attack_samples)

    for ident in legit_identities:
        feat_dict = extract_identity_features(ident)
        feature_rows.append([feat_dict[k] for k in FEATURE_NAMES])
        labels.append(0)

    # 2. Synthetic Attack Training Samples (Label 1)
    gen_attack = SyntheticIdentityAttackGenerator(seed=training_seed)
    for i in range(n_attack_samples):
        event = gen_attack.generate(round_id=f"train-atk-{i}")
        feat_dict = extract_identity_features(event.scenario)
        feature_rows.append([feat_dict[k] for k in FEATURE_NAMES])
        labels.append(1)

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    model = xgb.XGBClassifier(
        n_estimators=35,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


class SyntheticIdentityBlueDetector:
    """
    Evaluates Family 3 AttackEvents and computes synthetic identity fraud predictions.

    Uses an XGBoost classifier observing structured scenario features (demographics,
    contacts, account metadata, device context, and lifecycle history) with SHAP explanations.

    Satisfies the BlueTeamDetector protocol in simulation.interfaces.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        model_version: str = MODEL_VERSION,
        model: Optional[Any] = None,
        baseline_data_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the Family 3 detector.

        Args:
            threshold: Fraud decision threshold in [0.0, 1.0].
            model_version: Identifier string for model versioning.
            model: Optional pre-fitted classifier model.
            baseline_data_path: Optional path to legitimate training baseline.
        """
        self.threshold = threshold
        self.model_version = model_version

        if model is not None:
            self._model = model
        else:
            b_path = Path(baseline_data_path) if baseline_data_path else DEFAULT_BASELINE_PATH
            self._model = _train_default_model(baseline_path=b_path)

        try:
            self._explainer = shap.TreeExplainer(self._model)
        except Exception:
            self._explainer = None

    def detect(self, event: AttackEvent) -> PredictionResult:
        """
        Run detection on an AttackEvent and return a PredictionResult.

        Args:
            event: AttackEvent containing Family 3 scenario data.

        Returns:
            PredictionResult containing boolean prediction, risk score, explanation,
            and feature contributions.
        """
        # Extract features solely from observable scenario, never from ground_truth
        scenario = event.scenario
        feature_dict = extract_identity_features(scenario)
        feature_vector = np.array([[feature_dict[k] for k in FEATURE_NAMES]], dtype=np.float32)

        # Predict probability
        proba = self._model.predict_proba(feature_vector)[0, 1]
        risk_score = float(min(max(proba, 0.0), 1.0))
        risk_score = round(risk_score, 4)

        is_fraud = bool(risk_score >= self.threshold)

        # Compute SHAP feature contributions
        feature_contributions: Dict[str, float] = {}
        if self._explainer is not None:
            try:
                shap_vals = self._explainer.shap_values(feature_vector)[0]
                feature_contributions = {
                    FEATURE_NAMES[i]: round(float(shap_vals[i]), 4)
                    for i in range(len(FEATURE_NAMES))
                }
            except Exception:
                feature_contributions = {k: round(v, 4) for k, v in feature_dict.items()}
        else:
            feature_contributions = {k: round(v, 4) for k, v in feature_dict.items()}

        # Generate human-readable explanation
        if is_fraud:
            top_drivers = sorted(
                feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            key_signals = [
                f"{k} (impact: {v:+.3f})"
                for k, v in top_drivers
                if v > 0.05 or (v > 0 and feature_dict.get(k, 0) > 0.5)
            ]
            signals_str = ", ".join(key_signals[:3]) if key_signals else "synthetic identity anomaly signals"
            explanation = (
                f"Flagged synthetic identity risk (risk {risk_score:.2f} >= threshold {self.threshold:.2f}). "
                f"Primary drivers: {signals_str}."
            )
        else:
            explanation = (
                f"Identity profile verified as legitimate and consistent "
                f"(risk {risk_score:.2f} < threshold {self.threshold:.2f})."
            )

        return PredictionResult(
            prediction_id=f"pred-f3-{event.round_id}",
            prediction=is_fraud,
            risk_score=risk_score,
            model_version=self.model_version,
            explanation=explanation,
            feature_contributions=feature_contributions,
        )
