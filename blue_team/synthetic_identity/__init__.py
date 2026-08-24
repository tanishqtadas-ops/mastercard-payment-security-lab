"""
blue_team/synthetic_identity — Family 3 (Synthetic Identity) Blue-Team module.
"""

from .detector import (
    SyntheticIdentityBlueDetector,
    DEFAULT_DETECTION_THRESHOLD,
    MODEL_VERSION,
)
from .feature_extractor import (
    FEATURE_NAMES,
    extract_identity_features,
)

__all__ = [
    "SyntheticIdentityBlueDetector",
    "DEFAULT_DETECTION_THRESHOLD",
    "MODEL_VERSION",
    "FEATURE_NAMES",
    "extract_identity_features",
]
