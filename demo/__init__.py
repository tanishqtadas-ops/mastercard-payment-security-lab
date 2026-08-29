"""
demo — Precomputed Deterministic Arms-Race Demo Runner.

Provides judge-runnable presentation and demonstration workflows for the Mastercard
Payment Security Lab.
"""

from .run_demo import (
    DemoConfig,
    DemoRunResult,
    DemoRunner,
    run_demo,
)

__all__ = [
    "DemoConfig",
    "DemoRunResult",
    "DemoRunner",
    "run_demo",
]
