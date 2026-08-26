"""
dashboard — Presentation and product layer for the Mastercard Payment Security Lab.

Provides decoupled presentation data models and feed management for simulation rounds.
"""

from .presenter import (
    RoundDisplayData,
    extract_display_data,
    format_round_dict,
    format_round_summary,
)
from .feed import DashboardFeed

__all__ = [
    "RoundDisplayData",
    "extract_display_data",
    "format_round_dict",
    "format_round_summary",
    "DashboardFeed",
]
