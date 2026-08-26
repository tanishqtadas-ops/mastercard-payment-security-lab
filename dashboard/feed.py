"""
dashboard/feed.py — State management and ingestion feed for dashboard presentation.

Maintains an ordered history of simulation rounds and provides query methods
for dashboard UI/view components.
"""

from typing import List, Optional, Sequence, Union

from schemas import AttackFamily, RoundResult
from .presenter import RoundDisplayData, extract_display_data


class DashboardFeed:
    """
    In-memory feed and session store for dashboard round data.

    Ingests RoundResult objects from simulation runs, converts them to
    presentation-ready RoundDisplayData models, and provides filtering
    and query interfaces.
    """

    def __init__(self) -> None:
        self._history: List[RoundDisplayData] = []

    def ingest(self, result: Union[RoundResult, RoundDisplayData]) -> RoundDisplayData:
        """
        Ingest a single round result into the dashboard feed.

        Args:
            result: RoundResult or RoundDisplayData to ingest.

        Returns:
            The ingested RoundDisplayData.
        """
        if isinstance(result, RoundResult):
            display_data = extract_display_data(result)
        else:
            display_data = result

        self._history.append(display_data)
        return display_data

    def ingest_many(
        self,
        results: Sequence[Union[RoundResult, RoundDisplayData]],
    ) -> List[RoundDisplayData]:
        """
        Ingest multiple round results in order.

        Args:
            results: Sequence of round results or display data objects.

        Returns:
            List of ingested RoundDisplayData instances.
        """
        ingested: List[RoundDisplayData] = []
        for res in results:
            ingested.append(self.ingest(res))
        return ingested

    def get_rounds(self) -> List[RoundDisplayData]:
        """Return all ingested round records in chronological order."""
        return list(self._history)

    def get_latest_round(self) -> Optional[RoundDisplayData]:
        """Return the most recently ingested round, or None if empty."""
        return self._history[-1] if self._history else None

    def get_rounds_by_family(
        self,
        family: Union[AttackFamily, str],
    ) -> List[RoundDisplayData]:
        """
        Filter ingested rounds by attack family.

        Args:
            family: AttackFamily enum or matching string.

        Returns:
            List of RoundDisplayData instances matching the family.
        """
        target_name = family.value if isinstance(family, AttackFamily) else str(family)
        return [
            round_data for round_data in self._history
            if round_data.family == target_name or target_name in round_data.family
        ]

    def clear(self) -> None:
        """Reset and clear all history from the feed."""
        self._history.clear()

    @property
    def round_count(self) -> int:
        """Total number of ingested rounds."""
        return len(self._history)
