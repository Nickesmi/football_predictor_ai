from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from datetime import datetime
import json
import logging

logger = logging.getLogger("football_predictor")

class OddsProvider(ABC):
    """
    Interface for live odds streaming layer.
    Agnostic to the actual bookmaker/provider.
    """

    @abstractmethod
    def fetch_events(self, sport_key: str) -> List[Dict]:
        """Fetch list of live or upcoming events."""
        pass

    @abstractmethod
    def fetch_odds(self, event_id: str, markets: str = None) -> List[Dict]:
        """Fetch odds for a specific event."""
        pass

    @abstractmethod
    def normalize_market(self, raw_market: str) -> Optional[str]:
        """
        Normalize bookmaker-specific market names to internal model market names.
        Example: "h2h" -> "1X2", "totals" -> "Over/Under", etc.
        """
        pass

    @abstractmethod
    def get_last_update(self) -> Optional[datetime]:
        """Get timestamp of the last successful odds fetch."""
        pass
