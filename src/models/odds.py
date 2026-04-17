"""
Data models for odds snapshots and market lines.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OddsSnapshot:
    """A single odds reading for a market/selection."""
    match_id: str
    market: str           # "1X2", "O/U 2.5", "BTTS", etc.
    selection: str         # "home", "draw", "away", "over", "under", "yes", "no"
    odds: float            # decimal odds (e.g., 2.10)
    bookmaker: str = "sofascore"
    is_opening: bool = True

    @property
    def implied_prob(self) -> float:
        """Convert decimal odds to implied probability."""
        return 1.0 / self.odds if self.odds > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "market": self.market,
            "selection": self.selection,
            "odds": self.odds,
            "bookmaker": self.bookmaker,
            "is_opening": self.is_opening,
        }


@dataclass
class MarketLine:
    """A complete view of a betting market for a match."""
    match_id: str
    market: str
    selections: dict[str, float] = field(default_factory=dict)  # selection → odds

    @property
    def margin(self) -> float:
        """Bookmaker margin (overround). 0% = fair, >0 = bookie edge."""
        if not self.selections:
            return 0.0
        total_implied = sum(1.0 / o for o in self.selections.values() if o > 0)
        return round((total_implied - 1.0) * 100, 2)

    def implied_probs(self) -> dict[str, float]:
        """Selection → implied probability (raw, not adjusted for margin)."""
        return {sel: round(1.0 / odds, 4) for sel, odds in self.selections.items() if odds > 0}
