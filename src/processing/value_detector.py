"""
MVIM Value Detection Layer.

Compares the model's Intersection Confidence (IC) against
market-implied probability from betting odds.

From the Gemini MVIM Document (Section 5 — "The Value Find"):
  If calculated IC is 80% but market odds suggest 50% chance,
  this is a "Best Choice" selection.

Formula:
  implied_probability = (1 / decimal_odds) * 100
  value_edge = IC - implied_probability
  If value_edge > 10 → "Best Choice"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.processing.factor_analyzer import IntersectionFactor


@dataclass
class ValueSelection:
    """A single market where our model detects value against the odds."""
    pattern: str
    pillar: str
    ic: float                    # Intersection Confidence (combined_percentage)
    stability: float             # Stability score
    implied_probability: float   # From market odds
    value_edge: float            # ic - implied_probability
    verdict: str                 # "Best Choice", "Value", "Fair", "No Value"
    confidence: str              # From the IntersectionFactor

    @property
    def is_best_choice(self) -> bool:
        return self.verdict == "Best Choice"

    def __repr__(self) -> str:
        icon = "🏆" if self.is_best_choice else "✅" if self.verdict == "Value" else "➖"
        return (
            f"{icon} [{self.pillar}] {self.pattern}: "
            f"IC {self.ic:.1f}% vs Market {self.implied_probability:.1f}% "
            f"→ Edge {self.value_edge:+.1f}% [{self.verdict}]"
        )


class ValueDetector:
    """
    Compare intersection confidence against market odds to find value.

    Args:
        best_choice_edge: minimum edge for Best Choice (default 10%)
        value_edge: minimum edge for Value (default 5%)
    """

    def __init__(self, best_choice_edge: float = 10.0, value_edge_min: float = 5.0):
        self.best_choice_edge = best_choice_edge
        self.value_edge_min = value_edge_min

    def detect(
        self,
        intersections: list[IntersectionFactor],
        odds_data: Optional[dict[str, float]] = None,
    ) -> list[ValueSelection]:
        """
        Detect value selections by comparing IC vs implied probability.

        Args:
            intersections: list of IntersectionFactor from FactorAnalyzer
            odds_data: dict mapping pattern label → decimal odds
                       e.g. {"BTTS - Yes": 1.80, "Over 2.5 Goals FT": 1.65}
                       If None, a heuristic baseline is used.

        Returns:
            List of ValueSelection objects, sorted by value_edge descending.
        """
        if odds_data is None:
            odds_data = self._heuristic_odds()

        selections: list[ValueSelection] = []

        for factor in intersections:
            odds = odds_data.get(factor.label)
            if odds is None:
                # Try fuzzy match for close labels
                odds = self._fuzzy_match_odds(factor.label, odds_data)

            if odds is not None and odds > 1.0:
                implied = (1.0 / odds) * 100.0
                edge = factor.combined_percentage - implied
                verdict = self._classify_edge(edge)

                selections.append(ValueSelection(
                    pattern=factor.label,
                    pillar=factor.pillar,
                    ic=factor.combined_percentage,
                    stability=factor.stability_score,
                    implied_probability=round(implied, 1),
                    value_edge=round(edge, 1),
                    verdict=verdict,
                    confidence=factor.confidence,
                ))

        # Sort best value first
        selections.sort(key=lambda s: s.value_edge, reverse=True)
        return selections

    def _classify_edge(self, edge: float) -> str:
        if edge >= self.best_choice_edge:
            return "Best Choice"
        if edge >= self.value_edge_min:
            return "Value"
        if edge >= -self.value_edge_min:
            return "Fair"
        return "No Value"

    @staticmethod
    def _heuristic_odds() -> dict[str, float]:
        """
        Heuristic market odds for common patterns.
        These approximate average bookmaker lines for a typical match.
        """
        return {
            "BTTS - Yes": 1.80,
            "BTTS - No": 2.00,
            "Over 1.5 Goals FT": 1.30,
            "Over 2.5 Goals FT": 1.85,
            "Over 3.5 Goals FT": 3.20,
            "Over 0.5 Goals HT": 1.35,
            "Over 1.5 Goals HT": 3.10,
            "Goal in 1st Half": 1.35,
            "Team Scored": 1.25,
            "Failed to Score": 4.00,
            "Clean Sheet": 3.50,
        }

    @staticmethod
    def _fuzzy_match_odds(label: str, odds_data: dict[str, float]) -> Optional[float]:
        """Simple prefix-based fuzzy matching for odds lookup."""
        for key, odds in odds_data.items():
            if label.startswith(key) or key.startswith(label):
                return odds
        return None
