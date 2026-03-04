"""
Most Common Factor Analysis Engine (Issue #3).

Takes the TeamPatternReport objects produced by Issue #2's
PatternAnalyzer and:
  1. Identifies the most common factors for the Home team (A)
  2. Identifies the most common factors for the Away team (B)
  3. Computes the **intersection** — where patterns from both teams align
  4. Computes a combined Confidence % for each intersecting pattern

The intersection confidence is computed as:
    combined_pct = (home_pct + away_pct) / 2    (arithmetic mean)
    — This reflects how strongly BOTH teams' histories support the pattern.

Architecture:
    TeamPatternReport (Home A)  ──┐
                                  ├──→ FactorAnalyzer.analyze()
    TeamPatternReport (Away B)  ──┘         ↓
                                    MatchFactorReport
                                    ├── home_factors   (ranked)
                                    ├── away_factors   (ranked)
                                    └── intersection   (combined)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import logger
from src.models.patterns import PatternStat, TeamPatternReport


# ==================================================================
# Domain models
# ==================================================================

@dataclass
class IntersectionFactor:
    """
    A pattern that appears in BOTH the home team's and away team's
    historical data, with a combined confidence score.
    """
    label: str
    home_stat: PatternStat
    away_stat: PatternStat
    combined_percentage: float         # (home% + away%) / 2

    @property
    def confidence(self) -> str:
        """Combined confidence tier."""
        if self.combined_percentage >= 80:
            return "Very High"
        if self.combined_percentage >= 65:
            return "High"
        if self.combined_percentage >= 50:
            return "Medium"
        if self.combined_percentage >= 35:
            return "Low"
        return "Very Low"

    @property
    def agreement_strength(self) -> str:
        """
        How closely the two teams' percentages agree.
        A small gap means both teams strongly support the pattern.
        """
        gap = abs(self.home_stat.percentage - self.away_stat.percentage)
        if gap <= 10:
            return "Strong Agreement"
        if gap <= 25:
            return "Moderate Agreement"
        return "Weak Agreement"

    def __repr__(self) -> str:
        return (
            f"{self.label}: "
            f"Home {self.home_stat.percentage:.1f}% + "
            f"Away {self.away_stat.percentage:.1f}% → "
            f"Combined {self.combined_percentage:.1f}% "
            f"[{self.confidence}] ({self.agreement_strength})"
        )


@dataclass
class MatchFactorReport:
    """
    Complete factor analysis for a match: Home A vs Away B.

    Contains:
      - home_factors: Most common patterns for Team A at home (sorted by %)
      - away_factors: Most common patterns for Team B away (sorted by %)
      - intersection: Patterns where both teams' data aligns (sorted by combined %)
    """
    home_team: str
    away_team: str
    league_name: str
    season: str

    home_total_matches: int = 0
    away_total_matches: int = 0

    home_factors: list[PatternStat] = field(default_factory=list)
    away_factors: list[PatternStat] = field(default_factory=list)
    intersection: list[IntersectionFactor] = field(default_factory=list)

    def get_intersection_above(self, threshold: float = 65.0) -> list[IntersectionFactor]:
        """Return intersection factors with combined % >= threshold."""
        return [f for f in self.intersection if f.combined_percentage >= threshold]

    def get_strong_intersections(self) -> list[IntersectionFactor]:
        """Return only intersection factors with Strong Agreement."""
        return [
            f for f in self.intersection
            if f.agreement_strength == "Strong Agreement"
        ]

    def __repr__(self) -> str:
        return (
            f"MatchFactorReport({self.home_team} vs {self.away_team} | "
            f"{self.league_name} {self.season} | "
            f"Home: {len(self.home_factors)} factors, "
            f"Away: {len(self.away_factors)} factors, "
            f"Intersection: {len(self.intersection)} factors)"
        )


# ==================================================================
# Label normalization
# ==================================================================

# Mapping of label patterns that should be treated as equivalent
# when comparing home vs away reports. Some patterns have
# context-specific labels that refer to the same underlying market.
_NORMALIZE_MAP: dict[str, str] = {
    # Results — both "Home Win" and "Away Win" map to contextual "Team Win"
    # but we keep them separate since they're not the same market.
    # Instead, we normalize market-neutral labels.
    "HT Home Win": "HT Team Win",
    "HT Away Win": "HT Opponent Win",
    "Home Win": "Team Win",
    "Away Win": "Team Win",
    "Home Loss": "Team Loss",
    "Away Loss": "Team Loss",
    "HT Home Win": "HT Team Win",
    "HT Home Loss": "HT Team Loss",
    "HT Away Win": "HT Team Win",
    "HT Away Loss": "HT Team Loss",
}

# Labels that are MARKET-NEUTRAL — they mean the same thing regardless
# of home/away context. These are the labels we match on for intersection.
_MARKET_NEUTRAL_PREFIXES = (
    "BTTS",
    "Over", "Under",
    "Draw", "HT Draw",
    "Team Scored", "Failed to Score", "Clean Sheet",
    "Scored First", "Conceded First",
    "Team Scored in 1H", "Team Scored in 2H",
    "Conceded in 1H", "Conceded in 2H",
    "Goal in 1st Half", "Both Scored in 1H",
    "Card in 1st Half",
)


def _normalize_label(label: str) -> str:
    """
    Normalize a pattern label for cross-team comparison.

    Market-neutral labels (BTTS, O/U, etc.) stay unchanged.
    Context-specific labels are remapped to a common form.
    """
    return _NORMALIZE_MAP.get(label, label)


def _is_matchable(label: str) -> bool:
    """
    Check if a pattern label represents a market that can be
    meaningfully intersected between home and away teams.

    We exclude context-specific labels like "Home Win" vs "Away Win"
    because they aren't the same market. Instead, we include
    market-neutral patterns and specific scoring/event markets.
    """
    return any(label.startswith(prefix) for prefix in _MARKET_NEUTRAL_PREFIXES)


# ==================================================================
# Analyzer
# ==================================================================

class FactorAnalyzer:
    """
    Computes the most common factors for Home (A), Away (B),
    and their intersection.

    Usage::

        factor_analyzer = FactorAnalyzer()
        report = factor_analyzer.analyze(
            home_report,   # TeamPatternReport for Team A at home
            away_report,   # TeamPatternReport for Team B away
            threshold=50.0
        )
        # Top intersection factors
        for f in report.get_intersection_above(65.0):
            print(f)
    """

    def analyze(
        self,
        home_report: TeamPatternReport,
        away_report: TeamPatternReport,
        threshold: float = 50.0,
    ) -> MatchFactorReport:
        """
        Compute the most common factors for both teams and their intersection.

        Args:
            home_report: Pattern analysis for the home team.
            away_report: Pattern analysis for the away team.
            threshold: Minimum percentage for a pattern to be included
                       as a "most common factor". Default 50%.

        Returns:
            MatchFactorReport with home factors, away factors, and intersection.
        """
        logger.info(
            "Computing factor analysis: %s (HOME) vs %s (AWAY)",
            home_report.team_name, away_report.team_name,
        )

        # Step 1: Extract most common factors for each team
        home_factors = home_report.get_high_confidence_patterns(threshold=threshold)
        away_factors = away_report.get_high_confidence_patterns(threshold=threshold)

        logger.info(
            "Home factors (>= %.0f%%): %d | Away factors: %d",
            threshold, len(home_factors), len(away_factors),
        )

        # Step 2: Compute intersection
        intersection = self._compute_intersection(
            home_factors, away_factors, threshold
        )

        logger.info(
            "Intersection factors: %d (combined >= %.0f%%)",
            len(intersection), threshold,
        )

        return MatchFactorReport(
            home_team=home_report.team_name,
            away_team=away_report.team_name,
            league_name=home_report.league_name,
            season=home_report.season,
            home_total_matches=home_report.total_matches,
            away_total_matches=away_report.total_matches,
            home_factors=home_factors,
            away_factors=away_factors,
            intersection=intersection,
        )

    @staticmethod
    def _compute_intersection(
        home_factors: list[PatternStat],
        away_factors: list[PatternStat],
        threshold: float,
    ) -> list[IntersectionFactor]:
        """
        Find patterns that appear in BOTH teams' factor lists.

        For each matching pattern:
          - Combined % = (home% + away%) / 2
          - Only included if combined% >= threshold

        Returns the intersection sorted by combined_percentage descending.
        """
        # Build lookup: normalized_label → PatternStat for away team
        away_lookup: dict[str, PatternStat] = {}
        for stat in away_factors:
            norm = _normalize_label(stat.label)
            if _is_matchable(stat.label):
                away_lookup[norm] = stat

        intersection: list[IntersectionFactor] = []

        for home_stat in home_factors:
            if not _is_matchable(home_stat.label):
                continue

            norm = _normalize_label(home_stat.label)
            away_stat = away_lookup.get(norm)

            if away_stat is not None:
                combined = round(
                    (home_stat.percentage + away_stat.percentage) / 2, 1
                )
                if combined >= threshold:
                    intersection.append(IntersectionFactor(
                        label=home_stat.label,  # Use original label
                        home_stat=home_stat,
                        away_stat=away_stat,
                        combined_percentage=combined,
                    ))

        # Sort by combined percentage descending
        intersection.sort(key=lambda f: f.combined_percentage, reverse=True)

        return intersection
