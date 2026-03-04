"""
Domain models for computed statistical patterns (Issue #2).

These dataclasses hold the results of the pattern analysis —
occurrence counts, percentages, and confidence labels — for
every market the system tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PatternStat:
    """
    A single computed statistic with its occurrence count and percentage.

    Example:
        PatternStat(label="BTTS - Yes", count=12, total=15, percentage=80.0)
    """
    label: str
    count: int
    total: int
    percentage: float

    @property
    def confidence(self) -> str:
        """Human-readable confidence tier."""
        if self.percentage >= 80:
            return "Very High"
        if self.percentage >= 65:
            return "High"
        if self.percentage >= 50:
            return "Medium"
        if self.percentage >= 35:
            return "Low"
        return "Very Low"

    def __repr__(self) -> str:
        return (
            f"{self.label}: {self.count}/{self.total} "
            f"({self.percentage:.1f}%) [{self.confidence}]"
        )


@dataclass
class GoalsPattern:
    """All goal-related pattern stats for a set of matches."""
    # BTTS
    btts_yes: Optional[PatternStat] = None
    btts_no: Optional[PatternStat] = None

    # Over/Under Full Time
    over_0_5_ft: Optional[PatternStat] = None
    over_1_5_ft: Optional[PatternStat] = None
    over_2_5_ft: Optional[PatternStat] = None
    over_3_5_ft: Optional[PatternStat] = None
    over_4_5_ft: Optional[PatternStat] = None
    under_0_5_ft: Optional[PatternStat] = None
    under_1_5_ft: Optional[PatternStat] = None
    under_2_5_ft: Optional[PatternStat] = None
    under_3_5_ft: Optional[PatternStat] = None
    under_4_5_ft: Optional[PatternStat] = None

    # Over/Under Half Time
    over_0_5_ht: Optional[PatternStat] = None
    over_1_5_ht: Optional[PatternStat] = None
    over_2_5_ht: Optional[PatternStat] = None
    under_0_5_ht: Optional[PatternStat] = None
    under_1_5_ht: Optional[PatternStat] = None
    under_2_5_ht: Optional[PatternStat] = None

    # Average goals
    avg_goals_ft: float = 0.0
    avg_goals_ht: float = 0.0
    avg_goals_scored: float = 0.0    # by the context team
    avg_goals_conceded: float = 0.0  # by the context team


@dataclass
class ResultPattern:
    """Win / Draw / Loss pattern stats."""
    wins: Optional[PatternStat] = None
    draws: Optional[PatternStat] = None
    losses: Optional[PatternStat] = None

    # Half-time results
    ht_wins: Optional[PatternStat] = None
    ht_draws: Optional[PatternStat] = None
    ht_losses: Optional[PatternStat] = None

    # HT/FT combos (most common)
    ht_ft_distribution: dict[str, PatternStat] = field(default_factory=dict)


@dataclass
class TeamScoringPattern:
    """Patterns related to team scoring / failing to score."""
    scored_in_match: Optional[PatternStat] = None       # Team scored >= 1
    failed_to_score: Optional[PatternStat] = None       # Team scored 0
    clean_sheet: Optional[PatternStat] = None            # Opponent scored 0
    scored_first: Optional[PatternStat] = None           # Team scored the first goal
    conceded_first: Optional[PatternStat] = None         # Opponent scored first

    scored_in_1h: Optional[PatternStat] = None           # Team scored in first half
    scored_in_2h: Optional[PatternStat] = None           # Team scored in second half
    conceded_in_1h: Optional[PatternStat] = None         # Opponent scored in first half
    conceded_in_2h: Optional[PatternStat] = None         # Opponent scored in second half


@dataclass
class CornersPattern:
    """Corner-related pattern stats."""
    avg_corners_total: float = 0.0
    avg_corners_team: float = 0.0
    avg_corners_opponent: float = 0.0

    over_7_5: Optional[PatternStat] = None
    over_8_5: Optional[PatternStat] = None
    over_9_5: Optional[PatternStat] = None
    over_10_5: Optional[PatternStat] = None
    over_11_5: Optional[PatternStat] = None
    under_7_5: Optional[PatternStat] = None
    under_8_5: Optional[PatternStat] = None
    under_9_5: Optional[PatternStat] = None
    under_10_5: Optional[PatternStat] = None
    under_11_5: Optional[PatternStat] = None


@dataclass
class CardsPattern:
    """Card-related pattern stats."""
    avg_yellow_total: float = 0.0
    avg_yellow_team: float = 0.0
    avg_yellow_opponent: float = 0.0
    avg_red_total: float = 0.0

    over_2_5_cards: Optional[PatternStat] = None
    over_3_5_cards: Optional[PatternStat] = None
    over_4_5_cards: Optional[PatternStat] = None
    over_5_5_cards: Optional[PatternStat] = None
    under_2_5_cards: Optional[PatternStat] = None
    under_3_5_cards: Optional[PatternStat] = None
    under_4_5_cards: Optional[PatternStat] = None
    under_5_5_cards: Optional[PatternStat] = None

    # First half cards
    cards_in_1h: Optional[PatternStat] = None  # at least 1 card in 1H
    avg_cards_1h: float = 0.0


@dataclass
class FirstHalfPattern:
    """Aggregated first-half event patterns."""
    # Goals in 1st half
    goals_in_1h: Optional[PatternStat] = None           # At least 1 goal in 1H
    both_scored_1h: Optional[PatternStat] = None         # Both teams scored in 1H
    over_0_5_goals_1h: Optional[PatternStat] = None
    over_1_5_goals_1h: Optional[PatternStat] = None

    # Cards in 1st half
    cards_in_1h: Optional[PatternStat] = None            # At least 1 card in 1H
    over_0_5_cards_1h: Optional[PatternStat] = None
    over_1_5_cards_1h: Optional[PatternStat] = None

    # Corners in 1st half (when stats available)
    avg_corners_1h: float = 0.0

    # 1H result
    ht_home_win: Optional[PatternStat] = None
    ht_draw: Optional[PatternStat] = None
    ht_away_win: Optional[PatternStat] = None


@dataclass
class TeamPatternReport:
    """
    Complete pattern analysis for ONE team in ONE context
    (home or away) across a league/season.

    This is the output of the PatternAnalyzer for a single TeamMatchSet.
    """
    team_name: str
    context: str          # "home" or "away"
    league_name: str
    season: str
    total_matches: int

    goals: GoalsPattern = field(default_factory=GoalsPattern)
    results: ResultPattern = field(default_factory=ResultPattern)
    scoring: TeamScoringPattern = field(default_factory=TeamScoringPattern)
    corners: CornersPattern = field(default_factory=CornersPattern)
    cards: CardsPattern = field(default_factory=CardsPattern)
    first_half: FirstHalfPattern = field(default_factory=FirstHalfPattern)

    def get_high_confidence_patterns(self, threshold: float = 65.0) -> list[PatternStat]:
        """
        Return all patterns with percentage >= threshold,
        sorted by percentage descending.
        """
        all_stats: list[PatternStat] = []

        # Collect all PatternStat fields from sub-reports
        for sub in (self.goals, self.results, self.scoring, self.corners,
                    self.cards, self.first_half):
            for attr_name in vars(sub):
                val = getattr(sub, attr_name)
                if isinstance(val, PatternStat):
                    all_stats.append(val)
                elif isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, PatternStat):
                            all_stats.append(v)

        return sorted(
            [s for s in all_stats if s.percentage >= threshold],
            key=lambda s: s.percentage,
            reverse=True,
        )

    def __repr__(self) -> str:
        return (
            f"TeamPatternReport({self.team_name} | {self.context.upper()} | "
            f"{self.league_name} {self.season} | {self.total_matches} matches)"
        )
