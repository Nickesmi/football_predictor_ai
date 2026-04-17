"""
Domain models for computed statistical patterns (Issue #2).

These dataclasses hold the results of the pattern analysis —
occurrence counts, percentages, and confidence labels — for
every market the system tracks.
"""

from __future__ import annotations

import math
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
    def wilson_lower_bound(self) -> float:
        """
        Calculate the Wilson score interval lower bound (95% confidence).
        This strongly penalizes small sample sizes, preventing 1/1 (100%) from
        ranking higher than 18/20 (90%).
        """
        if self.total == 0:
            return 0.0
        
        z = 1.96  # 95% confidence
        p = self.count / self.total
        n = self.total
        
        denominator = 1 + z**2 / n
        center = p + z**2 / (2 * n)
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        
        bound = (center - spread) / denominator
        return bound * 100.0

    @property
    def confidence(self) -> str:
        """Human-readable confidence tier (now based on Wilson bounds)."""
        if self.wilson_lower_bound >= 75:
            return "Very High"
        if self.wilson_lower_bound >= 60:
            return "High"
        if self.wilson_lower_bound >= 45:
            return "Medium"
        if self.wilson_lower_bound >= 30:
            return "Low"
        return "Very Low"

    def __repr__(self) -> str:
        return (
            f"{self.label}: {self.count}/{self.total} "
            f"({self.percentage:.1f}%) [W:{self.wilson_lower_bound:.1f}%]"
        )


@dataclass
class GoalsPattern:
    """All goal-related pattern stats for a set of matches."""
    btts_yes: Optional[PatternStat] = None
    btts_no: Optional[PatternStat] = None

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

    over_0_5_ht: Optional[PatternStat] = None
    over_1_5_ht: Optional[PatternStat] = None
    over_2_5_ht: Optional[PatternStat] = None
    under_0_5_ht: Optional[PatternStat] = None
    under_1_5_ht: Optional[PatternStat] = None
    under_2_5_ht: Optional[PatternStat] = None

    avg_goals_ft: float = 0.0
    avg_goals_ht: float = 0.0
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0


@dataclass
class ResultPattern:
    """Win / Draw / Loss pattern stats."""
    wins: Optional[PatternStat] = None
    draws: Optional[PatternStat] = None
    losses: Optional[PatternStat] = None

    ht_wins: Optional[PatternStat] = None
    ht_draws: Optional[PatternStat] = None
    ht_losses: Optional[PatternStat] = None

    ht_ft_distribution: dict[str, PatternStat] = field(default_factory=dict)


@dataclass
class TeamScoringPattern:
    """Patterns related to team scoring / failing to score."""
    scored_in_match: Optional[PatternStat] = None
    failed_to_score: Optional[PatternStat] = None
    clean_sheet: Optional[PatternStat] = None
    scored_first: Optional[PatternStat] = None
    conceded_first: Optional[PatternStat] = None

    scored_in_1h: Optional[PatternStat] = None
    scored_in_2h: Optional[PatternStat] = None
    conceded_in_1h: Optional[PatternStat] = None
    conceded_in_2h: Optional[PatternStat] = None


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

    cards_in_1h: Optional[PatternStat] = None
    avg_cards_1h: float = 0.0


@dataclass
class FirstHalfPattern:
    """Aggregated first-half event patterns."""
    goals_in_1h: Optional[PatternStat] = None
    both_scored_1h: Optional[PatternStat] = None
    over_0_5_goals_1h: Optional[PatternStat] = None
    over_1_5_goals_1h: Optional[PatternStat] = None

    cards_in_1h: Optional[PatternStat] = None
    over_0_5_cards_1h: Optional[PatternStat] = None
    over_1_5_cards_1h: Optional[PatternStat] = None

    avg_corners_1h: float = 0.0

    ht_home_win: Optional[PatternStat] = None
    ht_draw: Optional[PatternStat] = None
    ht_away_win: Optional[PatternStat] = None


@dataclass
class TeamPatternReport:
    """
    Complete pattern analysis for ONE team in ONE context
    (home or away) across a league/season.
    """
    team_name: str
    context: str
    league_name: str
    season: str
    total_matches: int

    goals: GoalsPattern = field(default_factory=GoalsPattern)
    results: ResultPattern = field(default_factory=ResultPattern)
    scoring: TeamScoringPattern = field(default_factory=TeamScoringPattern)
    corners: CornersPattern = field(default_factory=CornersPattern)
    cards: CardsPattern = field(default_factory=CardsPattern)
    first_half: FirstHalfPattern = field(default_factory=FirstHalfPattern)

    def get_high_confidence_patterns(self, min_wilson: float = 60.0, min_matches: int = 8) -> list[PatternStat]:
        """
        Return all statistically significant patterns.
        
        Requires:
          1. Minimum total matches for the split (min_matches).
          2. Minimum Wilson lower bound (min_wilson).
        
        Sorted by the Wilson lower bound descending, as it is a strongly
        stable metric of predictive power.
        """
        # Early season instability mask
        if self.total_matches < min_matches:
            return []

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
            [s for s in all_stats if s.total >= min_matches and s.wilson_lower_bound >= min_wilson],
            key=lambda s: s.wilson_lower_bound,
            reverse=True,
        )

    def __repr__(self) -> str:
        return (
            f"TeamPatternReport({self.team_name} | {self.context.upper()} | "
            f"{self.league_name} {self.season} | {self.total_matches} matches)"
        )
