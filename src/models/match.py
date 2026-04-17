"""
Data models for Football Predictor AI.

These dataclasses provide a clean, typed representation of match data
that is API-agnostic. The API client maps raw JSON responses into these
models so that downstream processing never depends on the API shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class GoalEvent:
    """A single goal scored in a match."""
    minute: int
    scorer: str
    assist: Optional[str] = None
    is_home: bool = True              # True = home team scored
    half: str = ""                    # "1st Half" / "2nd Half"


@dataclass(frozen=True)
class CardEvent:
    """A booking (yellow / red card) in a match."""
    minute: int
    player: str
    card_type: str                    # "yellow" or "red"
    is_home: bool = True
    half: str = ""


@dataclass(frozen=True)
class MatchStatistics:
    """Aggregated match statistics (corners, shots, possession, etc.)."""
    corners_home: int = 0
    corners_away: int = 0
    shots_total_home: int = 0
    shots_total_away: int = 0
    shots_on_target_home: int = 0
    shots_on_target_away: int = 0
    fouls_home: int = 0
    fouls_away: int = 0
    yellow_cards_home: int = 0
    yellow_cards_away: int = 0
    red_cards_home: int = 0
    red_cards_away: int = 0
    possession_home: Optional[str] = None   # e.g. "58%"
    possession_away: Optional[str] = None

    # First-half specific stats (when available)
    corners_home_1h: int = 0
    corners_away_1h: int = 0
    yellow_cards_home_1h: int = 0
    yellow_cards_away_1h: int = 0


@dataclass(frozen=True)
class MatchResult:
    """
    A fully parsed match result with all data required for
    probabilistic pattern mining.

    This is the **core domain object** — every match fetched from any
    API is normalised into this shape.
    """
    # IDs & metadata
    match_id: str
    match_date: date
    league_id: str
    league_name: str
    season: str                        # e.g. "2024/2025"
    round: str = ""                    # e.g. "Regular Season - 15"

    # Teams
    home_team_id: str = ""
    home_team_name: str = ""
    away_team_id: str = ""
    away_team_name: str = ""

    # Full-time score
    home_score_ft: int = 0
    away_score_ft: int = 0

    # Half-time score
    home_score_ht: Optional[int] = None
    away_score_ht: Optional[int] = None

    # Derived flags (populated on construction)
    btts: bool = False                 # Both teams to score
    total_goals_ft: int = 0
    total_goals_ht: int = 0

    # Detailed events
    goals: list[GoalEvent] = field(default_factory=list)
    cards: list[CardEvent] = field(default_factory=list)
    statistics: Optional[MatchStatistics] = None

    # Match status
    status: str = "FT"                 # FT, NS (not started), etc.

    def __post_init__(self):
        """Compute derived fields after init (works with frozen dataclass)."""
        object.__setattr__(self, "total_goals_ft", self.home_score_ft + self.away_score_ft)
        if self.home_score_ht is not None and self.away_score_ht is not None:
            object.__setattr__(self, "total_goals_ht", self.home_score_ht + self.away_score_ht)
        object.__setattr__(
            self, "btts", self.home_score_ft > 0 and self.away_score_ft > 0
        )

    # ---- Convenience helpers for pattern analysis (Issue #2 will use these) ----

    @property
    def home_win(self) -> bool:
        return self.home_score_ft > self.away_score_ft

    @property
    def away_win(self) -> bool:
        return self.away_score_ft > self.home_score_ft

    @property
    def draw(self) -> bool:
        return self.home_score_ft == self.away_score_ft

    @property
    def home_clean_sheet(self) -> bool:
        return self.away_score_ft == 0

    @property
    def away_clean_sheet(self) -> bool:
        return self.home_score_ft == 0

    @property
    def over_1_5(self) -> bool:
        return self.total_goals_ft > 1

    @property
    def over_2_5(self) -> bool:
        return self.total_goals_ft > 2

    @property
    def over_3_5(self) -> bool:
        return self.total_goals_ft > 3

    @property
    def ht_result(self) -> str:
        """Half-time result: '1' (home), 'X' (draw), '2' (away)."""
        if self.home_score_ht is None or self.away_score_ht is None:
            return "?"
        if self.home_score_ht > self.away_score_ht:
            return "1"
        if self.home_score_ht < self.away_score_ht:
            return "2"
        return "X"

    @property
    def ft_result(self) -> str:
        """Full-time result: '1' (home), 'X' (draw), '2' (away)."""
        if self.home_win:
            return "1"
        if self.away_win:
            return "2"
        return "X"


@dataclass
class TeamMatchSet:
    """
    Container holding ALL context-specific matches for ONE team.
    For a Team A playing at Home in League L, this would hold
    all of Team A's home matches in that league/season.
    """
    team_id: str
    team_name: str
    league_id: str
    league_name: str
    season: str
    context: str                       # "home" or "away"
    matches: list[MatchResult] = field(default_factory=list)

    @property
    def total_matches(self) -> int:
        return len(self.matches)

    def __repr__(self) -> str:
        return (
            f"TeamMatchSet({self.team_name} | {self.context.upper()} | "
            f"{self.league_name} {self.season} | {self.total_matches} matches)"
        )
