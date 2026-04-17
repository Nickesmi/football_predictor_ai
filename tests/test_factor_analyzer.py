"""
Tests for the Factor Analyzer (Issue #3).

Uses hand-crafted TeamPatternReport data to validate:
  - Most common factor extraction for home and away teams
  - Intersection computation with combined confidence %
  - Agreement strength classification
  - Threshold filtering
  - Edge cases (empty data, no intersection, one team empty)
"""

from __future__ import annotations

import pytest

from src.models.patterns import (
    CardsPattern,
    CornersPattern,
    FirstHalfPattern,
    GoalsPattern,
    PatternStat,
    ResultPattern,
    TeamPatternReport,
    TeamScoringPattern,
)
from src.processing.factor_analyzer import (
    FactorAnalyzer,
    IntersectionFactor,
    MatchFactorReport,
    _normalize_label,
    _is_matchable,
)


# ==================================================================
# Test data builders
# ==================================================================

def _stat(label: str, count: int, total: int) -> PatternStat:
    pct = round((count / total) * 100, 1) if total > 0 else 0.0
    return PatternStat(label=label, count=count, total=total, percentage=pct)


def _build_home_report() -> TeamPatternReport:
    """Build a realistic home team pattern report (10 matches)."""
    return TeamPatternReport(
        team_name="Manchester United",
        context="home",
        league_name="Premier League",
        season="2024",
        total_matches=10,
        goals=GoalsPattern(
            btts_yes=_stat("BTTS - Yes", 7, 10),           # 70%
            btts_no=_stat("BTTS - No", 3, 10),             # 30%
            over_0_5_ft=_stat("Over 0.5 Goals FT", 10, 10),# 100%
            over_1_5_ft=_stat("Over 1.5 Goals FT", 8, 10), # 80%
            over_2_5_ft=_stat("Over 2.5 Goals FT", 6, 10), # 60%
            over_3_5_ft=_stat("Over 3.5 Goals FT", 3, 10), # 30%
            under_2_5_ft=_stat("Under 2.5 Goals FT", 4, 10), # 40%
            over_0_5_ht=_stat("Over 0.5 Goals HT", 8, 10), # 80%
            over_1_5_ht=_stat("Over 1.5 Goals HT", 4, 10), # 40%
            avg_goals_ft=2.7,
            avg_goals_scored=1.8,
            avg_goals_conceded=0.9,
        ),
        results=ResultPattern(
            wins=_stat("Home Win", 6, 10),                  # 60%
            draws=_stat("Draw", 3, 10),                     # 30%
            losses=_stat("Home Loss", 1, 10),               # 10%
            ht_wins=_stat("HT Home Win", 4, 10),            # 40%
            ht_draws=_stat("HT Draw", 5, 10),               # 50%
        ),
        scoring=TeamScoringPattern(
            scored_in_match=_stat("Team Scored", 9, 10),    # 90%
            failed_to_score=_stat("Failed to Score", 1, 10),# 10%
            clean_sheet=_stat("Clean Sheet", 3, 10),        # 30%
            scored_first=_stat("Scored First", 7, 9),       # 77.8%
            scored_in_1h=_stat("Team Scored in 1H", 6, 10), # 60%
            scored_in_2h=_stat("Team Scored in 2H", 7, 10), # 70%
            conceded_in_1h=_stat("Conceded in 1H", 4, 10),  # 40%
            conceded_in_2h=_stat("Conceded in 2H", 5, 10),  # 50%
        ),
        corners=CornersPattern(
            over_8_5=_stat("Over 8.5 Corners", 7, 10),     # 70%
            over_9_5=_stat("Over 9.5 Corners", 5, 10),     # 50%
            avg_corners_total=9.5,
        ),
        cards=CardsPattern(
            over_3_5_cards=_stat("Over 3.5 Yellow Cards", 6, 10), # 60%
            over_4_5_cards=_stat("Over 4.5 Yellow Cards", 3, 10), # 30%
            cards_in_1h=_stat("Card in 1st Half", 8, 10),         # 80%
            avg_yellow_total=3.8,
        ),
        first_half=FirstHalfPattern(
            goals_in_1h=_stat("Goal in 1st Half", 8, 10),  # 80%
            both_scored_1h=_stat("Both Scored in 1H", 3, 10), # 30%
            over_0_5_goals_1h=_stat("Over 0.5 Goals 1H", 8, 10), # 80%
        ),
    )


def _build_away_report() -> TeamPatternReport:
    """Build a realistic away team pattern report (12 matches)."""
    return TeamPatternReport(
        team_name="Chelsea",
        context="away",
        league_name="Premier League",
        season="2024",
        total_matches=12,
        goals=GoalsPattern(
            btts_yes=_stat("BTTS - Yes", 9, 12),           # 75%
            btts_no=_stat("BTTS - No", 3, 12),             # 25%
            over_0_5_ft=_stat("Over 0.5 Goals FT", 11, 12),# 91.7%
            over_1_5_ft=_stat("Over 1.5 Goals FT", 9, 12), # 75%
            over_2_5_ft=_stat("Over 2.5 Goals FT", 7, 12), # 58.3%
            over_3_5_ft=_stat("Over 3.5 Goals FT", 4, 12), # 33.3%
            under_2_5_ft=_stat("Under 2.5 Goals FT", 5, 12), # 41.7%
            over_0_5_ht=_stat("Over 0.5 Goals HT", 9, 12), # 75%
            over_1_5_ht=_stat("Over 1.5 Goals HT", 5, 12), # 41.7%
            avg_goals_ft=2.5,
            avg_goals_scored=1.0,
            avg_goals_conceded=1.5,
        ),
        results=ResultPattern(
            wins=_stat("Away Win", 4, 12),                  # 33.3%
            draws=_stat("Draw", 3, 12),                     # 25%
            losses=_stat("Away Loss", 5, 12),               # 41.7%
            ht_wins=_stat("HT Away Win", 3, 12),            # 25%
            ht_draws=_stat("HT Draw", 7, 12),               # 58.3%
        ),
        scoring=TeamScoringPattern(
            scored_in_match=_stat("Team Scored", 10, 12),   # 83.3%
            failed_to_score=_stat("Failed to Score", 2, 12),# 16.7%
            clean_sheet=_stat("Clean Sheet", 2, 12),        # 16.7%
            scored_first=_stat("Scored First", 5, 10),      # 50%
            scored_in_1h=_stat("Team Scored in 1H", 7, 12), # 58.3%
            scored_in_2h=_stat("Team Scored in 2H", 8, 12), # 66.7%
            conceded_in_1h=_stat("Conceded in 1H", 6, 12),  # 50%
            conceded_in_2h=_stat("Conceded in 2H", 7, 12),  # 58.3%
        ),
        corners=CornersPattern(
            over_8_5=_stat("Over 8.5 Corners", 8, 12),     # 66.7%
            over_9_5=_stat("Over 9.5 Corners", 6, 12),     # 50%
            avg_corners_total=9.2,
        ),
        cards=CardsPattern(
            over_3_5_cards=_stat("Over 3.5 Yellow Cards", 8, 12), # 66.7%
            over_4_5_cards=_stat("Over 4.5 Yellow Cards", 5, 12), # 41.7%
            cards_in_1h=_stat("Card in 1st Half", 9, 12),         # 75%
            avg_yellow_total=4.1,
        ),
        first_half=FirstHalfPattern(
            goals_in_1h=_stat("Goal in 1st Half", 9, 12),  # 75%
            both_scored_1h=_stat("Both Scored in 1H", 4, 12), # 33.3%
            over_0_5_goals_1h=_stat("Over 0.5 Goals 1H", 9, 12), # 75%
        ),
    )


# ==================================================================
# IntersectionFactor model tests
# ==================================================================


class TestIntersectionFactor:
    def test_confidence_tiers(self):
        f = IntersectionFactor(
            label="BTTS",
            home_stat=_stat("BTTS", 8, 10),
            away_stat=_stat("BTTS", 9, 10),
            combined_percentage=85.0,
        )
        assert f.confidence == "Very High"

    def test_agreement_strength_strong(self):
        f = IntersectionFactor(
            label="Test",
            home_stat=_stat("Test", 7, 10),    # 70%
            away_stat=_stat("Test", 8, 10),    # 80%
            combined_percentage=75.0,
        )
        assert f.agreement_strength == "Strong Agreement"  # gap = 10

    def test_agreement_strength_moderate(self):
        f = IntersectionFactor(
            label="Test",
            home_stat=_stat("Test", 5, 10),    # 50%
            away_stat=_stat("Test", 8, 10),    # 80%
            combined_percentage=65.0,
        )
        assert f.agreement_strength == "Weak Agreement"    # gap = 30

    def test_repr(self):
        f = IntersectionFactor(
            label="BTTS - Yes",
            home_stat=_stat("BTTS - Yes", 7, 10),
            away_stat=_stat("BTTS - Yes", 9, 12),
            combined_percentage=72.5,
        )
        r = repr(f)
        assert "BTTS - Yes" in r
        assert "72.5%" in r
        assert "High" in r


# ==================================================================
# Label normalization tests
# ==================================================================


class TestLabelNormalization:
    def test_market_neutral_labels_unchanged(self):
        assert _normalize_label("BTTS - Yes") == "BTTS - Yes"
        assert _normalize_label("Over 2.5 Goals FT") == "Over 2.5 Goals FT"
        assert _normalize_label("Under 0.5 Goals HT") == "Under 0.5 Goals HT"

    def test_context_labels_normalized(self):
        assert _normalize_label("Home Win") == "Team Win"
        assert _normalize_label("Away Win") == "Team Win"

    def test_is_matchable(self):
        assert _is_matchable("BTTS - Yes") is True
        assert _is_matchable("Over 2.5 Goals FT") is True
        assert _is_matchable("Under 8.5 Corners") is True
        assert _is_matchable("Team Scored") is True
        assert _is_matchable("Goal in 1st Half") is True
        assert _is_matchable("Card in 1st Half") is True
        assert _is_matchable("Draw") is True

    def test_non_matchable(self):
        # Context-specific results shouldn't match across teams
        assert _is_matchable("Home Win") is False
        assert _is_matchable("Away Loss") is False
        assert _is_matchable("HT Home Win") is False


# ==================================================================
# FactorAnalyzer tests
# ==================================================================


class TestFactorAnalyzer:
    def setup_method(self):
        self.analyzer = FactorAnalyzer()
        self.home_report = _build_home_report()
        self.away_report = _build_away_report()

    def test_basic_analysis(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        assert isinstance(result, MatchFactorReport)
        assert result.home_team == "Manchester United"
        assert result.away_team == "Chelsea"

    def test_home_factors_extracted(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        # Home factors >= 50% should include: Over 0.5 FT (100%),
        # Team Scored (90%), Over 1.5 FT (80%), etc.
        assert len(result.home_factors) > 0
        # Sorted descending
        for i in range(len(result.home_factors) - 1):
            assert result.home_factors[i].percentage >= result.home_factors[i + 1].percentage

    def test_away_factors_extracted(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        assert len(result.away_factors) > 0
        for i in range(len(result.away_factors) - 1):
            assert result.away_factors[i].percentage >= result.away_factors[i + 1].percentage

    def test_intersection_found(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        # Both teams have BTTS Yes, Over 0.5 FT, Over 1.5 FT, etc.
        assert len(result.intersection) > 0

    def test_intersection_sorted_by_combined(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        for i in range(len(result.intersection) - 1):
            assert (
                result.intersection[i].combined_percentage
                >= result.intersection[i + 1].combined_percentage
            )

    def test_intersection_combined_percentage(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        # Find BTTS - Yes: Home 70% + Away 75% → combined 72.5%
        btts = next(
            (f for f in result.intersection if f.label == "BTTS - Yes"), None
        )
        assert btts is not None
        assert btts.combined_percentage == 72.5
        assert btts.home_stat.percentage == 70.0
        assert btts.away_stat.percentage == 75.0

    def test_intersection_over_1_5(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        o15 = next(
            (f for f in result.intersection if f.label == "Over 1.5 Goals FT"), None
        )
        assert o15 is not None
        # Home 80% + Away 75% → 77.5%
        assert o15.combined_percentage == 77.5

    def test_intersection_team_scored(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        ts = next(
            (f for f in result.intersection if f.label == "Team Scored"), None
        )
        assert ts is not None
        # Home 90% + Away 83.3% → 86.65 → 86.7%
        assert ts.combined_percentage == pytest.approx(86.7, abs=0.1)

    def test_intersection_corners(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        corners = next(
            (f for f in result.intersection if f.label == "Over 8.5 Corners"), None
        )
        assert corners is not None
        # Home 70% + Away 66.7% → 68.35 → 68.4%
        assert corners.combined_percentage == pytest.approx(68.35, abs=0.2)

    def test_intersection_cards_1h(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        cards = next(
            (f for f in result.intersection if f.label == "Card in 1st Half"), None
        )
        assert cards is not None
        # Home 80% + Away 75% → 77.5%
        assert cards.combined_percentage == 77.5


# ==================================================================
# Threshold & filtering tests
# ==================================================================


class TestThresholdFiltering:
    def setup_method(self):
        self.analyzer = FactorAnalyzer()
        self.home_report = _build_home_report()
        self.away_report = _build_away_report()

    def test_high_threshold_fewer_results(self):
        result_50 = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        result_75 = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=75.0
        )
        # Higher threshold → fewer intersection factors
        assert len(result_75.intersection) <= len(result_50.intersection)

    def test_get_intersection_above(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        above_80 = result.get_intersection_above(80.0)
        above_60 = result.get_intersection_above(60.0)
        assert len(above_80) <= len(above_60)
        for f in above_80:
            assert f.combined_percentage >= 80.0

    def test_get_strong_intersections(self):
        result = self.analyzer.analyze(
            self.home_report, self.away_report, threshold=50.0
        )
        strong = result.get_strong_intersections()
        for f in strong:
            gap = abs(f.home_stat.percentage - f.away_stat.percentage)
            assert gap <= 10


# ==================================================================
# Edge cases
# ==================================================================


class TestEdgeCases:
    def setup_method(self):
        self.analyzer = FactorAnalyzer()

    def test_empty_home_report(self):
        home = TeamPatternReport(
            team_name="Empty",
            context="home",
            league_name="PL",
            season="2024",
            total_matches=0,
        )
        away = _build_away_report()
        result = self.analyzer.analyze(home, away, threshold=50.0)
        assert len(result.home_factors) == 0
        assert len(result.intersection) == 0
        assert len(result.away_factors) > 0

    def test_empty_away_report(self):
        home = _build_home_report()
        away = TeamPatternReport(
            team_name="Empty",
            context="away",
            league_name="PL",
            season="2024",
            total_matches=0,
        )
        result = self.analyzer.analyze(home, away, threshold=50.0)
        assert len(result.away_factors) == 0
        assert len(result.intersection) == 0
        assert len(result.home_factors) > 0

    def test_both_empty(self):
        home = TeamPatternReport(
            team_name="A", context="home", league_name="PL",
            season="2024", total_matches=0,
        )
        away = TeamPatternReport(
            team_name="B", context="away", league_name="PL",
            season="2024", total_matches=0,
        )
        result = self.analyzer.analyze(home, away, threshold=50.0)
        assert len(result.intersection) == 0

    def test_no_common_patterns(self):
        """Two teams where no patterns exceed threshold for both."""
        home = TeamPatternReport(
            team_name="Low", context="home", league_name="PL",
            season="2024", total_matches=10,
            goals=GoalsPattern(
                btts_yes=_stat("BTTS - Yes", 2, 10),       # 20%
                over_2_5_ft=_stat("Over 2.5 Goals FT", 1, 10), # 10%
            ),
        )
        away = TeamPatternReport(
            team_name="AlsoLow", context="away", league_name="PL",
            season="2024", total_matches=10,
            goals=GoalsPattern(
                btts_yes=_stat("BTTS - Yes", 3, 10),       # 30%
                over_2_5_ft=_stat("Over 2.5 Goals FT", 2, 10), # 20%
            ),
        )
        result = self.analyzer.analyze(home, away, threshold=50.0)
        # Neither team has >=50% patterns, so no intersection
        assert len(result.intersection) == 0

    def test_perfect_agreement(self):
        """Two teams with identical patterns → strong intersection."""
        goals = GoalsPattern(
            btts_yes=_stat("BTTS - Yes", 8, 10),           # 80%
            over_2_5_ft=_stat("Over 2.5 Goals FT", 7, 10), # 70%
        )
        scoring = TeamScoringPattern(
            scored_in_match=_stat("Team Scored", 9, 10),    # 90%
        )
        home = TeamPatternReport(
            team_name="A", context="home", league_name="PL",
            season="2024", total_matches=10,
            goals=goals, scoring=scoring,
        )
        away = TeamPatternReport(
            team_name="B", context="away", league_name="PL",
            season="2024", total_matches=10,
            goals=goals, scoring=scoring,
        )
        result = self.analyzer.analyze(home, away, threshold=50.0)

        # All matchable patterns should intersect
        btts = next(
            (f for f in result.intersection if f.label == "BTTS - Yes"), None
        )
        assert btts is not None
        assert btts.combined_percentage == 80.0
        assert btts.agreement_strength == "Strong Agreement"

    def test_report_repr(self):
        result = self.analyzer.analyze(
            _build_home_report(), _build_away_report(), threshold=50.0
        )
        r = repr(result)
        assert "Manchester United" in r
        assert "Chelsea" in r
        assert "Intersection" in r
