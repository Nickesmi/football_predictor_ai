"""
MVIM Factor Analysis Engine — Multi-Variable Intersection Model.

Implements the full decision engine from the Gemini MVIM document:
  1. Venue-isolated pattern extraction (Dataset H / Dataset A)
  2. Weighted intersection with Wilson bounds
  3. Conflict Rule: discard when |home% - away%| > 60
  4. IC Threshold: only surface patterns with combined_percentage >= 70
  5. Three-Pillar classification (Goals/Timing, Discipline, Outcome)
  6. League base-rate deviation scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import logger
from src.models.patterns import PatternStat, TeamPatternReport


# ==================================================================
# Three-Pillar Classification (MVIM Sections 3)
# ==================================================================

_PILLAR_GOALS = (
    "BTTS", "Over", "Under", "Goal in 1st Half", "Both Scored in 1H",
    "Team Scored in 1H", "Team Scored in 2H",
    "Conceded in 1H", "Conceded in 2H",
    "Team Scored", "Failed to Score",
    "Scored First", "Conceded First",
)

_PILLAR_DISCIPLINE = (
    "Card", "Over 2.5 Cards", "Over 3.5 Cards", "Over 4.5 Cards",
    "Over 5.5 Cards", "Under 2.5 Cards", "Under 3.5 Cards",
    "Under 4.5 Cards", "Under 5.5 Cards",
)

_PILLAR_OUTCOME = (
    "Team Win", "Draw", "Team Loss", "Clean Sheet",
    "HT Team Win", "HT Draw", "HT Team Loss", "HT Opponent Win",
    "Home Win", "Away Win", "Home Loss", "Away Loss",
    "HT Home Win", "HT Away Win", "HT Home Loss", "HT Away Loss",
)


def _classify_pillar(label: str) -> str:
    """Classify a pattern label into one of the MVIM's 3 pillars."""
    if any(label.startswith(p) for p in _PILLAR_DISCIPLINE):
        return "Discipline"
    if any(label.startswith(p) for p in _PILLAR_OUTCOME):
        return "Outcome"
    # Default to Goals/Timing (the largest category)
    return "Goals"


# ==================================================================
# Domain models
# ==================================================================

@dataclass
class IntersectionFactor:
    """
    A pattern that appears in BOTH teams' venue-specific data,
    passing the Conflict Rule and IC threshold gate.
    """
    label: str
    home_stat: PatternStat
    away_stat: PatternStat
    combined_percentage: float
    combined_wilson: float
    deviation_score: float
    pillar: str = "Goals"

    @property
    def stability_score(self) -> float:
        """The ultimate ranking metric: lower bound adjusted by deviation."""
        return self.combined_wilson + (self.deviation_score * 0.5)

    @property
    def confidence(self) -> str:
        """Combined confidence tier based on stability score."""
        if self.stability_score >= 75:
            return "Very High"
        if self.stability_score >= 60:
            return "High"
        if self.stability_score >= 45:
            return "Medium"
        if self.stability_score >= 30:
            return "Low"
        return "Very Low"

    @property
    def agreement_strength(self) -> str:
        gap = abs(self.home_stat.percentage - self.away_stat.percentage)
        if gap <= 10: return "Strong Agreement"
        if gap <= 25: return "Moderate Agreement"
        return "Weak Agreement"

    def __repr__(self) -> str:
        return (
            f"[{self.pillar}] {self.label}: "
            f"Combined {self.combined_percentage:.1f}% "
            f"[Wilson: {self.combined_wilson:.1f}% | Dev: {self.deviation_score:+.1f}%] "
            f"[{self.confidence}]"
        )


@dataclass
class MatchFactorReport:
    home_team: str
    away_team: str
    league_name: str
    season: str

    home_total_matches: int = 0
    away_total_matches: int = 0

    home_factors: list[PatternStat] = field(default_factory=list)
    away_factors: list[PatternStat] = field(default_factory=list)
    intersection: list[IntersectionFactor] = field(default_factory=list)
    conflicts_filtered: int = 0  # Count of conflicted patterns discarded

    def get_strong_intersections(self) -> list[IntersectionFactor]:
        return [f for f in self.intersection if f.confidence in ("High", "Very High")]

    def get_intersection_above(self, threshold: float) -> list[IntersectionFactor]:
        return [f for f in self.intersection if f.combined_percentage >= threshold]


# ==================================================================
# Label normalization & Baselines
# ==================================================================

_NORMALIZE_MAP: dict[str, str] = {
    "HT Home Win": "HT Team Win",
    "HT Away Win": "HT Opponent Win",
    "Home Win": "Team Win",
    "Away Win": "Team Win",
    "Home Loss": "Team Loss",
    "Away Loss": "Team Loss",
    "HT Home Loss": "HT Team Loss",
    "HT Away Loss": "HT Team Loss",
}

_MARKET_NEUTRAL_PREFIXES = (
    "BTTS", "Over", "Under", "Draw", "HT Draw",
    "Team Scored", "Failed to Score", "Clean Sheet",
    "Scored First", "Conceded First",
    "Team Scored in 1H", "Team Scored in 2H",
    "Conceded in 1H", "Conceded in 2H",
    "Goal in 1st Half", "Both Scored in 1H",
    "Card in 1st Half",
)

# Heuristic base rates for league averages
_BASELINES = {
    "BTTS - Yes": 53.0,
    "Over 1.5 Goals FT": 75.0,
    "Over 2.5 Goals FT": 52.0,
    "Over 3.5 Goals FT": 28.0,
    "Over 0.5 Goals HT": 68.0,
    "Over 1.5 Goals HT": 30.0,
    "Goal in 1st Half": 68.0,
}

def _get_baseline(label: str) -> float:
    for prefix, val in _BASELINES.items():
        if label.startswith(prefix):
            return val
    return 50.0  # Safe fallback

def _normalize_label(label: str) -> str:
    return _NORMALIZE_MAP.get(label, label)

def _is_matchable(label: str) -> bool:
    return any(label.startswith(prefix) for prefix in _MARKET_NEUTRAL_PREFIXES)


# ==================================================================
# MVIM Analyzer
# ==================================================================

# Conflict Rule constants (from Gemini MVIM Section 5)
CONFLICT_DISCARD_GAP = 60.0   # |home% - away%| > 60 → discard entirely
IC_MINIMUM_THRESHOLD = 70.0   # Only surface IC >= 70% (Gemini "Confidence Threshold")


class FactorAnalyzer:
    def analyze(
        self,
        home_report: TeamPatternReport,
        away_report: TeamPatternReport,
        min_wilson: float = 60.0,
    ) -> MatchFactorReport:
        logger.info("MVIM Analysis: %s (HOME) vs %s (AWAY)", home_report.team_name, away_report.team_name)

        home_factors = home_report.get_high_confidence_patterns(min_wilson=min_wilson)
        away_factors = away_report.get_high_confidence_patterns(min_wilson=min_wilson)

        intersection, conflicts = self._compute_intersection(home_factors, away_factors)

        logger.info(
            "MVIM Result: %d stable intersections | %d conflicts discarded",
            len(intersection), conflicts,
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
            conflicts_filtered=conflicts,
        )

    @staticmethod
    def _compute_intersection(
        home_factors: list[PatternStat],
        away_factors: list[PatternStat],
    ) -> tuple[list[IntersectionFactor], int]:
        away_lookup: dict[str, PatternStat] = {}
        for stat in away_factors:
            norm = _normalize_label(stat.label)
            if _is_matchable(stat.label):
                away_lookup[norm] = stat

        intersection: list[IntersectionFactor] = []
        conflicts = 0

        for home_stat in home_factors:
            if not _is_matchable(home_stat.label):
                continue
            norm = _normalize_label(home_stat.label)
            away_stat = away_lookup.get(norm)

            if away_stat is not None:
                # ── MVIM Conflict Rule (Section 5) ──────────────
                gap = abs(home_stat.percentage - away_stat.percentage)
                if gap > CONFLICT_DISCARD_GAP:
                    conflicts += 1
                    continue  # Conflicted — discard entirely

                # ── Weighted averages based on sample sizes ─────
                total_n = home_stat.total + away_stat.total
                hw = home_stat.total / total_n
                aw = away_stat.total / total_n

                comb_pct = (home_stat.percentage * hw) + (away_stat.percentage * aw)
                comb_wilson = (home_stat.wilson_lower_bound * hw) + (away_stat.wilson_lower_bound * aw)

                # ── MVIM IC Threshold (Section 5) ──────────────
                if comb_pct < IC_MINIMUM_THRESHOLD:
                    continue  # Below Gemini confidence gate

                # ── League base-rate deviation ─────────────────
                baseline = _get_baseline(home_stat.label)
                deviation = comb_pct - baseline

                # ── Conflict penalty for moderate disagreement ─
                conflict_penalty = max(0, (gap - 20) * 0.3)

                # ── Three-Pillar classification ────────────────
                pillar = _classify_pillar(home_stat.label)

                intersection.append(IntersectionFactor(
                    label=home_stat.label,
                    home_stat=home_stat,
                    away_stat=away_stat,
                    combined_percentage=round(comb_pct, 1),
                    combined_wilson=round(comb_wilson - conflict_penalty, 1),
                    deviation_score=round(deviation, 1),
                    pillar=pillar,
                ))

        # Sort by stability score descending
        intersection.sort(key=lambda f: f.stability_score, reverse=True)
        return intersection, conflicts
