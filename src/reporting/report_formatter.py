"""
Deterministic Report Formatter (Issue #4).

Converts the MatchFactorReport (from Issue #3) into structured,
human-readable output formats:
  - Plain text (terminal-friendly)
  - Markdown (for documentation, chat, or web display)
  - Structured dict (for JSON serialization / API responses)

IMPORTANT: This module is PURELY DETERMINISTIC. It does NOT guess
or predict anything. Every statement in the output is derived
directly from computed statistical percentages. No ML, no LLM,
no speculation — just formatted facts.

Architecture:
    MatchFactorReport (Issue #3)
         ↓
    ReportFormatter.format_text()    → plain text string
    ReportFormatter.format_markdown() → markdown string
    ReportFormatter.format_dict()     → dict (JSON-serializable)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.models.patterns import PatternStat, TeamPatternReport
from src.processing.factor_analyzer import (
    IntersectionFactor,
    MatchFactorReport,
)


class ReportFormatter:
    """
    Formats a MatchFactorReport into human-readable output.

    All output is strictly based on computed statistics.
    No predictions, no guesses, no ML inference.
    """

    def __init__(self, confidence_threshold: float = 65.0):
        """
        Args:
            confidence_threshold: Minimum combined % for intersection
                                  patterns to appear in the report.
        """
        self.threshold = confidence_threshold

    # ==================================================================
    # Plain Text
    # ==================================================================

    def format_text(
        self,
        report: MatchFactorReport,
        home_report: Optional[TeamPatternReport] = None,
        away_report: Optional[TeamPatternReport] = None,
    ) -> str:
        """
        Format the full analysis as a plain-text report.

        Args:
            report: The factor analysis report.
            home_report: Optional detailed home pattern report (for averages).
            away_report: Optional detailed away pattern report (for averages).

        Returns:
            A multi-line plain text string.
        """
        lines: list[str] = []
        sep = "=" * 72
        thin_sep = "-" * 72

        # Header
        lines.append(sep)
        lines.append(f"  MATCH ANALYSIS REPORT")
        lines.append(f"  {report.home_team} (HOME) vs {report.away_team} (AWAY)")
        lines.append(f"  {report.league_name} | Season {report.season}")
        lines.append(
            f"  Based on: {report.home_total_matches} home matches "
            f"& {report.away_total_matches} away matches"
        )
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(sep)

        # Averages section
        if home_report and away_report:
            lines.append("")
            lines.append("  KEY AVERAGES")
            lines.append(f"  {thin_sep[:60]}")
            hg = home_report.goals
            ag = away_report.goals
            lines.append(
                f"    {report.home_team} (Home): "
                f"Avg Scored {hg.avg_goals_scored} | "
                f"Avg Conceded {hg.avg_goals_conceded} | "
                f"Avg Total {hg.avg_goals_ft}"
            )
            lines.append(
                f"    {report.away_team} (Away): "
                f"Avg Scored {ag.avg_goals_scored} | "
                f"Avg Conceded {ag.avg_goals_conceded} | "
                f"Avg Total {ag.avg_goals_ft}"
            )
            if home_report.corners.avg_corners_total > 0:
                lines.append(
                    f"    Corners: Home avg {home_report.corners.avg_corners_total} | "
                    f"Away avg {away_report.corners.avg_corners_total}"
                )
            if home_report.cards.avg_yellow_total > 0:
                lines.append(
                    f"    Yellow Cards: Home avg {home_report.cards.avg_yellow_total} | "
                    f"Away avg {away_report.cards.avg_yellow_total}"
                )

        # Home factors
        lines.append("")
        lines.append(f"  TOP FACTORS: {report.home_team} (HOME)")
        lines.append(f"  {thin_sep[:60]}")
        for p in report.home_factors[:10]:
            lines.append(self._format_stat_line(p))

        # Away factors
        lines.append("")
        lines.append(f"  TOP FACTORS: {report.away_team} (AWAY)")
        lines.append(f"  {thin_sep[:60]}")
        for p in report.away_factors[:10]:
            lines.append(self._format_stat_line(p))

        # Intersection
        high = report.get_intersection_above(self.threshold)
        lines.append("")
        lines.append(
            f"  PATTERN INTERSECTION (Combined >= {self.threshold:.0f}%)"
        )
        lines.append(f"  {thin_sep[:60]}")
        if high:
            for f in high:
                lines.append(self._format_intersection_line(f))
        else:
            lines.append(
                f"    No patterns found above {self.threshold:.0f}% combined confidence."
            )

        # Strong agreements
        strong = [
            f for f in high
            if f.agreement_strength == "Strong Agreement"
        ]
        if strong:
            lines.append("")
            lines.append("  STRONGEST AGREEMENTS (both teams' data closely aligned)")
            lines.append(f"  {thin_sep[:60]}")
            for f in strong[:5]:
                lines.append(
                    f"    * {f.label}: {f.combined_percentage:.1f}% "
                    f"(Home {f.home_stat.percentage:.0f}% | "
                    f"Away {f.away_stat.percentage:.0f}%)"
                )

        # Disclaimer
        lines.append("")
        lines.append(sep)
        lines.append(
            "  DISCLAIMER: This report is based purely on historical"
        )
        lines.append(
            "  statistical patterns. All percentages reflect past occurrence"
        )
        lines.append(
            "  rates, NOT predictions. Past performance does not guarantee"
        )
        lines.append("  future results.")
        lines.append(sep)

        return "\n".join(lines)

    # ==================================================================
    # Markdown
    # ==================================================================

    def format_markdown(
        self,
        report: MatchFactorReport,
        home_report: Optional[TeamPatternReport] = None,
        away_report: Optional[TeamPatternReport] = None,
    ) -> str:
        """Format the full analysis as a Markdown document."""
        lines: list[str] = []

        # Header
        lines.append(
            f"# Match Analysis: {report.home_team} vs {report.away_team}"
        )
        lines.append("")
        lines.append(
            f"**{report.league_name}** | Season **{report.season}** | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        lines.append("")
        lines.append(
            f"Based on **{report.home_total_matches}** home matches "
            f"({report.home_team}) and **{report.away_total_matches}** "
            f"away matches ({report.away_team})."
        )

        # Averages
        if home_report and away_report:
            lines.append("")
            lines.append("## Key Averages")
            lines.append("")
            lines.append("| Metric | " + report.home_team + " (Home) | " + report.away_team + " (Away) |")
            lines.append("|--------|" + "-" * (len(report.home_team) + 9) + "|" + "-" * (len(report.away_team) + 9) + "|")
            hg = home_report.goals
            ag = away_report.goals
            lines.append(f"| Avg Goals Scored | {hg.avg_goals_scored} | {ag.avg_goals_scored} |")
            lines.append(f"| Avg Goals Conceded | {hg.avg_goals_conceded} | {ag.avg_goals_conceded} |")
            lines.append(f"| Avg Total Goals | {hg.avg_goals_ft} | {ag.avg_goals_ft} |")
            if home_report.corners.avg_corners_total > 0:
                lines.append(
                    f"| Avg Corners (Total) | {home_report.corners.avg_corners_total} "
                    f"| {away_report.corners.avg_corners_total} |"
                )
            if home_report.cards.avg_yellow_total > 0:
                lines.append(
                    f"| Avg Yellow Cards | {home_report.cards.avg_yellow_total} "
                    f"| {away_report.cards.avg_yellow_total} |"
                )

        # Intersection — the main event
        high = report.get_intersection_above(self.threshold)
        lines.append("")
        lines.append(
            f"## Pattern Intersection (Combined ≥ {self.threshold:.0f}%)"
        )
        lines.append("")
        if high:
            lines.append("| Pattern | Home % | Away % | Combined % | Confidence | Agreement |")
            lines.append("|---------|--------|--------|------------|------------|-----------|")
            for f in high:
                icon = "🟢" if f.combined_percentage >= 80 else "🟡" if f.combined_percentage >= 65 else "⚪"
                lines.append(
                    f"| {icon} {f.label} | {f.home_stat.percentage:.1f}% "
                    f"| {f.away_stat.percentage:.1f}% "
                    f"| **{f.combined_percentage:.1f}%** "
                    f"| {f.confidence} | {f.agreement_strength} |"
                )
        else:
            lines.append(
                f"*No patterns found above {self.threshold:.0f}% combined confidence.*"
            )

        # Home factors
        lines.append("")
        lines.append(f"## {report.home_team} — Home Factors")
        lines.append("")
        lines.append("| Pattern | Occurrences | Percentage | Confidence |")
        lines.append("|---------|-------------|------------|------------|")
        for p in report.home_factors[:10]:
            icon = "🟢" if p.percentage >= 80 else "🟡" if p.percentage >= 65 else "⚪"
            lines.append(
                f"| {icon} {p.label} | {p.count}/{p.total} "
                f"| {p.percentage:.1f}% | {p.confidence} |"
            )

        # Away factors
        lines.append("")
        lines.append(f"## {report.away_team} — Away Factors")
        lines.append("")
        lines.append("| Pattern | Occurrences | Percentage | Confidence |")
        lines.append("|---------|-------------|------------|------------|")
        for p in report.away_factors[:10]:
            icon = "🟢" if p.percentage >= 80 else "🟡" if p.percentage >= 65 else "⚪"
            lines.append(
                f"| {icon} {p.label} | {p.count}/{p.total} "
                f"| {p.percentage:.1f}% | {p.confidence} |"
            )

        # Disclaimer
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            "> **Disclaimer:** This report is based purely on historical "
            "statistical patterns. All percentages reflect past occurrence "
            "rates, NOT predictions. Past performance does not guarantee "
            "future results."
        )

        return "\n".join(lines)

    # ==================================================================
    # Dict (JSON-serializable)
    # ==================================================================

    def format_dict(
        self,
        report: MatchFactorReport,
        home_report: Optional[TeamPatternReport] = None,
        away_report: Optional[TeamPatternReport] = None,
    ) -> dict:
        """
        Format the analysis as a JSON-serializable dictionary.

        Useful for API responses, storage, or further processing.
        """
        high = report.get_intersection_above(self.threshold)

        result: dict = {
            "match": {
                "home_team": report.home_team,
                "away_team": report.away_team,
                "league": report.league_name,
                "season": report.season,
                "home_matches_analyzed": report.home_total_matches,
                "away_matches_analyzed": report.away_total_matches,
            },
            "generated_at": datetime.now().isoformat(),
            "confidence_threshold": self.threshold,
            "intersection": [
                {
                    "pattern": f.label,
                    "home_percentage": f.home_stat.percentage,
                    "away_percentage": f.away_stat.percentage,
                    "combined_percentage": f.combined_percentage,
                    "confidence": f.confidence,
                    "agreement": f.agreement_strength,
                    "home_occurrences": f"{f.home_stat.count}/{f.home_stat.total}",
                    "away_occurrences": f"{f.away_stat.count}/{f.away_stat.total}",
                }
                for f in high
            ],
            "home_factors": [
                {
                    "pattern": p.label,
                    "percentage": p.percentage,
                    "occurrences": f"{p.count}/{p.total}",
                    "confidence": p.confidence,
                }
                for p in report.home_factors[:15]
            ],
            "away_factors": [
                {
                    "pattern": p.label,
                    "percentage": p.percentage,
                    "occurrences": f"{p.count}/{p.total}",
                    "confidence": p.confidence,
                }
                for p in report.away_factors[:15]
            ],
        }

        # Add averages if available
        if home_report and away_report:
            result["averages"] = {
                "home": {
                    "avg_goals_scored": home_report.goals.avg_goals_scored,
                    "avg_goals_conceded": home_report.goals.avg_goals_conceded,
                    "avg_goals_total": home_report.goals.avg_goals_ft,
                    "avg_corners_total": home_report.corners.avg_corners_total,
                    "avg_yellow_cards": home_report.cards.avg_yellow_total,
                },
                "away": {
                    "avg_goals_scored": away_report.goals.avg_goals_scored,
                    "avg_goals_conceded": away_report.goals.avg_goals_conceded,
                    "avg_goals_total": away_report.goals.avg_goals_ft,
                    "avg_corners_total": away_report.corners.avg_corners_total,
                    "avg_yellow_cards": away_report.cards.avg_yellow_total,
                },
            }

        result["disclaimer"] = (
            "This report is based purely on historical statistical patterns. "
            "All percentages reflect past occurrence rates, NOT predictions. "
            "Past performance does not guarantee future results."
        )

        return result

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _format_stat_line(p: PatternStat) -> str:
        icon = ">>>" if p.percentage >= 80 else " >>" if p.percentage >= 65 else "  >"
        return (
            f"    {icon} {p.label}: "
            f"{p.count}/{p.total} ({p.percentage:.1f}%) "
            f"[{p.confidence}]"
        )

    @staticmethod
    def _format_intersection_line(f: IntersectionFactor) -> str:
        icon = ">>>" if f.combined_percentage >= 80 else " >>" if f.combined_percentage >= 65 else "  >"
        return (
            f"    {icon} {f.label}: "
            f"Home {f.home_stat.percentage:.0f}% + "
            f"Away {f.away_stat.percentage:.0f}% = "
            f"Combined {f.combined_percentage:.1f}% "
            f"[{f.confidence}] ({f.agreement_strength})"
        )
