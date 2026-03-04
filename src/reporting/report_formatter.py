"""
Deterministic Report Formatter (Issue #4).

Converts the MatchFactorReport (from Issue #3) into structured,
human-readable output formats. Now upgraded to include Wilson Confidence Bounds 
and statistical Deviation Scores.
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
    def __init__(self, confidence_threshold: float = 65.0):
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
        lines: list[str] = []
        sep = "=" * 72
        thin_sep = "-" * 72

        lines.append(sep)
        lines.append(f"  MATCH ANALYSIS REPORT")
        lines.append(f"  {report.home_team} (HOME) vs {report.away_team} (AWAY)")
        lines.append(f"  {report.league_name} | Season {report.season}")
        lines.append(f"  Based on: {report.home_total_matches} home matches & {report.away_total_matches} away matches")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(sep)

        high = report.get_intersection_above(self.threshold) if hasattr(report, 'get_intersection_above') else [f for f in report.intersection if f.confidence in ("High", "Very High")]
        
        lines.append("")
        lines.append(f"  PATTERN INTERSECTION (High Confidence)")
        lines.append(f"  {thin_sep[:60]}")
        if high:
            for f in high:
                lines.append(self._format_intersection_line(f))
        else:
            lines.append("    No highly stable patterns found.")

        lines.append("")
        lines.append(sep)
        lines.append("  DISCLAIMER: This report is based purely on historical statistical patterns with")
        lines.append("  Wilson bounds and base-rate deviation. Past performance does not guarantee future results.")
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
        lines: list[str] = []

        lines.append(f"# Match Analysis: {report.home_team} vs {report.away_team}")
        lines.append("")
        lines.append(f"**{report.league_name}** | Season **{report.season}** | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        high = report.get_intersection_above(self.threshold) if hasattr(report, 'get_intersection_above') else [f for f in report.intersection if f.confidence in ("High", "Very High")]

        lines.append("## Stable Recommended Patterns")
        lines.append("")
        if high:
            lines.append("| Pattern | Stability Score | Wilson Bound | Base Dev. | Confidence |")
            lines.append("|---------|-----------------|--------------|-----------|------------|")
            for f in high:
                icon = "🟢" if f.stability_score >= 75 else "🟡" if f.stability_score >= 60 else "⚪"
                lines.append(
                    f"| {icon} {f.label} | **{f.stability_score:.1f}** | {f.combined_wilson:.1f}% | {f.deviation_score:+.1f}% | {f.confidence} |"
                )
        else:
            lines.append("*No highly stable patterns found strictly passing statistical bounds.*")

        lines.append("")
        lines.append("---")
        lines.append("> **Disclaimer:** This report is based purely on historical statistical patterns, utilizing Wilson lower bounds and base-rate deviations to filter noise.")

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
        high = report.get_intersection_above(self.threshold) if hasattr(report, 'get_intersection_above') else [f for f in report.intersection if f.confidence in ("Medium", "High", "Very High")]

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
            "intersection": [
                {
                    "pattern": f.label,
                    "home_percentage": f.home_stat.percentage,
                    "away_percentage": f.away_stat.percentage,
                    "combined_percentage": f.combined_percentage,
                    "stability_score": getattr(f, 'stability_score', 0.0),
                    "wilson_bound": getattr(f, 'combined_wilson', 0.0),
                    "base_deviation": getattr(f, 'deviation_score', 0.0),
                    "confidence": f.confidence,
                    "agreement": getattr(f, 'agreement_strength', ""),
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

        if home_report and away_report:
            result["averages"] = {
                "home": {
                    "avg_goals_scored": home_report.goals.avg_goals_scored,
                    "avg_goals_conceded": home_report.goals.avg_goals_conceded,
                    "avg_goals_total": home_report.goals.avg_goals_ft,
                },
                "away": {
                    "avg_goals_scored": away_report.goals.avg_goals_scored,
                    "avg_goals_conceded": away_report.goals.avg_goals_conceded,
                    "avg_goals_total": away_report.goals.avg_goals_ft,
                },
            }

        result["disclaimer"] = (
            "This report is strictly based on historical statistical patterns. "
            "Predictions now employ Wilson Confidence Intervals, minimum sample "
            "size thresholds, and League Normalization deviation tracking to ensure stability."
        )

        return result

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _format_intersection_line(f: IntersectionFactor) -> str:
        icon = ">>>" if getattr(f, 'stability_score', 0) >= 75 else " >>" if getattr(f, 'stability_score', 0) >= 60 else "  >"
        return (
            f"    {icon} {f.label}: "
            f"Stability {getattr(f, 'stability_score', 0):.1f} | "
            f"Wilson Bound {getattr(f, 'combined_wilson', 0):.1f}% | "
            f"Dev {getattr(f, 'deviation_score', 0):+.1f}% "
            f"[{f.confidence}]"
        )
