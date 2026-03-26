"""
Optional LLM-based Natural Language Formatter (Issue #4).

Converts the structured statistical data into a fluent, natural-language
report using an LLM as a FORMATTING TOOL ONLY.

CRITICAL DESIGN CONSTRAINT:
  The LLM is given ONLY the computed statistics and is instructed to
  rewrite them as readable prose. It must NOT:
    - Make predictions or guesses
    - Add information not present in the data
    - Use opinion words ("likely", "should", "predicted")
    - Imply any causation or future results

  The system prompt enforces this strictly. The LLM is a
  "statistical copywriter", not a prediction engine.

Usage:
    formatter = LLMReportFormatter(api_key="...")
    prose = formatter.format_prose(factor_report, home_report, away_report)

If no LLM API key is available, use ReportFormatter instead.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from src.config import logger
from src.models.patterns import TeamPatternReport
from src.processing.factor_analyzer import MatchFactorReport
from src.reporting.report_formatter import ReportFormatter


# The system prompt that constrains the LLM to pure formatting
_SYSTEM_PROMPT = """\
You are a statistical report writer for football match analysis. Your ONLY job
is to convert structured statistical data into clear, natural-language prose.

STRICT RULES:
1. You must ONLY state facts that are present in the provided data.
2. You must NEVER predict, guess, or speculate about future outcomes.
3. You must NEVER use words like "likely", "should", "expected to", "predicted",
   "will", "probable", or any other predictive language.
4. Frame everything as historical patterns: "has occurred in X% of matches",
   "historically shows", "the data indicates that in past matches".
5. You must ALWAYS include the disclaimer that these are historical patterns,
   not predictions.
6. Report percentages exactly as provided — do not round or modify them.
7. Do not add any information not present in the data.
8. Do not express opinions about which team is "better" or "worse".
9. Structure your output with clear sections and bullet points.
10. Be concise but thorough — cover all major patterns in the data.

You are a formatting tool, not an analyst. Convert numbers to readable prose.
"""

_USER_PROMPT_TEMPLATE = """\
Please convert the following statistical match analysis into a clear,
natural-language report. Remember: ONLY format the data below into prose.
Do NOT predict, guess, or add any information not present in the data.

Match: {home_team} (Home) vs {away_team} (Away)
League: {league} | Season: {season}
Home matches analyzed: {home_matches}
Away matches analyzed: {away_matches}

=== KEY AVERAGES ===
{averages_section}

=== PATTERN INTERSECTION (Combined >= {threshold}%) ===
These patterns appear in BOTH teams' historical data:
{intersection_section}

=== TOP HOME FACTORS ({home_team}) ===
{home_factors_section}

=== TOP AWAY FACTORS ({away_team}) ===
{away_factors_section}

Please write this as a readable report with the following sections:
1. Match Overview (teams, league, sample size)
2. Key Statistical Patterns (the intersection — where both teams' data aligns)
3. Home Team Profile (top factors)
4. Away Team Profile (top factors)
5. Disclaimer (historical data, not predictions)
"""


class LLMReportFormatter:
    """
    Formats statistical reports into natural language using an LLM.

    The LLM is used STRICTLY as a formatting tool — all content
    comes from the computed statistics. The system prompt forbids
    any predictions or guesses.

    Supports OpenAI-compatible APIs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.3,  # Low temperature for deterministic output
    ):
        """
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use (default: gpt-4o-mini for cost efficiency).
            base_url: API endpoint (supports OpenAI-compatible APIs).
            temperature: Low value for more deterministic formatting.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self._fallback = ReportFormatter()

    def format_prose(
        self,
        report: MatchFactorReport,
        home_report: Optional[TeamPatternReport] = None,
        away_report: Optional[TeamPatternReport] = None,
        threshold: float = 65.0,
    ) -> str:
        """
        Generate a natural-language report from the factor analysis.

        If the LLM API is unavailable, falls back to the deterministic
        ReportFormatter.

        Args:
            report: The match factor report.
            home_report: Detailed home pattern report.
            away_report: Detailed away pattern report.
            threshold: Confidence threshold for intersection.

        Returns:
            A natural-language prose report string.
        """
        if not self.api_key:
            logger.warning(
                "No LLM API key set. Falling back to deterministic formatter."
            )
            return self._fallback.format_text(report, home_report, away_report)

        # Build the structured data for the LLM
        user_prompt = self._build_prompt(
            report, home_report, away_report, threshold
        )

        try:
            return self._call_llm(user_prompt)
        except Exception as e:
            logger.error("LLM formatting failed: %s. Using fallback.", e)
            return self._fallback.format_text(report, home_report, away_report)

    def _build_prompt(
        self,
        report: MatchFactorReport,
        home_report: Optional[TeamPatternReport],
        away_report: Optional[TeamPatternReport],
        threshold: float,
    ) -> str:
        """Build the user prompt with all statistical data."""

        # Averages section
        averages_lines = []
        if home_report and away_report:
            hg = home_report.goals
            ag = away_report.goals
            averages_lines.extend([
                f"  {report.home_team} (Home): Avg Scored {hg.avg_goals_scored}, "
                f"Avg Conceded {hg.avg_goals_conceded}, Avg Total {hg.avg_goals_ft}",
                f"  {report.away_team} (Away): Avg Scored {ag.avg_goals_scored}, "
                f"Avg Conceded {ag.avg_goals_conceded}, Avg Total {ag.avg_goals_ft}",
            ])
            if home_report.corners.avg_corners_total > 0:
                averages_lines.append(
                    f"  Corners: Home avg {home_report.corners.avg_corners_total}, "
                    f"Away avg {away_report.corners.avg_corners_total}"
                )
            if home_report.cards.avg_yellow_total > 0:
                averages_lines.append(
                    f"  Yellow Cards: Home avg {home_report.cards.avg_yellow_total}, "
                    f"Away avg {away_report.cards.avg_yellow_total}"
                )
        averages_section = "\n".join(averages_lines) if averages_lines else "  No averages available."

        # Intersection section
        high = report.get_intersection_above(threshold)
        intersection_lines = []
        for f in high:
            intersection_lines.append(
                f"  - {f.label}: Home {f.home_stat.percentage:.1f}% "
                f"({f.home_stat.count}/{f.home_stat.total}) + "
                f"Away {f.away_stat.percentage:.1f}% "
                f"({f.away_stat.count}/{f.away_stat.total}) = "
                f"Combined {f.combined_percentage:.1f}% "
                f"[{f.confidence}] ({f.agreement_strength})"
            )
        intersection_section = (
            "\n".join(intersection_lines)
            if intersection_lines
            else "  No patterns above threshold."
        )

        # Home / Away factors
        home_lines = [
            f"  - {p.label}: {p.count}/{p.total} ({p.percentage:.1f}%) [{p.confidence}]"
            for p in report.home_factors[:10]
        ]
        away_lines = [
            f"  - {p.label}: {p.count}/{p.total} ({p.percentage:.1f}%) [{p.confidence}]"
            for p in report.away_factors[:10]
        ]

        return _USER_PROMPT_TEMPLATE.format(
            home_team=report.home_team,
            away_team=report.away_team,
            league=report.league_name,
            season=report.season,
            home_matches=report.home_total_matches,
            away_matches=report.away_total_matches,
            threshold=threshold,
            averages_section=averages_section,
            intersection_section=intersection_section,
            home_factors_section="\n".join(home_lines) or "  No data.",
            away_factors_section="\n".join(away_lines) or "  No data.",
        )

    def _call_llm(self, user_prompt: str) -> str:
        """Call the OpenAI-compatible chat completions API."""
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 2000,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    @property
    def system_prompt(self) -> str:
        """Expose the system prompt for testing/auditing."""
        return _SYSTEM_PROMPT
