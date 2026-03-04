"""
Football Predictor AI  –  Main Entry Point

Usage:
    python main.py --home "Manchester United" --away "Chelsea" --league "Premier League" --season 2024

Implements the full pipeline:
  1. Search for team & league IDs (Issue #1)
  2. Fetch all HOME matches of Team A in League L (Issue #1)
  3. Fetch all AWAY matches of Team B in League L (Issue #1)
  4. Compute statistical patterns for each set (Issue #2)
  5. Display high-confidence patterns
"""

from __future__ import annotations

import argparse
import sys
from textwrap import dedent

from src.config import logger, APIFOOTBALL_API_KEY
from src.data.api_football_fetcher import APIFootballFetcher
from src.models.match import TeamMatchSet
from src.models.patterns import TeamPatternReport
from src.processing.pattern_analyzer import PatternAnalyzer


def print_match_set_summary(match_set: TeamMatchSet) -> None:
    """Print a human-readable summary of a TeamMatchSet."""
    print(f"\n{'='*70}")
    print(f"  {match_set}")
    print(f"{'='*70}")

    if not match_set.matches:
        print("  No finished matches found.\n")
        return

    for i, m in enumerate(match_set.matches, 1):
        ht_str = ""
        if m.home_score_ht is not None and m.away_score_ht is not None:
            ht_str = f" (HT: {m.home_score_ht}-{m.away_score_ht})"

        stats_str = ""
        if m.statistics:
            stats_str = (
                f" | Corners: {m.statistics.corners_home}-{m.statistics.corners_away}"
                f" | YC: {m.statistics.yellow_cards_home}-{m.statistics.yellow_cards_away}"
            )

        print(
            f"  {i:>2}. {m.match_date} | {m.home_team_name} {m.home_score_ft}-{m.away_score_ft} "
            f"{m.away_team_name}{ht_str} | BTTS: {'✓' if m.btts else '✗'} "
            f"| O2.5: {'✓' if m.over_2_5 else '✗'}{stats_str}"
        )
    print()


def print_pattern_report(report: TeamPatternReport, threshold: float = 60.0) -> None:
    """Print the full pattern analysis report."""
    ctx = report.context.upper()
    print(f"\n{'━'*70}")
    print(f"  📊 PATTERN ANALYSIS: {report.team_name} ({ctx})")
    print(f"     {report.league_name} {report.season} | {report.total_matches} matches")
    print(f"{'━'*70}")

    if report.total_matches == 0:
        print("  No data to analyze.\n")
        return

    # ---- Goals ----
    g = report.goals
    print(f"\n  ⚽ GOALS")
    print(f"     Avg Goals FT: {g.avg_goals_ft} | Avg Goals HT: {g.avg_goals_ht}")
    print(f"     Avg Scored: {g.avg_goals_scored} | Avg Conceded: {g.avg_goals_conceded}")
    print(f"     {g.btts_yes}")
    for stat in [g.over_0_5_ft, g.over_1_5_ft, g.over_2_5_ft, g.over_3_5_ft, g.over_4_5_ft]:
        if stat:
            print(f"     {stat}")
    if g.over_0_5_ht:
        print(f"     --- Half Time ---")
        for stat in [g.over_0_5_ht, g.over_1_5_ht, g.over_2_5_ht]:
            if stat:
                print(f"     {stat}")

    # ---- Results ----
    r = report.results
    print(f"\n  🏆 RESULTS (W/D/L)")
    for stat in [r.wins, r.draws, r.losses]:
        if stat:
            print(f"     {stat}")
    if r.ht_wins:
        print(f"     --- Half Time ---")
        for stat in [r.ht_wins, r.ht_draws, r.ht_losses]:
            if stat:
                print(f"     {stat}")
    if r.ht_ft_distribution:
        print(f"     --- HT/FT Combos ---")
        for combo, stat in list(r.ht_ft_distribution.items())[:5]:
            print(f"     {stat}")

    # ---- Team Scoring ----
    s = report.scoring
    print(f"\n  🎯 TEAM SCORING")
    for stat in [s.scored_in_match, s.failed_to_score, s.clean_sheet,
                 s.scored_first, s.conceded_first]:
        if stat:
            print(f"     {stat}")
    print(f"     --- By Half ---")
    for stat in [s.scored_in_1h, s.scored_in_2h, s.conceded_in_1h, s.conceded_in_2h]:
        if stat:
            print(f"     {stat}")

    # ---- Corners ----
    c = report.corners
    if c.avg_corners_total > 0:
        print(f"\n  🚩 CORNERS")
        print(f"     Avg Total: {c.avg_corners_total} | Team: {c.avg_corners_team} | Opponent: {c.avg_corners_opponent}")
        for stat in [c.over_7_5, c.over_8_5, c.over_9_5, c.over_10_5, c.over_11_5]:
            if stat:
                print(f"     {stat}")

    # ---- Cards ----
    cd = report.cards
    if cd.avg_yellow_total > 0:
        print(f"\n  🟨 CARDS")
        print(f"     Avg Yellow Total: {cd.avg_yellow_total} | Team: {cd.avg_yellow_team} | Opponent: {cd.avg_yellow_opponent}")
        for stat in [cd.over_2_5_cards, cd.over_3_5_cards, cd.over_4_5_cards, cd.over_5_5_cards]:
            if stat:
                print(f"     {stat}")
        if cd.cards_in_1h:
            print(f"     {cd.cards_in_1h}")

    # ---- First Half Events ----
    fh = report.first_half
    if fh.goals_in_1h:
        print(f"\n  ⏱️  FIRST HALF EVENTS")
        for stat in [fh.goals_in_1h, fh.both_scored_1h, fh.over_0_5_goals_1h,
                     fh.over_1_5_goals_1h, fh.cards_in_1h, fh.over_0_5_cards_1h,
                     fh.over_1_5_cards_1h]:
            if stat:
                print(f"     {stat}")

    # ---- High Confidence Summary ----
    high = report.get_high_confidence_patterns(threshold=threshold)
    if high:
        print(f"\n  🔥 HIGH CONFIDENCE PATTERNS (≥{threshold:.0f}%)")
        for p in high:
            icon = "🟢" if p.percentage >= 80 else "🟡"
            print(f"     {icon} {p}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Football Predictor AI – Statistical Pattern Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              python main.py --home "Manchester United" --away "Chelsea" --league "Premier League" --season 2024
              python main.py --home-id 33 --away-id 49 --league-id 39 --season 2024
              python main.py --home-id 33 --away-id 49 --league-id 39 --season 2024 --threshold 75
        """),
    )

    # Team identification – by name or by ID
    parser.add_argument("--home", type=str, help="Home team name (will search API)")
    parser.add_argument("--away", type=str, help="Away team name (will search API)")
    parser.add_argument("--home-id", type=int, help="Home team ID (skip search)")
    parser.add_argument("--away-id", type=int, help="Away team ID (skip search)")

    # League identification
    parser.add_argument("--league", type=str, help="League name (will search API)")
    parser.add_argument("--league-id", type=int, help="League ID (skip search)")
    parser.add_argument("--country", type=str, help="Country filter for league search")

    # Season
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 2024)")

    # Analysis options
    parser.add_argument(
        "--threshold", type=float, default=65.0,
        help="Confidence threshold for high-confidence patterns (default: 65.0)"
    )

    args = parser.parse_args()

    # Validate API key
    if not APIFOOTBALL_API_KEY:
        print(
            "❌ ERROR: APIFOOTBALL_API_KEY is not set.\n"
            "Please set it in your .env file or as an environment variable.\n"
            "Get a free key at: https://www.api-football.com/ or https://rapidapi.com/api-sports/api/api-football\n"
        )
        sys.exit(1)

    fetcher = APIFootballFetcher()
    analyzer = PatternAnalyzer()

    # ---- Resolve team IDs ----
    home_team_id = args.home_id
    away_team_id = args.away_id

    if not home_team_id:
        if not args.home:
            print("❌ ERROR: Provide --home (team name) or --home-id (team ID)")
            sys.exit(1)
        teams = fetcher.search_team(args.home)
        if not teams:
            print(f"❌ No teams found matching '{args.home}'")
            sys.exit(1)
        home_team_id = teams[0]["id"]
        print(f"✓ Home team: {teams[0]['name']} (ID: {home_team_id})")

    if not away_team_id:
        if not args.away:
            print("❌ ERROR: Provide --away (team name) or --away-id (team ID)")
            sys.exit(1)
        teams = fetcher.search_team(args.away)
        if not teams:
            print(f"❌ No teams found matching '{args.away}'")
            sys.exit(1)
        away_team_id = teams[0]["id"]
        print(f"✓ Away team: {teams[0]['name']} (ID: {away_team_id})")

    # ---- Resolve league ID ----
    league_id = args.league_id
    if not league_id:
        if not args.league:
            print("❌ ERROR: Provide --league (name) or --league-id (ID)")
            sys.exit(1)
        leagues = fetcher.search_league(args.league, country=args.country)
        if not leagues:
            print(f"❌ No leagues found matching '{args.league}'")
            sys.exit(1)
        league_id = leagues[0]["id"]
        print(f"✓ League: {leagues[0]['name']} (ID: {league_id})")

    # ---- Fetch data (Issue #1) ----
    print(f"\n🔄 Fetching match data for season {args.season}...")
    home_matches, away_matches = fetcher.fetch_match_context(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        league_id=league_id,
        season=args.season,
    )

    # ---- Display raw matches ----
    print_match_set_summary(home_matches)
    print_match_set_summary(away_matches)

    # ---- Analyze patterns (Issue #2) ----
    print(f"\n🔬 Computing statistical patterns...")
    home_report = analyzer.analyze(home_matches)
    away_report = analyzer.analyze(away_matches)

    print_pattern_report(home_report, threshold=args.threshold)
    print_pattern_report(away_report, threshold=args.threshold)

    print("✅ Analysis complete. Ready for factor intersection (Issue #3).\n")


if __name__ == "__main__":
    main()
