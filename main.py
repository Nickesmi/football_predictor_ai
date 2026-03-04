"""
Football Predictor AI  –  Main Entry Point

Usage:
    python main.py --home "Manchester United" --away "Chelsea" --league "Premier League" --season 2024

For Issue #1, this script demonstrates the data fetching pipeline:
  1. Search for team & league IDs
  2. Fetch all HOME matches of Team A in League L
  3. Fetch all AWAY matches of Team B in League L
  4. Print a summary of the fetched data
"""

from __future__ import annotations

import argparse
import sys
from textwrap import dedent

from src.config import logger, APIFOOTBALL_API_KEY
from src.data.api_football_fetcher import APIFootballFetcher
from src.models.match import TeamMatchSet


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

    # Quick aggregate stats
    total = match_set.total_matches
    btts_count = sum(1 for m in match_set.matches if m.btts)
    o25_count = sum(1 for m in match_set.matches if m.over_2_5)
    wins = sum(
        1 for m in match_set.matches
        if (match_set.context == "home" and m.home_win)
        or (match_set.context == "away" and m.away_win)
    )
    draws = sum(1 for m in match_set.matches if m.draw)
    losses = total - wins - draws

    print(f"\n  Summary ({total} matches):")
    print(f"    W/D/L:  {wins}/{draws}/{losses}")
    print(f"    BTTS:   {btts_count}/{total} ({100*btts_count/total:.0f}%)")
    print(f"    O2.5:   {o25_count}/{total} ({100*o25_count/total:.0f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Football Predictor AI – Data Fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              python main.py --home "Manchester United" --away "Chelsea" --league "Premier League" --season 2024
              python main.py --home-id 33 --away-id 49 --league-id 39 --season 2024
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

    # ---- Fetch data ----
    print(f"\n🔄 Fetching match data for season {args.season}...")
    home_matches, away_matches = fetcher.fetch_match_context(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        league_id=league_id,
        season=args.season,
    )

    # ---- Display results ----
    print_match_set_summary(home_matches)
    print_match_set_summary(away_matches)

    print("✅ Data collection complete. Ready for feature engineering (Issue #2).\n")


if __name__ == "__main__":
    main()
