"""
Pipeline orchestrator: runs all agents in sequence for a given date.

    1. Fixture Collector  → fetch & filter matches, store in DB
    2. Odds Fetcher        → fetch real bookmaker odds from The Odds API
    3. Probability Engine  → estimate market probabilities
    3b. Calibration        → adjust raw probs using historical performance
    4. Market Value         → find edge vs bookmaker odds
    5. Risk Control         → score confidence, filter by league thresholds
    6. Grading & Sizing     → assign grade + stake
    7. Store picks in DB
"""

import logging
import sqlite3
from typing import Optional

from src.db.database import get_db
from src.db.picks_repo import insert_pick, get_picks_by_date
from src.engine.fixture_collector import collect_fixtures
from src.engine.probability_engine import estimate_probabilities
from src.engine.market_value import find_value, filter_positive_edge
from src.engine.risk_control import get_league_profile, score_confidence, apply_risk_filter
from src.engine.calibration import ProbabilityCalibrator
from src.data.odds_fetcher import fetch_and_store_odds, get_api_key
from src.models.pick import Pick, assign_grade

logger = logging.getLogger("football_predictor")

# Singleton calibrator
_calibrator = ProbabilityCalibrator(n_bins=10)


def run_pipeline(
    date_str: str,
    raw_events: list[dict],
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """
    Run the full investment pipeline for a date.

    Args:
        date_str: "YYYY-MM-DD"
        raw_events: raw SofaScore event list (from _fetch_sofascore_events)
        conn: optional DB connection (defaults to singleton)

    Returns:
        {
            "date": str,
            "total_matches": int,
            "tracked_matches": int,
            "candidates_found": int,
            "picks": [Pick.to_display_dict(), ...],
            "summary": { grade counts, total stake }
        }
    """
    if conn is None:
        conn = get_db()

    # ── Agent 1: Fixture Collector ───────────────────────────────
    fixtures = collect_fixtures(raw_events, date_str, conn)
    if not fixtures:
        logger.warning(f"No tracked fixtures for {date_str}")
        return _empty_result(date_str, len(raw_events))

    # ── Agent 2: Odds Fetcher ────────────────────────────────────
    # Fetch real bookmaker odds if API key is available
    odds_status = "no_api_key"
    if get_api_key():
        try:
            league_ids = list(set(m.get("league_id", 0) for m in fixtures))
            odds_result = fetch_and_store_odds(conn, league_ids)
            odds_status = f"fetched:{sum(odds_result.values())}_matches"
        except Exception as e:
            logger.error(f"Odds fetch failed: {e}")
            odds_status = f"error:{e}"
    else:
        logger.info("ODDS_API_KEY not set — skipping real odds fetch")

    # ── Agent 3: Probability Engine ──────────────────────────────
    # Fit calibrator from historical data (if available)
    _calibrator.fit_from_db(conn)

    match_probs = {}
    for match in fixtures:
        try:
            probs = estimate_probabilities(match)

            # Apply calibration to all market probabilities
            for market in ["1X2", "O/U 2.5", "BTTS"]:
                if market in probs:
                    for sel in probs[market]:
                        raw = probs[market][sel]
                        probs[market][sel] = round(
                            _calibrator.calibrate(raw), 4
                        )

            match_probs[match["id"]] = probs
        except Exception as e:
            logger.error(f"Probability engine failed for {match['id']}: {e}")

    # ── Agent 4: Market Value ────────────────────────────────────
    all_candidates = []
    for match in fixtures:
        if match["id"] not in match_probs:
            continue
        probs = match_probs[match["id"]]
        candidates = find_value(match, probs, conn)
        positive = filter_positive_edge(candidates)
        all_candidates.extend(positive)

    # ── Agent 5: Risk Control ────────────────────────────────────
    for candidate in all_candidates:
        league_id = candidate.get("league_id", 0)
        profile = get_league_profile(conn, league_id)
        model_source = match_probs.get(candidate["match_id"], {}).get("source", "fallback")
        score_confidence(candidate, profile, model_source)

    # Apply risk filter (removes candidates below league-specific edge threshold)
    filtered = apply_risk_filter(all_candidates)

    # ── Grading & Sizing ─────────────────────────────────────────
    picks: list[Pick] = []
    for c in filtered:
        grade, stake = assign_grade(
            edge=c["edge"],
            confidence=c["confidence"],
            league_reliability=c["league_reliability"],
        )
        if grade == "Pass":
            continue

        pick = Pick(
            match_id=c["match_id"],
            home_team=c["home_team"],
            away_team=c["away_team"],
            league_name=c["league_name"],
            market=c["market"],
            selection=c["selection"],
            model_prob=c["model_prob"],
            implied_prob=c["implied_prob"],
            edge=c["edge"],
            odds_at_pick=c["odds"],
            confidence=c["confidence"],
            league_reliability=c["league_reliability"],
            grade=grade,
            stake_units=min(stake, c.get("max_stake_units", 2.0)),
        )
        picks.append(pick)

    # ── Store picks in DB ────────────────────────────────────────
    for pick in picks:
        try:
            insert_pick(conn, pick.to_db_dict())
        except Exception as e:
            logger.error(f"Failed to store pick: {e}")

    # ── Build result ─────────────────────────────────────────────
    grade_counts = {}
    total_stake = 0.0
    for p in picks:
        grade_counts[p.grade] = grade_counts.get(p.grade, 0) + 1
        total_stake += p.stake_units

    result = {
        "date": date_str,
        "total_events": len(raw_events),
        "tracked_matches": len(fixtures),
        "odds_status": odds_status,
        "candidates_found": len(all_candidates),
        "picks_after_filter": len(filtered),
        "final_picks": len(picks),
        "picks": [p.to_display_dict() for p in picks],
        "summary": {
            "grades": grade_counts,
            "total_stake_units": round(total_stake, 2),
            "avg_edge": round(
                sum(p.edge for p in picks) / max(len(picks), 1) * 100, 1
            ),
            "avg_confidence": round(
                sum(p.confidence for p in picks) / max(len(picks), 1), 3
            ),
        },
    }

    logger.info(
        f"Pipeline complete for {date_str}: "
        f"{result['tracked_matches']} matches → "
        f"{result['candidates_found']} candidates → "
        f"{result['final_picks']} picks "
        f"({result['summary']['total_stake_units']}u total) "
        f"[odds: {odds_status}]"
    )

    return result


def _empty_result(date_str: str, total_events: int) -> dict:
    return {
        "date": date_str,
        "total_events": total_events,
        "tracked_matches": 0,
        "odds_status": "no_fixtures",
        "candidates_found": 0,
        "picks_after_filter": 0,
        "final_picks": 0,
        "picks": [],
        "summary": {
            "grades": {},
            "total_stake_units": 0,
            "avg_edge": 0,
            "avg_confidence": 0,
        },
    }
