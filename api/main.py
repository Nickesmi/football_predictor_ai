"""
Football Predictor AI - API v5.0
Fetches REAL daily matches from SofaScore. Computes per-match unique predictions
using Poisson (goals) + statistical models (corners, cards).

COVERS: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, UCL, UEL
"""

from __future__ import annotations

import json
import math
import urllib.request
from datetime import date, datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import logger, APIFOOTBALL_API_KEY
from src.data.api_football_fetcher import APIFootballFetcher
from src.processing.pattern_analyzer import PatternAnalyzer
from src.processing.factor_analyzer import FactorAnalyzer
from src.reporting.report_formatter import ReportFormatter
from src.processing.value_detector import ValueDetector
from src.ml.predictor import XGBoostPredictor
from src.ml.poisson_model import PoissonGoalModel
from src.ml.team_stats_db import get_team_stats
from src.ml.feature_builder import TeamProfile

app = FastAPI(title="Football Predictor AI", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline components
fetcher = APIFootballFetcher()
pattern_analyzer = PatternAnalyzer()
factor_analyzer = FactorAnalyzer()
value_detector = ValueDetector()
xgb_predictor = XGBoostPredictor()

# ── Top leagues (SofaScore uniqueTournament IDs) ─────────
TOP_LEAGUES = {
    17: "Premier League",
    8: "LaLiga",
    23: "Serie A",
    35: "Bundesliga",
    34: "Ligue 1",
    7: "Champions League",
    679: "Europa League",
    37: "Eredivisie",
    242: "MLS",
    155: "Liga Profesional",
}

# Map SofaScore league names → our profile keys
LEAGUE_NAME_MAP = {
    "Premier League": "Premier League",
    "LaLiga": "LaLiga",
    "La Liga": "LaLiga",
    "Serie A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
    "Champions League": "Champions League",
    "UEFA Champions League": "Champions League",
    "Europa League": "Champions League",
    "UEFA Europa League": "Champions League",
    "Eredivisie": "Eredivisie",
    "VriendenLoterij Eredivisie": "Eredivisie",
    "Liga Profesional de Fútbol": "Liga Profesional",
    "Liga Profesional": "Liga Profesional",
}


def _poisson_over(lam: float, threshold: int) -> float:
    """P(X > threshold) for Poisson distributed X with rate lam."""
    if lam <= 0:
        return 0.0
    cum = sum(math.exp(-lam) * (lam ** k) / math.factorial(k) for k in range(threshold + 1))
    return max(0.0, min(100.0, (1 - cum) * 100))


def _compute_match_analysis(home_name: str, away_name: str, league_name: str = "Premier League") -> dict:
    """
    Compute UNIQUE per-match analysis: Poisson goals + corners/cards.
    
    Every match gets its own λ based on team-specific attacking/defensive strengths.
    """
    # Normalize league key
    league_key = LEAGUE_NAME_MAP.get(league_name, league_name)
    
    # Step 1: Get team-specific stats (scored, conceded, corners, cards)
    home_stats = get_team_stats(home_name, "home", league_key)
    away_stats = get_team_stats(away_name, "away", league_key)
    
    logger.info(
        "⚽ %s (H: %.1f/%.1f, C:%.1f, K:%.1f) vs %s (A: %.1f/%.1f, C:%.1f, K:%.1f) [%s]",
        home_name, home_stats.scored, home_stats.conceded, home_stats.corners, home_stats.cards,
        away_name, away_stats.scored, away_stats.conceded, away_stats.corners, away_stats.cards,
        league_key,
    )
    
    # Step 2: Poisson goal model
    poisson = PoissonGoalModel(league_key)
    pred = poisson.predict(
        home_scored=home_stats.scored,
        home_conceded=home_stats.conceded,
        away_scored=away_stats.scored,
        away_conceded=away_stats.conceded,
        home_team=home_name,
        away_team=away_name,
    )
    
    # Step 3: Corners model (Poisson on expected corners)
    expected_home_corners = home_stats.corners
    expected_away_corners = away_stats.corners
    total_expected_corners = expected_home_corners + expected_away_corners
    
    corners_data = {
        "expected_home": round(expected_home_corners, 1),
        "expected_away": round(expected_away_corners, 1),
        "expected_total": round(total_expected_corners, 1),
        "markets": [
            {"market": "Over 7.5 Corners", "probability": round(_poisson_over(total_expected_corners, 7), 1)},
            {"market": "Under 7.5 Corners", "probability": round(100 - _poisson_over(total_expected_corners, 7), 1)},
            {"market": "Over 8.5 Corners", "probability": round(_poisson_over(total_expected_corners, 8), 1)},
            {"market": "Under 8.5 Corners", "probability": round(100 - _poisson_over(total_expected_corners, 8), 1)},
            {"market": "Over 9.5 Corners", "probability": round(_poisson_over(total_expected_corners, 9), 1)},
            {"market": "Under 9.5 Corners", "probability": round(100 - _poisson_over(total_expected_corners, 9), 1)},
            {"market": "Over 10.5 Corners", "probability": round(_poisson_over(total_expected_corners, 10), 1)},
            {"market": "Under 10.5 Corners", "probability": round(100 - _poisson_over(total_expected_corners, 10), 1)},
            {"market": "Over 11.5 Corners", "probability": round(_poisson_over(total_expected_corners, 11), 1)},
            {"market": "Under 11.5 Corners", "probability": round(100 - _poisson_over(total_expected_corners, 11), 1)},
        ],
    }
    
    # Step 4: Cards model (Poisson on expected cards)
    expected_home_cards = home_stats.cards
    expected_away_cards = away_stats.cards
    total_expected_cards = expected_home_cards + expected_away_cards
    
    cards_data = {
        "expected_home": round(expected_home_cards, 1),
        "expected_away": round(expected_away_cards, 1),
        "expected_total": round(total_expected_cards, 1),
        "markets": [
            {"market": "Over 2.5 Cards", "probability": round(_poisson_over(total_expected_cards, 2), 1)},
            {"market": "Under 2.5 Cards", "probability": round(100 - _poisson_over(total_expected_cards, 2), 1)},
            {"market": "Over 3.5 Cards", "probability": round(_poisson_over(total_expected_cards, 3), 1)},
            {"market": "Under 3.5 Cards", "probability": round(100 - _poisson_over(total_expected_cards, 3), 1)},
            {"market": "Over 4.5 Cards", "probability": round(_poisson_over(total_expected_cards, 4), 1)},
            {"market": "Under 4.5 Cards", "probability": round(100 - _poisson_over(total_expected_cards, 4), 1)},
            {"market": "Over 5.5 Cards", "probability": round(_poisson_over(total_expected_cards, 5), 1)},
            {"market": "Under 5.5 Cards", "probability": round(100 - _poisson_over(total_expected_cards, 5), 1)},
            {"market": "Over 6.5 Cards", "probability": round(_poisson_over(total_expected_cards, 6), 1)},
            {"market": "Under 6.5 Cards", "probability": round(100 - _poisson_over(total_expected_cards, 6), 1)},
        ],
    }
    
    # Step 5: XGBoost reinforcement
    home_profile = TeamProfile(
        team_name=home_name, matches_played=19,
        avg_scored=home_stats.scored, avg_conceded=home_stats.conceded,
        avg_total_goals=home_stats.scored + home_stats.conceded,
        btts_rate=round(pred.btts_yes / 100, 3),
        clean_sheet_rate=round(pred.home_clean_sheet / 100, 3),
        failed_to_score_rate=round(max(0.05, 1 - pred.over_0_5 / 100), 3),
        over_1_5_rate=round(pred.over_1_5 / 100, 3),
        over_2_5_rate=round(pred.over_2_5 / 100, 3),
        over_0_5_ht_rate=round(min(0.95, pred.over_1_5 / 100 * 0.85), 3),
        form_last5=round(pred.home_win / 100 * 12, 1),
        goal_diff=round((home_stats.scored - home_stats.conceded) * 19, 1),
    )
    away_profile = TeamProfile(
        team_name=away_name, matches_played=18,
        avg_scored=away_stats.scored, avg_conceded=away_stats.conceded,
        avg_total_goals=away_stats.scored + away_stats.conceded,
        btts_rate=round(pred.btts_yes / 100, 3),
        clean_sheet_rate=round(pred.away_clean_sheet / 100, 3),
        failed_to_score_rate=round(max(0.05, 1 - pred.over_0_5 / 100), 3),
        over_1_5_rate=round(pred.over_1_5 / 100, 3),
        over_2_5_rate=round(pred.over_2_5 / 100, 3),
        over_0_5_ht_rate=round(min(0.95, pred.over_1_5 / 100 * 0.85), 3),
        form_last5=round(pred.away_win / 100 * 12, 1),
        goal_diff=round((away_stats.scored - away_stats.conceded) * 18, 1),
    )
    xgb_pred = xgb_predictor.predict(home_profile, away_profile)
    
    # Step 6: Value detections
    value_selections = []
    market_odds_map = {
        "Over 1.5 Goals": (pred.over_1_5, 1.35),
        "Over 2.5 Goals": (pred.over_2_5, 1.95),
        "BTTS - Yes":     (pred.btts_yes, 1.80),
        "Over 9.5 Corners": (corners_data["markets"][2]["probability"], 1.85),
        "Over 3.5 Cards": (cards_data["markets"][1]["probability"], 1.70),
    }
    for pattern, (prob, odds) in market_odds_map.items():
        implied = (1 / odds) * 100
        edge = round(prob - implied, 1)
        if edge > 10:
            verdict = "Best Choice"
        elif edge > 5:
            verdict = "Value"
        elif edge > -2:
            verdict = "Fair"
        else:
            continue
        pillar = "Corners" if "Corner" in pattern else "Cards" if "Card" in pattern else "Goals"
        value_selections.append({
            "pattern": pattern, "pillar": pillar,
            "ic": round(prob, 1), "implied_probability": round(implied, 1),
            "value_edge": edge, "verdict": verdict,
            "stability": round(prob * 0.85, 1),
            "confidence": "Very High" if prob >= 75 else "High" if prob >= 60 else "Medium",
        })
    
    # Step 7: Strongest edge ranking
    value_selections.sort(key=lambda v: v["value_edge"], reverse=True)
    
    # Step 8: Top Confident Picks (>70%)
    all_markets = []
    # Add Goals Markets
    all_markets.extend([
        {"market": "Over 0.5 Goals", "probability": round(pred.over_0_5, 1)},
        {"market": "Under 0.5 Goals", "probability": round(pred.under_0_5, 1)},
        {"market": "Over 1.5 Goals", "probability": round(pred.over_1_5, 1)},
        {"market": "Under 1.5 Goals", "probability": round(pred.under_1_5, 1)},
        {"market": "Over 2.5 Goals", "probability": round(pred.over_2_5, 1)},
        {"market": "Under 2.5 Goals", "probability": round(pred.under_2_5, 1)},
        {"market": "Over 3.5 Goals", "probability": round(pred.over_3_5, 1)},
        {"market": "Under 3.5 Goals", "probability": round(pred.under_3_5, 1)},
        {"market": "Over 4.5 Goals", "probability": round(pred.over_4_5, 1)},
        {"market": "Under 4.5 Goals", "probability": round(100 - pred.over_4_5, 1)},
        {"market": "BTTS - Yes", "probability": round(pred.btts_yes, 1)},
        {"market": "BTTS - No", "probability": round(pred.btts_no, 1)}
    ])
    
    max_result = max(pred.home_win, pred.draw, pred.away_win)
    if max_result != pred.away_win:
        all_markets.append({"market": f"1X ({home_name} or Draw)", "probability": round(pred.home_win + pred.draw, 1)})
    if max_result != pred.home_win:
        all_markets.append({"market": f"X2 ({away_name} or Draw)", "probability": round(pred.away_win + pred.draw, 1)})
    if max_result != pred.draw:
        all_markets.append({"market": "12 (Any Team to Win)", "probability": round(pred.home_win + pred.away_win, 1)})
    # Add Corners Markets
    all_markets.extend(corners_data["markets"])
    # Add Cards Markets
    all_markets.extend(cards_data["markets"])
    
    import random
    
    # We now return ALL markets with confidence strictly higher than 75%
    top_6_confident = [m for m in all_markets if m["probability"] > 75]
    
    # Sort them descending so the absolutely most confident are at the top
    top_6_confident.sort(key=lambda x: x["probability"], reverse=True)
    
    # Shuffle the list so they don't appear in strictly descending order
    random.shuffle(top_6_confident)
    
    return {
        "disclaimer": f"Hybrid Engine v5 — Poisson (λ={pred.lambda_home:.2f}+{pred.lambda_away:.2f}) + XGBoost | {league_key}",
        "poisson": pred.to_dict(),
        "corners": corners_data,
        "cards": cards_data,
        "xgboost_predictions": xgb_pred.to_dict().get("predictions", []),
        "value_selections": value_selections,
        "top_6_confident": top_6_confident,
        "averages": {
            "home": {
                "avg_goals_scored": home_stats.scored,
                "avg_goals_conceded": home_stats.conceded,
                "avg_goals_total": round(home_stats.scored + home_stats.conceded, 1),
                "avg_corners": home_stats.corners,
                "avg_cards": home_stats.cards,
            },
            "away": {
                "avg_goals_scored": away_stats.scored,
                "avg_goals_conceded": away_stats.conceded,
                "avg_goals_total": round(away_stats.scored + away_stats.conceded, 1),
                "avg_corners": away_stats.corners,
                "avg_cards": away_stats.cards,
            },
        },
    }


def _fetch_sofascore_events(date_str: str) -> list[dict]:
    url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date_str}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        return data.get("events", [])
    except Exception as e:
        logger.error(f"SofaScore fetch failed: {e}")
        return []


def _sofascore_to_fixture(event: dict) -> dict:
    tournament = event.get("tournament", {})
    unique_tournament = tournament.get("uniqueTournament", {})
    ut_id = unique_tournament.get("id", 0)
    category = tournament.get("category", {})
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    status = event.get("status", {})
    
    ts = event.get("startTimestamp", 0)
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        from datetime import timedelta
        local_dt = dt + timedelta(hours=3)
        time_str = local_dt.strftime("%H:%M")
        date_str = local_dt.strftime("%Y-%m-%d")
    except Exception:
        time_str = "TBD"
        date_str = ""
    
    status_type = status.get("type", "")
    if status_type == "finished": short_status = "FT"
    elif status_type == "inprogress": short_status = "LIVE"
    elif status_type == "notstarted": short_status = "NS"
    else: short_status = status.get("description", "")[:4]
    
    home_score = event.get("homeScore", {})
    away_score = event.get("awayScore", {})
    
    return {
        "id": str(event.get("id", "")),
        "date": date_str,
        "time": time_str,
        "status": short_status,
        "home_goals": home_score.get("current") if status_type != "notstarted" else None,
        "away_goals": away_score.get("current") if status_type != "notstarted" else None,
        "league": {
            "id": str(ut_id),
            "name": unique_tournament.get("name", tournament.get("name", "")),
            "country": category.get("name", ""),
            "logo": f"https://api.sofascore.com/api/v1/unique-tournament/{ut_id}/image",
        },
        "home_team": {
            "id": str(home.get("id", "")),
            "name": home.get("name", ""),
            "logo": f"https://api.sofascore.com/api/v1/team/{home.get('id', 0)}/image",
        },
        "away_team": {
            "id": str(away.get("id", "")),
            "name": away.get("name", ""),
            "logo": f"https://api.sofascore.com/api/v1/team/{away.get('id', 0)}/image",
        },
    }


# ── Endpoints ──────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "data_source": "sofascore",
        "analysis_mode": "live" if APIFOOTBALL_API_KEY else "per_match_poisson",
        "engine": "Hybrid Poisson Goals + Corners + Cards v5.0",
        "leagues": list(TOP_LEAGUES.values()),
    }


@app.get("/api/leagues")
def get_supported_leagues():
    return [{"id": str(k), "name": v} for k, v in TOP_LEAGUES.items()]


@app.get("/api/fixtures/{date_str}")
def get_fixtures_by_date(date_str: str):
    events = _fetch_sofascore_events(date_str)
    if not events:
        logger.warning(f"No events from SofaScore for {date_str}")
        return []
    
    fixtures = []
    for ev in events:
        ut = ev.get("tournament", {}).get("uniqueTournament", {})
        ut_id = ut.get("id", 0)
        if ut_id in TOP_LEAGUES:
            fixtures.append(_sofascore_to_fixture(ev))
    
    league_order = list(TOP_LEAGUES.keys())
    fixtures.sort(key=lambda f: (
        league_order.index(int(f["league"]["id"])) if int(f["league"]["id"]) in league_order else 999,
        f["time"]
    ))
    
    logger.info(f"Returning {len(fixtures)} fixtures for {date_str}")
    return fixtures


@app.get("/api/fixtures/today")
def get_today_fixtures():
    return get_fixtures_by_date(date.today().isoformat())


@app.get("/api/analysis/match/{fixture_id}")
def analyze_match(
    fixture_id: str, 
    home: str = "", 
    away: str = "", 
    league: str = "Premier League",
    status: str = "",
    start_time: str = ""
):
    """Per-match prediction. Every match gets UNIQUE probabilities."""
    home_name = home or "Unknown Home"
    away_name = away or "Unknown Away"
    
    if not APIFOOTBALL_API_KEY:
        logger.info(f"Per-match engine: {home_name} vs {away_name} [{league}]")
        analysis = _compute_match_analysis(
            home_name=home_name, 
            away_name=away_name, 
            league_name=league
        )
        analysis["match"] = {
            "home_team": home_name,
            "away_team": away_name,
            "league_name": league,
            "season": "2024/25",
            "date": date.today().isoformat(),
        }
        return analysis

    try:
        raw_data = fetcher.client.get("/fixtures", params={"id": int(fixture_id)})
        response = raw_data.get("response", [])
        if not response:
            raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found.")
        fix_info = response[0]
        home_id = fix_info["teams"]["home"]["id"]
        away_id = fix_info["teams"]["away"]["id"]
        league_id = fix_info["league"]["id"]
        season = fix_info["league"]["season"]
        home_matches, away_matches = fetcher.fetch_match_context(
            home_team_id=home_id, away_team_id=away_id,
            league_id=league_id, season=season,
        )
        home_report = pattern_analyzer.analyze(home_matches)
        away_report = pattern_analyzer.analyze(away_matches)
        factor_report = factor_analyzer.analyze(home_report, away_report, min_wilson=60.0)
        formatter = ReportFormatter(confidence_min_wilson=60.0)
        report_dict = formatter.format_dict(factor_report, home_report, away_report)
        report_dict["match"]["home_team_logo"] = fix_info["teams"]["home"]["logo"]
        report_dict["match"]["away_team_logo"] = fix_info["teams"]["away"]["logo"]
        report_dict["match"]["league_logo"] = fix_info["league"]["logo"]
        report_dict["match"]["date"] = fix_info["fixture"]["date"]
        return report_dict
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing fixture {fixture_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Results Verification ──────────────────────────

def _fetch_event_statistics(event_id: str) -> dict:
    """Fetch match statistics (corners, cards) from SofaScore for a finished event."""
    url = f"https://api.sofascore.com/api/v1/event/{event_id}/statistics"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        statistics = data.get("statistics", [])

        total_corners = 0
        yellow_cards = 0
        red_cards = 0

        # SofaScore returns statistics grouped by period (ALL, 1ST, 2ND)
        for period in statistics:
            period_name = period.get("period", "")
            if period_name != "ALL":
                continue
            groups = period.get("groups", [])
            for group in groups:
                group_name = group.get("groupName", "")
                items = group.get("statisticsItems", [])
                for item in items:
                    stat_name = item.get("name", "")
                    home_val = item.get("home", "0")
                    away_val = item.get("away", "0")
                    # Parse values (they can be strings like "5" or "43%")
                    try:
                        h = int(str(home_val).replace("%", ""))
                        a = int(str(away_val).replace("%", ""))
                    except (ValueError, TypeError):
                        continue

                    if stat_name == "Corner kicks":
                        total_corners = h + a
                    elif stat_name in ("Yellow cards", "Total yellow cards"):
                        yellow_cards += h + a
                    elif stat_name in ("Red cards", "Total red cards"):
                        red_cards += h + a

        total_cards = yellow_cards + red_cards
        return {"corners": total_corners, "cards": total_cards, "yellow_cards": yellow_cards, "red_cards": red_cards}
    except Exception as e:
        logger.warning(f"Could not fetch stats for event {event_id}: {e}")
        return {"corners": None, "cards": None, "yellow_cards": None, "red_cards": None}


def _evaluate_prediction(pick: dict, home_goals: int, away_goals: int,
                          total_corners: int | None, total_cards: int | None) -> dict:
    """
    Evaluate a single predicted market against actual match results.
    Returns the pick dict enriched with 'result': True/False/None.
    """
    market = pick.get("market", "")
    total_goals = home_goals + away_goals
    result = None  # None = cannot determine (e.g., missing stats)

    # ── Goals markets ──
    if market == "Over 0.5 Goals":
        result = total_goals > 0.5
    elif market == "Under 0.5 Goals":
        result = total_goals < 0.5
    elif market == "Over 1.5 Goals":
        result = total_goals > 1.5
    elif market == "Under 1.5 Goals":
        result = total_goals < 1.5
    elif market == "Over 2.5 Goals":
        result = total_goals > 2.5
    elif market == "Under 2.5 Goals":
        result = total_goals < 2.5
    elif market == "Over 3.5 Goals":
        result = total_goals > 3.5
    elif market == "Under 3.5 Goals":
        result = total_goals < 3.5
    elif market == "Over 4.5 Goals":
        result = total_goals > 4.5
    elif market == "Under 4.5 Goals":
        result = total_goals < 4.5

    # ── BTTS ──
    elif market == "BTTS - Yes":
        result = home_goals > 0 and away_goals > 0
    elif market == "BTTS - No":
        result = not (home_goals > 0 and away_goals > 0)

    # ── 1X / X2 / 12 ──
    elif "1X" in market:
        result = home_goals >= away_goals  # Home win or draw
    elif "X2" in market:
        result = away_goals >= home_goals  # Away win or draw
    elif "12" in market:
        result = home_goals != away_goals  # Any team wins (not draw)

    # ── Corners ──
    elif "Corners" in market:
        if total_corners is not None:
            if market == "Over 7.5 Corners":
                result = total_corners > 7.5
            elif market == "Under 7.5 Corners":
                result = total_corners < 7.5
            elif market == "Over 8.5 Corners":
                result = total_corners > 8.5
            elif market == "Under 8.5 Corners":
                result = total_corners < 8.5
            elif market == "Over 9.5 Corners":
                result = total_corners > 9.5
            elif market == "Under 9.5 Corners":
                result = total_corners < 9.5
            elif market == "Over 10.5 Corners":
                result = total_corners > 10.5
            elif market == "Under 10.5 Corners":
                result = total_corners < 10.5
            elif market == "Over 11.5 Corners":
                result = total_corners > 11.5
            elif market == "Under 11.5 Corners":
                result = total_corners < 11.5

    # ── Cards ──
    elif "Cards" in market:
        if total_cards is not None:
            if market == "Over 2.5 Cards":
                result = total_cards > 2.5
            elif market == "Under 2.5 Cards":
                result = total_cards < 2.5
            elif market == "Over 3.5 Cards":
                result = total_cards > 3.5
            elif market == "Under 3.5 Cards":
                result = total_cards < 3.5
            elif market == "Over 4.5 Cards":
                result = total_cards > 4.5
            elif market == "Under 4.5 Cards":
                result = total_cards < 4.5
            elif market == "Over 5.5 Cards":
                result = total_cards > 5.5
            elif market == "Under 5.5 Cards":
                result = total_cards < 5.5
            elif market == "Over 6.5 Cards":
                result = total_cards > 6.5
            elif market == "Under 6.5 Cards":
                result = total_cards < 6.5

    return {
        **pick,
        "result": result,
    }


@app.get("/api/results/{date_str}")
def get_results_verification(date_str: str):
    """
    For all finished matches on a given date, regenerate predictions
    and compare them against actual results.
    Returns per-match breakdown of correct/wrong top-6 picks.
    """
    events = _fetch_sofascore_events(date_str)
    if not events:
        return {"date": date_str, "matches": [], "summary": {}}

    results = []
    total_correct = 0
    total_wrong = 0
    total_unknown = 0
    total_picks = 0

    for ev in events:
        ut = ev.get("tournament", {}).get("uniqueTournament", {})
        ut_id = ut.get("id", 0)
        if ut_id not in TOP_LEAGUES:
            continue

        status = ev.get("status", {})
        if status.get("type") != "finished":
            continue

        fixture = _sofascore_to_fixture(ev)
        home_goals = fixture.get("home_goals")
        away_goals = fixture.get("away_goals")

        if home_goals is None or away_goals is None:
            continue

        home_name = fixture["home_team"]["name"]
        away_name = fixture["away_team"]["name"]
        league_name = fixture["league"]["name"]
        event_id = fixture["id"]

        # Fetch actual match statistics (corners, cards)
        stats = _fetch_event_statistics(event_id)
        total_corners = stats.get("corners")
        total_cards = stats.get("cards")
        yellow_cards = stats.get("yellow_cards")
        red_cards = stats.get("red_cards")

        # Regenerate predictions for this match
        try:
            analysis = _compute_match_analysis(home_name, away_name, league_name)
            top_picks = analysis.get("top_6_confident", [])
        except Exception as e:
            logger.warning(f"Could not compute analysis for {home_name} vs {away_name}: {e}")
            continue

        # Evaluate each pick
        evaluated_picks = []
        match_correct = 0
        match_wrong = 0
        match_unknown = 0

        for pick in top_picks:
            evaluated = _evaluate_prediction(pick, home_goals, away_goals, total_corners, total_cards)
            evaluated_picks.append(evaluated)
            if evaluated["result"] is True:
                match_correct += 1
            elif evaluated["result"] is False:
                match_wrong += 1
            else:
                match_unknown += 1

        total_correct += match_correct
        total_wrong += match_wrong
        total_unknown += match_unknown
        total_picks += len(evaluated_picks)

        results.append({
            "fixture": fixture,
            "actual": {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "total_goals": home_goals + away_goals,
                "total_corners": total_corners,
                "total_cards": total_cards,
                "yellow_cards": yellow_cards,
                "red_cards": red_cards,
            },
            "picks": evaluated_picks,
            "summary": {
                "correct": match_correct,
                "wrong": match_wrong,
                "unknown": match_unknown,
                "total": len(evaluated_picks),
            },
        })

    accuracy = round((total_correct / total_picks * 100), 1) if total_picks > 0 else 0.0

    return {
        "date": date_str,
        "matches": results,
        "summary": {
            "total_matches": len(results),
            "total_picks": total_picks,
            "total_correct": total_correct,
            "total_wrong": total_wrong,
            "total_unknown": total_unknown,
            "accuracy_pct": accuracy,
        },
    }
