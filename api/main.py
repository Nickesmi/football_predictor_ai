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
    238: "Primeira Liga",
    238: "Primeira Liga",
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


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF: P(X = k) for rate lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _poisson_over(lam: float, threshold: int) -> float:
    """P(X > threshold) for Poisson distributed X with rate lam."""
    if lam <= 0:
        return 0.0
    cum = sum(math.exp(-lam) * (lam ** k) / math.factorial(k) for k in range(threshold + 1))
    return max(0.0, min(100.0, (1 - cum) * 100))


# ── Reference bookmaker odds for standardised market lines ────────────
# Represent typical Pinnacle-style sharp-book prices (≈5 % total overround).
# Used to compute edge = model_probability − implied_probability.
_MARKET_REF_ODDS: dict[str, float] = {
    # Full-time goal totals
    "Over 0.5 Goals":         1.09,
    "Under 0.5 Goals":        8.00,
    "Over 1.5 Goals":         1.37,
    "Under 1.5 Goals":        3.10,
    "Over 2.5 Goals":         1.90,
    "Under 2.5 Goals":        1.95,
    "Over 3.5 Goals":         2.90,
    "Under 3.5 Goals":        1.38,
    "Over 4.5 Goals":         4.50,
    "Under 4.5 Goals":        1.18,
    "BTTS - Yes":             1.80,
    "BTTS - No":              1.95,
    # First-half goal totals
    "FH Over 0.5 Goals":      1.42,
    "FH Under 0.5 Goals":     2.70,
    "FH Over 1.5 Goals":      2.80,
    "FH Under 1.5 Goals":     1.42,
    # Corner totals
    "Over 7.5 Corners":       1.65,
    "Under 7.5 Corners":      2.20,
    "Over 8.5 Corners":       1.90,
    "Under 8.5 Corners":      1.90,
    "Over 9.5 Corners":       2.20,
    "Under 9.5 Corners":      1.65,
    "Over 10.5 Corners":      2.55,
    "Under 10.5 Corners":     1.50,
    "Over 11.5 Corners":      3.10,
    "Under 11.5 Corners":     1.38,
    # Yellow card totals
    "Over 2.5 Yellow Cards":  1.72,
    "Under 2.5 Yellow Cards": 2.10,
    "Over 3.5 Yellow Cards":  2.10,
    "Under 3.5 Yellow Cards": 1.72,
    "Over 4.5 Yellow Cards":  2.80,
    "Under 4.5 Yellow Cards": 1.40,
    "Over 5.5 Yellow Cards":  4.00,
    "Under 5.5 Yellow Cards": 1.22,
}

# Stability weight per category (how reliably the model predicts each type)
_CATEGORY_STABILITY: dict[str, float] = {
    "result":        0.90,
    "double_chance": 0.92,
    "goal":          0.85,
    "corners":       0.72,
    "cards":         0.65,
    "handicap":      0.82,
    "advanced":      0.75,
}

# Bookmaker overround applied to result / double-chance markets
# (no live odds → proxy: 10 % overround on each side of these markets)
_RESULT_VIG = 0.10
# Standard Asian-Handicap reference odds (Pinnacle pricing)
_AH_REF_ODDS = 1.88
# Team-supremacy reference odds (which team has more X)
_SUPREMACY_REF_ODDS = 1.90
# Fallback odds returned when implied probability is near zero (extreme favourite)
_FALLBACK_ODDS = 99.0


def _compute_team_supremacy(
    lam_home: float, lam_away: float, max_k: int = 20
) -> tuple[float, float, float]:
    """
    Return (P(home > away), P(home == away), P(away > home)) in percent,
    modelling both quantities as independent Poisson variables.
    """
    home_gt = tie = away_gt = 0.0
    for h in range(max_k + 1):
        ph = _poisson_pmf(h, lam_home)
        for a in range(max_k + 1):
            joint = ph * _poisson_pmf(a, lam_away)
            if h > a:
                home_gt += joint
            elif h == a:
                tie += joint
            else:
                away_gt += joint
    return round(home_gt * 100, 1), round(tie * 100, 1), round(away_gt * 100, 1)


def _build_market_candidates(
    pred,
    corners_data: dict,
    cards_data: dict,
    home_name: str,
    away_name: str,
    expected_home_corners: float,
    expected_away_corners: float,
    expected_home_cards: float,
    expected_away_cards: float,
) -> list[dict]:
    """
    Assemble the full universe of market candidates, each with:
    model probability, reference bookmaker odds, implied probability, edge,
    stability score, and category label.
    """
    candidates: list[dict] = []

    def _stability(category: str, prob: float) -> float:
        base = _CATEGORY_STABILITY.get(category, 0.70)
        return round(base * min(prob / 70.0, 1.0) * 100, 1)

    def _confidence(prob: float, edge: float) -> str:
        if prob >= 78 and edge >= 10:
            return "Very High"
        if prob >= 65 and edge >= 5:
            return "High"
        return "Medium"

    def add(market: str, category: str, prob: float, ref_odds: float) -> None:
        if ref_odds <= 1.0 or prob <= 0:
            return
        implied = round(100.0 / ref_odds, 1)
        edge = round(prob - implied, 1)
        candidates.append({
            "market": market,
            "category": category,
            "probability": round(prob, 1),
            "odds": round(ref_odds, 2),
            "implied_prob": implied,
            "edge": edge,
            "stability": _stability(category, prob),
            "confidence": _confidence(prob, edge),
        })

    def result_odds(prob: float) -> float:
        """Proxy bookmaker odds: 10 % overround applied to model prob."""
        implied = prob * (1.0 - _RESULT_VIG)
        return round(100.0 / implied, 2) if implied > 0 else _FALLBACK_ODDS

    # ── 1. 1X2 Result markets ──────────────────────────────────────────
    add("Home Win", "result", pred.home_win, result_odds(pred.home_win))
    add("Draw", "result", pred.draw, result_odds(pred.draw))
    add("Away Win", "result", pred.away_win, result_odds(pred.away_win))

    # ── 2. Double Chance markets ───────────────────────────────────────
    dc_1x = round(pred.home_win + pred.draw, 1)
    dc_x2 = round(pred.away_win + pred.draw, 1)
    dc_12 = round(pred.home_win + pred.away_win, 1)
    add("1X (Home or Draw)", "double_chance", dc_1x, result_odds(dc_1x))
    add("X2 (Away or Draw)", "double_chance", dc_x2, result_odds(dc_x2))
    add("12 (Any Team to Win)", "double_chance", dc_12, result_odds(dc_12))

    # ── 3. Full-time goal markets ──────────────────────────────────────
    for name, prob in [
        ("Over 0.5 Goals",   pred.over_0_5),
        ("Under 0.5 Goals",  pred.under_0_5),
        ("Over 1.5 Goals",   pred.over_1_5),
        ("Under 1.5 Goals",  pred.under_1_5),
        ("Over 2.5 Goals",   pred.over_2_5),
        ("Under 2.5 Goals",  pred.under_2_5),
        ("Over 3.5 Goals",   pred.over_3_5),
        ("Under 3.5 Goals",  pred.under_3_5),
        ("Over 4.5 Goals",   pred.over_4_5),
        ("Under 4.5 Goals",  100 - pred.over_4_5),
        ("BTTS - Yes",       pred.btts_yes),
        ("BTTS - No",        pred.btts_no),
    ]:
        if name in _MARKET_REF_ODDS:
            add(name, "goal", prob, _MARKET_REF_ODDS[name])

    # ── 4. First-half goal markets ─────────────────────────────────────
    fh_over_0_5  = pred.fh_over_0_5
    fh_over_1_5  = pred.fh_over_1_5
    for name, prob in [
        ("FH Over 0.5 Goals",  fh_over_0_5),
        ("FH Under 0.5 Goals", round(100 - fh_over_0_5, 1)),
        ("FH Over 1.5 Goals",  fh_over_1_5),
        ("FH Under 1.5 Goals", round(100 - fh_over_1_5, 1)),
    ]:
        if name in _MARKET_REF_ODDS:
            add(name, "goal", prob, _MARKET_REF_ODDS[name])

    # ── 5. Corners O/U totals ─────────────────────────────────────────
    for m in corners_data["markets"]:
        name = m["market"]
        if name in _MARKET_REF_ODDS:
            add(name, "corners", m["probability"], _MARKET_REF_ODDS[name])

    # ── 6. Yellow cards O/U totals ────────────────────────────────────
    for m in cards_data["markets"]:
        name = m["market"]
        if name in _MARKET_REF_ODDS:
            add(name, "cards", m["probability"], _MARKET_REF_ODDS[name])

    # ── 7. Asian Handicap — goals (from Poisson matrix) ───────────────
    for ah in pred.asian_handicap:
        add(ah["label"],                               "handicap", ah["home_prob"], _AH_REF_ODDS)
        add(ah["label"].replace("Home", "Away"),       "handicap", ah["away_prob"], _AH_REF_ODDS)

    # ── 8. Team supremacy — corners ───────────────────────────────────
    home_c_gt, _, away_c_gt = _compute_team_supremacy(
        expected_home_corners, expected_away_corners
    )
    add(f"{home_name} More Corners", "advanced", home_c_gt, _SUPREMACY_REF_ODDS)
    add(f"{away_name} More Corners", "advanced", away_c_gt, _SUPREMACY_REF_ODDS)

    # ── 9. Team supremacy — yellow cards ──────────────────────────────
    home_k_gt, _, away_k_gt = _compute_team_supremacy(
        expected_home_cards, expected_away_cards
    )
    add(f"{home_name} More Yellow Cards", "advanced", home_k_gt, _SUPREMACY_REF_ODDS)
    add(f"{away_name} More Yellow Cards", "advanced", away_k_gt, _SUPREMACY_REF_ODDS)

    return candidates


def _select_diverse_top_picks(
    candidates: list[dict],
    max_picks: int = 6,
    max_per_cat: int = 2,
    min_prob: float = 60.0,
    min_edge: float = 5.0,
) -> list[dict]:
    """
    Select the best picks from the candidate universe using:
    1. Filter  : probability ≥ min_prob AND edge ≥ min_edge (percentage points)
    2. Rank    : edge desc → probability desc → stability desc
    3. Diversity: at most max_per_cat picks from the same category
    """
    eligible = [
        c for c in candidates
        if c["probability"] >= min_prob and c["edge"] >= min_edge
    ]
    eligible.sort(key=lambda x: (-x["edge"], -x["probability"], -x["stability"]))

    selected: list[dict] = []
    cat_count: dict[str, int] = {}
    for pick in eligible:
        if len(selected) >= max_picks:
            break
        cat = pick["category"]
        if cat_count.get(cat, 0) < max_per_cat:
            cat_count[cat] = cat_count.get(cat, 0) + 1
            selected.append({**pick, "rank": len(selected) + 1})

    return selected


def _compute_corner_asian_handicap(
    expected_home_corners: float,
    expected_away_corners: float,
    max_corners: int = 20,
) -> list[dict]:
    """
    Compute Asian Handicap probabilities for corners.

    Lines are applied to the HOME team.  A negative line means home gives
    away corners; a positive line means home receives extra corners.
    """
    # Build joint corner probability matrix
    corner_matrix: dict[tuple[int, int], float] = {}
    for h in range(max_corners + 1):
        for a in range(max_corners + 1):
            corner_matrix[(h, a)] = (
                _poisson_pmf(h, expected_home_corners) *
                _poisson_pmf(a, expected_away_corners)
            )

    # Half-point lines only → no push possible (cleaner display)
    ah_lines = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    results = []
    for line in ah_lines:
        win_p = sum(
            p for (h, a), p in corner_matrix.items()
            if (h - a) > -line
        )
        ah_home = round(win_p * 100, 1)
        sign = "+" if line > 0 else ""
        results.append({
            "label": f"Home {sign}{line:g}",
            "line": line,
            "home_prob": ah_home,
            "away_prob": round(100.0 - ah_home, 1),
        })
    return results


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
    
    # Step 4: Yellow Cards model (Poisson on expected yellow cards per team)
    expected_home_cards = home_stats.cards
    expected_away_cards = away_stats.cards
    total_expected_cards = expected_home_cards + expected_away_cards
    
    cards_data = {
        "expected_home": round(expected_home_cards, 1),
        "expected_away": round(expected_away_cards, 1),
        "expected_total": round(total_expected_cards, 1),
        "markets": [
            {"market": "Over 2.5 Yellow Cards", "probability": round(_poisson_over(total_expected_cards, 2), 1)},
            {"market": "Under 2.5 Yellow Cards", "probability": round(100 - _poisson_over(total_expected_cards, 2), 1)},
            {"market": "Over 3.5 Yellow Cards", "probability": round(_poisson_over(total_expected_cards, 3), 1)},
            {"market": "Under 3.5 Yellow Cards", "probability": round(100 - _poisson_over(total_expected_cards, 3), 1)},
            {"market": "Over 4.5 Yellow Cards", "probability": round(_poisson_over(total_expected_cards, 4), 1)},
            {"market": "Under 4.5 Yellow Cards", "probability": round(100 - _poisson_over(total_expected_cards, 4), 1)},
            {"market": "Over 5.5 Yellow Cards", "probability": round(_poisson_over(total_expected_cards, 5), 1)},
            {"market": "Under 5.5 Yellow Cards", "probability": round(100 - _poisson_over(total_expected_cards, 5), 1)},
            {"market": "Over 6.5 Yellow Cards", "probability": round(_poisson_over(total_expected_cards, 6), 1)},
            {"market": "Under 6.5 Yellow Cards", "probability": round(100 - _poisson_over(total_expected_cards, 6), 1)},
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

    # ── Step 6: Multi-market professional engine ───────────────────────
    # Build full universe of candidates, then select the diverse top picks.
    market_candidates = _build_market_candidates(
        pred, corners_data, cards_data,
        home_name, away_name,
        expected_home_corners, expected_away_corners,
        expected_home_cards, expected_away_cards,
    )
    top_picks = _select_diverse_top_picks(market_candidates)

    # Corner Asian Handicap for the detailed display panel
    corner_asian_handicap = _compute_corner_asian_handicap(
        expected_home_corners, expected_away_corners
    )

    # ── Step 7: Derive value_selections from top_picks for legacy display ──
    value_selections = [
        {
            "pattern": p["market"],
            "pillar": p["category"].replace("_", " ").title(),
            "ic": p["probability"],
            "implied_probability": p["implied_prob"],
            "value_edge": p["edge"],
            "verdict": (
                "Best Choice" if p["edge"] >= 10
                else "Value" if p["edge"] >= 5
                else "Fair"
            ),
            "stability": p["stability"],
            "confidence": p["confidence"],
        }
        for p in top_picks
    ]

    return {
        "disclaimer": f"Hybrid Engine v6 — Poisson (λ={pred.lambda_home:.2f}+{pred.lambda_away:.2f}) + Multi-Market | {league_key}",
        "poisson": pred.to_dict(),
        "corners": corners_data,
        "corner_asian_handicap": corner_asian_handicap,
        "cards": cards_data,
        "xgboost_predictions": xgb_pred.to_dict().get("predictions", []),
        # New unified top-picks (edge-ranked, diversity-constrained)
        "top_picks": top_picks,
        # Legacy key — results verifier reads this key; keep it in sync
        "top_6_confident": top_picks,
        "value_selections": value_selections,
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
        fixtures.append(_sofascore_to_fixture(ev))
    
    fixtures.sort(key=lambda f: f["time"])
    
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

    # ── Yellow Cards ──
    elif "Yellow Cards" in market or "Cards" in market:
        if total_cards is not None:
            # Each tuple: (threshold, use_over)
            threshold_map = {
                "Over 2.5":  (2.5, True),
                "Under 2.5": (2.5, False),
                "Over 3.5":  (3.5, True),
                "Under 3.5": (3.5, False),
                "Over 4.5":  (4.5, True),
                "Under 4.5": (4.5, False),
                "Over 5.5":  (5.5, True),
                "Under 5.5": (5.5, False),
                "Over 6.5":  (6.5, True),
                "Under 6.5": (6.5, False),
            }
            for prefix, (threshold, use_over) in threshold_map.items():
                if market.startswith(prefix):
                    result = total_cards > threshold if use_over else total_cards < threshold
                    break

    return {
        **pick,
        "result": result,
    }


@app.get("/api/results/{date_str}")
def get_results_verification(date_str: str):
    """
    For all finished matches on a given date, regenerate predictions
    and compare them against actual results.

    CLEAN EVALUATION UNIVERSE:
    - Only settled picks (result = True/False) count in stats
    - Leagues with >25% NA matches are excluded entirely
    - Invariant: correct + wrong == total_picks (always)
    """
    events = _fetch_sofascore_events(date_str)
    if not events:
        return {"date": date_str, "matches": [], "summary": {}}

    # ── Phase 1: Build raw results per match ────────────────────
    raw_results = []

    for ev in events:
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

        # Evaluate each pick and tag settlement status
        evaluated_picks = []
        for pick in top_picks:
            evaluated = _evaluate_prediction(pick, home_goals, away_goals, total_corners, total_cards)
            # Add settlement fields
            evaluated["isSettled"] = evaluated["result"] is not None
            evaluated["isValidForEvaluation"] = evaluated["result"] is not None
            evaluated_picks.append(evaluated)

        raw_results.append({
            "fixture": fixture,
            "league_name": league_name,
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
        })

    # ── Phase 2: League-level quality filter ────────────────────
    # Group by league, exclude leagues with >25% NA matches
    league_stats = {}
    for match in raw_results:
        league = match["league_name"]
        if league not in league_stats:
            league_stats[league] = {"total_matches": 0, "na_matches": 0}
        league_stats[league]["total_matches"] += 1
        # A match is "NA" if all its picks are unsettled
        settled_count = sum(1 for p in match["picks"] if p["isSettled"])
        if settled_count == 0 and len(match["picks"]) > 0:
            league_stats[league]["na_matches"] += 1

    excluded_leagues = set()
    league_quality = {}
    for league, stats_data in league_stats.items():
        total = stats_data["total_matches"]
        na = stats_data["na_matches"]
        na_rate = na / total if total > 0 else 0
        is_excluded = na_rate > 0.25
        league_quality[league] = {
            "total_matches": total,
            "na_matches": na,
            "na_rate": round(na_rate * 100, 1),
            "excluded": is_excluded,
        }
        if is_excluded:
            excluded_leagues.add(league)

    # ── Phase 3: Build clean evaluation results ─────────────────
    clean_results = []
    total_correct = 0
    total_wrong = 0
    total_settled_picks = 0
    total_na_excluded = 0

    for match in raw_results:
        league = match["league_name"]
        league_excluded = league in excluded_leagues

        # Filter picks: only settled picks count
        settled_picks = [p for p in match["picks"] if p["isSettled"]]
        na_picks = [p for p in match["picks"] if not p["isSettled"]]

        if league_excluded:
            total_na_excluded += len(match["picks"])
            continue  # Skip entire league

        match_correct = sum(1 for p in settled_picks if p["result"] is True)
        match_wrong = sum(1 for p in settled_picks if p["result"] is False)

        total_correct += match_correct
        total_wrong += match_wrong
        total_settled_picks += len(settled_picks)
        total_na_excluded += len(na_picks)

        # Match summary uses ONLY settled picks
        clean_results.append({
            "fixture": match["fixture"],
            "actual": match["actual"],
            "picks": match["picks"],  # Keep all for display, but tag them
            "summary": {
                "correct": match_correct,
                "wrong": match_wrong,
                "unknown": len(na_picks),
                "total": len(settled_picks),  # Only settled count
            },
        })

    # ── Phase 4: overall summary using ONLY settled picks ───────
    # INVARIANT: correct + wrong == total_settled_picks
    accuracy = round(
        (total_correct / total_settled_picks * 100), 1
    ) if total_settled_picks > 0 else 0.0

    return {
        "date": date_str,
        "matches": clean_results,
        "summary": {
            "total_matches": len(clean_results),
            "total_picks": total_settled_picks,
            "total_correct": total_correct,
            "total_wrong": total_wrong,
            "total_unknown": 0,  # Always 0 — NAs are excluded
            "accuracy_pct": accuracy,
            "na_excluded": total_na_excluded,
            "leagues_excluded": len(excluded_leagues),
        },
        "league_quality": league_quality,
    }



# ═══════════════════════════════════════════════════════════════════════
# Investment Engine Endpoints
# ═══════════════════════════════════════════════════════════════════════

from src.db.database import get_db
from src.db.picks_repo import get_picks_by_date, get_unsettled_picks, settle_pick, get_portfolio_summary, get_league_pnl
from src.engine.pipeline import run_pipeline as _run_pipeline


@app.get("/api/pipeline/run/{date_str}")
def run_investment_pipeline(date_str: str):
    """Run the full investment pipeline for a date → returns graded picks."""
    events = _fetch_sofascore_events(date_str)
    if not events:
        return {"date": date_str, "error": "No events found", "picks": []}

    conn = get_db()
    result = _run_pipeline(date_str, events, conn)
    return result


@app.get("/api/picks/{date_str}")
def get_picks_for_date(date_str: str):
    """Get all stored picks for a date (from DB)."""
    conn = get_db()
    picks = get_picks_by_date(conn, date_str)
    return {"date": date_str, "picks": picks, "count": len(picks)}


@app.get("/api/portfolio/summary")
def get_portfolio():
    """Bankroll state: total P&L, ROI, hit rate, CLV."""
    conn = get_db()
    summary = get_portfolio_summary(conn)
    return summary


@app.post("/api/picks/settle")
def auto_settle_picks():
    """Auto-settle picks where match results are available."""
    conn = get_db()
    unsettled = get_unsettled_picks(conn)
    settled_count = 0

    for pick in unsettled:
        home_goals = pick.get("home_goals")
        away_goals = pick.get("away_goals")
        if home_goals is None or away_goals is None:
            continue

        result = _evaluate_pick_result(
            pick["market"], pick["selection"],
            home_goals, away_goals
        )
        if result is None:
            continue

        # Calculate P&L
        if result == "won":
            pnl = round(pick["stake_units"] * (pick["odds_at_pick"] - 1), 3)
        elif result == "lost":
            pnl = round(-pick["stake_units"], 3)
        else:
            pnl = 0.0

        settle_pick(conn, pick["id"], result, pnl)
        settled_count += 1

    return {"settled": settled_count, "remaining": len(unsettled) - settled_count}


@app.get("/api/leagues/profiles")
def get_league_profiles():
    """Get all league reliability profiles."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM league_profiles ORDER BY reliability_score DESC"
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/analytics/league-pnl")
def get_league_pnl_analytics():
    """P&L breakdown by league."""
    conn = get_db()
    return get_league_pnl(conn)


def _evaluate_pick_result(
    market: str, selection: str, home_goals: int, away_goals: int
) -> str | None:
    """Determine if a pick won or lost based on match result."""
    total_goals = home_goals + away_goals

    if market == "1X2":
        if selection == "home":
            return "won" if home_goals > away_goals else "lost"
        elif selection == "draw":
            return "won" if home_goals == away_goals else "lost"
        elif selection == "away":
            return "won" if away_goals > home_goals else "lost"

    elif market == "O/U 2.5":
        if selection == "over":
            return "won" if total_goals > 2.5 else "lost"
        elif selection == "under":
            return "won" if total_goals < 2.5 else "lost"

    elif market == "BTTS":
        both_scored = home_goals > 0 and away_goals > 0
        if selection == "yes":
            return "won" if both_scored else "lost"
        elif selection == "no":
            return "won" if not both_scored else "lost"

    return None


# ═══════════════════════════════════════════════════════════════════════
# ML Analytics Endpoints
# ═══════════════════════════════════════════════════════════════════════

from src.engine.calibration import ProbabilityCalibrator, ConfidenceBucketer
from src.data.odds_fetcher import fetch_and_store_odds, get_api_key as get_odds_key


@app.get("/api/analytics/calibration")
def get_calibration_report():
    """How well-calibrated are our probabilities? Predicted vs actual."""
    conn = get_db()
    calibrator = ProbabilityCalibrator(n_bins=10)
    calibrator.fit_from_db(conn)
    return {
        "report": calibrator.get_calibration_report(conn),
        "fitted": calibrator._fitted,
    }


@app.get("/api/analytics/confidence-buckets")
def get_confidence_buckets():
    """Performance breakdown by model confidence range."""
    conn = get_db()
    bucketer = ConfidenceBucketer()
    return {"buckets": bucketer.analyze(conn)}


@app.post("/api/odds/fetch")
def trigger_odds_fetch():
    """Manually trigger odds fetch from The Odds API."""
    if not get_odds_key():
        return {"error": "ODDS_API_KEY not set in .env", "status": "failed"}

    conn = get_db()
    result = fetch_and_store_odds(conn)
    count = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    return {
        "status": "ok",
        "leagues_fetched": result,
        "total_odds_in_db": count,
    }
