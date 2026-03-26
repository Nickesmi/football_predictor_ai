"""
Agent 3: Probability Engine

Wraps existing Poisson + XGBoost models to produce per-market probabilities.
"""

import logging
from src.ml.poisson_model import PoissonGoalModel
from src.ml.team_stats_db import get_team_stats
from src.ml.predictor import XGBoostPredictor

logger = logging.getLogger("football_predictor")

# Singleton models
_poisson = PoissonGoalModel()
_xgb = XGBoostPredictor()

# Map SofaScore league names → Poisson profile keys
LEAGUE_KEY_MAP = {
    "Premier League": "Premier League",
    "LaLiga": "LaLiga",
    "La Liga": "LaLiga",
    "Serie A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
    "Championship": "Premier League",  # closest proxy
    "Eredivisie": "Eredivisie",
    "VriendenLoterij Eredivisie": "Eredivisie",
    "Primeira Liga": "Serie A",
    "Liga Portugal": "Serie A",
    "Liga Portugal Betclic": "Serie A",
    "Belgian Pro League": "Ligue 1",
    "Jupiler Pro League": "Ligue 1",
    "Pro League": "Ligue 1",
    "Süper Lig": "Süper Lig",
    "Trendyol Süper Lig": "Süper Lig",
    "Super Lig": "Süper Lig",
    "Scottish Premiership": "Premier League",
    "Champions League": "Champions League",
    "UEFA Champions League": "Champions League",
    "Europa League": "Champions League",
    "UEFA Europa League": "Champions League",
    "Conference League": "Champions League",
    "UEFA Europa Conference League": "Champions League",
}


def estimate_probabilities(match: dict) -> dict:
    """
    Produce probability estimates for all supported markets.

    Returns dict with:
        "1X2": {"home": p, "draw": p, "away": p},
        "O/U 2.5": {"over": p, "under": p},
        "BTTS": {"yes": p, "no": p},
        "goals": {"exp_home": float, "exp_away": float},
        "source": "poisson" | "xgboost" | "hybrid"
    """
    home_team = match["home_team"]
    away_team = match["away_team"]
    league_name = match.get("league_name", "")
    league_key = LEAGUE_KEY_MAP.get(league_name, "default")

    # Get team stats - home team gets home venue stats, away gets away
    home_stats = get_team_stats(home_team, "home", league_key)
    away_stats = get_team_stats(away_team, "away", league_key)

    # --- Poisson model ---
    try:
        # TeamVenueStats has .scored, .conceded, .corners, .cards attributes
        # Handle both dict and TeamVenueStats objects
        if hasattr(home_stats, 'scored'):
            home_gf = home_stats.scored
            home_ga = home_stats.conceded
        else:
            home_gf = home_stats.get("scored", home_stats.get("gf", 1.4))
            home_ga = home_stats.get("conceded", home_stats.get("ga", 1.1))

        if hasattr(away_stats, 'scored'):
            away_gf = away_stats.scored
            away_ga = away_stats.conceded
        else:
            away_gf = away_stats.get("scored", away_stats.get("gf", 1.2))
            away_ga = away_stats.get("conceded", away_stats.get("ga", 1.3))

        # Create a league-specific Poisson instance
        poisson = PoissonGoalModel(league=league_key)

        prediction = poisson.predict(
            home_scored=float(home_gf),
            home_conceded=float(home_ga),
            away_scored=float(away_gf),
            away_conceded=float(away_ga),
            home_team=home_team,
            away_team=away_team,
        )

        exp_home = prediction.lambda_home
        exp_away = prediction.lambda_away

        # Convert from 0-100 scale to 0-1 scale
        probs_1x2 = {
            "home": prediction.home_win / 100.0,
            "draw": prediction.draw / 100.0,
            "away": prediction.away_win / 100.0,
        }

        probs_ou = {
            "over": prediction.over_2_5 / 100.0,
            "under": prediction.under_2_5 / 100.0,
        }

        probs_btts = {
            "yes": prediction.btts_yes / 100.0,
            "no": prediction.btts_no / 100.0,
        }

        source = "poisson"

    except Exception as e:
        logger.warning(f"Poisson failed for {home_team} vs {away_team}: {e}")
        # Fallback to baseline
        exp_home, exp_away = 1.4, 1.1
        probs_1x2 = {"home": 0.42, "draw": 0.27, "away": 0.31}
        probs_ou = {"over": 0.52, "under": 0.48}
        probs_btts = {"yes": 0.50, "no": 0.50}
        source = "fallback"

    # Normalize 1X2 to sum to 1.0
    total = sum(probs_1x2.values())
    if total > 0 and abs(total - 1.0) > 0.01:
        probs_1x2 = {k: v / total for k, v in probs_1x2.items()}

    return {
        "1X2": {k: round(v, 4) for k, v in probs_1x2.items()},
        "O/U 2.5": {k: round(v, 4) for k, v in probs_ou.items()},
        "BTTS": {k: round(v, 4) for k, v in probs_btts.items()},
        "goals": {"exp_home": round(exp_home, 2), "exp_away": round(exp_away, 2)},
        "source": source,
    }
