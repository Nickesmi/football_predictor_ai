from datetime import datetime, timezone
import logging
from typing import Dict

from src.db.database import get_db
from src.db.match_repo import get_matches_by_date
from src.data.odds_fetcher import TheOddsAPIProvider, LEAGUE_TO_SPORT
from src.engine.execution_engine import find_executable_bets, _is_tradable_market, EXCLUDED_MARKET_TYPES, MIN_ODDS, MAX_ODDS, MIN_CALIBRATED_PROB, MIN_EDGE, MIN_EV, compute_edge, compute_ev, get_bookmaker_weight
from src.ml.hybrid_predictor import HybridPredictor
from src.ml.feature_builder import FeatureBuilder

logger = logging.getLogger("football_predictor")

def scan_live_odds() -> Dict:
    """
    Orchestration layer to fetch live odds, run through the ExecutionEngine,
    and return positive EV executable bets.
    """
    conn = get_db()
    provider = TheOddsAPIProvider(conn)
    predictor = HybridPredictor()
    fb = FeatureBuilder()
    
    # Track stats
    stats = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "matches_scanned": 0,
        "odds_snapshots_saved": 0,
        "markets_matched": 0,
        "executable_bets": [],
        "rejections": {
            "stale": 0,
            "odds_out_of_range": 0,
            "no_model_match": 0,
            "negative_ev": 0,
            "untradable_market": 0
        }
    }

    # 1. Load upcoming matches for today
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matches = get_matches_by_date(conn, today_str)

    # We only care about matches that haven't started/finished (status NS)
    upcoming_matches = [m for m in matches if m.get("status") in ("NS", "TBD")]
    
    for match in upcoming_matches:
        home_team = match["home_team"]
        away_team = match["away_team"]
        league_name = match.get("league_name", "")
        league_id = match.get("league_id")
        
        sport_key = LEAGUE_TO_SPORT.get(league_id)
        if not sport_key:
            continue
            
        stats["matches_scanned"] += 1
            
        # 2. Fetch live odds + store snapshots
        best_odds = provider.get_normalized_odds_for_match(sport_key, home_team, away_team)
        stats["odds_snapshots_saved"] += len(best_odds)

        if not best_odds:
            continue
            
        # Build profiles for prediction
        try:
            home_profile = fb.build_team_profile(home_team, league_id)
            away_profile = fb.build_team_profile(away_team, league_id)
        except Exception as e:
            logger.warning(f"FeatureBuilder failed for {home_team} vs {away_team}: {e}")
            continue

        # Generate internal model predictions
        prediction = predictor.predict(home_profile, away_profile, home_team, away_team)
        
        # We need the full_analysis format to pass to execution engine
        analysis = {"full_analysis": {"Unified": prediction.unified_markets}}
        
        # Run execution engine
        match_bets = find_executable_bets(
            analysis=analysis,
            bookmaker_odds=best_odds,
            home_name=home_team,
            away_name=away_team,
            league_name=league_name,
            match_id=match.get("id", "")
        )
        
        for bet in match_bets:
            stats["executable_bets"].append(bet.to_dict())
            stats["markets_matched"] += 1

        # Calculate rejection stats manually for the payload
        # Build lookup
        market_lookup = {m["market"]: m for m in prediction.unified_markets}
        
        for bk_odds in best_odds:
            market_name = bk_odds.get("market", "")
            odds = bk_odds.get("odds", 0)
            
            if not _is_tradable_market(market_name, home_team, away_team):
                stats["rejections"]["untradable_market"] += 1
                continue
                
            if not (MIN_ODDS <= odds <= MAX_ODDS):
                stats["rejections"]["odds_out_of_range"] += 1
                continue
                
            model = market_lookup.get(market_name)
            if not model:
                stats["rejections"]["no_model_match"] += 1
                continue
                
            cal_prob = model.get("probability", 0)
            if cal_prob < MIN_CALIBRATED_PROB:
                stats["rejections"]["negative_ev"] += 1
                continue
                
            bk_weight = get_bookmaker_weight(bk_odds.get("bookmaker", "unknown"))
            edge = compute_edge(cal_prob, odds)
            ev = compute_ev(cal_prob, odds)
            
            if edge * bk_weight < MIN_EDGE or ev < MIN_EV:
                stats["rejections"]["negative_ev"] += 1

    return stats
