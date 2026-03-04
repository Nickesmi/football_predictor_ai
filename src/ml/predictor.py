"""
Runtime Predictor — takes team profiles and returns calibrated probabilities.

This replaces the manual pattern intersection as the PRIMARY prediction engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.config import logger
from src.ml.feature_builder import FeatureBuilder, TeamProfile, MatchFeatures, FEATURE_COLUMNS
from src.ml.trainer import XGBoostTrainer, TARGET_MARKETS


@dataclass
class MarketPrediction:
    """Calibrated probability for a single market."""
    market: str
    probability: float        # 0.0 to 1.0 (calibrated)
    confidence_pct: float     # probability * 100
    top_features: list[dict]  # Feature importance for this market

    @property
    def confidence_tier(self) -> str:
        if self.confidence_pct >= 75:
            return "Very High"
        if self.confidence_pct >= 60:
            return "High"
        if self.confidence_pct >= 45:
            return "Medium"
        if self.confidence_pct >= 30:
            return "Low"
        return "Very Low"

    @property
    def display_name(self) -> str:
        names = {
            "btts": "BTTS - Yes",
            "over_1_5": "Over 1.5 Goals FT",
            "over_2_5": "Over 2.5 Goals FT",
            "over_3_5": "Over 3.5 Goals FT",
            "home_win": "Home Win",
            "draw": "Draw",
            "ht_over_0_5": "Over 0.5 Goals HT",
        }
        return names.get(self.market, self.market)


@dataclass
class MatchPrediction:
    """Full prediction for a match across all markets."""
    home_team: str
    away_team: str
    predictions: list[MarketPrediction] = field(default_factory=list)
    features: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predictions": [
                {
                    "market": p.display_name,
                    "market_key": p.market,
                    "probability": round(p.confidence_pct, 1),
                    "confidence": p.confidence_tier,
                    "top_drivers": p.top_features[:3],
                }
                for p in sorted(self.predictions, key=lambda x: x.probability, reverse=True)
            ],
        }


class XGBoostPredictor:
    """Runtime prediction engine using trained XGBoost models."""

    def __init__(self):
        self.trainer = XGBoostTrainer()
        self.builder = FeatureBuilder()
        self._loaded = False

    def ensure_loaded(self) -> bool:
        """Load models from disk if not already loaded."""
        if not self._loaded:
            self._loaded = self.trainer.load_models()
        return self._loaded

    def predict(self, home: TeamProfile, away: TeamProfile) -> MatchPrediction:
        """
        Run XGBoost prediction for a match.

        Args:
            home: TeamProfile for the home team (venue-specific stats).
            away: TeamProfile for the away team (venue-specific stats).

        Returns:
            MatchPrediction with calibrated probabilities for all markets.
        """
        if not self.ensure_loaded():
            logger.warning("No trained models found — returning demo predictions.")
            return self._demo_prediction(home, away)

        features = self.builder.build(home, away)
        X = features.to_dataframe()[FEATURE_COLUMNS].values

        predictions = []
        for market in TARGET_MARKETS:
            model = self.trainer.models.get(market)
            if model is None:
                continue

            proba = model.predict_proba(X)[0, 1]
            top_features = self.trainer.metrics.get(market, {}).get("top_features", [])

            predictions.append(MarketPrediction(
                market=market,
                probability=float(proba),
                confidence_pct=round(float(proba) * 100, 1),
                top_features=top_features,
            ))

        return MatchPrediction(
            home_team=home.team_name,
            away_team=away.team_name,
            predictions=predictions,
            features=features.to_dict(),
        )

    def _demo_prediction(self, home: TeamProfile, away: TeamProfile) -> MatchPrediction:
        """Generate plausible demo predictions when no models are trained."""
        features = self.builder.build(home, away)

        # Heuristic predictions from interaction features
        total_exp = features.expected_total_goals
        btts_est = features.btts_combined
        home_strength = 0.5 + (features.strength_diff * 0.02)

        predictions = [
            MarketPrediction("btts", btts_est, round(btts_est * 100, 1), []),
            MarketPrediction("over_1_5", min(0.95, total_exp * 0.32), round(min(95, total_exp * 32), 1), []),
            MarketPrediction("over_2_5", min(0.85, total_exp * 0.22), round(min(85, total_exp * 22), 1), []),
            MarketPrediction("over_3_5", min(0.60, total_exp * 0.12), round(min(60, total_exp * 12), 1), []),
            MarketPrediction("home_win", np.clip(home_strength, 0.2, 0.8), round(np.clip(home_strength, 0.2, 0.8) * 100, 1), []),
            MarketPrediction("draw", 0.25, 25.0, []),
            MarketPrediction("ht_over_0_5", min(0.85, total_exp * 0.28), round(min(85, total_exp * 28), 1), []),
        ]

        return MatchPrediction(
            home_team=home.team_name,
            away_team=away.team_name,
            predictions=predictions,
            features=features.to_dict(),
        )
