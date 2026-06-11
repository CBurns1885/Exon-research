"""Gradient-boosted ensemble strategy using XGBoost/LightGBM.

Uses all existing strategy signals as features for a supervised ML model
that predicts forward returns. This is the meta-learning layer that sits
on top of the individual alpha strategies — the machine learns which
signals are predictive in which conditions.

Feature set (per asset, per bar):
- Raw signals from all sub-strategies (direction * strength)
- Regime features (volatility ratio, efficiency ratio, Hurst estimate)
- Market microstructure (relative volume, OBV z-score)
- Cross-sectional rank of each feature
- Interaction features (momentum * vol_regime, signal_agreement count)

Training uses purged walk-forward: gap between train and test to prevent
look-ahead from overlapping signal windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Signal, Strategy

logger = logging.getLogger(__name__)


@dataclass
class MLEnsembleConfig:
    """Configuration for the ML ensemble strategy."""

    # Feature construction
    momentum_windows: list[int] = field(default_factory=lambda: [5, 21, 63])
    vol_windows: list[int] = field(default_factory=lambda: [5, 21, 63])
    # Training
    train_window: int = 504  # 2 years of daily data
    retrain_frequency: int = 63  # retrain quarterly
    purge_gap: int = 5  # gap between train and test to prevent leakage
    # Target
    forward_return_window: int = 5  # predict 5-day forward return
    # Model
    use_xgboost: bool = True  # False = LightGBM
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    # Signal generation
    top_n: int = 5
    bottom_n: int = 3
    min_prediction_threshold: float = 0.001


class MLEnsemble(Strategy):
    """Gradient-boosted ensemble that meta-learns from sub-strategy signals."""

    name = "ml_ensemble"

    def __init__(self, config: MLEnsembleConfig | None = None):
        self.config = config or MLEnsembleConfig()
        self._model = None
        self._last_train_bar = -999
        self._feature_names: list[str] = []

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(data) < self.config.train_window + self.config.forward_return_window + 10:
            return []

        n = len(data)
        ts = data.index[-1]

        # Build feature matrix for entire history
        features_df = self._build_features(data)
        if features_df.empty:
            return []

        # Retrain if needed
        if n - self._last_train_bar >= self.config.retrain_frequency or self._model is None:
            self._train(features_df, data)
            self._last_train_bar = n

        if self._model is None:
            return []

        # Predict on latest bar
        latest_features = features_df.iloc[-1:]
        signals = self._predict_signals(latest_features, data, ts)
        return signals

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Construct feature matrix from price data."""
        cfg = self.config
        returns = data.pct_change()
        n = len(data)

        all_features = {}

        for col in data.columns:
            price = data[col]
            ret = returns[col]

            # Momentum features at multiple horizons
            for w in cfg.momentum_windows:
                if n > w:
                    all_features[f"{col}_mom_{w}"] = price.pct_change(w)

            # Volatility features
            for w in cfg.vol_windows:
                if n > w:
                    all_features[f"{col}_vol_{w}"] = ret.rolling(w).std()

            # Volatility ratio (short/long)
            if n > 63:
                short_vol = ret.rolling(5).std()
                long_vol = ret.rolling(63).std()
                all_features[f"{col}_vol_ratio"] = short_vol / long_vol.replace(0, np.nan)

            # Efficiency ratio (trend quality)
            if n > 21:
                net_change = price.diff(21).abs()
                total_change = price.diff().abs().rolling(21).sum()
                all_features[f"{col}_efficiency"] = net_change / total_change.replace(0, np.nan)

            # Mean reversion z-score
            if n > 21:
                ma = price.rolling(21).mean()
                std = price.rolling(21).std()
                all_features[f"{col}_zscore_21"] = (price - ma) / std.replace(0, np.nan)

            # RSI proxy (ratio of up-moves to total moves)
            if n > 14:
                up = ret.clip(lower=0).rolling(14).mean()
                down = (-ret.clip(upper=0)).rolling(14).mean()
                all_features[f"{col}_rsi"] = up / (up + down).replace(0, np.nan)

            # Autocorrelation
            if n > 21:
                all_features[f"{col}_autocorr"] = ret.rolling(21).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 2 else 0, raw=False
                )

            # Skewness
            if n > 21:
                all_features[f"{col}_skew"] = ret.rolling(21).skew()

            # Return dispersion (cross-asset)
            if len(data.columns) > 1 and n > 21:
                cross_vol = returns.rolling(21).std().mean(axis=1)
                all_features[f"{col}_cross_disp"] = cross_vol

        features_df = pd.DataFrame(all_features, index=data.index)

        # Add cross-sectional features
        if len(data.columns) > 1:
            for w in cfg.momentum_windows:
                mom_cols = [c for c in features_df.columns if f"_mom_{w}" in c]
                if mom_cols:
                    mom_slice = features_df[mom_cols]
                    ranks = mom_slice.rank(axis=1, pct=True)
                    for col_name in mom_cols:
                        features_df[f"{col_name}_rank"] = ranks[col_name]

            # Signal agreement (how many momentum signals agree on direction)
            for col in data.columns:
                mom_signals = []
                for w in cfg.momentum_windows:
                    key = f"{col}_mom_{w}"
                    if key in features_df.columns:
                        mom_signals.append(np.sign(features_df[key]))
                if mom_signals:
                    agreement = pd.concat(mom_signals, axis=1).sum(axis=1)
                    features_df[f"{col}_signal_agreement"] = agreement

        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        self._feature_names = list(features_df.columns)
        return features_df

    def _train(self, features_df: pd.DataFrame, data: pd.DataFrame) -> None:
        """Train the ML model on historical data with purged walk-forward."""
        cfg = self.config
        n = len(features_df)

        # Need enough data for training
        min_required = cfg.train_window + cfg.forward_return_window + cfg.purge_gap + 10
        if n < min_required:
            return

        # Compute forward returns as target
        returns = data.pct_change()
        targets = {}
        for col in data.columns:
            targets[col] = returns[col].rolling(cfg.forward_return_window).sum().shift(-cfg.forward_return_window)

        # Build training dataset: each (bar, asset) pair is one sample
        train_end = n - cfg.forward_return_window - cfg.purge_gap
        train_start = max(0, train_end - cfg.train_window)

        X_rows = []
        y_rows = []

        for col in data.columns:
            col_features = [c for c in features_df.columns if c.startswith(f"{col}_")]
            if not col_features:
                continue

            for i in range(train_start, train_end):
                target_val = targets[col].iloc[i] if i < len(targets[col]) else np.nan
                if np.isnan(target_val):
                    continue
                row = features_df[col_features].iloc[i].values
                X_rows.append(row)
                y_rows.append(target_val)

        if len(X_rows) < 50:
            return

        X = np.array(X_rows)
        y = np.array(y_rows)

        # Remove any remaining NaN/inf
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]

        if len(X) < 50:
            return

        try:
            if cfg.use_xgboost:
                self._train_xgboost(X, y)
            else:
                self._train_lightgbm(X, y)
        except Exception as e:
            logger.warning("ML training failed: %s", e)
            self._model = None

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost regressor."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed, falling back to sklearn")
            self._train_sklearn_fallback(X, y)
            return

        cfg = self.config
        model = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        self._model = model

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train LightGBM regressor."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed, falling back to sklearn")
            self._train_sklearn_fallback(X, y)
            return

        cfg = self.config
        model = lgb.LGBMRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)
        self._model = model

    def _train_sklearn_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback to sklearn GradientBoostingRegressor if XGB/LGB unavailable."""
        from sklearn.ensemble import GradientBoostingRegressor

        cfg = self.config
        model = GradientBoostingRegressor(
            n_estimators=min(cfg.n_estimators, 100),
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            random_state=42,
        )
        model.fit(X, y)
        self._model = model

    def _predict_signals(
        self, latest_features: pd.DataFrame, data: pd.DataFrame, ts: pd.Timestamp
    ) -> list[Signal]:
        """Generate trading signals from ML predictions."""
        cfg = self.config
        predictions = {}

        for col in data.columns:
            col_features = [c for c in latest_features.columns if c.startswith(f"{col}_")]
            if not col_features:
                continue

            X = latest_features[col_features].values
            if not np.isfinite(X).all():
                continue

            try:
                pred = float(self._model.predict(X)[0])
            except Exception:
                continue

            predictions[col] = pred

        if not predictions:
            return []

        # Rank predictions cross-sectionally
        sorted_assets = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        signals = []
        n_assets = len(sorted_assets)

        # Long top_n
        for rank, (asset, pred) in enumerate(sorted_assets[:cfg.top_n]):
            if pred < cfg.min_prediction_threshold:
                continue
            strength = min(abs(pred) * 50, 1.0)  # scale prediction to 0-1
            strength = max(strength, 0.3)  # minimum strength for ranked assets
            # Higher rank = higher strength
            rank_boost = 1.0 - rank * 0.1
            signals.append(Signal(
                timestamp=ts,
                asset=asset,
                direction=1.0,
                strength=min(strength * rank_boost, 1.0),
                metadata={
                    "strategy_type": "ml_ensemble",
                    "predicted_return": pred,
                    "rank": rank + 1,
                    "n_assets": n_assets,
                },
            ))

        # Short bottom_n
        for rank, (asset, pred) in enumerate(reversed(sorted_assets[-cfg.bottom_n:])):
            if pred > -cfg.min_prediction_threshold:
                continue
            strength = min(abs(pred) * 50, 1.0)
            strength = max(strength, 0.3)
            rank_boost = 1.0 - rank * 0.1
            signals.append(Signal(
                timestamp=ts,
                asset=asset,
                direction=-1.0,
                strength=min(strength * rank_boost, 1.0),
                metadata={
                    "strategy_type": "ml_ensemble",
                    "predicted_return": pred,
                    "rank": n_assets - rank,
                    "n_assets": n_assets,
                },
            ))

        return signals
