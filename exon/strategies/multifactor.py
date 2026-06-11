"""Cross-sectional multi-factor alpha model.

This is the workhorse of any systematic fund. Instead of trading one
signal, we construct a composite alpha from multiple orthogonal factors,
rank the universe cross-sectionally on the composite, and go long the
top quintile / short the bottom quintile.

Factors implemented (all z-scored cross-sectionally):

1. Momentum (MOM): 1-week to 1-month cumulative return
2. Short-Term Reversal (STR): 1-24h return (contrarian at short horizon)
3. Volatility (VOL): lower vol assets earn a premium (low-vol anomaly)
4. Liquidity/Volume (LIQ): proxy from return magnitude — less liquid = higher expected return
5. Autocorrelation (AC): positive autocorrelation → trend continuation, negative → reversal
6. Skewness (SKEW): negative skew assets tend to outperform (lottery effect)

The factor weights can be fixed or estimated via rolling IC (information
coefficient = rank correlation between factor and forward returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .base import Signal, Strategy


class MultiFactor(Strategy):
    """Cross-sectional multi-factor ranking model.

    Ranks the entire universe on a composite alpha score and
    goes long the top N assets, short the bottom N.
    """

    name = "multifactor"

    def __init__(
        self,
        momentum_windows: list[int] | None = None,
        reversal_window: int = 12,
        vol_window: int = 168,
        autocorr_window: int = 48,
        autocorr_lag: int = 1,
        skew_window: int = 168,
        top_n: int = 3,
        bottom_n: int = 3,
        ic_lookback: int = 720,
        use_adaptive_weights: bool = True,
        factor_weights: dict[str, float] | None = None,
    ):
        """
        Parameters
        ----------
        momentum_windows : lookbacks for momentum factor (blended)
        reversal_window : hours for short-term reversal signal
        vol_window : lookback for volatility factor
        autocorr_window : lookback for autocorrelation estimation
        autocorr_lag : lag for autocorrelation
        skew_window : lookback for skewness factor
        top_n, bottom_n : how many assets to go long/short
        ic_lookback : rolling window for IC-based adaptive factor weights
        use_adaptive_weights : if True, weight factors by rolling IC
        factor_weights : fixed weights if not adaptive; defaults to equal
        """
        self.momentum_windows = momentum_windows or [72, 168, 336]
        self.reversal_window = reversal_window
        self.vol_window = vol_window
        self.autocorr_window = autocorr_window
        self.autocorr_lag = autocorr_lag
        self.skew_window = skew_window
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.ic_lookback = ic_lookback
        self.use_adaptive_weights = use_adaptive_weights
        self.factor_weights = factor_weights or {
            "momentum": 0.25,
            "reversal": 0.15,
            "volatility": 0.15,
            "liquidity": 0.10,
            "autocorrelation": 0.20,
            "skewness": 0.15,
        }
        self._ic_history: dict[str, list[float]] = {k: [] for k in self.factor_weights}

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        returns = np.log(data / data.shift(1)).dropna()
        timestamp = data.index[-1]
        min_history = max(
            max(self.momentum_windows), self.vol_window, self.skew_window, self.autocorr_window
        ) + 50

        if len(returns) < min_history:
            return []

        assets = data.columns.tolist()
        if len(assets) < self.top_n + self.bottom_n:
            return []

        # --- Compute factors cross-sectionally ---
        factor_scores = {}

        # 1. Momentum: blended cumulative return across windows
        mom_scores = []
        for w in self.momentum_windows:
            cum_ret = returns.iloc[-w:].sum()
            mom_scores.append(self._cross_sectional_z(cum_ret))
        factor_scores["momentum"] = pd.concat(mom_scores, axis=1).mean(axis=1)

        # 2. Short-Term Reversal: negative of very recent returns
        recent_ret = returns.iloc[-self.reversal_window:].sum()
        factor_scores["reversal"] = -self._cross_sectional_z(recent_ret)

        # 3. Volatility: negative vol (lower vol = higher score)
        vol = returns.iloc[-self.vol_window:].std()
        factor_scores["volatility"] = -self._cross_sectional_z(vol)

        # 4. Liquidity proxy: negative of average absolute return (less "liquid" = higher)
        avg_abs_ret = returns.iloc[-self.vol_window:].abs().mean()
        factor_scores["liquidity"] = -self._cross_sectional_z(avg_abs_ret)

        # 5. Autocorrelation: positive AC → trend, negative → reversal tendency
        ac = returns.iloc[-self.autocorr_window:].apply(
            lambda s: s.autocorr(lag=self.autocorr_lag)
        )
        factor_scores["autocorrelation"] = self._cross_sectional_z(ac.fillna(0))

        # 6. Skewness: negative skew → fatter left tail → higher expected return
        skew = returns.iloc[-self.skew_window:].skew()
        factor_scores["skewness"] = -self._cross_sectional_z(skew.fillna(0))

        # --- Compute adaptive weights via rolling IC ---
        weights = self.factor_weights.copy()
        if self.use_adaptive_weights and len(returns) > self.ic_lookback + 24:
            weights = self._compute_adaptive_weights(returns, factor_scores)

        # --- Composite alpha ---
        composite = pd.Series(0.0, index=assets)
        for factor_name, scores in factor_scores.items():
            w = weights.get(factor_name, 0)
            composite = composite.add(scores * w, fill_value=0)

        # --- Rank and select ---
        ranked = composite.sort_values(ascending=False)
        long_assets = ranked.index[:self.top_n].tolist()
        short_assets = ranked.index[-self.bottom_n:].tolist()

        signals = []
        for i, asset in enumerate(long_assets):
            # Strength decays by rank position
            rank_weight = 1.0 - 0.3 * (i / max(self.top_n - 1, 1))
            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=asset,
                    direction=1.0,
                    strength=min(rank_weight * abs(composite[asset]), 1.0),
                    metadata={
                        "composite_alpha": composite[asset],
                        "rank": i,
                        "factors": {k: float(v.get(asset, 0)) for k, v in factor_scores.items()},
                    },
                )
            )

        for i, asset in enumerate(short_assets):
            rank_weight = 1.0 - 0.3 * (i / max(self.bottom_n - 1, 1))
            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=asset,
                    direction=-1.0,
                    strength=min(rank_weight * abs(composite[asset]), 1.0),
                    metadata={
                        "composite_alpha": composite[asset],
                        "rank": len(assets) - self.bottom_n + i,
                        "factors": {k: float(v.get(asset, 0)) for k, v in factor_scores.items()},
                    },
                )
            )

        return signals

    def _compute_adaptive_weights(
        self,
        returns: pd.DataFrame,
        current_factors: dict[str, pd.Series],
    ) -> dict[str, float]:
        """Estimate factor weights from rolling Information Coefficient.

        IC = Spearman rank correlation between factor score at time t
        and realised forward return at time t+1.
        Higher IC → factor has been more predictive → higher weight.
        """
        # Use historical data to compute IC for each factor
        lookback = min(self.ic_lookback, len(returns) - 24)
        if lookback < 100:
            return self.factor_weights

        hist_returns = returns.iloc[-lookback:-1]
        fwd_returns = returns.iloc[-lookback + 1:]

        ics = {}
        for factor_name in self.factor_weights:
            ic_values = []
            # Sample IC at multiple points in the lookback
            for t in range(0, lookback - 50, 24):
                # Compute factor at time t
                chunk = hist_returns.iloc[t:t+48]
                if len(chunk) < 24:
                    continue

                if factor_name == "momentum":
                    score = chunk.sum()
                elif factor_name == "reversal":
                    score = -chunk.iloc[-self.reversal_window:].sum()
                elif factor_name == "volatility":
                    score = -chunk.std()
                elif factor_name == "liquidity":
                    score = -chunk.abs().mean()
                elif factor_name == "autocorrelation":
                    score = chunk.apply(lambda s: s.autocorr(lag=1)).fillna(0)
                elif factor_name == "skewness":
                    score = -chunk.skew().fillna(0)
                else:
                    continue

                # Forward return
                fwd_idx = min(t + 48, len(fwd_returns) - 1)
                fwd = fwd_returns.iloc[fwd_idx]

                # Spearman IC
                common = score.dropna().index.intersection(fwd.dropna().index)
                if len(common) < 4:
                    continue
                rho, _ = spearmanr(score.loc[common], fwd.loc[common])
                if not np.isnan(rho):
                    ic_values.append(rho)

            ics[factor_name] = np.mean(ic_values) if ic_values else 0.0

        # Convert ICs to weights (positive IC = keep, negative = flip sign handled by factor def)
        abs_ics = {k: max(abs(v), 0.01) for k, v in ics.items()}
        total = sum(abs_ics.values())
        return {k: v / total for k, v in abs_ics.items()}

    @staticmethod
    def _cross_sectional_z(series: pd.Series) -> pd.Series:
        """Z-score normalise across assets (cross-sectionally)."""
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std
