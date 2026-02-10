"""Hurst exponent regime filter — a meta-strategy.

The Hurst exponent H measures the long-range dependence structure of
a time series:

    H < 0.5 → mean-reverting (anti-persistent)
    H = 0.5 → random walk (no exploitable structure)
    H > 0.5 → trending (persistent)

Estimated via Rescaled Range (R/S) analysis, which is robust to
non-Gaussianity — important for crypto's fat tails.

This strategy doesn't generate alpha on its own. Instead, it acts as
a gating function for other strategies:
- When H > 0.5: activate momentum strategies, suppress mean-reversion
- When H < 0.5: activate mean-reversion strategies, suppress momentum
- When H ≈ 0.5: reduce all exposure (no edge in a random walk)

It also directly trades: go long trending assets (high H + positive
drift), short mean-reverting assets that are extended (low H + extreme
z-score). The Hurst exponent IS the position sizing function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


def estimate_hurst(series: np.ndarray, max_lag: int | None = None) -> float:
    """Estimate Hurst exponent via Rescaled Range (R/S) analysis.

    Internally differences the input (converts prices to returns)
    because R/S analysis must be run on increments, not levels.
    Passing returns directly also works (will be differenced again,
    but R/S is robust to this).

    Parameters
    ----------
    series : 1D array of prices (will be differenced) or returns
    max_lag : maximum partition size (default: len/4)

    Returns
    -------
    Hurst exponent estimate (0, 1)
    """
    # Difference to get returns — R/S on price levels always gives H≈1
    series = np.diff(series)
    n = len(series)
    if n < 20:
        return 0.5

    if max_lag is None:
        max_lag = max(n // 4, 10)

    lags = []
    rs_values = []

    for lag in range(10, max_lag + 1, max(1, (max_lag - 10) // 20)):
        # Partition series into non-overlapping blocks of size `lag`
        rs_block = []
        for start in range(0, n - lag + 1, lag):
            block = series[start : start + lag]
            if len(block) < lag:
                continue

            mean = block.mean()
            deviations = block - mean
            cumulative = np.cumsum(deviations)
            R = cumulative.max() - cumulative.min()
            S = block.std(ddof=1)

            if S > 0:
                rs_block.append(R / S)

        if rs_block:
            lags.append(lag)
            rs_values.append(np.mean(rs_block))

    if len(lags) < 3:
        return 0.5

    # H = slope of log(R/S) vs log(n)
    log_lags = np.log(lags)
    log_rs = np.log(rs_values)

    # OLS for slope
    x = log_lags
    y = log_rs
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.5
    H = ((x - x_mean) * (y - y_mean)).sum() / denom

    return float(np.clip(H, 0.01, 0.99))


class HurstRegimeFilter(Strategy):
    """Meta-strategy that gates signals based on per-asset Hurst exponent.

    For each asset:
    - H > threshold_trend: trade momentum (long if recent return > 0, else short)
    - H < threshold_mr: trade mean-reversion (counter-trend)
    - H ≈ 0.5: no position (random walk, no edge)

    Position size is proportional to |H - 0.5| — further from 0.5 = stronger regime.
    """

    name = "hurst_regime"

    def __init__(
        self,
        hurst_window: int = 168,
        momentum_window: int = 72,
        mr_z_window: int = 48,
        threshold_trend: float = 0.55,
        threshold_mr: float = 0.45,
        dead_zone: float = 0.03,
        vol_target: float = 0.15,
    ):
        """
        Parameters
        ----------
        hurst_window : number of bars for Hurst estimation
        momentum_window : lookback for momentum signal when H > threshold
        mr_z_window : lookback for mean-reversion z-score when H < threshold
        threshold_trend : H above this → trending regime
        threshold_mr : H below this → mean-reverting regime
        dead_zone : band around 0.5 where we trade nothing
        vol_target : annualised vol target for sizing
        """
        self.hurst_window = hurst_window
        self.momentum_window = momentum_window
        self.mr_z_window = mr_z_window
        self.threshold_trend = threshold_trend
        self.threshold_mr = threshold_mr
        self.dead_zone = dead_zone
        self.vol_target = vol_target

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        returns = np.log(data / data.shift(1)).dropna()
        timestamp = data.index[-1]
        min_history = max(self.hurst_window, self.momentum_window, self.mr_z_window) + 20

        if len(returns) < min_history:
            return []

        signals = []

        for asset in data.columns:
            prices = data[asset].dropna()
            ret = returns[asset].dropna()
            if len(ret) < min_history:
                continue

            # Estimate Hurst exponent on recent price data
            H = estimate_hurst(prices.iloc[-self.hurst_window:].values)

            # Regime confidence: |H - 0.5| normalised
            regime_strength = abs(H - 0.5) * 2.0  # ranges 0 to ~1

            # Dead zone: skip if H is too close to 0.5
            if abs(H - 0.5) < self.dead_zone:
                continue

            # Volatility scaling
            vol = ret.rolling(72).std().iloc[-1]
            annual_vol = vol * np.sqrt(8760)
            vol_scale = min(self.vol_target / annual_vol, 2.0) if annual_vol > 0 else 0

            if H > self.threshold_trend:
                # TRENDING REGIME: trade momentum
                cum_ret = ret.iloc[-self.momentum_window:].sum()
                direction = np.sign(cum_ret)
                if direction == 0:
                    continue

                # Strength: regime confidence × momentum magnitude × vol scale
                mom_z = cum_ret / (ret.iloc[-self.momentum_window:].std() * np.sqrt(self.momentum_window))
                if np.isnan(mom_z):
                    continue
                strength = min(regime_strength * min(abs(mom_z) * 0.3, 1.0) * vol_scale, 1.0)

                if strength > 0.05:
                    signals.append(
                        Signal(
                            timestamp=timestamp,
                            asset=asset,
                            direction=direction,
                            strength=strength,
                            metadata={
                                "hurst": H,
                                "regime": "trending",
                                "momentum_z": mom_z,
                                "regime_strength": regime_strength,
                            },
                        )
                    )

            elif H < self.threshold_mr:
                # MEAN-REVERTING REGIME: trade counter-trend
                ma = prices.rolling(self.mr_z_window).mean().iloc[-1]
                std = prices.rolling(self.mr_z_window).std().iloc[-1]
                if std == 0:
                    continue
                z = (prices.iloc[-1] - ma) / std

                if abs(z) < 1.0:
                    continue  # not extended enough

                direction = -np.sign(z)  # counter-trend

                # Strength: regime confidence × deviation × vol scale
                dev_score = min((abs(z) - 1.0) / 2.0, 1.0)
                strength = min(regime_strength * dev_score * vol_scale, 1.0)

                if strength > 0.05:
                    signals.append(
                        Signal(
                            timestamp=timestamp,
                            asset=asset,
                            direction=direction,
                            strength=strength,
                            metadata={
                                "hurst": H,
                                "regime": "mean_reverting",
                                "z_score": z,
                                "regime_strength": regime_strength,
                            },
                        )
                    )

        return signals
