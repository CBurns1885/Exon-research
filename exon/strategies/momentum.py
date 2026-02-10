"""Advanced momentum strategies.

Goes beyond simple moving-average crossovers to include:
- Time-series momentum (absolute returns lookback)
- Cross-sectional momentum (relative ranking)
- Residual momentum (alpha after stripping factor exposure)
- Adaptive momentum with regime awareness
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy, normalise_signal


class TimeSeriesMomentum(Strategy):
    """Classic time-series momentum: go long assets with positive past returns,
    short those with negative past returns.

    Uses multiple lookback windows and volatility scaling.
    """

    name = "ts_momentum"

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        vol_target: float = 0.15,
        vol_window: int = 72,
    ):
        self.lookbacks = lookbacks or [24, 72, 168, 336, 720]
        self.vol_target = vol_target
        self.vol_window = vol_window

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        returns = np.log(data / data.shift(1))
        signals = []
        timestamp = data.index[-1]

        for asset in data.columns:
            ret = returns[asset].dropna()
            if len(ret) < max(self.lookbacks):
                continue

            # Blended momentum across lookbacks — z-score each window
            mom_scores = []
            for lb in self.lookbacks:
                cum_ret = ret.iloc[-lb:].sum()
                lb_vol = ret.iloc[-lb:].std() * np.sqrt(lb)
                z = cum_ret / lb_vol if lb_vol > 0 else 0.0
                mom_scores.append(z)
            raw_signal = np.mean(mom_scores)

            # Volatility scaling: target vol / realised vol
            vol = ret.rolling(self.vol_window).std().iloc[-1]
            annual_vol = vol * np.sqrt(8760)
            vol_scale = min(self.vol_target / annual_vol, 2.0) if annual_vol > 0 else 0

            direction = np.sign(raw_signal)
            strength = min(abs(raw_signal) * vol_scale, 1.0)

            if abs(strength) > 0.05:
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=direction,
                        strength=strength,
                        metadata={"raw_mom": raw_signal, "vol": annual_vol},
                    )
                )
        return signals


class CrossSectionalMomentum(Strategy):
    """Rank assets by recent performance: long the top N, short the bottom N."""

    name = "xs_momentum"

    def __init__(
        self,
        lookback: int = 168,
        top_n: int = 3,
        bottom_n: int = 3,
        holding_period: int = 24,
    ):
        self.lookback = lookback
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.holding_period = holding_period

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        returns = np.log(data / data.shift(1))
        timestamp = data.index[-1]

        # Rank by cumulative return over lookback
        cum_returns = returns.iloc[-self.lookback :].sum().sort_values(ascending=False)

        signals = []
        n_assets = len(cum_returns)
        long_assets = cum_returns.index[: self.top_n].tolist()
        short_assets = cum_returns.index[-self.bottom_n :].tolist()

        for asset in long_assets:
            rank = long_assets.index(asset)
            strength = 1.0 - (rank / max(self.top_n, 1)) * 0.5
            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=asset,
                    direction=1.0,
                    strength=strength,
                    metadata={"cum_return": cum_returns[asset], "rank": rank},
                )
            )

        for asset in short_assets:
            rank = short_assets.index(asset)
            strength = 1.0 - (rank / max(self.bottom_n, 1)) * 0.5
            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=asset,
                    direction=-1.0,
                    strength=strength,
                    metadata={"cum_return": cum_returns[asset]},
                )
            )

        return signals


class AdaptiveMomentum(Strategy):
    """Momentum with regime-adaptive lookback windows.

    In trending regimes, uses longer lookbacks; in mean-reverting regimes,
    reduces exposure or shortens lookbacks.
    """

    name = "adaptive_momentum"

    def __init__(
        self,
        base_lookback: int = 168,
        trend_threshold: float = 0.3,
        vol_ratio_window_short: int = 24,
        vol_ratio_window_long: int = 168,
    ):
        self.base_lookback = base_lookback
        self.trend_threshold = trend_threshold
        self.vol_short = vol_ratio_window_short
        self.vol_long = vol_ratio_window_long

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        returns = np.log(data / data.shift(1))
        timestamp = data.index[-1]
        signals = []

        for asset in data.columns:
            prices = data[asset].dropna()
            ret = returns[asset].dropna()
            if len(ret) < self.base_lookback * 2:
                continue

            # Trend strength (efficiency ratio)
            net_change = prices.iloc[-1] - prices.iloc[-self.base_lookback]
            sum_abs = prices.diff().abs().iloc[-self.base_lookback :].sum()
            efficiency = abs(net_change / sum_abs) if sum_abs > 0 else 0

            # Volatility regime
            vol_short = ret.rolling(self.vol_short).std().iloc[-1]
            vol_long = ret.rolling(self.vol_long).std().iloc[-1]
            vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0

            # Adaptive lookback: longer in trends, shorter in chop
            if efficiency > self.trend_threshold:
                lookback = int(self.base_lookback * 1.5)
                confidence_boost = 1.2
            else:
                lookback = int(self.base_lookback * 0.5)
                confidence_boost = 0.6

            # Reduce exposure in high-vol regimes
            vol_penalty = min(1.0, 1.0 / vol_ratio) if vol_ratio > 1.5 else 1.0

            cum_ret = ret.iloc[-lookback:].sum()
            direction = np.sign(cum_ret)
            raw_strength = min(abs(cum_ret) * 5, 1.0)
            strength = raw_strength * confidence_boost * vol_penalty

            if strength > 0.05:
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=direction,
                        strength=min(strength, 1.0),
                        metadata={
                            "efficiency": efficiency,
                            "vol_ratio": vol_ratio,
                            "adapted_lookback": lookback,
                        },
                    )
                )

        return signals
