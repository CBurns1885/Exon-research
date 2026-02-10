"""Volume-weighted momentum strategy for crypto markets.

Crypto volume is uniquely informative: exchange volumes are transparent,
wash-trading aside, and genuine volume surges precede major moves more
reliably than in equities. This strategy combines price momentum with
volume confirmation to filter false breakouts and size into high-
conviction moves.

Three volume signals are fused:
1. Relative Volume (RVOL): current volume vs rolling average
2. Volume-Price Trend (VPT): cumulative volume-weighted price direction
3. On-Balance Volume divergence: OBV trend vs price trend disagreement
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class VolumeWeightedMomentum(Strategy):
    """Momentum strategy that requires volume confirmation.

    Only takes momentum signals when volume supports the move,
    and scales position size by volume conviction.
    """

    name = "volume_momentum"

    def __init__(
        self,
        momentum_window: int = 72,
        volume_window: int = 168,
        rvol_threshold: float = 1.5,
        vpt_window: int = 48,
        obv_divergence_window: int = 72,
        vol_target: float = 0.15,
    ):
        """
        Parameters
        ----------
        momentum_window : lookback for price momentum signal
        volume_window : lookback for average volume baseline
        rvol_threshold : relative volume above which we consider volume "elevated"
        vpt_window : lookback for Volume-Price Trend slope
        obv_divergence_window : lookback for OBV vs price divergence detection
        vol_target : annualised volatility target for sizing
        """
        self.momentum_window = momentum_window
        self.volume_window = volume_window
        self.rvol_threshold = rvol_threshold
        self.vpt_window = vpt_window
        self.obv_divergence_window = obv_divergence_window
        self.vol_target = vol_target

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        """Generate signals from price DataFrame.

        If `data` contains only close prices (no volume), the strategy
        falls back to a volume-proxy using absolute return magnitude as
        a substitute for volume intensity.

        If OHLCV DataFrames are passed per-asset in kwargs["ohlcv"],
        real volume is used.
        """
        ohlcv_map: dict[str, pd.DataFrame] = kwargs.get("ohlcv", {})
        signals = []
        timestamp = data.index[-1]
        min_history = max(self.momentum_window, self.volume_window, self.obv_divergence_window) + 10

        for asset in data.columns:
            prices = data[asset].dropna()
            if len(prices) < min_history:
                continue

            returns = np.log(prices / prices.shift(1))

            # Get volume data (real or synthesised from return magnitude)
            if asset in ohlcv_map and "volume" in ohlcv_map[asset].columns:
                volume = ohlcv_map[asset]["volume"].reindex(prices.index).fillna(0)
            else:
                # Proxy: absolute return as volume-like intensity measure
                volume = returns.abs() * prices

            # --- Signal 1: Relative Volume (RVOL) ---
            avg_vol = volume.rolling(self.volume_window).mean()
            rvol = volume / avg_vol.replace(0, np.nan)
            current_rvol = rvol.iloc[-1] if not np.isnan(rvol.iloc[-1]) else 1.0
            rvol_elevated = current_rvol > self.rvol_threshold

            # --- Signal 2: Volume-Price Trend (VPT) ---
            pct_change = prices.pct_change()
            vpt = (pct_change * volume).cumsum()
            vpt_slope = (vpt.iloc[-1] - vpt.iloc[-self.vpt_window]) / self.vpt_window
            vpt_direction = np.sign(vpt_slope)

            # --- Signal 3: OBV Divergence ---
            obv = (np.sign(returns) * volume).cumsum()
            price_trend = self._linear_slope(prices, self.obv_divergence_window)
            obv_trend = self._linear_slope(obv, self.obv_divergence_window)
            # Divergence: OBV trend disagrees with price trend
            divergence = obv_trend * np.sign(price_trend) < 0

            # --- Price momentum ---
            mom = returns.iloc[-self.momentum_window :].sum()
            mom_direction = np.sign(mom)
            mom_z = mom / (returns.iloc[-self.momentum_window :].std() * np.sqrt(self.momentum_window))
            if np.isnan(mom_z):
                continue

            # --- Combine ---
            # Base: momentum direction with z-score magnitude
            base_strength = min(abs(mom_z) * 0.3, 1.0)

            # Volume confirmation boost
            if rvol_elevated and vpt_direction == mom_direction:
                # Strong confirmation: both RVOL and VPT align with momentum
                volume_boost = min(current_rvol / self.rvol_threshold, 2.0) * 0.3
            elif rvol_elevated or vpt_direction == mom_direction:
                # Partial confirmation
                volume_boost = 0.1
            else:
                # No volume support: reduce conviction
                volume_boost = -0.15

            # Divergence override: if OBV diverges from price, flip or suppress
            if divergence:
                if obv_trend > 0 and price_trend < 0:
                    # Bullish divergence: OBV rising while price falling
                    mom_direction = 1.0
                    volume_boost += 0.2
                elif obv_trend < 0 and price_trend > 0:
                    # Bearish divergence: OBV falling while price rising
                    mom_direction = -1.0
                    volume_boost += 0.2

            # Vol-targeting
            recent_vol = returns.rolling(72).std().iloc[-1] * np.sqrt(8760)
            vol_scale = min(self.vol_target / recent_vol, 2.0) if recent_vol > 0 else 0

            strength = min(max(base_strength + volume_boost, 0) * vol_scale, 1.0)

            if strength > 0.05:
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=mom_direction,
                        strength=strength,
                        metadata={
                            "rvol": current_rvol,
                            "vpt_direction": vpt_direction,
                            "obv_divergence": divergence,
                            "momentum_z": mom_z,
                        },
                    )
                )

        return signals

    @staticmethod
    def _linear_slope(series: pd.Series, window: int) -> float:
        """Simple linear regression slope over the last `window` bars."""
        y = series.iloc[-window:].values
        if len(y) < window:
            return 0.0
        x = np.arange(len(y), dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return ((x - x_mean) * (y - y_mean)).sum() / denom
