"""Market microstructure-based equity strategy.

Uses price and volume patterns to detect institutional order flow:
1. VWAP reversion: stocks that deviate significantly from their intraday
   VWAP tend to revert (institutional benchmarks)
2. Volume profile: detects accumulation/distribution via OBV and volume-
   weighted metrics
3. Relative volume: unusual volume signals institutional activity

For daily bars, this operates as a volume-weighted momentum overlay
that detects accumulation (smart money buying) vs distribution (selling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class MarketMicrostructure(Strategy):
    """Detects institutional flow via volume-price analysis.

    Combines On-Balance Volume (OBV), accumulation/distribution line,
    and relative volume signals to identify institutional positioning.
    """

    name = "market_microstructure"

    def __init__(
        self,
        obv_window: int = 21,
        volume_window: int = 63,
        rvol_threshold: float = 1.5,
        price_volume_window: int = 10,
        vol_target: float = 0.15,
    ):
        self.obv_window = obv_window
        self.volume_window = volume_window
        self.rvol_threshold = rvol_threshold
        self.price_volume_window = price_volume_window
        self.vol_target = vol_target

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        """Generate signals based on volume-price analysis.

        Expects data with columns as assets. If volume data is available
        in kwargs['volume'], it uses actual volume; otherwise falls back
        to a price-based proxy.
        """
        min_bars = max(self.volume_window, self.obv_window) + 20
        if len(data) < min_bars:
            return []

        timestamp = data.index[-1]
        volume = kwargs.get("volume")  # Optional volume DataFrame

        signals = []
        for col in data.columns:
            prices = data[col].dropna()
            if len(prices) < min_bars:
                continue

            vol_series = volume[col] if volume is not None and col in volume.columns else None
            sig = self._analyse_flow(prices, vol_series, col, timestamp)
            if sig is not None:
                signals.append(sig)

        return signals

    def _analyse_flow(
        self,
        prices: pd.Series,
        volume: pd.Series | None,
        asset: str,
        timestamp: pd.Timestamp,
    ) -> Signal | None:
        """Analyse volume-price flow for a single asset."""
        returns = prices.pct_change().dropna()

        # 1. OBV (On-Balance Volume) trend
        if volume is not None and len(volume) == len(prices):
            obv = self._compute_obv(returns, volume.values[1:])
            obv_trend = self._obv_signal(obv)
        else:
            # Proxy: use absolute returns as volume proxy
            pseudo_vol = returns.abs()
            obv = self._compute_obv(returns, pseudo_vol.values)
            obv_trend = self._obv_signal(obv)

        # 2. Accumulation/Distribution via price position in range
        ad_signal = self._accumulation_distribution(prices)

        # 3. Relative volume (if available)
        rvol_signal = 0.0
        if volume is not None and len(volume) >= self.volume_window:
            recent_vol = volume.iloc[-self.price_volume_window:].mean()
            avg_vol = volume.iloc[-self.volume_window:].mean()
            if avg_vol > 0:
                rvol = recent_vol / avg_vol
                if rvol > self.rvol_threshold:
                    # High volume + positive price = accumulation
                    recent_ret = returns.iloc[-self.price_volume_window:].sum()
                    rvol_signal = np.sign(recent_ret) * min((rvol - 1.0) / 2.0, 1.0)

        # Composite signal
        composite = 0.4 * obv_trend + 0.35 * ad_signal + 0.25 * rvol_signal

        if abs(composite) < 0.05:
            return None

        direction = np.sign(composite)

        # Vol targeting
        annual_vol = returns.iloc[-self.volume_window:].std() * np.sqrt(252)
        vol_scale = min(self.vol_target / annual_vol, 2.0) if annual_vol > 0 else 1.0
        strength = min(abs(composite) * vol_scale, 1.0)

        if strength < 0.05:
            return None

        return Signal(
            timestamp=timestamp,
            asset=asset,
            direction=direction,
            strength=strength,
            metadata={
                "obv_trend": float(obv_trend),
                "ad_signal": float(ad_signal),
                "rvol_signal": float(rvol_signal),
                "composite": float(composite),
            },
        )

    def _compute_obv(self, returns: pd.Series, volume: np.ndarray) -> pd.Series:
        """Compute On-Balance Volume."""
        signs = np.sign(returns.values)
        # Align lengths
        n = min(len(signs), len(volume))
        obv = np.cumsum(signs[-n:] * volume[-n:])
        return pd.Series(obv, index=returns.index[-n:])

    def _obv_signal(self, obv: pd.Series) -> float:
        """OBV trend signal: is OBV trending up or down?"""
        if len(obv) < self.obv_window:
            return 0.0
        obv_sma = obv.rolling(self.obv_window).mean()
        if obv_sma.iloc[-1] == 0:
            return 0.0
        obv_z = (obv.iloc[-1] - obv_sma.iloc[-1]) / max(obv.rolling(self.obv_window).std().iloc[-1], 1e-10)
        return float(np.clip(obv_z / 3.0, -1, 1))

    def _accumulation_distribution(self, prices: pd.Series) -> float:
        """Accumulation/Distribution based on close position within High-Low range.

        For close-only data, uses rolling range as proxy.
        """
        window = self.price_volume_window
        if len(prices) < window + 5:
            return 0.0

        recent = prices.iloc[-window:]
        high = recent.max()
        low = recent.min()

        if high == low:
            return 0.0

        # Where is the current close relative to the range?
        close = prices.iloc[-1]
        clv = (2 * close - low - high) / (high - low)  # -1 to +1

        # Weight by how narrow the range is (tight range + directional close = conviction)
        range_pct = (high - low) / prices.iloc[-self.volume_window:].mean()

        return float(np.clip(clv * min(range_pct * 10, 1.0), -1, 1))
