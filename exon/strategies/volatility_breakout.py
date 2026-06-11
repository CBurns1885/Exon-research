"""Volatility breakout strategy tuned for crypto markets.

Crypto assets exhibit distinct volatility clustering: long periods of
compression followed by explosive directional moves. This strategy
detects the compression via ATR squeeze and trades the breakout from
Donchian channels, with position sizing inversely proportional to
current volatility (risk-normalised).

Particularly effective on BTC, ETH, and high-cap alts where breakout
momentum tends to persist for 24-72 hours before mean-reverting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class VolatilityBreakout(Strategy):
    """Donchian channel breakout with ATR squeeze filter.

    Entry logic:
        1. Detect volatility compression (ATR squeeze ratio < threshold)
        2. Wait for price to break above/below the Donchian channel
        3. Enter in breakout direction with vol-scaled position size

    The squeeze filter avoids whipsaws in range-bound markets and
    concentrates capital on the high-conviction breakout events that
    crypto is known for.
    """

    name = "vol_breakout"

    def __init__(
        self,
        channel_window: int = 48,
        atr_window: int = 24,
        squeeze_window: int = 168,
        squeeze_threshold: float = 0.75,
        vol_target: float = 0.15,
        confirmation_bars: int = 2,
    ):
        """
        Parameters
        ----------
        channel_window : lookback for Donchian high/low channel
        atr_window : lookback for Average True Range calculation
        squeeze_window : lookback for long-term ATR baseline (squeeze denominator)
        squeeze_threshold : ATR ratio below which we consider volatility "squeezed"
        vol_target : annualised volatility target for position sizing
        confirmation_bars : number of consecutive bars price must hold beyond
                           the channel to confirm the breakout
        """
        self.channel_window = channel_window
        self.atr_window = atr_window
        self.squeeze_window = squeeze_window
        self.squeeze_threshold = squeeze_threshold
        self.vol_target = vol_target
        self.confirmation_bars = confirmation_bars

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        signals = []
        timestamp = data.index[-1]
        min_history = max(self.channel_window, self.squeeze_window) + self.confirmation_bars

        for asset in data.columns:
            prices = data[asset].dropna()
            if len(prices) < min_history:
                continue

            # Donchian channel
            high_channel = prices.rolling(self.channel_window).max()
            low_channel = prices.rolling(self.channel_window).min()

            # ATR (using close-to-close as proxy since we only have close prices)
            abs_changes = prices.diff().abs()
            atr_short = abs_changes.rolling(self.atr_window).mean()
            atr_long = abs_changes.rolling(self.squeeze_window).mean()

            current_price = prices.iloc[-1]
            current_high = high_channel.iloc[-1 - self.confirmation_bars]
            current_low = low_channel.iloc[-1 - self.confirmation_bars]
            current_atr = atr_short.iloc[-1]
            squeeze_ratio = (
                atr_short.iloc[-1] / atr_long.iloc[-1]
                if atr_long.iloc[-1] > 0
                else 1.0
            )

            # Check for squeeze condition (recent vol < long-term vol)
            in_squeeze = squeeze_ratio < self.squeeze_threshold

            # Check for confirmed breakout: price has been beyond channel
            # for confirmation_bars consecutive periods
            breaking_high = all(
                prices.iloc[-1 - i] > high_channel.iloc[-1 - self.confirmation_bars]
                for i in range(self.confirmation_bars)
            )
            breaking_low = all(
                prices.iloc[-1 - i] < low_channel.iloc[-1 - self.confirmation_bars]
                for i in range(self.confirmation_bars)
            )

            if not (breaking_high or breaking_low):
                continue

            direction = 1.0 if breaking_high else -1.0

            # Strength: squeeze + breakout magnitude
            # Tighter squeeze = higher conviction breakout
            squeeze_score = max(0.0, 1.0 - squeeze_ratio) if in_squeeze else 0.3

            # Breakout magnitude relative to channel width
            channel_width = current_high - current_low
            if channel_width > 0:
                if breaking_high:
                    magnitude = (current_price - current_high) / channel_width
                else:
                    magnitude = (current_low - current_price) / channel_width
                magnitude_score = min(magnitude * 2, 1.0)
            else:
                magnitude_score = 0.5

            # Volatility-based position sizing
            annual_vol = current_atr / current_price * np.sqrt(8760) if current_price > 0 else 1.0
            vol_scale = min(self.vol_target / annual_vol, 2.0) if annual_vol > 0 else 0

            strength = min(
                (0.4 * squeeze_score + 0.6 * magnitude_score) * vol_scale,
                1.0,
            )

            if strength > 0.05:
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=direction,
                        strength=strength,
                        metadata={
                            "squeeze_ratio": squeeze_ratio,
                            "in_squeeze": in_squeeze,
                            "channel_high": current_high,
                            "channel_low": current_low,
                            "atr": current_atr,
                        },
                    )
                )

        return signals
