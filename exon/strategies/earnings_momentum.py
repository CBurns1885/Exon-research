"""Earnings momentum / post-earnings announcement drift (PEAD) strategy.

One of the most well-documented anomalies in equity markets:
stocks that beat earnings expectations tend to continue drifting
in the direction of the surprise for 30-60 trading days.

Since we can't access fundamental data through Alpaca directly,
this strategy uses a price-based proxy for earnings surprises:
- Detects sudden gap moves (>2 std of daily returns)
- Classifies them as potential earnings events
- Trades the drift in the direction of the surprise

This is a simplified but effective proxy used by many quant funds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class EarningsMomentum(Strategy):
    """Post-earnings-announcement-drift (PEAD) proxy strategy.

    Detects large gap/jump moves and trades the continuation drift.
    In production, this would be enhanced with actual earnings date
    calendars and consensus estimate data.
    """

    name = "earnings_momentum"

    def __init__(
        self,
        gap_threshold_std: float = 2.5,
        drift_window: int = 40,
        vol_lookback: int = 63,
        decay_rate: float = 0.95,
        max_concurrent: int = 10,
    ):
        """
        Parameters
        ----------
        gap_threshold_std : minimum z-score for a move to be flagged as an event
        drift_window : how many bars to hold the drift trade
        vol_lookback : lookback for volatility estimation
        decay_rate : daily signal decay (0.95 = halved in ~14 days)
        max_concurrent : max number of concurrent drift trades
        """
        self.gap_threshold_std = gap_threshold_std
        self.drift_window = drift_window
        self.vol_lookback = vol_lookback
        self.decay_rate = decay_rate
        self.max_concurrent = max_concurrent

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        min_bars = self.vol_lookback + self.drift_window + 10
        if len(data) < min_bars:
            return []

        timestamp = data.index[-1]
        returns = data.pct_change().dropna()

        signals = []
        for col in data.columns:
            sig = self._check_drift(returns[col], col, timestamp)
            if sig is not None:
                signals.append(sig)

        # Limit concurrent positions
        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals[:self.max_concurrent]

    def _check_drift(
        self,
        returns: pd.Series,
        asset: str,
        timestamp: pd.Timestamp,
    ) -> Signal | None:
        """Check if an asset has a recent event and is in the drift window."""
        if len(returns) < self.vol_lookback + 10:
            return None

        # Estimate normal volatility (excluding recent event window)
        baseline_vol = returns.iloc[-self.vol_lookback - self.drift_window:-self.drift_window].std()
        if baseline_vol <= 0:
            return None

        # Look for event days in the drift window
        recent = returns.iloc[-self.drift_window:]
        z_scores = recent / baseline_vol

        # Find the most extreme day (potential earnings event)
        event_idx = z_scores.abs().idxmax()
        event_z = z_scores.loc[event_idx]

        if abs(event_z) < self.gap_threshold_std:
            return None  # no significant event

        # How many bars since the event?
        event_pos = list(recent.index).index(event_idx)
        bars_since_event = len(recent) - event_pos - 1

        if bars_since_event > self.drift_window:
            return None  # event too old

        # Direction = same as the surprise
        direction = np.sign(event_z)

        # Strength decays over time
        decay = self.decay_rate ** bars_since_event
        raw_strength = min(abs(event_z) / 5.0, 1.0)
        strength = raw_strength * decay

        # Check if the drift is still intact (price hasn't fully reverted)
        post_event_ret = recent.iloc[event_pos:].sum()
        if np.sign(post_event_ret) != direction:
            # Price has already reverted — don't chase
            strength *= 0.3

        if strength < 0.05:
            return None

        return Signal(
            timestamp=timestamp,
            asset=asset,
            direction=direction,
            strength=strength,
            metadata={
                "event_z": float(event_z),
                "bars_since_event": bars_since_event,
                "event_date": str(event_idx),
                "post_event_return": float(post_event_ret),
                "baseline_vol": float(baseline_vol),
            },
        )
