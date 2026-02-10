"""Base strategy interface and common utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Signal:
    """A trading signal for a single asset at a point in time."""

    timestamp: pd.Timestamp
    asset: str
    direction: float  # -1 to +1 (short to long)
    strength: float  # 0 to 1 (confidence)
    metadata: dict = field(default_factory=dict)

    @property
    def position_size(self) -> float:
        """Suggested position size = direction * strength."""
        return self.direction * self.strength


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        """Generate trading signals from market data.

        Parameters
        ----------
        data : DataFrame with at minimum close prices indexed by datetime

        Returns
        -------
        List of Signal objects
        """
        ...

    def signals_to_weights(self, signals: list[Signal]) -> pd.Series:
        """Convert a list of signals to a portfolio weight Series."""
        if not signals:
            return pd.Series(dtype=float)
        weights = {}
        for s in signals:
            weights[s.asset] = s.position_size
        series = pd.Series(weights)
        # Normalise to sum of absolute weights = 1
        total = series.abs().sum()
        if total > 0:
            series = series / total
        return series


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Detect crossover events. Returns +1 on cross up, -1 on cross down."""
    prev_diff = (fast.shift(1) - slow.shift(1))
    curr_diff = (fast - slow)
    cross_up = (prev_diff <= 0) & (curr_diff > 0)
    cross_down = (prev_diff >= 0) & (curr_diff < 0)
    signal = pd.Series(0.0, index=fast.index)
    signal[cross_up] = 1.0
    signal[cross_down] = -1.0
    return signal


def normalise_signal(raw: pd.Series, window: int = 168) -> pd.Series:
    """Z-score normalise a raw signal for comparability."""
    mean = raw.rolling(window).mean()
    std = raw.rolling(window).std()
    return ((raw - mean) / std.replace(0, np.nan)).fillna(0).clip(-3, 3) / 3
