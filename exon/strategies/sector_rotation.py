"""Sector rotation strategy for US equities.

Classic quant equity strategy: rotates capital between sectors based on
relative momentum and mean-reversion signals. Uses sector ETFs as proxies
(XLK, XLF, XLE, XLV, etc.) or classifies individual stocks by sector.

Theory:
- Sectors exhibit regime-dependent performance driven by macro factors
  (rates, inflation, credit, growth)
- Momentum within sectors persists at 1-3 month horizons
- Cross-sectional ranking exploits relative strength divergences
- Risk-off/risk-on transitions rotate across defensive/cyclical sectors
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy

# Default sector ETF universe
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


class SectorRotation(Strategy):
    """Rotate between equity sectors based on multi-signal ranking.

    Combines:
    1. Momentum score (12-1 month: skip most recent month for reversal)
    2. Relative strength vs SPY
    3. Volatility-adjusted returns
    4. Mean-reversion overlay for extreme moves
    """

    name = "sector_rotation"

    def __init__(
        self,
        momentum_window: int = 252,
        skip_window: int = 21,
        vol_window: int = 63,
        top_n: int = 3,
        bottom_n: int = 2,
        spy_column: str = "SPY",
        rebalance_threshold: float = 0.05,
    ):
        self.momentum_window = momentum_window
        self.skip_window = skip_window
        self.vol_window = vol_window
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.spy_column = spy_column
        self.rebalance_threshold = rebalance_threshold

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        min_bars = self.momentum_window + self.skip_window + 10
        if len(data) < min_bars:
            return []

        timestamp = data.index[-1]
        returns = np.log(data / data.shift(1)).dropna()

        if len(returns) < min_bars - 1:
            return []

        # Score each sector
        scores = {}
        for col in data.columns:
            if col == self.spy_column:
                continue

            # 1. 12-1 month momentum (skip recent month for short-term reversal)
            total_ret = returns[col].iloc[-self.momentum_window:-self.skip_window].sum()

            # 2. Relative strength vs market (SPY)
            if self.spy_column in returns.columns:
                spy_ret = returns[self.spy_column].iloc[-self.momentum_window:-self.skip_window].sum()
                rel_strength = total_ret - spy_ret
            else:
                rel_strength = 0.0

            # 3. Volatility-adjusted return (Sharpe-like)
            period_vol = returns[col].iloc[-self.vol_window:].std()
            if period_vol > 0:
                risk_adj = total_ret / (period_vol * np.sqrt(self.vol_window))
            else:
                risk_adj = 0.0

            # 4. Mean-reversion signal for recent extremes
            recent_ret = returns[col].iloc[-self.skip_window:].sum()
            recent_z = recent_ret / (period_vol * np.sqrt(self.skip_window)) if period_vol > 0 else 0

            # Composite score: momentum + relative strength + risk-adjusted
            # penalise if recently overextended (mean reversion)
            mr_penalty = -0.2 * recent_z  # slight counter-trend for extremes
            composite = 0.4 * risk_adj + 0.35 * rel_strength / max(period_vol, 0.01) + 0.25 * mr_penalty

            scores[col] = composite

        if not scores:
            return []

        ranked = pd.Series(scores).sort_values(ascending=False)

        signals = []
        # Long top sectors
        for asset in ranked.head(self.top_n).index:
            score = ranked[asset]
            strength = min(abs(score) / 3.0, 1.0)
            if strength > 0.05:
                signals.append(Signal(
                    timestamp=timestamp,
                    asset=asset,
                    direction=1.0,
                    strength=strength,
                    metadata={
                        "composite_score": score,
                        "rank": int(list(ranked.index).index(asset)) + 1,
                        "sector": SECTOR_ETFS.get(asset, "unknown"),
                    },
                ))

        # Short bottom sectors
        for asset in ranked.tail(self.bottom_n).index:
            score = ranked[asset]
            strength = min(abs(score) / 3.0, 1.0)
            if strength > 0.05:
                signals.append(Signal(
                    timestamp=timestamp,
                    asset=asset,
                    direction=-1.0,
                    strength=strength,
                    metadata={
                        "composite_score": score,
                        "rank": int(list(ranked.index).index(asset)) + 1,
                        "sector": SECTOR_ETFS.get(asset, "unknown"),
                    },
                ))

        return signals
