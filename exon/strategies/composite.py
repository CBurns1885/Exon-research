"""Composite strategy that combines signals from multiple sub-strategies.

Weights sub-strategy signals based on regime, recent performance, or
fixed allocations to produce a single blended signal per asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class CompositeStrategy(Strategy):
    """Blends signals from multiple strategies with configurable weights."""

    name = "composite"

    def __init__(
        self,
        strategies: list[tuple[Strategy, float]],
        regime_aware: bool = True,
    ):
        """
        Parameters
        ----------
        strategies : list of (Strategy, weight) tuples
        regime_aware : if True, adjusts weights based on market regime
        """
        self.strategies = strategies
        self.regime_aware = regime_aware

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        regime = kwargs.get("regime")  # optional regime label
        asset_signals: dict[str, list[tuple[float, Signal]]] = {}

        for strategy, base_weight in self.strategies:
            weight = self._adjust_weight(strategy, base_weight, regime)
            try:
                signals = strategy.generate_signals(data, **kwargs)
            except Exception:
                continue

            for sig in signals:
                if sig.asset not in asset_signals:
                    asset_signals[sig.asset] = []
                asset_signals[sig.asset].append((weight, sig))

        # Blend signals per asset
        combined = []
        timestamp = data.index[-1]

        for asset, weighted_sigs in asset_signals.items():
            total_weight = sum(w for w, _ in weighted_sigs)
            if total_weight == 0:
                continue

            blended_direction = sum(w * s.direction * s.strength for w, s in weighted_sigs)
            blended_direction /= total_weight

            direction = np.sign(blended_direction)
            strength = min(abs(blended_direction), 1.0)

            if strength > 0.02:
                combined.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=direction,
                        strength=strength,
                        metadata={
                            "n_strategies": len(weighted_sigs),
                            "sources": [s.metadata.get("strategy", type(s).__name__)
                                        for _, s in weighted_sigs],
                        },
                    )
                )

        return combined

    def _adjust_weight(self, strategy: Strategy, base_weight: float, regime) -> float:
        """Adjust strategy weight based on current market regime."""
        if not self.regime_aware or regime is None:
            return base_weight

        # Regime 0 = low vol, 1 = normal, 2 = high vol (from regime detector)
        name = strategy.name

        if name == "kalman_spread" or name == "kalman_scanner":
            # Kalman spread: best in normal vol, poor in crisis (correlation breaks)
            regime_multiplier = {0: 1.0, 1: 1.3, 2: 0.5}.get(regime, 1.0)
        elif name == "multifactor":
            # Multi-factor: fairly regime-neutral, slight edge in normal
            regime_multiplier = {0: 1.0, 1: 1.1, 2: 0.8}.get(regime, 1.0)
        elif name == "hurst_regime":
            # Hurst: it IS the regime filter, works everywhere
            regime_multiplier = {0: 1.0, 1: 1.0, 2: 1.0}.get(regime, 1.0)
        elif "wavelet" in name:
            # Wavelet momentum: multi-scale, slightly better in trending
            regime_multiplier = {0: 1.2, 1: 1.0, 2: 0.7}.get(regime, 1.0)
        elif "breakout" in name or name == "vol_breakout":
            # Breakout thrives in volatility expansion, dies in compression
            regime_multiplier = {0: 0.5, 1: 0.8, 2: 1.5}.get(regime, 1.0)
        elif "lead_lag" in name:
            # Lead-lag works in all regimes but best in trending moves
            regime_multiplier = {0: 1.2, 1: 1.0, 2: 0.8}.get(regime, 1.0)
        elif "volume" in name:
            # Volume momentum is regime-neutral; vol confirmation helps everywhere
            regime_multiplier = {0: 1.0, 1: 1.1, 2: 0.9}.get(regime, 1.0)
        elif "momentum" in name:
            # Momentum works better in trending/low-vol regimes
            regime_multiplier = {0: 1.3, 1: 1.0, 2: 0.5}.get(regime, 1.0)
        elif "mean_reversion" in name or "mr" in name:
            # Mean reversion works better in range-bound/normal regimes
            regime_multiplier = {0: 0.7, 1: 1.2, 2: 0.8}.get(regime, 1.0)
        elif "pairs" in name or "stat_arb" in name:
            # Stat arb tends to be regime-neutral but suffers in crisis
            regime_multiplier = {0: 1.0, 1: 1.1, 2: 0.6}.get(regime, 1.0)
        else:
            regime_multiplier = 1.0

        return base_weight * regime_multiplier
