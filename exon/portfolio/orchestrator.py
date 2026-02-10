"""Strategy orchestrator — the brain that allocates capital across models.

At a real fund, all models run all the time. The question is: how much
capital does each model get? The orchestrator decides this dynamically
based on three inputs:

1. **Regime detection**: GMM + Hurst analysis determines the current
   market state. Momentum models get more capital in trending regimes,
   mean-reversion models get more in range-bound regimes, etc.

2. **Rolling model performance**: Models that have been performing well
   recently get more capital (Bayesian updating of model quality).
   This is a form of online learning — the system adapts to which
   models are working in the current environment.

3. **Signal conviction**: Models with stronger signals (higher z-scores,
   wider spreads) get proportionally more capital for those specific
   trades.

The orchestrator runs at every rebalance:
    1. All strategies generate signals (always-on)
    2. Regime detector classifies current state
    3. Rolling Sharpe for each strategy updates allocation weights
    4. Signals are weighted by (base_allocation × regime_mult × perf_mult)
    5. Portfolio optimizer produces final weights
    6. Risk manager gates the output
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.base import Strategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Tracks rolling performance of a strategy for adaptive allocation."""

    name: str
    returns: list[float] = field(default_factory=list)
    signals_generated: int = 0
    trades_executed: int = 0

    @property
    def rolling_sharpe(self) -> float:
        if len(self.returns) < 20:
            return 0.0
        r = np.array(self.returns[-720:])  # last 30 days of hourly
        std = r.std()
        if std == 0:
            return 0.0
        return r.mean() / std * np.sqrt(8760)

    @property
    def hit_rate(self) -> float:
        if len(self.returns) < 10:
            return 0.5
        r = np.array(self.returns[-720:])
        return (r > 0).sum() / len(r)

    def record_return(self, ret: float):
        self.returns.append(ret)


@dataclass
class OrchestratorConfig:
    """Configuration for the strategy orchestrator."""

    # How much to weight regime-based allocation vs performance-based
    regime_weight: float = 0.5
    performance_weight: float = 0.3
    base_weight: float = 0.2  # residual from config file allocations

    # Performance smoothing
    performance_lookback: int = 720  # hours
    min_performance_history: int = 48  # hours before perf-weighting kicks in

    # Regime → strategy type multipliers
    # regime 0=low vol, 1=normal, 2=high vol
    regime_allocations: dict = field(default_factory=lambda: {
        0: {  # Low volatility (trending)
            "momentum": 1.5, "mean_reversion": 0.5, "stat_arb": 1.0,
            "breakout": 0.5, "factor": 1.0, "regime_filter": 1.0,
        },
        1: {  # Normal
            "momentum": 1.0, "mean_reversion": 1.2, "stat_arb": 1.3,
            "breakout": 0.8, "factor": 1.1, "regime_filter": 1.0,
        },
        2: {  # High volatility
            "momentum": 0.5, "mean_reversion": 0.8, "stat_arb": 0.5,
            "breakout": 1.5, "factor": 0.8, "regime_filter": 1.0,
        },
    })


def classify_strategy_type(name: str) -> str:
    """Map strategy name to a broad category for regime allocation."""
    if any(k in name for k in ["momentum", "wavelet", "xs_mom", "ts_mom", "adaptive"]):
        return "momentum"
    elif any(k in name for k in ["mr", "mean_reversion", "bollinger", "ou_"]):
        return "mean_reversion"
    elif any(k in name for k in ["kalman", "pairs", "stat_arb", "pca"]):
        return "stat_arb"
    elif "breakout" in name:
        return "breakout"
    elif any(k in name for k in ["multifactor", "factor"]):
        return "factor"
    elif "hurst" in name:
        return "regime_filter"
    else:
        return "factor"


class StrategyOrchestrator:
    """Dynamically allocates capital across all strategies.

    All strategies run at every rebalance. The orchestrator controls
    how much weight each strategy's signals receive in the final
    portfolio, based on regime + recent performance.
    """

    def __init__(
        self,
        strategies: list[tuple[Strategy, float]],
        config: OrchestratorConfig | None = None,
    ):
        """
        Parameters
        ----------
        strategies : list of (Strategy, base_allocation) tuples
        config : orchestrator configuration
        """
        self.strategies = strategies
        self.config = config or OrchestratorConfig()
        self.performance: dict[str, StrategyPerformance] = {}

        for strat, _ in strategies:
            self.performance[strat.name] = StrategyPerformance(name=strat.name)

    def generate_weighted_signals(
        self,
        data: pd.DataFrame,
        regime: int | None = None,
        **kwargs,
    ) -> list[Signal]:
        """Run all strategies and weight their signals.

        Parameters
        ----------
        data : price DataFrame
        regime : current market regime (0, 1, or 2)

        Returns
        -------
        List of weighted, blended signals
        """
        # Step 1: compute dynamic allocations
        allocations = self._compute_allocations(regime)

        # Step 2: run all strategies and collect weighted signals
        asset_signals: dict[str, list[tuple[float, Signal]]] = {}

        for strategy, base_alloc in self.strategies:
            weight = allocations.get(strategy.name, base_alloc)

            try:
                signals = strategy.generate_signals(data, **kwargs)
                self.performance[strategy.name].signals_generated += len(signals)
            except Exception as e:
                logger.debug("Strategy %s failed: %s", strategy.name, e)
                continue

            for sig in signals:
                asset_signals.setdefault(sig.asset, []).append((weight, sig))

        # Step 3: blend signals per asset
        timestamp = data.index[-1]
        blended = []

        for asset, weighted_sigs in asset_signals.items():
            total_weight = sum(w for w, _ in weighted_sigs)
            if total_weight == 0:
                continue

            blended_dir = sum(w * s.direction * s.strength for w, s in weighted_sigs)
            blended_dir /= total_weight

            direction = np.sign(blended_dir)
            strength = min(abs(blended_dir), 1.0)

            if strength > 0.02:
                contributing = [
                    (type(s).__class__.__name__, w) for w, s in weighted_sigs
                ]
                blended.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=direction,
                        strength=strength,
                        metadata={
                            "n_models": len(weighted_sigs),
                            "regime": regime,
                        },
                    )
                )

        return blended

    def _compute_allocations(self, regime: int | None) -> dict[str, float]:
        """Compute dynamic allocation per strategy."""
        cfg = self.config
        allocations = {}

        for strategy, base_alloc in self.strategies:
            name = strategy.name

            # Component 1: base allocation from config
            base = base_alloc

            # Component 2: regime-based multiplier
            regime_mult = 1.0
            if regime is not None:
                strat_type = classify_strategy_type(name)
                regime_mults = cfg.regime_allocations.get(regime, {})
                regime_mult = regime_mults.get(strat_type, 1.0)

            # Component 3: performance-based multiplier
            perf_mult = 1.0
            perf = self.performance.get(name)
            if perf and len(perf.returns) >= cfg.min_performance_history:
                sharpe = perf.rolling_sharpe
                # Map Sharpe to multiplier: Sharpe of 2 → mult of 1.5, -1 → 0.5
                perf_mult = np.clip(1.0 + sharpe * 0.25, 0.3, 2.0)

            # Weighted combination
            final = (
                cfg.base_weight * base
                + cfg.regime_weight * base * regime_mult
                + cfg.performance_weight * base * perf_mult
            )
            allocations[name] = max(final, 0.01)

        # Normalise so allocations sum to 1
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v / total for k, v in allocations.items()}

        return allocations

    def record_strategy_returns(
        self,
        strategy_name: str,
        period_return: float,
    ):
        """Record a strategy's realised return for performance tracking."""
        if strategy_name in self.performance:
            self.performance[strategy_name].record_return(period_return)

    def get_allocation_summary(self, regime: int | None = None) -> pd.DataFrame:
        """Get current allocation weights for all strategies."""
        allocs = self._compute_allocations(regime)
        rows = []
        for name, weight in allocs.items():
            perf = self.performance.get(name)
            rows.append({
                "strategy": name,
                "allocation": weight,
                "rolling_sharpe": perf.rolling_sharpe if perf else 0,
                "hit_rate": perf.hit_rate if perf else 0.5,
                "signals_generated": perf.signals_generated if perf else 0,
                "type": classify_strategy_type(name),
            })
        return pd.DataFrame(rows).set_index("strategy").sort_values(
            "allocation", ascending=False
        )
