"""Tests for the strategy orchestrator."""

import numpy as np
import pandas as pd
import pytest

from exon.strategies.base import Signal, Strategy
from exon.portfolio.orchestrator import (
    StrategyOrchestrator,
    OrchestratorConfig,
    StrategyPerformance,
    classify_strategy_type,
)


# ---------- Helpers ----------

class DummyStrategy(Strategy):
    """Simple strategy that returns fixed signals for testing."""

    def __init__(self, name: str, signals: list[Signal] | None = None):
        self.name = name
        self._signals = signals or []

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        return self._signals


class FailingStrategy(Strategy):
    """Strategy that always raises."""

    name = "failing"

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        raise RuntimeError("boom")


def _make_data(n=100):
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"BTC-USD": 100 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))},
        index=dates,
    )


def _make_signal(asset="BTC-USD", direction=1.0, strength=0.5):
    return Signal(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        asset=asset,
        direction=direction,
        strength=strength,
    )


# ---------- classify_strategy_type ----------

class TestClassifyStrategyType:
    def test_momentum_types(self):
        assert classify_strategy_type("ts_momentum") == "momentum"
        assert classify_strategy_type("wavelet_momentum") == "momentum"
        assert classify_strategy_type("xs_mom") == "momentum"
        assert classify_strategy_type("adaptive") == "momentum"

    def test_mean_reversion_types(self):
        assert classify_strategy_type("bollinger_mr") == "mean_reversion"
        assert classify_strategy_type("ou_mean_reversion") == "mean_reversion"

    def test_stat_arb_types(self):
        assert classify_strategy_type("kalman_spread") == "stat_arb"
        assert classify_strategy_type("pairs_trading") == "stat_arb"
        assert classify_strategy_type("pca_stat_arb") == "stat_arb"

    def test_breakout(self):
        assert classify_strategy_type("vol_breakout") == "breakout"

    def test_factor(self):
        assert classify_strategy_type("multifactor") == "factor"

    def test_regime_filter(self):
        assert classify_strategy_type("hurst_regime") == "regime_filter"

    def test_unknown_defaults_to_factor(self):
        assert classify_strategy_type("some_unknown") == "factor"


# ---------- StrategyPerformance ----------

class TestStrategyPerformance:
    def test_rolling_sharpe_insufficient_data(self):
        perf = StrategyPerformance(name="test")
        assert perf.rolling_sharpe == 0.0

    def test_rolling_sharpe_positive(self):
        perf = StrategyPerformance(name="test")
        rng = np.random.default_rng(42)
        for _ in range(100):
            perf.record_return(0.001 + rng.normal(0, 0.005))
        assert perf.rolling_sharpe > 0

    def test_hit_rate_insufficient_data(self):
        perf = StrategyPerformance(name="test")
        assert perf.hit_rate == 0.5

    def test_hit_rate_all_positive(self):
        perf = StrategyPerformance(name="test")
        for _ in range(50):
            perf.record_return(0.01)
        assert perf.hit_rate == 1.0

    def test_hit_rate_half(self):
        perf = StrategyPerformance(name="test")
        for i in range(100):
            perf.record_return(0.01 if i % 2 == 0 else -0.01)
        assert abs(perf.hit_rate - 0.5) < 0.01

    def test_rolling_sharpe_zero_std(self):
        perf = StrategyPerformance(name="test")
        for _ in range(30):
            perf.record_return(0.0)
        assert perf.rolling_sharpe == 0.0


# ---------- OrchestratorConfig ----------

class TestOrchestratorConfig:
    def test_default_weights_sum_to_one(self):
        cfg = OrchestratorConfig()
        total = cfg.regime_weight + cfg.performance_weight + cfg.base_weight
        assert abs(total - 1.0) < 1e-9

    def test_default_regime_allocations(self):
        cfg = OrchestratorConfig()
        assert 0 in cfg.regime_allocations
        assert 1 in cfg.regime_allocations
        assert 2 in cfg.regime_allocations


# ---------- StrategyOrchestrator ----------

class TestStrategyOrchestrator:
    def test_init_tracks_all_strategies(self):
        s1 = DummyStrategy("momentum_fast")
        s2 = DummyStrategy("kalman_spread")
        orch = StrategyOrchestrator([(s1, 0.5), (s2, 0.5)])
        assert "momentum_fast" in orch.performance
        assert "kalman_spread" in orch.performance

    def test_allocations_sum_to_one(self):
        s1 = DummyStrategy("ts_momentum")
        s2 = DummyStrategy("kalman_spread")
        s3 = DummyStrategy("vol_breakout")
        orch = StrategyOrchestrator([(s1, 0.4), (s2, 0.3), (s3, 0.3)])
        allocs = orch._compute_allocations(regime=1)
        assert abs(sum(allocs.values()) - 1.0) < 1e-9

    def test_allocations_without_regime(self):
        s1 = DummyStrategy("ts_momentum")
        s2 = DummyStrategy("kalman_spread")
        orch = StrategyOrchestrator([(s1, 0.6), (s2, 0.4)])
        allocs = orch._compute_allocations(regime=None)
        assert abs(sum(allocs.values()) - 1.0) < 1e-9

    def test_regime_affects_allocations(self):
        """In low-vol regime, momentum should get higher allocation than breakout."""
        s_mom = DummyStrategy("ts_momentum")
        s_brk = DummyStrategy("vol_breakout")
        orch = StrategyOrchestrator(
            [(s_mom, 0.5), (s_brk, 0.5)],
            config=OrchestratorConfig(regime_weight=0.8, performance_weight=0.0, base_weight=0.2),
        )
        allocs_low_vol = orch._compute_allocations(regime=0)
        # Momentum gets 1.5x in low vol, breakout gets 0.5x
        assert allocs_low_vol["ts_momentum"] > allocs_low_vol["vol_breakout"]

        allocs_high_vol = orch._compute_allocations(regime=2)
        # Breakout gets 1.5x in high vol, momentum gets 0.5x
        assert allocs_high_vol["vol_breakout"] > allocs_high_vol["ts_momentum"]

    def test_performance_affects_allocations(self):
        """Strategy with better Sharpe should get more allocation."""
        s1 = DummyStrategy("ts_momentum")
        s2 = DummyStrategy("kalman_spread")
        orch = StrategyOrchestrator(
            [(s1, 0.5), (s2, 0.5)],
            config=OrchestratorConfig(
                regime_weight=0.0,
                performance_weight=0.8,
                base_weight=0.2,
                min_performance_history=20,
            ),
        )
        # Give s1 great returns, s2 bad returns
        rng = np.random.default_rng(42)
        for _ in range(100):
            orch.record_strategy_returns("ts_momentum", 0.005 + rng.normal(0, 0.002))
            orch.record_strategy_returns("kalman_spread", -0.003 + rng.normal(0, 0.002))

        allocs = orch._compute_allocations(regime=None)
        assert allocs["ts_momentum"] > allocs["kalman_spread"]

    def test_generate_weighted_signals(self):
        sig1 = _make_signal("BTC-USD", 1.0, 0.8)
        sig2 = _make_signal("BTC-USD", 1.0, 0.6)
        s1 = DummyStrategy("ts_momentum", [sig1])
        s2 = DummyStrategy("kalman_spread", [sig2])
        orch = StrategyOrchestrator([(s1, 0.5), (s2, 0.5)])

        data = _make_data()
        blended = orch.generate_weighted_signals(data, regime=1)
        assert len(blended) > 0
        assert blended[0].asset == "BTC-USD"
        assert blended[0].direction == 1.0

    def test_conflicting_signals_blend(self):
        """Opposing signals should partially cancel."""
        sig_long = _make_signal("BTC-USD", 1.0, 0.8)
        sig_short = _make_signal("BTC-USD", -1.0, 0.3)
        s1 = DummyStrategy("ts_momentum", [sig_long])
        s2 = DummyStrategy("bollinger_mr", [sig_short])
        orch = StrategyOrchestrator([(s1, 0.5), (s2, 0.5)])

        data = _make_data()
        blended = orch.generate_weighted_signals(data)
        # Should still be net long (0.8 > 0.3)
        if blended:
            assert blended[0].direction == 1.0
            # Blended strength should be less than the stronger signal
            assert blended[0].strength < 0.8

    def test_failing_strategy_is_skipped(self):
        sig = _make_signal("BTC-USD", 1.0, 0.5)
        s1 = DummyStrategy("ts_momentum", [sig])
        s2 = FailingStrategy()
        orch = StrategyOrchestrator([(s1, 0.5), (s2, 0.5)])

        data = _make_data()
        blended = orch.generate_weighted_signals(data)
        # Should still get the signal from s1
        assert len(blended) > 0

    def test_no_signal_below_threshold(self):
        """Very weak signals should be filtered out."""
        sig = _make_signal("BTC-USD", 1.0, 0.01)  # very weak
        s1 = DummyStrategy("ts_momentum", [sig])
        orch = StrategyOrchestrator([(s1, 1.0)])

        data = _make_data()
        blended = orch.generate_weighted_signals(data)
        # Strength 0.01 * weight should be below 0.02 threshold
        assert len(blended) == 0

    def test_record_strategy_returns(self):
        s1 = DummyStrategy("ts_momentum")
        orch = StrategyOrchestrator([(s1, 1.0)])
        orch.record_strategy_returns("ts_momentum", 0.01)
        orch.record_strategy_returns("ts_momentum", -0.005)
        assert len(orch.performance["ts_momentum"].returns) == 2

    def test_record_unknown_strategy_is_noop(self):
        s1 = DummyStrategy("ts_momentum")
        orch = StrategyOrchestrator([(s1, 1.0)])
        orch.record_strategy_returns("nonexistent", 0.01)  # should not raise

    def test_allocation_summary(self):
        s1 = DummyStrategy("ts_momentum")
        s2 = DummyStrategy("kalman_spread")
        orch = StrategyOrchestrator([(s1, 0.6), (s2, 0.4)])
        summary = orch.get_allocation_summary(regime=1)
        assert isinstance(summary, pd.DataFrame)
        assert "allocation" in summary.columns
        assert "rolling_sharpe" in summary.columns
        assert "type" in summary.columns
        assert abs(summary["allocation"].sum() - 1.0) < 1e-9

    def test_multi_asset_signals(self):
        """Signals for multiple assets should all come through."""
        sig_btc = _make_signal("BTC-USD", 1.0, 0.7)
        sig_eth = _make_signal("ETH-USD", -1.0, 0.5)
        s1 = DummyStrategy("ts_momentum", [sig_btc, sig_eth])
        orch = StrategyOrchestrator([(s1, 1.0)])

        data = _make_data()
        blended = orch.generate_weighted_signals(data)
        assets = {s.asset for s in blended}
        assert "BTC-USD" in assets
        assert "ETH-USD" in assets

    def test_minimum_allocation_floor(self):
        """No strategy should get allocated exactly 0."""
        s1 = DummyStrategy("ts_momentum")
        s2 = DummyStrategy("kalman_spread")
        orch = StrategyOrchestrator([(s1, 0.01), (s2, 0.99)])
        allocs = orch._compute_allocations(regime=2)
        # Even with regime penalty, shouldn't go to zero
        assert allocs["ts_momentum"] > 0
        assert allocs["kalman_spread"] > 0
