"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from exon.backtest.engine import BacktestEngine, BacktestConfig
from exon.backtest.metrics import (
    drawdown_analysis,
    rolling_sharpe,
    stability_of_returns,
    monte_carlo_sharpe_test,
    trade_statistics,
    compare_strategies,
)
from exon.strategies.momentum import TimeSeriesMomentum, CrossSectionalMomentum


def make_prices(n=800, n_assets=4, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data = {}
    for i in range(n_assets):
        drift = 0.00005 * (i + 1)
        noise = rng.normal(drift, 0.015, n)
        data[f"ASSET{i}-USD"] = 100 * np.exp(np.cumsum(noise))
    return pd.DataFrame(data, index=dates)


class TestBacktestEngine:
    def test_basic_run(self):
        prices = make_prices()
        strategy = TimeSeriesMomentum(lookbacks=[24, 72])
        engine = BacktestEngine(BacktestConfig(initial_capital=100_000))
        result = engine.run(strategy, prices)

        assert len(result.equity_curve) == len(prices)
        assert result.equity_curve.iloc[0] == 100_000
        assert result.n_trades >= 0
        assert result.strategy_name == "ts_momentum"

    def test_summary_keys(self):
        prices = make_prices()
        strategy = TimeSeriesMomentum(lookbacks=[24])
        engine = BacktestEngine()
        result = engine.run(strategy, prices)
        summary = result.summary()

        expected_keys = [
            "strategy", "total_return", "annualised_return",
            "sharpe_ratio", "max_drawdown", "n_trades",
        ]
        for key in expected_keys:
            assert key in summary

    def test_transaction_costs_reduce_returns(self):
        prices = make_prices()
        strategy = TimeSeriesMomentum(lookbacks=[24, 72])

        no_cost = BacktestEngine(BacktestConfig(maker_fee=0, slippage_bps=0))
        with_cost = BacktestEngine(BacktestConfig(maker_fee=0.006, slippage_bps=10))

        r1 = no_cost.run(strategy, prices)
        r2 = with_cost.run(strategy, prices)

        # With costs should have lower or equal returns
        assert r2.equity_curve.iloc[-1] <= r1.equity_curve.iloc[-1]

    def test_walk_forward(self):
        prices = make_prices(n=1000)
        strategy = TimeSeriesMomentum(lookbacks=[24])
        engine = BacktestEngine()
        results = engine.walk_forward(strategy, prices, train_size=400, test_size=100, step=100)
        assert len(results) > 0


class TestMetrics:
    def _get_result(self):
        prices = make_prices()
        strategy = TimeSeriesMomentum(lookbacks=[24, 72])
        engine = BacktestEngine()
        return engine.run(strategy, prices)

    def test_drawdown_analysis(self):
        result = self._get_result()
        dd = drawdown_analysis(result.equity_curve)
        if not dd.empty:
            assert "depth" in dd.columns
            assert (dd["depth"] <= 0).all()

    def test_rolling_sharpe(self):
        # Use direct synthetic returns for this metric test
        returns = pd.Series(
            np.random.default_rng(42).normal(0.0001, 0.01, 500),
            index=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        rs = rolling_sharpe(returns, window=100)
        assert len(rs) > 0

    def test_stability(self):
        result = self._get_result()
        # Filter out zero-only returns from warm-up
        active_returns = result.returns[result.returns != 0]
        if len(active_returns) > 10:
            s = stability_of_returns(active_returns)
            assert 0 <= s <= 1
        else:
            # Not enough active returns to test
            pass

    def test_monte_carlo(self):
        result = self._get_result()
        mc = monte_carlo_sharpe_test(result.returns, n_simulations=500)
        assert "actual_sharpe" in mc
        assert "p_value" in mc
        assert 0 <= mc["p_value"] <= 1

    def test_trade_statistics(self):
        result = self._get_result()
        stats = trade_statistics(result.trades)
        assert "n_trades" in stats

    def test_compare(self):
        prices = make_prices()
        engine = BacktestEngine()
        r1 = engine.run(TimeSeriesMomentum(lookbacks=[24]), prices)
        r2 = engine.run(CrossSectionalMomentum(top_n=2, bottom_n=2), prices)
        comp = compare_strategies([r1, r2])
        assert len(comp) == 2
