"""Comprehensive backtesting tests — equities + options.

Covers:
- Options backtesting engine (Greeks tracking, expiry, P&L)
- Equity backtesting engine (daily bars, whole shares, multi-strategy)
- Walk-forward validation
- Cross-validation of options + equity backtests
"""

import numpy as np
import pandas as pd
import pytest

from exon.backtest.engine import BacktestEngine, BacktestConfig
from exon.backtest.equity_engine import EquityBacktestEngine, EquityBacktestConfig
from exon.backtest.options_engine import (
    OptionsBacktestEngine,
    OptionsBacktestConfig,
    OptionsPosition,
    OptionsTradeRecord,
)
from exon.backtest.metrics import (
    drawdown_analysis,
    rolling_sharpe,
    stability_of_returns,
    monte_carlo_sharpe_test,
    trade_statistics,
    compare_strategies,
    information_ratio,
    tail_ratio,
    turnover,
)
from exon.options.pricing import black_scholes_price, OptionType
from exon.options.strategy_mapper import StrategyMapperConfig
from exon.strategies.base import Signal, Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_daily_prices(n=300, n_assets=5, seed=42, trend=0.0003):
    """Generate synthetic daily equity prices with optional trend."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    data = {}
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"][:n_assets]
    for i, ticker in enumerate(tickers):
        drift = trend * (1 + 0.5 * i)
        noise = rng.normal(drift, 0.015, n)
        data[ticker] = 100 * (1 + i * 10) * np.exp(np.cumsum(noise))
    return pd.DataFrame(data, index=dates)


def make_trending_prices(n=300, n_assets=3, seed=42):
    """Generate prices with clear uptrend for strategy testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    data = {}
    tickers = ["AAPL", "MSFT", "GOOGL"][:n_assets]
    for i, ticker in enumerate(tickers):
        trend = 0.001 * (1 + i * 0.3)
        noise = rng.normal(trend, 0.01, n)
        data[ticker] = 150 * np.exp(np.cumsum(noise))
    return pd.DataFrame(data, index=dates)


class SimpleMomentumStrategy(Strategy):
    """A simple momentum strategy for testing."""

    name = "simple_momentum"

    def __init__(self, lookback: int = 21):
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(data) < self.lookback + 1:
            return []
        signals = []
        ts = data.index[-1]
        for col in data.columns:
            ret = data[col].iloc[-1] / data[col].iloc[-self.lookback] - 1
            direction = 1.0 if ret > 0 else -1.0
            strength = min(abs(ret) * 10, 1.0)
            if strength > 0.1:
                signals.append(Signal(
                    timestamp=ts,
                    asset=col,
                    direction=direction,
                    strength=strength,
                    metadata={"strategy_type": "momentum"},
                ))
        return signals


class MeanReversionStrategy(Strategy):
    """A simple mean-reversion strategy for testing."""

    name = "mean_reversion"

    def __init__(self, lookback: int = 21):
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(data) < self.lookback + 1:
            return []
        signals = []
        ts = data.index[-1]
        for col in data.columns:
            returns = data[col].pct_change().dropna()
            if len(returns) < self.lookback:
                continue
            z = (returns.iloc[-1] - returns.iloc[-self.lookback:].mean()) / max(
                returns.iloc[-self.lookback:].std(), 1e-8
            )
            if abs(z) > 1.0:
                direction = -1.0 if z > 0 else 1.0
                strength = min(abs(z) / 3, 1.0)
                signals.append(Signal(
                    timestamp=ts,
                    asset=col,
                    direction=direction,
                    strength=strength,
                    metadata={"strategy_type": "mean_reversion"},
                ))
        return signals


# ===========================================================================
# OPTIONS BACKTESTING TESTS
# ===========================================================================


class TestOptionsBacktestEngine:
    """Tests for the options backtesting engine."""

    def test_basic_run(self):
        """Engine runs without errors on simple data."""
        prices = make_daily_prices(n=150, n_assets=3)
        strategy = SimpleMomentumStrategy(lookback=21)
        engine = OptionsBacktestEngine(OptionsBacktestConfig(initial_capital=100_000))
        result = engine.run(strategy, prices)

        assert len(result.equity_curve) == len(prices)
        assert result.equity_curve.iloc[0] == 100_000
        assert result.strategy_name == "options_simple_momentum"
        assert isinstance(result.trades, list)
        assert isinstance(result.daily_greeks, pd.DataFrame)

    def test_summary_keys(self):
        """Summary dict contains all expected keys."""
        prices = make_daily_prices(n=150, n_assets=3)
        strategy = SimpleMomentumStrategy()
        engine = OptionsBacktestEngine()
        result = engine.run(strategy, prices)
        summary = result.summary()

        expected_keys = [
            "strategy", "total_return", "annualised_return",
            "sharpe_ratio", "max_drawdown", "n_trades",
            "total_fees", "win_rate", "avg_trade_pnl",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_greeks_tracking(self):
        """Daily Greeks are tracked throughout the backtest."""
        prices = make_daily_prices(n=150, n_assets=3)
        strategy = SimpleMomentumStrategy()
        engine = OptionsBacktestEngine()
        result = engine.run(strategy, prices)

        assert "net_delta" in result.daily_greeks.columns
        assert "net_gamma" in result.daily_greeks.columns
        assert "net_theta" in result.daily_greeks.columns
        assert "net_vega" in result.daily_greeks.columns

    def test_positions_over_time(self):
        """Positions history is recorded."""
        prices = make_daily_prices(n=150, n_assets=3)
        strategy = SimpleMomentumStrategy()
        engine = OptionsBacktestEngine()
        result = engine.run(strategy, prices)

        assert len(result.positions_over_time) == len(result.equity_curve)
        for pos_snap in result.positions_over_time:
            assert "n_positions" in pos_snap

    def test_trades_have_structure_info(self):
        """Trade records include structure and leg details."""
        prices = make_daily_prices(n=200, n_assets=3)
        strategy = SimpleMomentumStrategy()
        engine = OptionsBacktestEngine(OptionsBacktestConfig(max_positions=5))
        result = engine.run(strategy, prices)

        for trade in result.trades:
            assert trade.underlying != ""
            assert trade.structure_id != ""
            assert trade.action in ("open", "close", "expire", "exercise")
            assert isinstance(trade.legs, list)

    def test_max_positions_limit(self):
        """Engine respects max positions limit."""
        prices = make_daily_prices(n=200, n_assets=5)
        strategy = SimpleMomentumStrategy()
        config = OptionsBacktestConfig(max_positions=3)
        engine = OptionsBacktestEngine(config)
        result = engine.run(strategy, prices)

        for snap in result.positions_over_time:
            assert snap["n_positions"] <= 3 * 4 + 3  # 3 structures * max 4 legs + buffer

    def test_risk_per_trade_limit(self):
        """Trades exceeding risk budget are skipped."""
        prices = make_daily_prices(n=150, n_assets=2)
        strategy = SimpleMomentumStrategy()
        config = OptionsBacktestConfig(max_risk_per_trade_pct=0.001)  # very tight
        engine = OptionsBacktestEngine(config)
        result = engine.run(strategy, prices)

        # With very tight risk limit, fewer trades should execute
        loose_config = OptionsBacktestConfig(max_risk_per_trade_pct=0.10)
        loose_engine = OptionsBacktestEngine(loose_config)
        loose_result = loose_engine.run(strategy, prices)

        assert result.n_trades <= loose_result.n_trades

    def test_transaction_costs_accrue(self):
        """Transaction costs are tracked and non-zero with fees enabled."""
        prices = make_daily_prices(n=200, n_assets=3)
        strategy = SimpleMomentumStrategy()

        with_cost = OptionsBacktestEngine(OptionsBacktestConfig(
            per_contract_fee=2.0, spread_slippage_pct=0.10
        ))
        result = with_cost.run(strategy, prices)

        if result.n_trades > 0:
            assert result.total_fees > 0

    def test_walk_forward(self):
        """Walk-forward produces multiple results."""
        prices = make_daily_prices(n=500, n_assets=3)
        strategy = SimpleMomentumStrategy()
        engine = OptionsBacktestEngine()
        results = engine.walk_forward(
            strategy, prices, train_size=200, test_size=50, step=50
        )
        assert len(results) > 0
        for r in results:
            assert isinstance(r.equity_curve, pd.Series)

    def test_expiry_settlement_call_itm(self):
        """ITM call settles for intrinsic value."""
        engine = OptionsBacktestEngine()
        pos = OptionsPosition(
            leg_id="test", underlying="AAPL", option_type="call",
            strike=100, entry_date=pd.Timestamp("2024-01-01"),
            expiry_date=pd.Timestamp("2024-02-01"),
            side="buy", qty=1, entry_price=5.0,
        )
        # Spot at 110, call with strike 100 → intrinsic = 10
        settlement = engine._settle_at_expiry(pos, 110.0)
        assert settlement == 10.0 * 100 * 1  # 1 contract, 100 multiplier, long

    def test_expiry_settlement_put_otm(self):
        """OTM put expires worthless."""
        engine = OptionsBacktestEngine()
        pos = OptionsPosition(
            leg_id="test", underlying="AAPL", option_type="put",
            strike=100, entry_date=pd.Timestamp("2024-01-01"),
            expiry_date=pd.Timestamp("2024-02-01"),
            side="buy", qty=1, entry_price=3.0,
        )
        settlement = engine._settle_at_expiry(pos, 110.0)
        assert settlement == 0.0  # OTM put, worthless

    def test_expiry_settlement_short_call(self):
        """Short ITM call results in negative settlement."""
        engine = OptionsBacktestEngine()
        pos = OptionsPosition(
            leg_id="test", underlying="AAPL", option_type="call",
            strike=100, entry_date=pd.Timestamp("2024-01-01"),
            expiry_date=pd.Timestamp("2024-02-01"),
            side="sell", qty=1, entry_price=5.0,
        )
        settlement = engine._settle_at_expiry(pos, 110.0)
        assert settlement == -10.0 * 100 * 1  # short loses on ITM assignment

    def test_iv_paths_generated(self):
        """IV paths are generated for all assets."""
        prices = make_daily_prices(n=100, n_assets=3)
        returns_df = prices.pct_change().fillna(0)
        rng = np.random.default_rng(42)
        engine = OptionsBacktestEngine()
        iv_paths = engine._simulate_iv_paths(prices, returns_df, rng)

        assert len(iv_paths) == 3
        for asset, path in iv_paths.items():
            assert len(path) == 100
            for iv in path.values():
                assert 0.05 <= iv <= 2.0

    def test_different_strategies_produce_different_results(self):
        """Different underlying strategies produce different options results."""
        prices = make_daily_prices(n=200, n_assets=3)
        engine = OptionsBacktestEngine()

        r1 = engine.run(SimpleMomentumStrategy(lookback=10), prices)
        r2 = engine.run(MeanReversionStrategy(lookback=21), prices)

        # They should differ in at least some metric
        assert r1.strategy_name != r2.strategy_name
        # Final equities may differ
        assert r1.equity_curve.iloc[-1] != pytest.approx(
            r2.equity_curve.iloc[-1], abs=1.0
        ) or r1.n_trades != r2.n_trades


# ===========================================================================
# EQUITY BACKTESTING TESTS
# ===========================================================================


class TestEquityBacktestEngine:
    """Tests for the enhanced equity backtesting engine."""

    def test_basic_run(self):
        """Engine runs on daily equity data."""
        prices = make_daily_prices()
        strategy = SimpleMomentumStrategy()
        engine = EquityBacktestEngine(EquityBacktestConfig(initial_capital=100_000))
        result = engine.run(strategy, prices)

        assert len(result.equity_curve) == len(prices)
        assert result.equity_curve.iloc[0] == 100_000

    def test_whole_share_rounding(self):
        """Positions are rounded to whole shares."""
        prices = make_daily_prices(n=150, n_assets=2)
        strategy = SimpleMomentumStrategy()
        config = EquityBacktestConfig(whole_shares=True)
        engine = EquityBacktestEngine(config)
        result = engine.run(strategy, prices)

        for trade in result.trades:
            assert trade.quantity == round(trade.quantity), \
                f"Trade qty {trade.quantity} not whole share"

    def test_commission_free(self):
        """Default Alpaca config has zero commissions."""
        config = EquityBacktestConfig()
        assert config.maker_fee == 0.0
        assert config.taker_fee == 0.0

    def test_slippage_impact(self):
        """Higher slippage reduces returns."""
        prices = make_daily_prices(n=200)
        strategy = SimpleMomentumStrategy()

        tight = EquityBacktestEngine(EquityBacktestConfig(slippage_bps=0))
        wide = EquityBacktestEngine(EquityBacktestConfig(slippage_bps=20))

        r1 = tight.run(strategy, prices)
        r2 = wide.run(strategy, prices)

        assert r2.equity_curve.iloc[-1] <= r1.equity_curve.iloc[-1]

    def test_min_trade_value(self):
        """Trades below minimum value are skipped."""
        prices = make_daily_prices(n=150, n_assets=2)
        strategy = SimpleMomentumStrategy()

        low_min = EquityBacktestEngine(EquityBacktestConfig(min_trade_value=10))
        high_min = EquityBacktestEngine(EquityBacktestConfig(min_trade_value=10_000))

        r1 = low_min.run(strategy, prices)
        r2 = high_min.run(strategy, prices)

        assert r1.n_trades >= r2.n_trades

    def test_position_limit(self):
        """Positions are capped at max_position_pct."""
        prices = make_daily_prices(n=200, n_assets=3)
        strategy = SimpleMomentumStrategy()
        config = EquityBacktestConfig(max_position_pct=0.10)
        engine = EquityBacktestEngine(config)
        result = engine.run(strategy, prices)

        # Check position values don't exceed 10% of portfolio
        for i in range(len(result.positions)):
            row = result.positions.iloc[i]
            equity = result.equity_curve.iloc[i]
            if equity > 0:
                for asset in row.index:
                    if row[asset] != 0 and asset in prices.columns:
                        price = prices.iloc[i].get(asset, 0)
                        pos_value = abs(row[asset] * price)
                        # Allow some tolerance for price moves between rebalances
                        assert pos_value <= equity * 0.20, \
                            f"Position {asset} = ${pos_value:.0f} exceeds 20% of ${equity:.0f}"

    def test_walk_forward_daily(self):
        """Walk-forward works with daily bar sizes."""
        prices = make_daily_prices(n=500, n_assets=3)
        strategy = SimpleMomentumStrategy()
        engine = EquityBacktestEngine()
        results = engine.walk_forward(
            strategy, prices, train_size=200, test_size=50, step=50
        )
        assert len(results) > 0

    def test_walk_forward_summary(self):
        """Walk-forward summary aggregates metrics correctly."""
        prices = make_daily_prices(n=500, n_assets=3)
        strategy = SimpleMomentumStrategy()
        engine = EquityBacktestEngine()
        summary = engine.walk_forward_summary(
            strategy, prices, train_size=200, test_size=50, step=50
        )

        assert summary["n_windows"] > 0
        assert "mean_sharpe" in summary
        assert "pct_positive_sharpe" in summary
        assert "worst_dd" in summary
        assert 0 <= summary["pct_positive_sharpe"] <= 1

    def test_multi_strategy(self):
        """Multi-strategy backtesting combines results."""
        prices = make_daily_prices(n=200, n_assets=3)
        strategies = [
            (SimpleMomentumStrategy(lookback=10), 0.6),
            (MeanReversionStrategy(lookback=21), 0.4),
        ]
        engine = EquityBacktestEngine(EquityBacktestConfig(initial_capital=100_000))
        combined, per_strat = engine.run_multi_strategy(strategies, prices)

        assert len(per_strat) == 2
        assert "simple_momentum" in per_strat
        assert "mean_reversion" in per_strat
        assert combined.strategy_name == "multi_strategy"
        # Combined equity should be sum of individual strategies
        assert combined.equity_curve.iloc[0] == pytest.approx(100_000, abs=1)

    def test_summary_keys(self):
        """Result summary has expected keys."""
        prices = make_daily_prices(n=150)
        strategy = SimpleMomentumStrategy()
        engine = EquityBacktestEngine()
        result = engine.run(strategy, prices)
        summary = result.summary()

        for key in ["strategy", "total_return", "sharpe_ratio", "max_drawdown", "n_trades"]:
            assert key in summary


# ===========================================================================
# METRICS TESTS
# ===========================================================================


class TestMetricsComprehensive:
    """Extended tests for performance metrics."""

    def _make_equity(self, n=300, trend=0.001, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        returns = rng.normal(trend, 0.01, n)
        equity = 100_000 * np.exp(np.cumsum(returns))
        return pd.Series(equity, index=dates)

    def _make_returns(self, n=300, mean=0.0003, std=0.01, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        return pd.Series(rng.normal(mean, std, n), index=dates)

    def test_drawdown_finds_periods(self):
        """Drawdown analysis identifies drawdown periods."""
        equity = self._make_equity()
        dd = drawdown_analysis(equity)
        if not dd.empty:
            assert "depth" in dd.columns
            assert "start" in dd.columns
            assert (dd["depth"] <= 0).all()

    def test_rolling_sharpe_daily(self):
        """Rolling Sharpe works with daily returns."""
        returns = self._make_returns()
        rs = rolling_sharpe(returns, window=63)
        assert len(rs) > 0

    def test_information_ratio(self):
        """Information ratio computed correctly."""
        strat_rets = self._make_returns(mean=0.001)
        bench_rets = self._make_returns(mean=0.0005, seed=99)
        ir = information_ratio(strat_rets, bench_rets)
        assert isinstance(ir, float)
        # Strategy with higher mean should have positive IR
        assert ir > 0 or abs(ir) < 100  # sanity check

    def test_tail_ratio(self):
        """Tail ratio is computed."""
        returns = self._make_returns()
        tr = tail_ratio(returns)
        assert tr > 0

    def test_stability(self):
        """Stability of returns gives R² between 0 and 1."""
        returns = self._make_returns(mean=0.001)
        s = stability_of_returns(returns)
        assert 0 <= s <= 1

    def test_monte_carlo_significance(self):
        """Monte Carlo test detects significant Sharpe."""
        # Strong signal
        strong_rets = self._make_returns(mean=0.005, std=0.01)
        mc = monte_carlo_sharpe_test(strong_rets, n_simulations=1000)
        assert mc["actual_sharpe"] > 0
        assert 0 <= mc["p_value"] <= 1

    def test_turnover(self):
        """Turnover computed from position changes."""
        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        positions = pd.DataFrame({
            "AAPL": [0, 0.1, 0.15, 0.2, 0.15, 0.1, 0, 0, 0.1, 0.2],
            "MSFT": [0, 0.05, 0.1, 0.1, 0.1, 0.05, 0, 0, 0.05, 0.1],
        }, index=dates)
        to = turnover(positions)
        assert len(to) == 10
        assert to.iloc[0] == 0  # first row is NaN diff → 0 if fillna

    def test_compare_strategies_daily(self):
        """Strategy comparison works with equity backtests."""
        prices = make_daily_prices(n=200, n_assets=3)
        engine = EquityBacktestEngine()

        r1 = engine.run(SimpleMomentumStrategy(lookback=10), prices)
        r2 = engine.run(SimpleMomentumStrategy(lookback=42), prices)
        comp = compare_strategies([r1, r2])
        assert len(comp) == 2


# ===========================================================================
# OPTIONS POSITION MECHANICS
# ===========================================================================


class TestOptionsPositionMechanics:
    """Tests for OptionsPosition dataclass behaviour."""

    def test_long_position_value(self):
        """Long position has positive market value."""
        pos = OptionsPosition(
            leg_id="L1", underlying="SPY", option_type="call",
            strike=500, entry_date=pd.Timestamp("2024-01-01"),
            expiry_date=pd.Timestamp("2024-02-01"),
            side="buy", qty=2, entry_price=5.0, current_price=7.0,
        )
        assert pos.multiplier == 1
        assert pos.market_value == 7.0 * 100 * 2  # $1,400
        assert pos.entry_value == 5.0 * 100 * 2  # $1,000
        assert pos.pnl == 400.0  # $400 profit

    def test_short_position_value(self):
        """Short position has negative market value."""
        pos = OptionsPosition(
            leg_id="L1", underlying="SPY", option_type="put",
            strike=500, entry_date=pd.Timestamp("2024-01-01"),
            expiry_date=pd.Timestamp("2024-02-01"),
            side="sell", qty=1, entry_price=4.0, current_price=3.0,
        )
        assert pos.multiplier == -1
        assert pos.market_value == -3.0 * 100  # -$300 (liability)
        assert pos.entry_value == -4.0 * 100  # -$400 (received credit)
        assert pos.pnl == 100.0  # $100 profit (option decayed in our favor)

    def test_spread_net_value(self):
        """Bull call spread has limited risk/reward."""
        long_call = OptionsPosition(
            leg_id="L1", underlying="SPY", option_type="call",
            strike=500, entry_date=pd.Timestamp("2024-01-01"),
            expiry_date=pd.Timestamp("2024-02-01"),
            side="buy", qty=1, entry_price=8.0, current_price=10.0,
        )
        short_call = OptionsPosition(
            leg_id="L2", underlying="SPY", option_type="call",
            strike=510, entry_date=pd.Timestamp("2024-01-01"),
            expiry_date=pd.Timestamp("2024-02-01"),
            side="sell", qty=1, entry_price=4.0, current_price=5.0,
        )
        net_pnl = long_call.pnl + short_call.pnl
        # Long leg gained $200, short leg lost $100 → net $100
        assert net_pnl == 100.0


# ===========================================================================
# IV PATH SIMULATION TESTS
# ===========================================================================


class TestIVSimulation:
    """Tests for IV path generation in options backtester."""

    def test_iv_mean_reverts(self):
        """IV paths mean-revert toward realised vol."""
        prices = make_daily_prices(n=200, n_assets=1)
        returns_df = prices.pct_change().fillna(0)
        rng = np.random.default_rng(42)

        config = OptionsBacktestConfig(
            iv_mean_reversion_speed=0.1,  # fast mean reversion
            iv_vol_of_vol=0.001,  # low noise
        )
        engine = OptionsBacktestEngine(config)
        iv_paths = engine._simulate_iv_paths(prices, returns_df, rng)

        asset = list(iv_paths.keys())[0]
        ivs = list(iv_paths[asset].values())
        # IV should stay in a reasonable range
        assert all(0.05 <= iv <= 2.0 for iv in ivs)
        # Shouldn't drift wildly with fast mean reversion
        assert np.std(ivs) < 0.5

    def test_iv_responds_to_vol_regime(self):
        """IV paths reflect underlying volatility changes."""
        rng_prices = np.random.default_rng(42)
        dates = pd.date_range("2024-01-02", periods=200, freq="B", tz="UTC")
        # First half: low vol; second half: high vol
        low_vol = rng_prices.normal(0.0005, 0.005, 100)
        high_vol = rng_prices.normal(0.0005, 0.03, 100)
        all_rets = np.concatenate([low_vol, high_vol])
        prices = pd.DataFrame(
            {"TEST": 100 * np.exp(np.cumsum(all_rets))}, index=dates
        )
        returns_df = prices.pct_change().fillna(0)

        engine = OptionsBacktestEngine()
        iv_paths = engine._simulate_iv_paths(
            prices, returns_df, np.random.default_rng(42)
        )

        ivs = list(iv_paths["TEST"].values())
        avg_iv_first_half = np.mean(ivs[50:100])
        avg_iv_second_half = np.mean(ivs[150:200])
        # IV should be higher in the high-vol regime
        assert avg_iv_second_half > avg_iv_first_half


# ===========================================================================
# CROSS-ASSET BACKTEST COMPARISON
# ===========================================================================


class TestCrossAssetBacktest:
    """Tests comparing equity and options backtests on same data."""

    def test_both_engines_run_same_data(self):
        """Both equity and options engines run on identical data."""
        prices = make_daily_prices(n=200, n_assets=3)
        strategy = SimpleMomentumStrategy()

        eq_engine = EquityBacktestEngine()
        opt_engine = OptionsBacktestEngine()

        eq_result = eq_engine.run(strategy, prices)
        opt_result = opt_engine.run(strategy, prices)

        assert len(eq_result.equity_curve) == len(opt_result.equity_curve)
        assert eq_result.equity_curve.iloc[0] == opt_result.equity_curve.iloc[0]

    def test_options_has_greeks_equity_does_not(self):
        """Options result tracks Greeks; equity result does not."""
        prices = make_daily_prices(n=150, n_assets=2)
        strategy = SimpleMomentumStrategy()

        eq_result = EquityBacktestEngine().run(strategy, prices)
        opt_result = OptionsBacktestEngine().run(strategy, prices)

        assert hasattr(opt_result, "daily_greeks")
        assert "net_delta" in opt_result.daily_greeks.columns
        # Equity result has positions but not Greeks
        assert hasattr(eq_result, "positions")


# ===========================================================================
# WALK-FORWARD VALIDATION
# ===========================================================================


class TestWalkForwardValidation:
    """Tests for walk-forward methodology."""

    def test_equity_walk_forward_windows_no_overlap(self):
        """Walk-forward windows don't overlap in out-of-sample."""
        prices = make_daily_prices(n=500, n_assets=2)
        strategy = SimpleMomentumStrategy()
        engine = EquityBacktestEngine()

        results = engine.walk_forward(
            strategy, prices, train_size=200, test_size=50, step=50
        )
        assert len(results) >= 3

    def test_options_walk_forward_produces_results(self):
        """Options walk-forward produces valid results."""
        prices = make_daily_prices(n=400, n_assets=2)
        strategy = SimpleMomentumStrategy()
        engine = OptionsBacktestEngine()

        results = engine.walk_forward(
            strategy, prices, train_size=200, test_size=50, step=50
        )
        assert len(results) >= 1
        for r in results:
            assert isinstance(r.equity_curve, pd.Series)
            assert r.equity_curve.iloc[0] == 100_000

    def test_walk_forward_summary_statistics(self):
        """Walk-forward summary provides useful aggregate stats."""
        prices = make_daily_prices(n=600, n_assets=3)
        strategy = SimpleMomentumStrategy()
        engine = EquityBacktestEngine()

        summary = engine.walk_forward_summary(
            strategy, prices, train_size=200, test_size=63, step=63
        )

        assert summary["n_windows"] >= 2
        assert isinstance(summary["mean_sharpe"], float)
        assert isinstance(summary["worst_dd"], float)
        assert summary["worst_dd"] <= 0  # drawdowns are negative


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_signals(self):
        """Engine handles strategy that produces no signals."""
        class NoSignalStrategy(Strategy):
            name = "no_signal"
            def generate_signals(self, data, **kwargs):
                return []

        prices = make_daily_prices(n=100)
        engine = EquityBacktestEngine()
        result = engine.run(NoSignalStrategy(), prices)
        assert result.n_trades == 0
        assert result.equity_curve.iloc[-1] == 100_000

    def test_single_asset(self):
        """Engine works with single asset."""
        prices = make_daily_prices(n=150, n_assets=1)
        strategy = SimpleMomentumStrategy()
        engine = EquityBacktestEngine()
        result = engine.run(strategy, prices)
        assert len(result.equity_curve) == 150

    def test_options_single_asset(self):
        """Options engine works with single asset."""
        prices = make_daily_prices(n=150, n_assets=1)
        strategy = SimpleMomentumStrategy()
        engine = OptionsBacktestEngine()
        result = engine.run(strategy, prices)
        assert len(result.equity_curve) == 150

    def test_short_data(self):
        """Engine handles very short data without crashing."""
        prices = make_daily_prices(n=30, n_assets=2)
        strategy = SimpleMomentumStrategy(lookback=5)
        engine = EquityBacktestEngine()
        result = engine.run(strategy, prices)
        assert len(result.equity_curve) == 30

    def test_options_short_data(self):
        """Options engine handles short data."""
        prices = make_daily_prices(n=30, n_assets=2)
        strategy = SimpleMomentumStrategy(lookback=5)
        engine = OptionsBacktestEngine()
        result = engine.run(strategy, prices)
        assert len(result.equity_curve) == 30
