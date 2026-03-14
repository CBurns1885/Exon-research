"""Tests for equities-specific modules: Alpaca client, equity data pipeline,
Alpaca executor, sector rotation, earnings momentum, and market microstructure.
"""

import numpy as np
import pandas as pd
import pytest

from exon.strategies.base import Signal
from exon.strategies.sector_rotation import SectorRotation
from exon.strategies.earnings_momentum import EarningsMomentum
from exon.strategies.market_microstructure import MarketMicrostructure


# ---------- Test data generators ----------

def make_sector_data(n=300, n_sectors=6, seed=42):
    """Simulated sector ETF data with differentiated momentum profiles."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)

    data = {}
    # SPY benchmark
    spy_returns = rng.normal(0.0004, 0.01, n)
    data["SPY"] = 100 * np.exp(np.cumsum(spy_returns))

    sector_names = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY"][:n_sectors]
    for i, name in enumerate(sector_names):
        # Each sector has different drift (some outperform, some underperform)
        drift = 0.0004 + 0.0003 * (i - n_sectors // 2)
        vol = 0.012 + 0.002 * i
        # Correlated with SPY + idiosyncratic
        beta = 0.7 + 0.1 * i
        returns = drift + beta * spy_returns + rng.normal(0, vol, n)
        data[name] = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame(data, index=dates)


def make_earnings_data(n=200, n_stocks=5, seed=42):
    """Simulated stock data with injected earnings-like gap events."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)

    data = {}
    # Place event near the end so it's within drift_window of the end
    event_bar = n - 20
    for i in range(n_stocks):
        returns = rng.normal(0.0003, 0.015, n)
        if event_bar < n:
            returns[event_bar] = 0.10 * (1 if i % 2 == 0 else -1)  # 10% gap
        data[f"STOCK{i}"] = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame(data, index=dates)


def make_volume_data(n=200, n_stocks=4, seed=42):
    """Simulated price + volume data for microstructure analysis."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)

    prices = {}
    volumes = {}
    for i in range(n_stocks):
        drift = 0.001 * (i - 1)
        returns = rng.normal(drift, 0.015, n)

        # Inject volume spike at bar 180 with positive return
        if i == 0:
            returns[175:185] = 0.005  # sustained positive move
        prices[f"VOL{i}"] = 100 * np.exp(np.cumsum(returns))

        base_vol = 1_000_000 * (1 + 0.5 * i)
        vol = base_vol + rng.normal(0, base_vol * 0.2, n)
        if i == 0:
            vol[175:185] *= 3  # volume spike
        volumes[f"VOL{i}"] = np.abs(vol)

    return pd.DataFrame(prices, index=dates), pd.DataFrame(volumes, index=dates)


# ---------- Sector Rotation Tests ----------

class TestSectorRotation:
    def test_generates_long_short(self):
        data = make_sector_data()
        strategy = SectorRotation(
            momentum_window=200, skip_window=21, vol_window=63,
            top_n=2, bottom_n=2, spy_column="SPY",
        )
        signals = strategy.generate_signals(data)
        assert len(signals) > 0
        longs = [s for s in signals if s.direction > 0]
        shorts = [s for s in signals if s.direction < 0]
        assert len(longs) == 2
        assert len(shorts) == 2

    def test_no_spy_in_signals(self):
        """SPY is the benchmark, should not appear as a signal."""
        data = make_sector_data()
        strategy = SectorRotation(
            momentum_window=200, skip_window=21,
            top_n=2, bottom_n=2, spy_column="SPY",
        )
        signals = strategy.generate_signals(data)
        assets = [s.asset for s in signals]
        assert "SPY" not in assets

    def test_metadata_has_rank_and_score(self):
        data = make_sector_data()
        strategy = SectorRotation(
            momentum_window=200, skip_window=21,
            top_n=2, bottom_n=1, spy_column="SPY",
        )
        signals = strategy.generate_signals(data)
        for s in signals:
            assert "composite_score" in s.metadata
            assert "rank" in s.metadata
            assert "sector" in s.metadata

    def test_insufficient_data_returns_empty(self):
        data = make_sector_data(n=50)
        strategy = SectorRotation(momentum_window=200, skip_window=21)
        signals = strategy.generate_signals(data)
        assert signals == []

    def test_without_spy(self):
        """Should still work without SPY in the data."""
        data = make_sector_data()
        data = data.drop(columns=["SPY"])
        strategy = SectorRotation(
            momentum_window=200, skip_window=21,
            top_n=2, bottom_n=2, spy_column="SPY",
        )
        signals = strategy.generate_signals(data)
        # Should generate signals using momentum without relative strength
        assert len(signals) > 0

    def test_no_duplicate_assets(self):
        data = make_sector_data()
        strategy = SectorRotation(top_n=2, bottom_n=2, momentum_window=200)
        signals = strategy.generate_signals(data)
        assets = [s.asset for s in signals]
        assert len(assets) == len(set(assets))


# ---------- Earnings Momentum Tests ----------

class TestEarningsMomentum:
    def test_detects_gap_events(self):
        data = make_earnings_data()
        strategy = EarningsMomentum(
            gap_threshold_std=2.0, drift_window=45,
            vol_lookback=63, decay_rate=0.95,
        )
        signals = strategy.generate_signals(data)
        assert len(signals) > 0

    def test_direction_matches_gap(self):
        """Stocks with positive gaps should get positive signals."""
        data = make_earnings_data()
        strategy = EarningsMomentum(
            gap_threshold_std=2.0, drift_window=45,
            vol_lookback=63,
        )
        signals = strategy.generate_signals(data)
        # STOCK0 had positive gap, should be long
        stock0_sigs = [s for s in signals if s.asset == "STOCK0"]
        if stock0_sigs:
            assert stock0_sigs[0].direction == 1.0

    def test_metadata_has_event_info(self):
        data = make_earnings_data()
        strategy = EarningsMomentum(gap_threshold_std=2.0, drift_window=45)
        signals = strategy.generate_signals(data)
        for s in signals:
            assert "event_z" in s.metadata
            assert "bars_since_event" in s.metadata
            assert "baseline_vol" in s.metadata

    def test_max_concurrent_limit(self):
        data = make_earnings_data(n_stocks=20)
        strategy = EarningsMomentum(
            gap_threshold_std=1.5, drift_window=45, max_concurrent=3,
        )
        signals = strategy.generate_signals(data)
        assert len(signals) <= 3

    def test_insufficient_data_returns_empty(self):
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2023-01-01", periods=30)
        data = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 0.5, 30))}, index=dates)
        strategy = EarningsMomentum(vol_lookback=63, drift_window=40)
        signals = strategy.generate_signals(data)
        assert signals == []

    def test_signal_decays_over_time(self):
        """Signals should be weaker as time since event increases."""
        data = make_earnings_data(n=250)
        strategy = EarningsMomentum(
            gap_threshold_std=2.0, drift_window=80,
            decay_rate=0.95,
        )
        signals = strategy.generate_signals(data)
        for s in signals:
            bars_since = s.metadata["bars_since_event"]
            # Strength should incorporate decay
            assert s.strength <= 1.0
            if bars_since > 20:
                # After 20 bars with 0.95 decay: 0.95^20 ≈ 0.36
                assert s.strength < 0.5


# ---------- Market Microstructure Tests ----------

class TestMarketMicrostructure:
    def test_generates_signals(self):
        prices, volumes = make_volume_data()
        strategy = MarketMicrostructure(
            obv_window=21, volume_window=63,
            rvol_threshold=1.5,
        )
        signals = strategy.generate_signals(prices, volume=volumes)
        assert isinstance(signals, list)

    def test_metadata_fields(self):
        prices, volumes = make_volume_data()
        strategy = MarketMicrostructure(obv_window=21, volume_window=63)
        signals = strategy.generate_signals(prices, volume=volumes)
        for s in signals:
            assert "obv_trend" in s.metadata
            assert "ad_signal" in s.metadata
            assert "composite" in s.metadata

    def test_works_without_volume(self):
        """Should still work using price-based proxy when volume unavailable."""
        prices, _ = make_volume_data()
        strategy = MarketMicrostructure(obv_window=21, volume_window=63)
        signals = strategy.generate_signals(prices)  # no volume kwarg
        assert isinstance(signals, list)

    def test_insufficient_data_returns_empty(self):
        prices, volumes = make_volume_data(n=20)
        strategy = MarketMicrostructure(obv_window=21, volume_window=63)
        signals = strategy.generate_signals(prices, volume=volumes)
        assert signals == []

    def test_strengths_bounded(self):
        prices, volumes = make_volume_data()
        strategy = MarketMicrostructure(obv_window=21, volume_window=63)
        signals = strategy.generate_signals(prices, volume=volumes)
        for s in signals:
            assert 0 < s.strength <= 1.0
            assert s.direction in (-1.0, 1.0)


# ---------- Alpaca Client Tests (unit, no API calls) ----------

class TestAlpacaConfig:
    def test_default_config(self):
        from exon.data.alpaca_client import AlpacaConfig
        config = AlpacaConfig()
        assert config.paper is True
        assert config.data_feed == "iex"
        assert config.api_key == ""
        assert config.api_secret == ""

    def test_custom_config(self):
        from exon.data.alpaca_client import AlpacaConfig
        config = AlpacaConfig(api_key="test", api_secret="secret", paper=False, data_feed="sip")
        assert config.paper is False
        assert config.data_feed == "sip"


class TestAlpacaClientInit:
    def test_lazy_init(self):
        """Client should not attempt API connection on init."""
        from exon.data.alpaca_client import AlpacaClient, AlpacaConfig
        config = AlpacaConfig(api_key="test", api_secret="test")
        client = AlpacaClient(config=config)
        assert client._trading_client is None
        assert client._data_client is None


# ---------- Equity Data Pipeline Tests ----------

class TestEquityDataPipeline:
    def test_to_wide_multiindex(self):
        """Test wide format conversion with multi-index data."""
        from exon.data.equity_data import EquityDataPipeline
        dates = pd.bdate_range("2024-01-01", periods=5)
        idx = pd.MultiIndex.from_product([["AAPL", "MSFT"], dates], names=["symbol", "timestamp"])
        data = pd.DataFrame({
            "close": np.random.randn(10) + 100,
            "volume": np.abs(np.random.randn(10) * 1e6),
        }, index=idx)
        wide = EquityDataPipeline._to_wide(data, "close")
        assert "AAPL" in wide.columns
        assert "MSFT" in wide.columns
        assert len(wide) == 5

    def test_to_wide_empty(self):
        from exon.data.equity_data import EquityDataPipeline
        wide = EquityDataPipeline._to_wide(pd.DataFrame(), "close")
        assert wide.empty

    def test_get_returns_log(self):
        from exon.data.equity_data import EquityDataPipeline
        pipeline = EquityDataPipeline(client=None)
        dates = pd.bdate_range("2024-01-01", periods=100)
        prices = pd.DataFrame({
            "AAPL": 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01)),
            "MSFT": 200 * np.exp(np.cumsum(np.random.randn(100) * 0.01)),
        }, index=dates)
        returns = pipeline.get_returns(prices, method="log")
        assert len(returns) == 99
        assert "AAPL" in returns.columns

    def test_get_returns_pct(self):
        from exon.data.equity_data import EquityDataPipeline
        pipeline = EquityDataPipeline(client=None)
        dates = pd.bdate_range("2024-01-01", periods=50)
        prices = pd.DataFrame({"AAPL": 100 + np.arange(50)}, index=dates)
        returns = pipeline.get_returns(prices, method="pct")
        assert len(returns) == 49

    def test_annualisation_constants(self):
        from exon.data.equity_data import TRADING_DAYS_PER_YEAR, BARS_PER_DAY
        assert TRADING_DAYS_PER_YEAR == 252
        assert BARS_PER_DAY["1Day"] == 1
        assert BARS_PER_DAY["1Hour"] == 7


# ---------- Alpaca Executor Tests ----------

class TestAlpacaExecutor:
    def test_simulate_order(self):
        from exon.execution.alpaca_executor import AlpacaExecutionEngine, OrderRequest, OrderSide
        engine = AlpacaExecutionEngine(client=None, dry_run=True)
        order = OrderRequest(symbol="AAPL", side=OrderSide.BUY, qty=10)
        result = engine._simulate_order(order, {"AAPL": 150.0})
        assert result.status == "FILLED_SIM"
        assert result.filled_qty == 10
        assert result.avg_price > 150.0  # slippage added for buy

    def test_simulate_sell_order(self):
        from exon.execution.alpaca_executor import AlpacaExecutionEngine, OrderRequest, OrderSide
        engine = AlpacaExecutionEngine(client=None, dry_run=True)
        order = OrderRequest(symbol="MSFT", side=OrderSide.SELL, qty=5)
        result = engine._simulate_order(order, {"MSFT": 300.0})
        assert result.avg_price < 300.0  # slippage subtracted for sell
        assert result.fees == 0  # commission-free

    def test_compute_orders(self):
        from exon.execution.alpaca_executor import AlpacaExecutionEngine
        engine = AlpacaExecutionEngine(client=None, dry_run=True)
        target = pd.Series({"AAPL": 0.3, "MSFT": 0.2})
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        positions = {}
        orders = engine._compute_orders(target, 100_000, prices, positions)
        assert len(orders) == 2
        symbols = {o.symbol for o in orders}
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_skip_small_trades(self):
        from exon.execution.alpaca_executor import AlpacaExecutionEngine
        engine = AlpacaExecutionEngine(client=None, dry_run=True)
        target = pd.Series({"AAPL": 0.001})  # tiny allocation
        prices = {"AAPL": 150.0}
        positions = {}
        orders = engine._compute_orders(target, 10_000, prices, positions)
        # $10 * 0.001 = $10 < $50 threshold
        assert len(orders) == 0

    def test_round_to_whole_shares(self):
        from exon.execution.alpaca_executor import AlpacaExecutionEngine
        engine = AlpacaExecutionEngine(client=None, dry_run=True)
        target = pd.Series({"AAPL": 0.05})
        prices = {"AAPL": 150.0}
        positions = {}
        orders = engine._compute_orders(target, 100_000, prices, positions)
        if orders:
            assert orders[0].qty == int(orders[0].qty)

    def test_execute_target_weights_dry_run(self):
        from exon.execution.alpaca_executor import AlpacaExecutionEngine
        engine = AlpacaExecutionEngine(client=None, dry_run=True)
        target = pd.Series({"AAPL": 0.3, "MSFT": 0.2})
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        results = engine.execute_target_weights(target, 100_000, prices, {})
        assert len(results) == 2
        for r in results:
            assert r.status == "FILLED_SIM"

    def test_dry_run_positions_empty(self):
        from exon.execution.alpaca_executor import AlpacaExecutionEngine
        engine = AlpacaExecutionEngine(client=None, dry_run=True)
        assert engine.get_current_positions() == {}


# ---------- Config Tests ----------

class TestEquitiesConfig:
    def test_load_equities_yaml(self):
        from pathlib import Path
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "equities.yaml"
        assert config_path.exists()
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "alpaca" in cfg
        assert cfg["alpaca"]["paper"] is True
        assert "universe" in cfg
        assert "SPY" == cfg["universe"]["benchmark"]
        assert len(cfg["universe"]["symbols"]) >= 10
        assert cfg["backtest"]["maker_fee"] == 0.0  # commission-free
        assert cfg["backtest"]["slippage_bps"] == 2

    def test_equities_strategies_defined(self):
        from pathlib import Path
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "equities.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        strats = cfg["strategies"]
        assert "sector_rotation" in strats
        assert "earnings_momentum" in strats
        assert "market_microstructure" in strats
        assert strats["sector_rotation"]["enabled"] is True

    def test_alpaca_env_override(self):
        import os
        from exon.config import load_config
        os.environ["ALPACA_API_KEY"] = "test_key_123"
        os.environ["ALPACA_API_SECRET"] = "test_secret_456"
        try:
            cfg = load_config()
            assert cfg.get("alpaca", {}).get("api_key") == "test_key_123"
            assert cfg.get("alpaca", {}).get("api_secret") == "test_secret_456"
        finally:
            del os.environ["ALPACA_API_KEY"]
            del os.environ["ALPACA_API_SECRET"]


# ---------- Integration: Strategies with equities-like data ----------

class TestStrategiesWithEquityData:
    """Verify that existing crypto strategies work on equity-like data (daily bars)."""

    def _make_equity_universe(self, n=300, n_stocks=8, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2023-01-01", periods=n)
        data = {}
        for i in range(n_stocks):
            drift = 0.0003 * (i - n_stocks // 2)
            vol = 0.015 + 0.005 * (i % 3)
            data[f"STOCK{i}"] = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        return pd.DataFrame(data, index=dates)

    def test_ts_momentum_on_daily(self):
        from exon.strategies.momentum import TimeSeriesMomentum
        data = self._make_equity_universe()
        strategy = TimeSeriesMomentum(lookbacks=[21, 63, 126], vol_target=0.15)
        signals = strategy.generate_signals(data)
        assert isinstance(signals, list)

    def test_kalman_scanner_on_equities(self):
        from exon.strategies.kalman_spread import KalmanSpreadScanner
        data = self._make_equity_universe(n_stocks=5)
        scanner = KalmanSpreadScanner(max_pairs=3, z_entry=1.0)
        signals = scanner.generate_signals(data)
        assert isinstance(signals, list)

    def test_multifactor_on_equities(self):
        from exon.strategies.multifactor import MultiFactor
        data = self._make_equity_universe(n=400, n_stocks=10)
        strategy = MultiFactor(top_n=3, bottom_n=3, use_adaptive_weights=False)
        signals = strategy.generate_signals(data)
        assert len(signals) > 0

    def test_hurst_regime_on_equities(self):
        from exon.strategies.hurst_regime import HurstRegimeFilter
        data = self._make_equity_universe(n=400, n_stocks=3)
        strategy = HurstRegimeFilter(hurst_window=63, threshold_trend=0.55, threshold_mr=0.45)
        signals = strategy.generate_signals(data)
        assert isinstance(signals, list)

    def test_composite_on_equities(self):
        from exon.strategies.momentum import TimeSeriesMomentum
        from exon.strategies.mean_reversion import BollingerMeanReversion
        from exon.strategies.composite import CompositeStrategy

        data = self._make_equity_universe()
        strategies = [
            (TimeSeriesMomentum(lookbacks=[21, 63]), 0.5),
            (BollingerMeanReversion(window=21, entry_std=2.0), 0.5),
        ]
        composite = CompositeStrategy(strategies, regime_aware=False)
        signals = composite.generate_signals(data)
        assert isinstance(signals, list)
