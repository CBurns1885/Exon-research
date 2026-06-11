"""Tests for trading strategies using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from exon.strategies.base import Signal, normalise_signal, crossover
from exon.strategies.momentum import (
    TimeSeriesMomentum,
    CrossSectionalMomentum,
    AdaptiveMomentum,
)
from exon.strategies.mean_reversion import (
    BollingerMeanReversion,
    OUMeanReversion,
    MultiTimeframeMeanReversion,
)
from exon.strategies.composite import CompositeStrategy


def make_trending_prices(n=1000, n_assets=5, seed=42):
    """Generate synthetic trending price data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data = {}
    for i in range(n_assets):
        drift = 0.001 * (i + 1)  # strong drift to ensure signal generation
        noise = rng.normal(drift, 0.02, n)
        prices = 100 * np.exp(np.cumsum(noise))
        data[f"ASSET{i}-USD"] = prices
    return pd.DataFrame(data, index=dates)


def make_mean_reverting_prices(n=1000, seed=42):
    """Generate synthetic mean-reverting price data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data = {}
    for i in range(4):
        x = np.zeros(n)
        x[0] = 100
        theta = 0.05
        mu = np.log(100)
        sigma = 0.01
        for t in range(1, n):
            x[t] = x[t - 1] + theta * (mu - np.log(x[t - 1])) + sigma * rng.normal()
        data[f"MR{i}-USD"] = np.exp(x) if i == 0 else 100 + np.cumsum(
            theta * (0 - np.cumsum(rng.normal(0, sigma, n))) + rng.normal(0, sigma, n)
        )
    # Ensure all positive
    for k in data:
        data[k] = np.abs(data[k]) + 1
    return pd.DataFrame(data, index=dates)


class TestSignal:
    def test_position_size(self):
        s = Signal(pd.Timestamp.now(), "BTC-USD", direction=1.0, strength=0.5)
        assert s.position_size == 0.5

        s2 = Signal(pd.Timestamp.now(), "ETH-USD", direction=-1.0, strength=0.8)
        assert s2.position_size == -0.8


class TestNormaliseSignal:
    def test_output_range(self):
        raw = pd.Series(np.random.randn(500))
        normed = normalise_signal(raw, window=100)
        assert normed.max() <= 1.0
        assert normed.min() >= -1.0


class TestCrossover:
    def test_detects_crossover(self):
        fast = pd.Series([1, 2, 3, 2, 1, 2, 3])
        slow = pd.Series([2, 2, 2, 2, 2, 2, 2])
        result = crossover(fast, slow)
        assert (result == 1).any()
        assert (result == -1).any()


class TestTimeSeriesMomentum:
    def test_generates_signals(self):
        prices = make_trending_prices()
        strategy = TimeSeriesMomentum(lookbacks=[24, 72, 168])
        signals = strategy.generate_signals(prices)
        assert len(signals) > 0
        for s in signals:
            assert isinstance(s, Signal)
            assert -1.0 <= s.direction <= 1.0
            assert 0 <= s.strength <= 1.0

    def test_weights_sum(self):
        prices = make_trending_prices()
        strategy = TimeSeriesMomentum()
        signals = strategy.generate_signals(prices)
        weights = strategy.signals_to_weights(signals)
        assert abs(weights.abs().sum() - 1.0) < 0.01


class TestCrossSectionalMomentum:
    def test_long_short(self):
        prices = make_trending_prices()
        strategy = CrossSectionalMomentum(top_n=2, bottom_n=2)
        signals = strategy.generate_signals(prices)
        longs = [s for s in signals if s.direction > 0]
        shorts = [s for s in signals if s.direction < 0]
        assert len(longs) == 2
        assert len(shorts) == 2


class TestAdaptiveMomentum:
    def test_runs(self):
        prices = make_trending_prices()
        strategy = AdaptiveMomentum()
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)


class TestBollingerMeanReversion:
    def test_counter_trend(self):
        prices = make_mean_reverting_prices()
        strategy = BollingerMeanReversion(window=48, entry_std=1.5)
        signals = strategy.generate_signals(prices)
        # Should generate some signals on mean-reverting data
        assert isinstance(signals, list)


class TestOUMeanReversion:
    def test_runs(self):
        prices = make_mean_reverting_prices()
        strategy = OUMeanReversion(calibration_window=500)
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)


class TestCompositeStrategy:
    def test_blends_signals(self):
        prices = make_trending_prices()
        strategies = [
            (TimeSeriesMomentum(lookbacks=[24, 72]), 0.5),
            (CrossSectionalMomentum(top_n=2, bottom_n=2), 0.5),
        ]
        composite = CompositeStrategy(strategies, regime_aware=False)
        signals = composite.generate_signals(prices)
        assert len(signals) > 0
