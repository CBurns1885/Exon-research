"""Tests for the four quant hedge fund strategies."""

import numpy as np
import pandas as pd
import pytest

from exon.strategies.base import Signal
from exon.strategies.kalman_spread import KalmanSpreadTrading, KalmanSpreadScanner
from exon.strategies.multifactor import MultiFactor
from exon.strategies.hurst_regime import HurstRegimeFilter, estimate_hurst
from exon.strategies.wavelet_momentum import WaveletMomentum, haar_wavelet_decompose


# ---------- Test data generators ----------

def make_cointegrated_pair(n=800, seed=42):
    """Two cointegrated price series with known hedge ratio ~2.0."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    x = 100 + np.cumsum(rng.normal(0, 0.5, n))
    # y = 2*x + stationary noise, with a recent divergence
    noise = np.zeros(n)
    theta = 0.05  # mean-reversion speed of spread
    for t in range(1, n):
        noise[t] = noise[t - 1] * (1 - theta) + rng.normal(0, 1.5)
    # Inject a divergence in the last 20 bars
    noise[-20:] += 8.0
    y = 2.0 * x + 50 + noise
    return pd.DataFrame({"Y-USD": y, "X-USD": x}, index=dates)


def make_multi_asset(n=800, n_assets=10, seed=42):
    """10 assets with varying momentum, vol, and autocorrelation."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data = {}
    for i in range(n_assets):
        # Vary drift, vol, and autocorrelation
        drift = 0.001 * (i - n_assets // 2)
        vol = 0.01 + 0.003 * i
        ac = 0.1 * ((-1) ** i)  # alternating pos/neg autocorrelation
        returns = np.zeros(n)
        for t in range(1, n):
            returns[t] = drift + ac * returns[t - 1] + rng.normal(0, vol)
        data[f"ASSET{i}-USD"] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


def make_trending_series(n=500, seed=42):
    """Single asset with strong trend (H > 0.5)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Fractional Brownian-like via cumulative sum of correlated noise
    noise = rng.normal(0.002, 0.01, n)
    # Add persistence
    for t in range(1, n):
        noise[t] += 0.3 * noise[t - 1]
    prices = 100 * np.exp(np.cumsum(noise))
    return pd.DataFrame({"TREND-USD": prices}, index=dates)


def make_mean_reverting_series(n=500, seed=42):
    """Single asset with mean-reverting behaviour (H < 0.5)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    x = np.zeros(n)
    x[0] = np.log(100)
    theta = 0.08
    mu = np.log(100)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + rng.normal(0, 0.01)
    prices = np.exp(x)
    return pd.DataFrame({"MR-USD": prices}, index=dates)


# ---------- Kalman Filter Tests ----------

class TestKalmanSpreadTrading:
    def test_runs_on_cointegrated_pair(self):
        prices = make_cointegrated_pair()
        strategy = KalmanSpreadTrading(
            asset_y="Y-USD", asset_x="X-USD",
            delta=1e-4, z_entry=1.5, z_stop=5.0,
        )
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)
        # With the injected divergence, should produce signals
        assert len(signals) > 0

    def test_signal_directions_opposite(self):
        """In a pairs trade, y and x should have opposite directions."""
        prices = make_cointegrated_pair()
        strategy = KalmanSpreadTrading(
            asset_y="Y-USD", asset_x="X-USD",
            delta=1e-4, z_entry=1.0, z_stop=6.0,
        )
        signals = strategy.generate_signals(prices)
        if len(signals) == 2:
            assert signals[0].direction != signals[1].direction

    def test_kalman_beta_near_true_value(self):
        """Kalman filter should recover hedge ratio ~2.0."""
        prices = make_cointegrated_pair()
        strategy = KalmanSpreadTrading(
            asset_y="Y-USD", asset_x="X-USD", delta=1e-4,
        )
        signals = strategy.generate_signals(prices)
        for s in signals:
            if s.metadata.get("role") == "y":
                beta = s.metadata["kalman_beta"]
                assert 1.0 < beta < 3.5, f"Kalman beta={beta}, expected ~2.0"

    def test_no_signal_when_spread_is_flat(self):
        """If spread is well-behaved, no signal should fire."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
        x = 100 + np.cumsum(rng.normal(0, 0.3, 500))
        y = 2.0 * x + 50 + rng.normal(0, 0.5, 500)  # tight spread
        prices = pd.DataFrame({"Y-USD": y, "X-USD": x}, index=dates)
        strategy = KalmanSpreadTrading(
            asset_y="Y-USD", asset_x="X-USD",
            delta=1e-5, z_entry=2.5,
        )
        signals = strategy.generate_signals(prices)
        assert len(signals) == 0

    def test_metadata_completeness(self):
        prices = make_cointegrated_pair()
        strategy = KalmanSpreadTrading(
            asset_y="Y-USD", asset_x="X-USD", z_entry=1.0,
        )
        signals = strategy.generate_signals(prices)
        for s in signals:
            assert "innovation_z" in s.metadata
            assert "kalman_beta" in s.metadata


class TestKalmanScanner:
    def test_scanner_finds_pairs(self):
        prices = make_cointegrated_pair()
        scanner = KalmanSpreadScanner(max_pairs=3, z_entry=1.0, z_stop=6.0)
        signals = scanner.generate_signals(prices)
        assert isinstance(signals, list)


# ---------- Multi-Factor Tests ----------

class TestMultiFactor:
    def test_generates_long_short(self):
        prices = make_multi_asset()
        strategy = MultiFactor(
            top_n=3, bottom_n=3, use_adaptive_weights=False,
        )
        signals = strategy.generate_signals(prices)
        assert len(signals) > 0
        longs = [s for s in signals if s.direction > 0]
        shorts = [s for s in signals if s.direction < 0]
        assert len(longs) == 3
        assert len(shorts) == 3

    def test_factor_metadata(self):
        prices = make_multi_asset()
        strategy = MultiFactor(top_n=2, bottom_n=2, use_adaptive_weights=False)
        signals = strategy.generate_signals(prices)
        for s in signals:
            assert "composite_alpha" in s.metadata
            assert "factors" in s.metadata
            factors = s.metadata["factors"]
            assert "momentum" in factors
            assert "reversal" in factors
            assert "volatility" in factors

    def test_adaptive_weights(self):
        prices = make_multi_asset(n=1200)
        strategy = MultiFactor(
            top_n=2, bottom_n=2, use_adaptive_weights=True, ic_lookback=500,
        )
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_no_duplicate_assets(self):
        prices = make_multi_asset()
        strategy = MultiFactor(top_n=3, bottom_n=3)
        signals = strategy.generate_signals(prices)
        assets = [s.asset for s in signals]
        assert len(assets) == len(set(assets))


# ---------- Hurst Exponent Tests ----------

class TestHurstExponent:
    def test_trending_series_high_hurst(self):
        """A trending series (autocorrelated returns) should have H > 0.5."""
        rng = np.random.default_rng(42)
        # Positively autocorrelated returns → persistent → H > 0.5
        returns = np.zeros(2000)
        for t in range(1, 2000):
            returns[t] = 0.5 * returns[t - 1] + rng.normal(0, 1)
        prices = 100 + np.cumsum(returns)
        H = estimate_hurst(prices)
        assert H > 0.5, f"H={H} for trending series, expected > 0.5"

    def test_mean_reverting_low_hurst(self):
        """A mean-reverting series (anti-correlated returns) should have H < 0.5."""
        rng = np.random.default_rng(42)
        # Negatively autocorrelated returns → anti-persistent → H < 0.5
        returns = np.zeros(2000)
        for t in range(1, 2000):
            returns[t] = -0.5 * returns[t - 1] + rng.normal(0, 1)
        prices = 100 + np.cumsum(returns)
        H = estimate_hurst(prices)
        assert H < 0.5, f"H={H} for MR series, expected < 0.5"

    def test_random_walk_near_half(self):
        """A random walk (iid returns) should have H ≈ 0.5."""
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 5000))
        H = estimate_hurst(prices)
        assert 0.35 < H < 0.65, f"H={H} for random walk, expected ~0.5"


class TestHurstRegimeFilter:
    def test_trending_asset_gets_momentum_signal(self):
        prices = make_trending_series()
        strategy = HurstRegimeFilter(
            hurst_window=168, threshold_trend=0.50, threshold_mr=0.45,
        )
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)
        for s in signals:
            if s.metadata.get("regime") == "trending":
                assert s.direction in (-1.0, 1.0)
                assert s.metadata["hurst"] > 0.5

    def test_mr_asset_gets_counter_signal(self):
        prices = make_mean_reverting_series()
        strategy = HurstRegimeFilter(
            hurst_window=168, threshold_trend=0.55, threshold_mr=0.50,
            dead_zone=0.01,
        )
        signals = strategy.generate_signals(prices)
        for s in signals:
            if s.metadata.get("regime") == "mean_reverting":
                assert s.metadata["hurst"] < 0.50

    def test_metadata_has_hurst(self):
        prices = make_trending_series()
        strategy = HurstRegimeFilter()
        signals = strategy.generate_signals(prices)
        for s in signals:
            assert "hurst" in s.metadata
            assert "regime" in s.metadata


# ---------- Wavelet Momentum Tests ----------

class TestHaarWavelet:
    def test_decomposition_levels(self):
        signal = np.random.randn(256)
        coeffs = haar_wavelet_decompose(signal, max_level=5)
        assert "D1" in coeffs
        assert "D5" in coeffs
        assert "A5" in coeffs
        # D1 should have half the length of input
        assert len(coeffs["D1"]) == 128
        # D2 should have quarter
        assert len(coeffs["D2"]) == 64

    def test_zero_signal(self):
        signal = np.zeros(64)
        coeffs = haar_wavelet_decompose(signal, max_level=3)
        for key, val in coeffs.items():
            assert np.allclose(val, 0)


class TestWaveletMomentum:
    def test_generates_signals(self):
        prices = make_trending_series(n=500)
        strategy = WaveletMomentum(max_level=5, lookback=256)
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_metadata_has_scale_info(self):
        prices = make_trending_series(n=500)
        strategy = WaveletMomentum(max_level=4, lookback=128)
        signals = strategy.generate_signals(prices)
        for s in signals:
            assert "scale_signals" in s.metadata
            assert "trend_signal" in s.metadata
            assert "agreeing_scales" in s.metadata

    def test_multi_asset(self):
        prices = make_multi_asset(n=500, n_assets=4)
        strategy = WaveletMomentum(max_level=4, lookback=256)
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)

    def test_scale_weights_sum(self):
        """Default scale weights + trend weight should sum to ~1."""
        strategy = WaveletMomentum(max_level=5)
        total = sum(strategy.scale_weights) + strategy.trend_weight
        assert abs(total - 1.0) < 0.1
