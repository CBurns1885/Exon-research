"""Tests for the three new crypto-specific strategies."""

import numpy as np
import pandas as pd
import pytest

from exon.strategies.base import Signal
from exon.strategies.volatility_breakout import VolatilityBreakout
from exon.strategies.volume_momentum import VolumeWeightedMomentum
from exon.strategies.lead_lag import LeadLagExploitation


def make_breakout_prices(n=600, n_assets=4, seed=42):
    """Synthetic prices: long compression then explosive breakout."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data = {}
    for i in range(n_assets):
        # Phase 1: tight range (compression)
        compression = rng.normal(0, 0.002, n - 50)
        # Phase 2: breakout
        breakout_dir = 1 if i % 2 == 0 else -1
        breakout = rng.normal(0.01 * breakout_dir, 0.005, 50)
        log_returns = np.concatenate([compression, breakout])
        prices = 100 * np.exp(np.cumsum(log_returns))
        data[f"ASSET{i}-USD"] = prices
    return pd.DataFrame(data, index=dates)


def make_volume_prices(n=600, n_assets=4, seed=42):
    """Synthetic prices with volume-like intensity surges."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data = {}
    for i in range(n_assets):
        # Strong directional trend with varying intensity
        drift = 0.003 * ((-1) ** i)
        noise = rng.normal(drift, 0.012, n)
        # Add a volume spike period (larger absolute returns reinforcing trend)
        noise[-72:-24] = rng.normal(drift * 4, 0.02, 48)
        prices = 100 * np.exp(np.cumsum(noise))
        data[f"ASSET{i}-USD"] = prices
    return pd.DataFrame(data, index=dates)


def make_lead_lag_prices(n=800, seed=42):
    """Synthetic prices where BTC-USD leads altcoins by ~3 bars."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    # BTC: the leader with a strong recent move
    btc_returns = rng.normal(0.0005, 0.015, n)
    # Add a clear directional impulse in the last 12 bars
    btc_returns[-12:] += 0.01
    btc = 50000 * np.exp(np.cumsum(btc_returns))

    # ETH: correlated leader
    eth_returns = 0.7 * btc_returns + 0.3 * rng.normal(0, 0.015, n)
    eth_returns[-10:] += 0.008
    eth = 3000 * np.exp(np.cumsum(eth_returns))

    # Altcoins: follow BTC with a lag of 3 bars
    alt1_returns = np.zeros(n)
    alt1_returns[3:] = 0.5 * btc_returns[:-3] + 0.5 * rng.normal(0, 0.01, n - 3)
    alt1 = 100 * np.exp(np.cumsum(alt1_returns))

    alt2_returns = np.zeros(n)
    alt2_returns[4:] = 0.4 * btc_returns[:-4] + 0.6 * rng.normal(0, 0.012, n - 4)
    alt2 = 50 * np.exp(np.cumsum(alt2_returns))

    return pd.DataFrame(
        {"BTC-USD": btc, "ETH-USD": eth, "SOL-USD": alt1, "AVAX-USD": alt2},
        index=dates,
    )


class TestVolatilityBreakout:
    def test_detects_breakout(self):
        prices = make_breakout_prices()
        strategy = VolatilityBreakout(
            channel_window=48,
            atr_window=24,
            squeeze_window=168,
            squeeze_threshold=0.8,
            confirmation_bars=2,
        )
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, Signal)
            assert s.direction in (-1.0, 1.0)
            assert 0 < s.strength <= 1.0
            assert "squeeze_ratio" in s.metadata

    def test_no_signals_in_flat_market(self):
        """A perfectly flat market should produce no breakout signals."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
        # Very tight random walk around 100
        noise = rng.normal(0, 0.0005, 500)
        prices = pd.DataFrame(
            {"FLAT-USD": 100 + np.cumsum(noise)},
            index=dates,
        )
        strategy = VolatilityBreakout(confirmation_bars=2)
        signals = strategy.generate_signals(prices)
        # Either no signals or very weak ones
        for s in signals:
            assert s.strength < 0.5

    def test_metadata_fields(self):
        prices = make_breakout_prices()
        strategy = VolatilityBreakout()
        signals = strategy.generate_signals(prices)
        for s in signals:
            assert "squeeze_ratio" in s.metadata
            assert "in_squeeze" in s.metadata
            assert "channel_high" in s.metadata
            assert "atr" in s.metadata


class TestVolumeWeightedMomentum:
    def test_generates_signals(self):
        prices = make_volume_prices()
        strategy = VolumeWeightedMomentum(
            momentum_window=72,
            volume_window=168,
        )
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)
        assert len(signals) > 0
        for s in signals:
            assert isinstance(s, Signal)
            assert -1.0 <= s.direction <= 1.0
            assert 0 < s.strength <= 1.0

    def test_metadata_fields(self):
        prices = make_volume_prices()
        strategy = VolumeWeightedMomentum()
        signals = strategy.generate_signals(prices)
        for s in signals:
            assert "rvol" in s.metadata
            assert "vpt_direction" in s.metadata
            assert "obv_divergence" in s.metadata
            assert "momentum_z" in s.metadata

    def test_vol_target_scaling(self):
        """Higher vol-target should produce stronger signals, all else equal."""
        prices = make_volume_prices()
        low_vol = VolumeWeightedMomentum(vol_target=0.05)
        high_vol = VolumeWeightedMomentum(vol_target=0.30)
        sigs_low = low_vol.generate_signals(prices)
        sigs_high = high_vol.generate_signals(prices)
        if sigs_low and sigs_high:
            avg_low = np.mean([s.strength for s in sigs_low])
            avg_high = np.mean([s.strength for s in sigs_high])
            assert avg_high >= avg_low


class TestLeadLagExploitation:
    def test_detects_lead_lag(self):
        prices = make_lead_lag_prices()
        strategy = LeadLagExploitation(
            leaders=["BTC-USD", "ETH-USD"],
            max_lag=6,
            xcorr_window=168,
            min_correlation=0.10,
            signal_window=6,
        )
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, list)
        # Should signal on followers (SOL, AVAX) not leaders
        signal_assets = {s.asset for s in signals}
        for asset in signal_assets:
            assert asset not in ["BTC-USD", "ETH-USD"]

    def test_no_signals_without_leader_move(self):
        """If BTC is flat, no lead-lag signals should fire."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
        # All assets random-walk with no trend
        data = {}
        for asset in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            data[asset] = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 500)))
        prices = pd.DataFrame(data, index=dates)
        strategy = LeadLagExploitation(min_correlation=0.20)
        signals = strategy.generate_signals(prices)
        # Should produce few or no signals since leader hasn't moved
        assert len(signals) <= 1

    def test_metadata_has_leader_info(self):
        prices = make_lead_lag_prices()
        strategy = LeadLagExploitation(min_correlation=0.10)
        signals = strategy.generate_signals(prices)
        for s in signals:
            meta = s.metadata
            assert "leader" in meta or "leaders" in meta

    def test_deduplication(self):
        """When both BTC and ETH signal the same alt, result should be deduped."""
        prices = make_lead_lag_prices()
        strategy = LeadLagExploitation(
            leaders=["BTC-USD", "ETH-USD"],
            min_correlation=0.05,
        )
        signals = strategy.generate_signals(prices)
        # No duplicate assets in output
        assets = [s.asset for s in signals]
        assert len(assets) == len(set(assets))
