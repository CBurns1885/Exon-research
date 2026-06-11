"""Tests for research modules."""

import numpy as np
import pandas as pd
import pytest

from exon.research.correlations import (
    correlation_matrix,
    rolling_correlation,
    correlation_stability,
    cluster_assets,
    top_correlated_pairs,
)
from exon.research.cointegration import (
    engle_granger_test,
    adf_test,
    z_score,
    half_life,
    scan_cointegrated_pairs,
)
from exon.research.pca import decompose_returns, residual_momentum
from exon.research.regime import (
    detect_regimes_gmm,
    regime_statistics,
    volatility_regime,
    trend_strength,
)


def make_returns(n=1000, n_assets=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Correlated returns
    base = rng.normal(0, 0.01, n)
    data = {}
    for i in range(n_assets):
        idio = rng.normal(0, 0.005, n)
        data[f"ASSET{i}"] = base * (0.5 + 0.1 * i) + idio
    return pd.DataFrame(data, index=dates)


def make_cointegrated_prices(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    x = 100 + np.cumsum(rng.normal(0, 0.5, n))
    y = 2.0 * x + 50 + rng.normal(0, 2, n)  # cointegrated with x
    return pd.DataFrame({"X": x, "Y": y}, index=dates)


class TestCorrelations:
    def test_matrix_shape(self):
        returns = make_returns()
        corr = correlation_matrix(returns)
        assert corr.shape == (5, 5)
        assert (np.diag(corr.values) == 1.0).all()

    def test_rolling_correlation(self):
        returns = make_returns()
        rc = rolling_correlation(returns, ("ASSET0", "ASSET1"), window=100)
        assert len(rc) == len(returns)
        assert rc.dropna().between(-1, 1).all()

    def test_stability(self):
        returns = make_returns()
        stab = correlation_stability(returns)
        assert "stability" in stab.columns
        assert len(stab) > 0

    def test_cluster_assets(self):
        returns = make_returns()
        clusters = cluster_assets(returns, n_clusters=2)
        assert len(clusters) == 5
        assert clusters.nunique() <= 2

    def test_top_pairs(self):
        returns = make_returns()
        pairs = top_correlated_pairs(returns, n=3)
        assert len(pairs) <= 6  # top 3 + bottom 3
        assert "correlation" in pairs.columns


class TestCointegration:
    def test_engle_granger(self):
        prices = make_cointegrated_prices()
        result = engle_granger_test(prices["Y"], prices["X"])
        assert result["p_value"] < 0.05  # should be cointegrated
        assert abs(result["hedge_ratio"] - 2.0) < 0.5

    def test_adf(self):
        prices = make_cointegrated_prices()
        # Spread should be stationary
        spread = prices["Y"] - 2.0 * prices["X"]
        result = adf_test(spread)
        assert result["p_value"] < 0.05

    def test_z_score(self):
        spread = pd.Series(np.random.randn(500))
        z = z_score(spread, window=50)
        assert len(z) == 500

    def test_half_life_positive(self):
        prices = make_cointegrated_prices()
        spread = prices["Y"] - 2.0 * prices["X"]
        hl = half_life(spread)
        assert hl > 0
        assert hl < 500

    def test_scan(self):
        prices = make_cointegrated_prices()
        pairs = scan_cointegrated_pairs(prices, p_threshold=0.10)
        assert len(pairs) > 0


class TestPCA:
    def test_decomposition(self):
        returns = make_returns()
        result = decompose_returns(returns, n_components=3)
        assert result.n_components == 3
        assert result.components.shape == (3, 5)
        assert result.factors.shape[1] == 3
        assert result.residuals.shape[1] == 5
        assert result.cumulative_variance[-1] <= 1.0

    def test_residual_momentum(self):
        returns = make_returns()
        rm = residual_momentum(returns, n_components=2, lookback=100)
        assert len(rm) == 5


class TestRegime:
    def test_detect_regimes(self):
        returns = make_returns()
        regime_df = detect_regimes_gmm(returns["ASSET0"], n_regimes=3)
        assert "regime" in regime_df.columns
        assert regime_df["regime"].nunique() <= 3

    def test_regime_statistics(self):
        returns = make_returns()
        regime_df = detect_regimes_gmm(returns["ASSET0"])
        stats = regime_statistics(regime_df)
        assert "mean_return" in stats.columns
        assert abs(stats["pct_time"].sum() - 1.0) < 0.01

    def test_volatility_regime(self):
        returns = make_returns()
        vr = volatility_regime(returns["ASSET0"])
        assert len(vr) > 0

    def test_trend_strength(self):
        prices = pd.Series(np.cumsum(np.random.randn(500)) + 100)
        ts = trend_strength(prices, window=50)
        assert ts.between(-1, 1).all()
