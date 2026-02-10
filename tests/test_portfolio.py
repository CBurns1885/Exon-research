"""Tests for portfolio optimisation and risk management."""

import numpy as np
import pandas as pd
import pytest

from exon.portfolio.optimizer import (
    mean_variance_optimize,
    risk_parity,
    kelly_criterion,
    blend_signal_weights,
    constrain_turnover,
)
from exon.risk.manager import RiskManager, RiskLimits


def make_returns(n=500, n_assets=4, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    assets = [f"A{i}" for i in range(n_assets)]
    data = rng.normal(0.0001, 0.01, (n, n_assets))
    return pd.DataFrame(data, index=dates, columns=assets)


class TestMeanVariance:
    def test_weights_sum_to_one(self):
        returns = make_returns()
        mu = returns.mean()
        cov = returns.cov()
        w = mean_variance_optimize(mu, cov, max_weight=0.50)
        assert abs(w.sum() - 1.0) < 0.05

    def test_position_limits(self):
        returns = make_returns()
        mu = returns.mean()
        cov = returns.cov()
        w = mean_variance_optimize(mu, cov, max_weight=0.30)
        assert w.abs().max() <= 0.30 + 0.01


class TestRiskParity:
    def test_weights_positive(self):
        returns = make_returns()
        cov = returns.cov()
        w = risk_parity(cov)
        assert (w > 0).all()
        assert abs(w.sum() - 1.0) < 0.01


class TestKelly:
    def test_fractional_kelly(self):
        returns = make_returns()
        mu = returns.mean()
        cov = returns.cov()
        w = kelly_criterion(mu, cov, fraction=0.25, max_weight=0.25)
        assert w.abs().max() <= 0.25 + 0.01


class TestBlending:
    def test_blend(self):
        w1 = pd.Series({"A": 0.5, "B": 0.5})
        w2 = pd.Series({"B": 0.3, "C": 0.7})
        blended = blend_signal_weights(
            {"s1": w1, "s2": w2},
            {"s1": 0.6, "s2": 0.4},
        )
        assert abs(blended.abs().sum() - 1.0) < 0.01


class TestTurnover:
    def test_constrains(self):
        current = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        target = pd.Series({"A": 0.1, "B": 0.5, "C": 0.4})
        adjusted = constrain_turnover(target, current, max_turnover=0.15)
        actual_turnover = (adjusted - current).abs().sum()
        assert actual_turnover <= 0.15 + 0.01


class TestRiskManager:
    def test_within_limits(self):
        returns = make_returns()
        mgr = RiskManager(RiskLimits(
            max_position_pct=0.30,
            max_leverage=1.1,
            min_cash_pct=0.0,
            max_portfolio_var_pct=0.10,
        ))
        weights = pd.Series({"A0": 0.25, "A1": 0.25, "A2": 0.25, "A3": 0.25})
        report = mgr.check_portfolio(weights, 100_000, returns)
        assert report.is_within_limits

    def test_breach_position(self):
        returns = make_returns()
        mgr = RiskManager(RiskLimits(max_position_pct=0.20))
        weights = pd.Series({"A0": 0.50, "A1": 0.50})
        report = mgr.check_portfolio(weights, 100_000, returns)
        assert not report.is_within_limits

    def test_adjust_weights(self):
        returns = make_returns()
        mgr = RiskManager(RiskLimits(max_position_pct=0.25, max_leverage=1.0))
        weights = pd.Series({"A0": 0.50, "A1": 0.50, "A2": 0.50})
        adjusted = mgr.adjust_weights(weights, 100_000, returns)
        assert adjusted.abs().sum() <= 1.0 + 0.01
