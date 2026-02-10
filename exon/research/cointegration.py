"""Cointegration analysis for pairs / statistical arbitrage.

Identifies cointegrated pairs suitable for mean-reversion strategies,
computes hedge ratios, and monitors spread stationarity over time.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    max_lag: str = "AIC",
) -> dict:
    """Run Engle-Granger two-step cointegration test.

    Returns dict with t-statistic, p-value, hedge_ratio, and spread.
    """
    t_stat, p_value, crit_values = coint(y, x, maxlag=10, autolag=max_lag)

    # OLS hedge ratio: y = alpha + beta * x + epsilon
    model = OLS(y, add_constant(x)).fit()
    hedge_ratio = model.params.iloc[1]
    intercept = model.params.iloc[0]
    spread = y - hedge_ratio * x - intercept

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "crit_1pct": crit_values[0],
        "crit_5pct": crit_values[1],
        "crit_10pct": crit_values[2],
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread": spread,
    }


def adf_test(series: pd.Series, max_lag: str = "AIC") -> dict:
    """Augmented Dickey-Fuller stationarity test."""
    result = adfuller(series.dropna(), autolag=max_lag)
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
    }


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
) -> pd.Series:
    """Compute the spread for a cointegrated pair."""
    return y - hedge_ratio * x - intercept


def z_score(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of a spread."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int = 168,
) -> pd.Series:
    """Rolling OLS hedge ratio (beta) between two price series."""
    ratios = []
    idx = []
    for i in range(window, len(y)):
        y_win = y.iloc[i - window : i]
        x_win = x.iloc[i - window : i]
        model = OLS(y_win, add_constant(x_win)).fit()
        ratios.append(model.params.iloc[1])
        idx.append(y.index[i])
    return pd.Series(ratios, index=idx, name="hedge_ratio")


def half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life via AR(1) regression.

    Returns the half-life in number of periods.
    """
    spread = spread.dropna()
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    aligned = pd.concat([delta, lag], axis=1).dropna()
    aligned.columns = ["delta", "lag"]
    model = OLS(aligned["delta"], add_constant(aligned["lag"])).fit()
    phi = model.params.iloc[1]
    if phi >= 0:
        return float("inf")  # not mean reverting
    return -np.log(2) / phi


def scan_cointegrated_pairs(
    prices: pd.DataFrame,
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Scan all pairs in a price universe for cointegration.

    Returns DataFrame sorted by p-value with hedge ratios and half-lives.
    """
    symbols = prices.columns.tolist()
    results = []

    for a, b in itertools.combinations(symbols, 2):
        y, x = prices[a].dropna(), prices[b].dropna()
        common = y.index.intersection(x.index)
        if len(common) < 100:
            continue
        y, x = y.loc[common], x.loc[common]

        try:
            coint_result = engle_granger_test(y, x)
        except Exception as e:
            logger.debug("Coint failed for %s/%s: %s", a, b, e)
            continue

        if coint_result["p_value"] > p_threshold:
            continue

        hl = half_life(coint_result["spread"])

        results.append(
            {
                "asset_y": a,
                "asset_x": b,
                "p_value": coint_result["p_value"],
                "t_stat": coint_result["t_stat"],
                "hedge_ratio": coint_result["hedge_ratio"],
                "half_life": hl,
            }
        )

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
