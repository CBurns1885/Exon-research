"""Performance metrics and statistical tests for backtest evaluation.

Provides comprehensive metrics beyond basic Sharpe, including
drawdown analysis, trade statistics, and statistical significance tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def drawdown_analysis(equity: pd.Series) -> pd.DataFrame:
    """Analyse all drawdown periods.

    Returns DataFrame with start, trough, recovery dates,
    depth, and duration for each drawdown.
    """
    peak = equity.cummax()
    dd = (equity - peak) / peak

    in_drawdown = dd < 0
    groups = (~in_drawdown).cumsum()

    records = []
    for _, group in dd[in_drawdown].groupby(groups[in_drawdown]):
        if group.empty:
            continue
        records.append(
            {
                "start": group.index[0],
                "trough_date": group.idxmin(),
                "end": group.index[-1],
                "depth": group.min(),
                "duration_hours": len(group),
            }
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("depth").reset_index(drop=True)


def rolling_sharpe(returns: pd.Series, window: int = 720) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(8760)).dropna()


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Information ratio vs a benchmark."""
    excess = strategy_returns - benchmark_returns
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(8760)


def tail_ratio(returns: pd.Series, percentile: float = 5.0) -> float:
    """Ratio of right tail (gains) to left tail (losses)."""
    right = np.percentile(returns.dropna(), 100 - percentile)
    left = abs(np.percentile(returns.dropna(), percentile))
    if left == 0:
        return float("inf") if right > 0 else 0.0
    return right / left


def stability_of_returns(returns: pd.Series) -> float:
    """R-squared of cumulative returns regressed against time.

    Higher values indicate more consistent compounding.
    """
    cum = returns.cumsum()
    x = np.arange(len(cum))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, cum.values)
    return r_value**2


def monte_carlo_sharpe_test(
    returns: pd.Series,
    n_simulations: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap test: is the Sharpe ratio statistically significant?

    Shuffles returns to destroy any real signal and computes the Sharpe
    distribution under the null hypothesis.
    """
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(8760) if returns.std() > 0 else 0

    null_sharpes = []
    rng = np.random.default_rng(42)
    vals = returns.dropna().values

    for _ in range(n_simulations):
        shuffled = rng.permutation(vals)
        s = shuffled.mean() / shuffled.std() * np.sqrt(8760) if shuffled.std() > 0 else 0
        null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= actual_sharpe).mean()

    return {
        "actual_sharpe": actual_sharpe,
        "null_mean": null_sharpes.mean(),
        "null_std": null_sharpes.std(),
        "p_value": p_value,
        "significant": p_value < (1 - confidence),
        "confidence": confidence,
    }


def turnover(positions: pd.DataFrame) -> pd.Series:
    """Calculate portfolio turnover as absolute change in weights per period."""
    weight_changes = positions.diff().abs().sum(axis=1)
    return weight_changes


def trade_statistics(trades: list) -> dict:
    """Compute detailed trade statistics."""
    if not trades:
        return {"n_trades": 0}

    costs = [t.cost for t in trades]
    slippages = [t.slippage for t in trades]
    values = [t.quantity * t.price for t in trades]

    return {
        "n_trades": len(trades),
        "total_cost": sum(costs),
        "avg_cost": np.mean(costs),
        "total_slippage": sum(slippages),
        "avg_trade_value": np.mean(values),
        "median_trade_value": np.median(values),
        "buys": sum(1 for t in trades if t.side == "buy"),
        "sells": sum(1 for t in trades if t.side == "sell"),
    }


def compare_strategies(results: list) -> pd.DataFrame:
    """Compare multiple backtest results side by side."""
    rows = []
    for r in results:
        rows.append(r.summary())
    return pd.DataFrame(rows).set_index("strategy")
