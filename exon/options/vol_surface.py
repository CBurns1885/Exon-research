"""Volatility surface analysis and vol-based trading strategies.

Provides tools for:
1. IV rank/percentile computation
2. Term structure analysis (front vs back month IV)
3. Skew analysis (put vs call IV)
4. Realised vs implied vol comparison (variance risk premium)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class VolSurfaceSnapshot:
    """A snapshot of the volatility surface for a single underlying."""

    underlying: str
    atm_iv: float
    iv_rank: float  # 0-1 percentile vs history
    iv_percentile: float  # 0-1
    realised_vol: float
    variance_risk_premium: float  # IV - RV
    put_call_skew: float  # 25d put IV - 25d call IV
    term_structure_slope: float  # back month IV - front month IV


def compute_iv_rank(
    current_iv: float,
    historical_ivs: pd.Series,
) -> float:
    """IV Rank: where current IV sits relative to 52-week range.

    IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV)
    """
    if historical_ivs.empty:
        return 0.5
    high = historical_ivs.max()
    low = historical_ivs.min()
    if high == low:
        return 0.5
    return float(np.clip((current_iv - low) / (high - low), 0, 1))


def compute_iv_percentile(
    current_iv: float,
    historical_ivs: pd.Series,
) -> float:
    """IV Percentile: what % of days had IV below current level."""
    if historical_ivs.empty:
        return 0.5
    return float((historical_ivs < current_iv).sum() / len(historical_ivs))


def realised_volatility(
    returns: pd.Series,
    window: int = 21,
    annualise: bool = True,
) -> pd.Series:
    """Compute rolling realised volatility.

    Parameters
    ----------
    returns : log or pct returns
    window : lookback in bars
    annualise : if True, annualise assuming 252 trading days
    """
    rv = returns.rolling(window).std()
    if annualise:
        rv = rv * np.sqrt(252)
    return rv


def variance_risk_premium(
    implied_vol: float,
    realised_vol_val: float,
) -> float:
    """Variance Risk Premium (VRP) = IV - RV.

    Positive VRP → options are "expensive" → sell premium
    Negative VRP → options are "cheap" → buy premium
    """
    return implied_vol - realised_vol_val


def estimate_iv_from_returns(
    returns: pd.Series,
    windows: list[int] | None = None,
) -> dict[str, float]:
    """Estimate implied volatility proxies from historical returns.

    When live IV data isn't available (e.g., backtesting), we estimate
    using multiple realised vol windows as IV proxies:
    - Short-term RV (5d) ≈ near-term market expectation
    - Medium-term RV (21d) ≈ monthly ATM IV proxy
    - Long-term RV (63d) ≈ quarterly IV proxy
    - EWMA vol ≈ GARCH-like adaptive IV estimate
    """
    if windows is None:
        windows = [5, 21, 63]

    result = {}
    for w in windows:
        if len(returns) >= w:
            rv = returns.iloc[-w:].std() * np.sqrt(252)
            result[f"rv_{w}d"] = float(rv)

    # EWMA volatility (lambda=0.94, like RiskMetrics)
    if len(returns) >= 21:
        ewma = _ewma_vol(returns.values, lam=0.94)
        result["ewma_vol"] = float(ewma * np.sqrt(252))

    # Use 21d RV as primary IV proxy
    result["iv_proxy"] = result.get("rv_21d", result.get("rv_5d", 0.20))

    return result


def _ewma_vol(returns: np.ndarray, lam: float = 0.94) -> float:
    """Exponentially weighted moving average volatility."""
    var = returns[-1] ** 2
    for i in range(len(returns) - 2, max(len(returns) - 252, -1), -1):
        var = lam * var + (1 - lam) * returns[i] ** 2
    return float(np.sqrt(var))


def analyse_vol_surface(
    underlying: str,
    returns: pd.Series,
    current_iv: float | None = None,
    iv_history: pd.Series | None = None,
) -> VolSurfaceSnapshot:
    """Produce a complete volatility surface snapshot.

    Can work with or without live IV data — falls back to
    realised vol estimates when IV is unavailable.
    """
    # Realised vol
    rv = float(returns.iloc[-21:].std() * np.sqrt(252)) if len(returns) >= 21 else 0.20

    # IV (use provided or estimate)
    if current_iv is None:
        iv_est = estimate_iv_from_returns(returns)
        current_iv = iv_est.get("iv_proxy", rv * 1.1)

    # IV rank and percentile
    if iv_history is not None and len(iv_history) > 20:
        iv_rank = compute_iv_rank(current_iv, iv_history)
        iv_pctile = compute_iv_percentile(current_iv, iv_history)
    else:
        # Estimate from RV history as proxy
        rv_series = realised_volatility(returns, window=21)
        rv_clean = rv_series.dropna()
        iv_rank = compute_iv_rank(current_iv, rv_clean) if len(rv_clean) > 20 else 0.5
        iv_pctile = compute_iv_percentile(current_iv, rv_clean) if len(rv_clean) > 20 else 0.5

    vrp = variance_risk_premium(current_iv, rv)

    # Skew proxy (using returns asymmetry)
    skew = 0.0
    if len(returns) >= 63:
        down_vol = returns[returns < 0].iloc[-63:].std() * np.sqrt(252) if (returns < 0).sum() > 10 else rv
        up_vol = returns[returns > 0].iloc[-63:].std() * np.sqrt(252) if (returns > 0).sum() > 10 else rv
        skew = float(down_vol - up_vol)  # positive = downside vol > upside

    return VolSurfaceSnapshot(
        underlying=underlying,
        atm_iv=current_iv,
        iv_rank=iv_rank,
        iv_percentile=iv_pctile,
        realised_vol=rv,
        variance_risk_premium=vrp,
        put_call_skew=skew,
        term_structure_slope=0.0,  # requires multi-expiry data
    )
