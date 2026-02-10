"""Market regime detection using Hidden Markov Models and statistical methods.

Identifies trending, mean-reverting, and volatile regimes to dynamically
adjust strategy allocation — a core concept in Renaissance-style systems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def detect_regimes_gmm(
    returns: pd.Series,
    n_regimes: int = 3,
    features: list[str] | None = None,
    vol_window: int = 24,
    mom_window: int = 48,
) -> pd.DataFrame:
    """Detect market regimes using Gaussian Mixture Models.

    Builds feature matrix from returns (volatility, momentum, return level)
    and clusters into regimes.

    Parameters
    ----------
    returns : Series of asset returns
    n_regimes : number of regimes to detect
    features : which features to use; defaults to all
    vol_window : rolling window for volatility feature
    mom_window : rolling window for momentum feature

    Returns
    -------
    DataFrame with columns: return, volatility, momentum, regime, regime_prob
    """
    df = pd.DataFrame({"return": returns})
    df["volatility"] = df["return"].rolling(vol_window).std()
    df["momentum"] = df["return"].rolling(mom_window).mean()
    df = df.dropna()

    if features is None:
        features = ["return", "volatility", "momentum"]

    X = df[features].values

    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=42)
    df["regime"] = gmm.fit_predict(X)

    probs = gmm.predict_proba(X)
    df["regime_prob"] = probs.max(axis=1)

    # Label regimes by volatility (0=low vol, highest=high vol)
    regime_vols = df.groupby("regime")["volatility"].mean().sort_values()
    label_map = {old: new for new, old in enumerate(regime_vols.index)}
    df["regime"] = df["regime"].map(label_map)

    return df


def regime_statistics(regime_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime statistics."""
    stats = regime_df.groupby("regime").agg(
        mean_return=("return", "mean"),
        std_return=("return", "std"),
        mean_vol=("volatility", "mean"),
        mean_momentum=("momentum", "mean"),
        count=("return", "count"),
        sharpe=("return", lambda x: x.mean() / x.std() * np.sqrt(8760) if x.std() > 0 else 0),
    )
    stats["pct_time"] = stats["count"] / stats["count"].sum()
    return stats


def rolling_regime(
    returns: pd.Series,
    window: int = 720,
    n_regimes: int = 3,
    step: int = 24,
) -> pd.Series:
    """Rolling regime detection — re-fit GMM on rolling windows."""
    regimes = pd.Series(index=returns.index, dtype=float)

    for end in range(window, len(returns), step):
        chunk = returns.iloc[end - window : end]
        try:
            result = detect_regimes_gmm(chunk, n_regimes=n_regimes)
            if not result.empty:
                regimes.iloc[end - 1] = result["regime"].iloc[-1]
        except Exception:
            continue

    return regimes.ffill()


def volatility_regime(
    returns: pd.Series,
    short_window: int = 24,
    long_window: int = 168,
) -> pd.Series:
    """Simple volatility-based regime indicator.

    Returns ratio of short-term to long-term volatility.
    >1 means volatility expansion, <1 means contraction.
    """
    short_vol = returns.rolling(short_window).std()
    long_vol = returns.rolling(long_window).std()
    return (short_vol / long_vol).dropna()


def trend_strength(
    prices: pd.Series,
    window: int = 168,
) -> pd.Series:
    """Measure trend strength using price efficiency ratio.

    Ratio of net price change to sum of absolute changes.
    Values near +/-1 indicate strong trend, near 0 indicates chop.
    """
    net_change = prices - prices.shift(window)
    abs_changes = prices.diff().abs().rolling(window).sum()
    return (net_change / abs_changes).dropna()
