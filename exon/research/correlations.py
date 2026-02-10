"""Correlation analysis for crypto asset universe.

Provides rolling correlations, correlation clustering, stability analysis,
and correlation regime detection — core tools for identifying exploitable
cross-asset relationships.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute full correlation matrix."""
    return returns.corr(method=method)


def rolling_correlation(
    returns: pd.DataFrame,
    pair: tuple[str, str],
    window: int = 168,
) -> pd.Series:
    """Rolling pairwise correlation between two assets."""
    return returns[pair[0]].rolling(window).corr(returns[pair[1]])


def rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int = 168,
) -> dict[str, pd.Series]:
    """Rolling correlations for all unique pairs."""
    cols = returns.columns.tolist()
    result = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            result[f"{a}__{b}"] = rolling_correlation(returns, (a, b), window)
    return result


def correlation_stability(
    returns: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Measure how stable correlations are across different lookback windows.

    Returns a DataFrame indexed by pair with columns for each window's
    average correlation and overall standard deviation across windows.
    """
    if windows is None:
        windows = [24, 72, 168, 336, 720]

    cols = returns.columns.tolist()
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    records = []
    for a, b in pairs:
        corrs = {}
        for w in windows:
            rc = returns[a].rolling(w).corr(returns[b]).dropna()
            corrs[f"corr_{w}"] = rc.mean()
        row = {"pair": f"{a}__{b}", **corrs}
        vals = list(corrs.values())
        row["stability"] = 1.0 - np.std(vals)  # higher = more stable
        records.append(row)
    return pd.DataFrame(records).set_index("pair").sort_values("stability", ascending=False)


def cluster_assets(
    returns: pd.DataFrame,
    n_clusters: int = 4,
    method: str = "ward",
) -> pd.Series:
    """Hierarchical clustering of assets based on correlation distance."""
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    dist_arr = dist.values.copy()
    np.fill_diagonal(dist_arr, 0)
    dist = pd.DataFrame(dist_arr, index=dist.index, columns=dist.columns)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    return pd.Series(labels, index=corr.columns, name="cluster")


def top_correlated_pairs(
    returns: pd.DataFrame,
    n: int = 10,
    min_periods: int = 100,
) -> pd.DataFrame:
    """Find the most correlated and anti-correlated pairs."""
    corr = returns.corr(min_periods=min_periods)
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            pairs.append({"asset_a": a, "asset_b": b, "correlation": corr.iloc[i, j]})
    df = pd.DataFrame(pairs).dropna()
    most_pos = df.nlargest(n, "correlation")
    most_neg = df.nsmallest(n, "correlation")
    return pd.concat([most_pos, most_neg]).reset_index(drop=True)
