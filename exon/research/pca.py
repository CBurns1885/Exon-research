"""Principal Component Analysis for crypto return decomposition.

Decomposes the return covariance structure into orthogonal factors,
identifying the dominant drivers (market beta, sector rotation, etc.)
and isolating idiosyncratic alpha signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    """Container for PCA decomposition results."""

    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    components: pd.DataFrame  # (n_components, n_assets) — factor loadings
    factors: pd.DataFrame  # (n_periods, n_components) — factor returns
    residuals: pd.DataFrame  # (n_periods, n_assets) — idiosyncratic returns


def decompose_returns(
    returns: pd.DataFrame,
    n_components: int | None = None,
    variance_threshold: float = 0.90,
) -> PCAResult:
    """Decompose asset returns into principal component factors.

    Parameters
    ----------
    returns : DataFrame of asset returns (rows=time, cols=assets)
    n_components : fixed number of components; if None, use variance_threshold
    variance_threshold : cumulative variance to retain (used when n_components is None)
    """
    clean = returns.dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)

    # Determine number of components
    if n_components is None:
        pca_full = PCA().fit(scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cum_var, variance_threshold) + 1)
        n_components = min(n_components, min(scaled.shape) - 1)

    pca = PCA(n_components=n_components)
    factor_values = pca.fit_transform(scaled)

    components_df = pd.DataFrame(
        pca.components_,
        columns=clean.columns,
        index=[f"PC{i+1}" for i in range(n_components)],
    )
    factors_df = pd.DataFrame(
        factor_values,
        index=clean.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    # Residuals = returns minus explained portion
    reconstructed = pd.DataFrame(
        scaler.inverse_transform(pca.inverse_transform(factor_values)),
        index=clean.index,
        columns=clean.columns,
    )
    residuals_df = clean - reconstructed

    return PCAResult(
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
        components=components_df,
        factors=factors_df,
        residuals=residuals_df,
    )


def rolling_pca(
    returns: pd.DataFrame,
    window: int = 720,
    n_components: int = 3,
    step: int = 24,
) -> dict[str, list]:
    """Rolling PCA to observe how factor structure evolves over time.

    Returns dict with timestamps and corresponding explained variance ratios.
    """
    timestamps = []
    variances = []
    top_loadings = []

    for end in range(window, len(returns), step):
        chunk = returns.iloc[end - window : end]
        try:
            result = decompose_returns(chunk, n_components=n_components)
            timestamps.append(chunk.index[-1])
            variances.append(result.explained_variance_ratio.tolist())
            # Top loading asset for PC1
            top_loadings.append(result.components.iloc[0].abs().idxmax())
        except Exception:
            continue

    return {
        "timestamps": timestamps,
        "explained_variances": variances,
        "pc1_dominant_asset": top_loadings,
    }


def residual_momentum(
    returns: pd.DataFrame,
    n_components: int = 3,
    lookback: int = 168,
) -> pd.Series:
    """Compute residual (idiosyncratic) momentum after stripping factor exposure.

    This is a key Renaissance-style signal: momentum in the residual
    after removing systematic factor moves.
    """
    result = decompose_returns(returns, n_components=n_components)
    # Cumulative residual returns over lookback
    return result.residuals.iloc[-lookback:].sum()
