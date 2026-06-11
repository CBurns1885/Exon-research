"""Portfolio optimisation: mean-variance, risk parity, and Kelly criterion.

Combines signals from multiple strategies into optimal portfolio weights
with constraints on concentration, turnover, and sector exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def mean_variance_optimize(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    max_weight: float = 0.20,
    long_only: bool = False,
) -> pd.Series:
    """Mean-variance optimisation with position limits.

    max(w'mu - (gamma/2) * w'Sigma*w)
    subject to sum(w) = 1, |w_i| <= max_weight
    """
    n = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values

    def objective(w):
        ret = w @ mu
        risk = w @ sigma @ w
        return -(ret - risk_aversion / 2 * risk)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if long_only:
        bounds = [(0, max_weight)] * n
    else:
        bounds = [(-max_weight, max_weight)] * n

    x0 = np.ones(n) / n
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    weights = pd.Series(result.x, index=expected_returns.index)
    return weights


def risk_parity(
    cov_matrix: pd.DataFrame,
    budget: pd.Series | None = None,
) -> pd.Series:
    """Risk parity: equalise risk contribution from each asset.

    Optionally accepts a risk budget (e.g., from signal strength).
    """
    n = len(cov_matrix)
    sigma = cov_matrix.values
    if budget is None:
        budget_arr = np.ones(n) / n
    else:
        budget_arr = budget.values / budget.values.sum()

    def risk_contribution(w):
        port_var = w @ sigma @ w
        marginal = sigma @ w
        rc = w * marginal / np.sqrt(port_var) if port_var > 0 else np.zeros(n)
        return rc

    def objective(w):
        rc = risk_contribution(w)
        rc_norm = rc / rc.sum() if rc.sum() > 0 else np.zeros(n)
        return np.sum((rc_norm - budget_arr) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01, 0.5)] * n
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=cov_matrix.columns)


def kelly_criterion(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    fraction: float = 0.25,  # fractional Kelly for safety
    max_weight: float = 0.25,
) -> pd.Series:
    """Kelly criterion portfolio weights.

    Full Kelly = Sigma^{-1} @ mu
    We use fractional Kelly (typically 1/4) for robustness.
    """
    try:
        sigma_inv = np.linalg.inv(cov_matrix.values)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(cov_matrix.values)

    full_kelly = sigma_inv @ expected_returns.values
    weights = full_kelly * fraction

    # Clip and normalise
    weights = np.clip(weights, -max_weight, max_weight)
    weights = weights / np.abs(weights).sum() if np.abs(weights).sum() > 0 else weights

    return pd.Series(weights, index=expected_returns.index)


def blend_signal_weights(
    signal_weights: dict[str, pd.Series],
    strategy_allocations: dict[str, float],
) -> pd.Series:
    """Blend portfolio weights from multiple strategies.

    Parameters
    ----------
    signal_weights : dict mapping strategy name -> weight Series
    strategy_allocations : dict mapping strategy name -> allocation fraction
    """
    combined = pd.Series(dtype=float)

    for name, weights in signal_weights.items():
        alloc = strategy_allocations.get(name, 0.0)
        scaled = weights * alloc
        combined = combined.add(scaled, fill_value=0.0)

    # Normalise
    total = combined.abs().sum()
    if total > 0:
        combined = combined / total

    return combined


def constrain_turnover(
    target_weights: pd.Series,
    current_weights: pd.Series,
    max_turnover: float = 0.30,
) -> pd.Series:
    """Limit portfolio turnover to reduce transaction costs."""
    diff = target_weights.subtract(current_weights, fill_value=0.0)
    total_turnover = diff.abs().sum()

    if total_turnover <= max_turnover:
        return target_weights

    # Scale down changes proportionally
    scale = max_turnover / total_turnover
    adjusted = current_weights.add(diff * scale, fill_value=0.0)

    return adjusted
