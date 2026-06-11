"""Pairs trading and statistical arbitrage strategies.

Implements cointegration-based pairs trading with dynamic hedge ratios
and multi-leg statistical arbitrage using PCA residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .base import Signal, Strategy
from ..research.cointegration import engle_granger_test, half_life, z_score


class PairsTrading(Strategy):
    """Classic pairs trading on cointegrated assets.

    Trades the spread between two cointegrated assets when it deviates
    significantly from equilibrium, expecting mean reversion.
    """

    name = "pairs_trading"

    def __init__(
        self,
        asset_y: str = "",
        asset_x: str = "",
        lookback: int = 720,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.0,
        hedge_ratio_window: int = 168,
    ):
        self.asset_y = asset_y
        self.asset_x = asset_x
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.hedge_ratio_window = hedge_ratio_window

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        if self.asset_y not in data.columns or self.asset_x not in data.columns:
            return []

        y = data[self.asset_y].dropna()
        x = data[self.asset_x].dropna()
        common = y.index.intersection(x.index)
        y, x = y.loc[common], x.loc[common]

        if len(y) < self.lookback:
            return []

        # Rolling hedge ratio
        y_win = y.iloc[-self.hedge_ratio_window :]
        x_win = x.iloc[-self.hedge_ratio_window :]
        model = OLS(y_win, add_constant(x_win)).fit()
        hr = model.params.iloc[1]
        intercept = model.params.iloc[0]

        # Compute spread and z-score
        spread = y - hr * x - intercept
        z = z_score(spread, window=self.lookback // 4)
        current_z = z.iloc[-1]

        if np.isnan(current_z):
            return []

        timestamp = data.index[-1]
        signals = []

        if abs(current_z) > self.z_stop:
            # Stop-loss: close position
            return []

        if abs(current_z) > self.z_entry:
            # Enter: short the spread if z > entry, long if z < -entry
            direction_y = -np.sign(current_z)
            direction_x = np.sign(current_z)
            strength = min((abs(current_z) - self.z_entry) / (self.z_stop - self.z_entry), 1.0)

            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=self.asset_y,
                    direction=direction_y,
                    strength=strength,
                    metadata={"z": current_z, "hedge_ratio": hr, "role": "y"},
                )
            )
            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=self.asset_x,
                    direction=direction_x,
                    strength=strength * abs(hr),
                    metadata={"z": current_z, "hedge_ratio": hr, "role": "x"},
                )
            )

        return signals


class StatArbPCA(Strategy):
    """Statistical arbitrage using PCA residuals.

    Decomposes the return space into factors, then trades on the
    residual (idiosyncratic) component when it deviates from zero —
    the core Renaissance Technologies approach.
    """

    name = "stat_arb_pca"

    def __init__(
        self,
        n_factors: int = 3,
        lookback: int = 720,
        z_entry: float = 1.5,
        z_exit: float = 0.3,
        residual_window: int = 168,
    ):
        self.n_factors = n_factors
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.residual_window = residual_window

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        from ..research.pca import decompose_returns

        returns = np.log(data / data.shift(1)).dropna()
        if len(returns) < self.lookback:
            return []

        window = returns.iloc[-self.lookback :]

        try:
            pca_result = decompose_returns(window, n_components=self.n_factors)
        except Exception:
            return []

        timestamp = data.index[-1]
        signals = []

        for asset in data.columns:
            if asset not in pca_result.residuals.columns:
                continue

            resid = pca_result.residuals[asset]
            if len(resid) < self.residual_window:
                continue

            # Z-score of cumulative residual
            cum_resid = resid.rolling(self.residual_window).sum()
            mu = cum_resid.mean()
            sigma = cum_resid.std()
            if sigma == 0:
                continue
            z = (cum_resid.iloc[-1] - mu) / sigma

            if abs(z) > self.z_entry:
                direction = -np.sign(z)  # mean revert the residual
                strength = min((abs(z) - self.z_entry) / 3.0, 1.0)

                signals.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=direction,
                        strength=strength,
                        metadata={
                            "residual_z": z,
                            "cum_residual": cum_resid.iloc[-1],
                            "factor_exposure": pca_result.components[asset].tolist(),
                        },
                    )
                )

        return signals
