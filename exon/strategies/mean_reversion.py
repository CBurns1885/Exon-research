"""Advanced mean-reversion strategies.

Includes:
- Bollinger Band mean reversion with dynamic widths
- Ornstein-Uhlenbeck calibrated reversion
- Multi-timeframe mean reversion
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class BollingerMeanReversion(Strategy):
    """Mean reversion using Bollinger Bands with adaptive width.

    Enters counter-trend when price deviates significantly from
    moving average, with position sizing proportional to deviation.
    """

    name = "bollinger_mr"

    def __init__(
        self,
        window: int = 48,
        entry_std: float = 2.0,
        exit_std: float = 0.5,
        vol_lookback: int = 168,
    ):
        self.window = window
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.vol_lookback = vol_lookback

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        timestamp = data.index[-1]
        signals = []

        for asset in data.columns:
            prices = data[asset].dropna()
            if len(prices) < self.vol_lookback:
                continue

            ma = prices.rolling(self.window).mean()
            std = prices.rolling(self.window).std()

            current_price = prices.iloc[-1]
            current_ma = ma.iloc[-1]
            current_std = std.iloc[-1]

            if current_std == 0:
                continue

            z = (current_price - current_ma) / current_std

            # Adaptive band width based on recent volatility
            recent_vol = prices.pct_change().rolling(self.vol_lookback).std().iloc[-1]
            historical_vol = prices.pct_change().std()
            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0

            adaptive_entry = self.entry_std * max(0.8, min(vol_ratio, 1.5))

            if abs(z) > adaptive_entry:
                direction = -np.sign(z)  # counter-trend
                # Strength proportional to how far beyond the band we are
                overshoot = (abs(z) - adaptive_entry) / adaptive_entry
                strength = min(0.3 + overshoot * 0.5, 1.0)

                signals.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=direction,
                        strength=strength,
                        metadata={"z_score": z, "adaptive_entry": adaptive_entry},
                    )
                )

        return signals


class OUMeanReversion(Strategy):
    """Mean reversion calibrated to an Ornstein-Uhlenbeck process.

    Estimates the OU parameters (theta, mu, sigma) from recent data
    and trades based on expected reversion speed.
    """

    name = "ou_mean_reversion"

    def __init__(
        self,
        calibration_window: int = 720,
        min_half_life: float = 2.0,
        max_half_life: float = 168.0,
        entry_threshold: float = 1.5,
    ):
        self.calibration_window = calibration_window
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.entry_threshold = entry_threshold

    def _estimate_ou_params(self, prices: pd.Series) -> dict:
        """Estimate OU parameters via OLS on discrete approximation.

        dX = theta * (mu - X) * dt + sigma * dW
        """
        log_prices = np.log(prices)
        x = log_prices.values
        dx = np.diff(x)
        x_lag = x[:-1]

        # OLS: dx = a + b * x_lag
        n = len(dx)
        sx = x_lag.sum()
        sy = dx.sum()
        sxx = (x_lag**2).sum()
        sxy = (x_lag * dx).sum()

        b = (n * sxy - sx * sy) / (n * sxx - sx**2) if (n * sxx - sx**2) != 0 else 0
        a = (sy - b * sx) / n

        theta = -b  # mean reversion speed
        mu = a / theta if theta != 0 else x.mean()
        residuals = dx - a - b * x_lag
        sigma = residuals.std()

        half_life = np.log(2) / theta if theta > 0 else float("inf")

        return {
            "theta": theta,
            "mu": mu,
            "sigma": sigma,
            "half_life": half_life,
        }

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        timestamp = data.index[-1]
        signals = []

        for asset in data.columns:
            prices = data[asset].dropna()
            if len(prices) < self.calibration_window:
                continue

            window = prices.iloc[-self.calibration_window :]
            params = self._estimate_ou_params(window)

            hl = params["half_life"]
            if not (self.min_half_life <= hl <= self.max_half_life):
                continue  # not a good mean-reversion candidate

            # Current deviation from equilibrium
            current = np.log(prices.iloc[-1])
            deviation = (current - params["mu"]) / params["sigma"]

            if abs(deviation) < self.entry_threshold:
                continue

            direction = -np.sign(deviation)
            # Strength: combination of deviation size and reversion speed
            speed_score = 1.0 - (hl - self.min_half_life) / (
                self.max_half_life - self.min_half_life
            )
            dev_score = min((abs(deviation) - self.entry_threshold) / 2.0, 1.0)
            strength = 0.5 * speed_score + 0.5 * dev_score

            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=asset,
                    direction=direction,
                    strength=min(strength, 1.0),
                    metadata={
                        "half_life": hl,
                        "theta": params["theta"],
                        "deviation": deviation,
                    },
                )
            )

        return signals


class MultiTimeframeMeanReversion(Strategy):
    """Mean reversion across multiple timeframes.

    Combines z-scores from different lookback windows, only trading
    when multiple timeframes agree on the reversion signal.
    """

    name = "mtf_mean_reversion"

    def __init__(
        self,
        windows: list[int] | None = None,
        threshold: float = 1.5,
        min_agreement: int = 3,
    ):
        self.windows = windows or [12, 24, 48, 96, 168]
        self.threshold = threshold
        self.min_agreement = min_agreement

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        timestamp = data.index[-1]
        signals = []

        for asset in data.columns:
            prices = data[asset].dropna()
            if len(prices) < max(self.windows) * 2:
                continue

            z_scores = []
            for w in self.windows:
                ma = prices.rolling(w).mean().iloc[-1]
                std = prices.rolling(w).std().iloc[-1]
                if std > 0:
                    z = (prices.iloc[-1] - ma) / std
                    z_scores.append(z)

            if len(z_scores) < self.min_agreement:
                continue

            # Count how many timeframes show extreme deviation
            extreme = [z for z in z_scores if abs(z) > self.threshold]
            if len(extreme) < self.min_agreement:
                continue

            # All extreme z-scores should agree on direction
            directions = [np.sign(z) for z in extreme]
            if len(set(directions)) > 1:
                continue  # disagreement

            direction = -directions[0]  # counter-trend
            avg_z = np.mean([abs(z) for z in extreme])
            strength = min((avg_z - self.threshold) / 2.0, 1.0) * (
                len(extreme) / len(self.windows)
            )

            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=asset,
                    direction=direction,
                    strength=min(strength, 1.0),
                    metadata={
                        "z_scores": dict(zip(self.windows, z_scores)),
                        "n_extreme": len(extreme),
                    },
                )
            )

        return signals
