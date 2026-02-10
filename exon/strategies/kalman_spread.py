"""Kalman Filter spread trading strategy.

Rolling OLS for hedge ratios is fragile: it lags structural breaks,
assigns equal weight to all observations in the window, and gives you
no measure of uncertainty. The Kalman filter solves all three problems.

The state-space model treats the hedge ratio as a hidden state that
evolves over time (random walk), and the observed spread as a noisy
measurement. The filter recursively updates:

    State:       beta_t = beta_{t-1} + w_t,   w ~ N(0, Q)
    Observation: y_t = beta_t * x_t + v_t,    v ~ N(0, R)

This gives us:
- An adaptive hedge ratio that responds instantly to regime changes
- A Kalman gain that tells us how much to trust new observations
- A prediction error (innovation) whose z-score IS the trading signal
- A forecast variance that naturally sizes our position by confidence

This is what stat arb desks at Citadel, Two Sigma, and DE Shaw
actually run. The innovation z-score is the alpha signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class KalmanSpreadTrading(Strategy):
    """Pairs trading with Kalman-filtered dynamic hedge ratios.

    The innovation (prediction error) of the Kalman filter IS the
    mean-reversion signal: when the actual spread diverges from the
    Kalman-predicted spread, we trade the reversion.
    """

    name = "kalman_spread"

    def __init__(
        self,
        asset_y: str = "",
        asset_x: str = "",
        delta: float = 1e-4,
        obs_noise: float = 1.0,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.5,
        min_kalman_gain_periods: int = 50,
    ):
        """
        Parameters
        ----------
        asset_y, asset_x : the two assets in the pair (y = dependent)
        delta : state transition covariance scalar — controls how fast
                the hedge ratio can move. Smaller = smoother, larger = more adaptive.
                1e-4 is a good default for hourly crypto.
        obs_noise : observation noise variance (R). Larger = trust the model more.
        z_entry : innovation z-score threshold to enter a trade
        z_exit : z-score threshold to close a trade
        z_stop : z-score threshold for stop-loss (model breakdown)
        min_kalman_gain_periods : warm-up before trading
        """
        self.asset_y = asset_y
        self.asset_x = asset_x
        self.delta = delta
        self.obs_noise = obs_noise
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.min_kalman_gain_periods = min_kalman_gain_periods

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        if self.asset_y not in data.columns or self.asset_x not in data.columns:
            return []

        y = data[self.asset_y].dropna()
        x = data[self.asset_x].dropna()
        common = y.index.intersection(x.index)
        y, x = y.loc[common], x.loc[common]

        if len(y) < self.min_kalman_gain_periods + 10:
            return []

        result = self._run_kalman_filter(y.values, x.values)

        # Current state
        current_z = result["innovation_z"][-1]
        current_beta = result["beta"][-1]
        current_alpha = result["alpha"][-1]
        current_P = result["P"][-1]

        if np.isnan(current_z) or len(result["innovation_z"]) < self.min_kalman_gain_periods:
            return []

        timestamp = data.index[-1]
        signals = []

        # Stop-loss: model might be broken
        if abs(current_z) > self.z_stop:
            return []

        if abs(current_z) > self.z_entry:
            # Trade the innovation reversion
            # If z > 0: spread is above prediction → short spread (short y, long x)
            # If z < 0: spread is below prediction → long spread (long y, short x)
            direction_y = -np.sign(current_z)
            direction_x = np.sign(current_z)

            # Strength: z-score magnitude, scaled by Kalman confidence
            # Lower P (state uncertainty) = higher confidence = larger position
            confidence = 1.0 / (1.0 + np.sqrt(np.trace(current_P)))
            z_strength = min((abs(current_z) - self.z_entry) / (self.z_stop - self.z_entry), 1.0)
            strength = z_strength * confidence

            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=self.asset_y,
                    direction=direction_y,
                    strength=min(strength, 1.0),
                    metadata={
                        "innovation_z": current_z,
                        "kalman_beta": current_beta,
                        "kalman_alpha": current_alpha,
                        "state_uncertainty": np.trace(current_P),
                        "role": "y",
                    },
                )
            )
            signals.append(
                Signal(
                    timestamp=timestamp,
                    asset=self.asset_x,
                    direction=direction_x,
                    strength=min(strength * abs(current_beta), 1.0),
                    metadata={
                        "innovation_z": current_z,
                        "kalman_beta": current_beta,
                        "role": "x",
                    },
                )
            )

        return signals

    def _run_kalman_filter(self, y: np.ndarray, x: np.ndarray) -> dict:
        """Run the Kalman filter on the full series.

        State vector: [alpha, beta]  (intercept and hedge ratio)
        Observation:  y_t = alpha + beta * x_t + noise

        Returns dict with beta, alpha, innovation_z, P history.
        """
        n = len(y)

        # State: [alpha, beta]
        theta = np.zeros(2)  # initial state estimate
        P = np.eye(2)  # initial state covariance

        # State transition noise covariance
        Q = self.delta * np.eye(2)
        # Observation noise variance
        R = self.obs_noise

        betas = np.zeros(n)
        alphas = np.zeros(n)
        innovations = np.zeros(n)
        innovation_vars = np.zeros(n)
        P_history = []

        for t in range(n):
            # Observation vector: [1, x_t]
            F = np.array([1.0, x[t]])

            # --- Predict ---
            # theta_{t|t-1} = theta_{t-1|t-1}  (random walk model)
            # P_{t|t-1} = P_{t-1|t-1} + Q
            P_pred = P + Q

            # --- Innovation ---
            y_pred = F @ theta
            innovation = y[t] - y_pred
            S = F @ P_pred @ F + R  # innovation variance

            # --- Update ---
            K = P_pred @ F / S  # Kalman gain
            theta = theta + K * innovation
            P = P_pred - np.outer(K, F) @ P_pred

            # Store
            betas[t] = theta[1]
            alphas[t] = theta[0]
            innovations[t] = innovation
            innovation_vars[t] = S
            P_history.append(P.copy())

        # Z-score of innovations
        innovation_z = innovations / np.sqrt(np.maximum(innovation_vars, 1e-10))

        return {
            "beta": betas,
            "alpha": alphas,
            "innovation": innovations,
            "innovation_var": innovation_vars,
            "innovation_z": innovation_z,
            "P": P_history,
        }


class KalmanSpreadScanner(Strategy):
    """Scans all pairs in a universe and trades the best Kalman spreads.

    Unlike KalmanSpreadTrading which takes a fixed pair, this strategy
    dynamically selects which pairs to trade based on Kalman filter
    diagnostics (innovation stationarity, state stability).
    """

    name = "kalman_scanner"

    def __init__(
        self,
        max_pairs: int = 5,
        delta: float = 1e-4,
        obs_noise: float = 1.0,
        z_entry: float = 2.0,
        z_stop: float = 4.5,
        min_history: int = 200,
    ):
        self.max_pairs = max_pairs
        self.delta = delta
        self.obs_noise = obs_noise
        self.z_entry = z_entry
        self.z_stop = z_stop
        self.min_history = min_history

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        import itertools

        cols = data.columns.tolist()
        if len(cols) < 2:
            return []

        # Score all pairs by Kalman filter quality
        pair_scores = []
        for a, b in itertools.combinations(cols, 2):
            y = data[a].dropna()
            x = data[b].dropna()
            common = y.index.intersection(x.index)
            if len(common) < self.min_history:
                continue

            kf = KalmanSpreadTrading(
                asset_y=a, asset_x=b,
                delta=self.delta, obs_noise=self.obs_noise,
                z_entry=self.z_entry, z_stop=self.z_stop,
            )
            result = kf._run_kalman_filter(y.loc[common].values, x.loc[common].values)

            # Score: we want pairs where innovations are stationary and
            # the current z-score is in the tradeable range
            z_series = result["innovation_z"][self.min_history:]
            if len(z_series) == 0:
                continue
            current_z = z_series[-1]
            # Stationarity proxy: std of rolling z should be stable
            z_std = np.std(z_series)
            # Tradeable: |z| > entry but < stop
            tradeable = self.z_entry < abs(current_z) < self.z_stop

            if tradeable and 0.5 < z_std < 3.0:
                pair_scores.append((abs(current_z), a, b, kf))

        # Take top N pairs by z-score magnitude
        pair_scores.sort(reverse=True)
        all_signals = []
        for _, a, b, kf in pair_scores[: self.max_pairs]:
            kf.asset_y = a
            kf.asset_x = b
            sigs = kf.generate_signals(data)
            all_signals.extend(sigs)

        return all_signals
