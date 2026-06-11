"""Wavelet momentum decomposition strategy.

Traditional momentum uses a single lookback window, which is arbitrary.
The Discrete Wavelet Transform (DWT) solves this by decomposing returns
into frequency bands — think of it as a principled multi-resolution
momentum analysis:

    Level 1 (D1): ~2-4 hour cycles   (noise / microstructure)
    Level 2 (D2): ~4-8 hour cycles   (intraday patterns)
    Level 3 (D3): ~8-16 hour cycles  (day-trading horizon)
    Level 4 (D4): ~16-32 hour cycles (swing trading horizon)
    Level 5 (D5): ~32-64 hour cycles (weekly structure)
    Approx  (A5): >64 hour trend     (macro trend)

Each level captures momentum at a different timescale. We compute a
momentum signal at each scale, weight them (with the macro trend
getting the most weight), and combine into a composite.

This uses the Haar wavelet (simplest, most robust to noise) implemented
from scratch — no external wavelet library needed.

The key insight: momentum at the weekly scale (D4-D5) has the best
Sharpe in crypto, while hourly momentum (D1-D2) is pure noise.
By decomposing, we can suppress noise and amplify signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


def haar_wavelet_decompose(signal: np.ndarray, max_level: int = 5) -> dict[str, np.ndarray]:
    """Discrete Wavelet Transform using the Haar wavelet.

    Returns dict mapping level names to coefficient arrays:
    - "D1", "D2", ..., "D{max_level}": detail coefficients (oscillations)
    - "A{max_level}": approximation (low-frequency trend)
    """
    coefficients = {}
    current = signal.copy().astype(float)

    for level in range(1, max_level + 1):
        n = len(current)
        if n < 2:
            break

        # Ensure even length
        if n % 2 != 0:
            current = current[:-1]
            n = n - 1

        half = n // 2
        approx = np.zeros(half)
        detail = np.zeros(half)

        for i in range(half):
            approx[i] = (current[2 * i] + current[2 * i + 1]) / np.sqrt(2)
            detail[i] = (current[2 * i] - current[2 * i + 1]) / np.sqrt(2)

        coefficients[f"D{level}"] = detail
        current = approx

    coefficients[f"A{max_level}"] = current
    return coefficients


def wavelet_reconstruct_level(
    coefficients: dict[str, np.ndarray],
    level: str,
    original_length: int,
) -> np.ndarray:
    """Reconstruct the signal contribution from a single wavelet level.

    Upsamples the coefficients back to the original signal length.
    """
    coeff = coefficients[level]
    # Simple upsampling: repeat each coefficient to match original length
    factor = original_length // len(coeff)
    return np.repeat(coeff, factor)[:original_length]


class WaveletMomentum(Strategy):
    """Multi-scale momentum via discrete wavelet decomposition.

    Decomposes returns into frequency bands, computes momentum at each
    scale, and produces a composite signal weighted toward the macro trend.
    """

    name = "wavelet_momentum"

    def __init__(
        self,
        max_level: int = 5,
        lookback: int = 256,
        scale_weights: list[float] | None = None,
        trend_weight: float = 0.35,
        vol_target: float = 0.15,
    ):
        """
        Parameters
        ----------
        max_level : number of wavelet decomposition levels
        lookback : number of bars for wavelet analysis (should be power of 2 ish)
        scale_weights : weight for D1...D{max_level} (detail scales).
                       Default escalates: low weight for high-freq noise,
                       high weight for low-freq signal.
        trend_weight : weight for the approximation (macro trend) component
        vol_target : annualised vol target
        """
        self.max_level = max_level
        self.lookback = lookback
        self.trend_weight = trend_weight
        self.vol_target = vol_target

        if scale_weights is not None:
            self.scale_weights = scale_weights
        else:
            # Default: exponentially increasing weight toward lower frequencies
            # D1 (noise) gets very little, D5 (weekly) gets a lot
            raw = [2 ** i for i in range(max_level)]
            total = sum(raw) + self.trend_weight * sum(raw) / raw[-1]
            self.scale_weights = [w / total * (1 - self.trend_weight) for w in raw]

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        returns = np.log(data / data.shift(1)).dropna()
        timestamp = data.index[-1]

        if len(returns) < self.lookback:
            return []

        signals = []

        for asset in data.columns:
            ret = returns[asset].dropna()
            if len(ret) < self.lookback:
                continue

            window = ret.iloc[-self.lookback:].values

            # Wavelet decomposition
            coeffs = haar_wavelet_decompose(window, max_level=self.max_level)

            # Momentum signal at each scale
            scale_signals = []
            for level in range(1, self.max_level + 1):
                key = f"D{level}"
                if key not in coeffs or len(coeffs[key]) == 0:
                    scale_signals.append(0.0)
                    continue

                detail = coeffs[key]
                # Momentum at this scale: sum of recent coefficients
                # Use last quarter of coefficients at this scale
                recent = detail[-max(len(detail) // 4, 1):]
                scale_mom = recent.mean()

                # Normalise by scale's standard deviation
                scale_std = detail.std()
                if scale_std > 0:
                    scale_signals.append(scale_mom / scale_std)
                else:
                    scale_signals.append(0.0)

            # Macro trend from approximation
            approx_key = f"A{self.max_level}"
            trend_signal = 0.0
            if approx_key in coeffs and len(coeffs[approx_key]) > 1:
                approx = coeffs[approx_key]
                # Trend: normalised slope of approximation coefficients
                if len(approx) >= 2:
                    slope = (approx[-1] - approx[0])
                    trend_signal = slope / approx.std() if approx.std() > 0 else 0

            # Composite: weighted combination of scale signals + trend
            composite = 0.0
            for i, ss in enumerate(scale_signals):
                if i < len(self.scale_weights):
                    composite += self.scale_weights[i] * ss
            composite += self.trend_weight * trend_signal

            direction = np.sign(composite)
            if direction == 0:
                continue

            # Strength: the composite is already a weighted z-score blend,
            # so values > 0.3 are meaningful. Vol-scale for position sizing.
            vol = ret.rolling(72).std().iloc[-1]
            annual_vol = vol * np.sqrt(8760)
            vol_scale = min(self.vol_target / annual_vol, 2.0) if annual_vol > 0 else 0

            raw_strength = min(abs(composite), 2.0) / 2.0
            strength = min(raw_strength + raw_strength * vol_scale, 1.0)

            if strength > 0.05:
                # Decompose for metadata: which scales agree/disagree
                agreeing_scales = sum(
                    1 for ss in scale_signals if np.sign(ss) == direction
                )
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=direction,
                        strength=strength,
                        metadata={
                            "composite": composite,
                            "trend_signal": trend_signal,
                            "scale_signals": {
                                f"D{i+1}": round(ss, 4) for i, ss in enumerate(scale_signals)
                            },
                            "agreeing_scales": agreeing_scales,
                            "total_scales": len(scale_signals),
                        },
                    )
                )

        return signals
