"""Black-Scholes pricing, Greeks, and implied volatility utilities.

Provides a pure-Python options pricing engine so strategies can
compute theoretical values, Greeks, and IV without external dependencies.
Also used for backtesting where live greeks aren't available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Option Greeks for a single contract."""

    delta: float
    gamma: float
    theta: float  # per day
    vega: float  # per 1% IV move
    rho: float

    @property
    def as_dict(self) -> dict:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }


@dataclass
class OptionPrice:
    """Complete pricing output for a single option."""

    theoretical_price: float
    intrinsic_value: float
    time_value: float
    greeks: Greeks
    option_type: OptionType
    spot: float
    strike: float
    tte: float  # time to expiry in years
    iv: float
    risk_free_rate: float


def black_scholes_price(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    r: float = 0.05,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Black-Scholes option price.

    Parameters
    ----------
    spot : current underlying price
    strike : option strike price
    tte : time to expiry in years
    vol : annualised implied volatility (e.g. 0.30 for 30%)
    r : risk-free rate
    option_type : CALL or PUT
    """
    if tte <= 0:
        if option_type == OptionType.CALL:
            return max(spot - strike, 0)
        return max(strike - spot, 0)

    d1 = (math.log(spot / strike) + (r + 0.5 * vol**2) * tte) / (vol * math.sqrt(tte))
    d2 = d1 - vol * math.sqrt(tte)

    if option_type == OptionType.CALL:
        return spot * norm.cdf(d1) - strike * math.exp(-r * tte) * norm.cdf(d2)
    else:
        return strike * math.exp(-r * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def compute_greeks(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    r: float = 0.05,
    option_type: OptionType = OptionType.CALL,
) -> Greeks:
    """Compute all Greeks for a single option."""
    if tte <= 1e-10:
        # At expiry
        itm = (spot > strike) if option_type == OptionType.CALL else (spot < strike)
        return Greeks(
            delta=1.0 if itm else 0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
        )

    sqrt_t = math.sqrt(tte)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol**2) * tte) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    # Gamma (same for call and put)
    gamma = pdf_d1 / (spot * vol * sqrt_t)

    # Vega (same for call and put, per 1% IV move)
    vega = spot * pdf_d1 * sqrt_t / 100.0

    if option_type == OptionType.CALL:
        delta = cdf_d1
        theta = (
            -spot * pdf_d1 * vol / (2 * sqrt_t)
            - r * strike * math.exp(-r * tte) * cdf_d2
        ) / 365.0  # per day
        rho = strike * tte * math.exp(-r * tte) * cdf_d2 / 100.0
    else:
        delta = cdf_d1 - 1.0
        theta = (
            -spot * pdf_d1 * vol / (2 * sqrt_t)
            + r * strike * math.exp(-r * tte) * norm.cdf(-d2)
        ) / 365.0
        rho = -strike * tte * math.exp(-r * tte) * norm.cdf(-d2) / 100.0

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def full_price(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    r: float = 0.05,
    option_type: OptionType = OptionType.CALL,
) -> OptionPrice:
    """Compute price + all greeks in one call."""
    price = black_scholes_price(spot, strike, tte, vol, r, option_type)
    greeks = compute_greeks(spot, strike, tte, vol, r, option_type)

    if option_type == OptionType.CALL:
        intrinsic = max(spot - strike, 0)
    else:
        intrinsic = max(strike - spot, 0)

    return OptionPrice(
        theoretical_price=price,
        intrinsic_value=intrinsic,
        time_value=price - intrinsic,
        greeks=greeks,
        option_type=option_type,
        spot=spot,
        strike=strike,
        tte=tte,
        iv=vol,
        risk_free_rate=r,
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    tte: float,
    r: float = 0.05,
    option_type: OptionType = OptionType.CALL,
    tol: float = 1e-6,
) -> float:
    """Compute implied volatility from market price using Brent's method.

    Returns IV as a decimal (e.g. 0.30 for 30%).
    Returns NaN if no solution found.
    """
    if tte <= 0:
        return float("nan")

    # Intrinsic value check
    if option_type == OptionType.CALL:
        intrinsic = max(spot - strike * math.exp(-r * tte), 0)
    else:
        intrinsic = max(strike * math.exp(-r * tte) - spot, 0)

    if market_price < intrinsic - tol:
        return float("nan")

    def objective(vol):
        return black_scholes_price(spot, strike, tte, vol, r, option_type) - market_price

    try:
        iv = brentq(objective, 0.001, 10.0, xtol=tol)
        return iv
    except (ValueError, RuntimeError):
        return float("nan")


def put_call_parity_check(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    tte: float,
    r: float = 0.05,
    tolerance: float = 0.05,
) -> dict:
    """Check put-call parity: C - P = S - K*exp(-rT).

    Returns dict with parity values and whether it holds within tolerance.
    """
    theoretical_diff = spot - strike * math.exp(-r * tte)
    actual_diff = call_price - put_price
    deviation = actual_diff - theoretical_diff

    return {
        "call_minus_put": actual_diff,
        "theoretical_diff": theoretical_diff,
        "deviation": deviation,
        "parity_holds": abs(deviation) <= tolerance * spot,
    }
