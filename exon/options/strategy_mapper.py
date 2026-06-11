"""Options strategy mapper — translates equity signals into options structures.

The core insight: our existing strategies produce direction + strength signals.
This module maps those signals onto the optimal options structure based on:

1. Signal direction & conviction → structure type
2. Current IV regime → premium vs. direction bias
3. Time horizon → expiry selection
4. Risk budget → position sizing via Greeks

Signal-to-Structure Mapping:
┌────────────────────┬──────────────┬───────────────────────┐
│ Signal Profile     │ IV Regime    │ Options Structure     │
├────────────────────┼──────────────┼───────────────────────┤
│ Strong directional │ Low IV       │ Long call/put         │
│ Strong directional │ High IV      │ Debit spread          │
│ Weak directional   │ High IV      │ Credit spread         │
│ Conflicting/None   │ High IV      │ Short iron condor     │
│ Conflicting/None   │ Low IV       │ Long straddle/strangle│
│ Mean-reversion     │ Any          │ Short strangle        │
│ Breakout           │ Low IV       │ Long straddle         │
└────────────────────┴──────────────┴───────────────────────┘
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..strategies.base import Signal


class OptionsStructure(Enum):
    """Supported options trade structures."""

    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_PUT_SPREAD = "bull_put_spread"  # credit spread
    BEAR_CALL_SPREAD = "bear_call_spread"  # credit spread
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"


@dataclass
class OptionsLeg:
    """A single leg of an options trade."""

    symbol: str  # OCC contract symbol
    side: str  # "buy" or "sell"
    ratio: int = 1
    option_type: str = "call"  # "call" or "put"
    strike: float = 0.0
    expiration: str = ""


@dataclass
class OptionsTradeRecommendation:
    """A recommended options trade derived from an equity signal."""

    underlying: str
    structure: OptionsStructure
    legs: list[OptionsLeg] = field(default_factory=list)
    direction: float = 0.0  # net delta direction
    max_loss: float = 0.0  # maximum loss per contract
    max_gain: float = 0.0  # maximum gain (inf for naked)
    target_dte: int = 30
    conviction: float = 0.0  # 0-1, from original signal strength
    metadata: dict = field(default_factory=dict)

    @property
    def is_defined_risk(self) -> bool:
        return self.max_loss > 0 and self.max_loss < float("inf")

    @property
    def risk_reward_ratio(self) -> float:
        if self.max_loss > 0 and self.max_gain > 0:
            return self.max_gain / self.max_loss
        return 0.0


class IVRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class StrategyMapperConfig:
    """Configuration for signal-to-options mapping."""

    # IV percentile thresholds
    iv_low_threshold: float = 0.30   # below 30th percentile = low IV
    iv_high_threshold: float = 0.70  # above 70th percentile = high IV

    # Signal strength thresholds
    strong_signal_threshold: float = 0.6
    weak_signal_threshold: float = 0.25

    # Default DTE targets by structure type
    directional_dte: int = 30
    spread_dte: int = 45
    premium_dte: int = 30
    straddle_dte: int = 21

    # Strike selection
    otm_delta_target: float = 0.30
    spread_width_pct: float = 0.05  # 5% between spread strikes

    # Position sizing
    max_risk_per_trade_pct: float = 0.02  # 2% of portfolio
    max_contracts: int = 10


class StrategyMapper:
    """Maps equity trading signals to options trade structures."""

    def __init__(self, config: StrategyMapperConfig | None = None):
        self.config = config or StrategyMapperConfig()

    def map_signal(
        self,
        signal: Signal,
        spot_price: float,
        current_iv: float,
        iv_percentile: float,
        historical_vol: float,
        portfolio_value: float = 100_000,
    ) -> OptionsTradeRecommendation:
        """Map an equity signal to an options trade recommendation.

        Parameters
        ----------
        signal : the equity trading signal
        spot_price : current underlying price
        current_iv : current ATM implied volatility
        iv_percentile : IV rank (0-1) relative to history
        historical_vol : realised volatility
        portfolio_value : for position sizing
        """
        iv_regime = self._classify_iv(iv_percentile)
        structure = self._select_structure(signal, iv_regime)
        dte = self._select_dte(structure)
        legs = self._build_legs(structure, signal, spot_price, dte)

        # Estimate max loss/gain
        max_loss, max_gain = self._estimate_risk_reward(
            structure, spot_price, legs
        )

        return OptionsTradeRecommendation(
            underlying=signal.asset,
            structure=structure,
            legs=legs,
            direction=signal.direction,
            max_loss=max_loss,
            max_gain=max_gain,
            target_dte=dte,
            conviction=signal.strength,
            metadata={
                "iv_regime": iv_regime.value,
                "iv_percentile": iv_percentile,
                "current_iv": current_iv,
                "historical_vol": historical_vol,
                "signal_direction": signal.direction,
                "signal_strength": signal.strength,
                "spot_price": spot_price,
            },
        )

    def map_signals(
        self,
        signals: list[Signal],
        spot_prices: dict[str, float],
        ivs: dict[str, float],
        iv_percentiles: dict[str, float],
        hvols: dict[str, float],
        portfolio_value: float = 100_000,
    ) -> list[OptionsTradeRecommendation]:
        """Map multiple signals to options recommendations."""
        recommendations = []
        for sig in signals:
            asset = sig.asset
            if asset not in spot_prices:
                continue
            rec = self.map_signal(
                sig,
                spot_prices[asset],
                ivs.get(asset, 0.25),
                iv_percentiles.get(asset, 0.50),
                hvols.get(asset, 0.20),
                portfolio_value,
            )
            recommendations.append(rec)
        return recommendations

    def _classify_iv(self, iv_percentile: float) -> IVRegime:
        if iv_percentile < self.config.iv_low_threshold:
            return IVRegime.LOW
        elif iv_percentile > self.config.iv_high_threshold:
            return IVRegime.HIGH
        return IVRegime.NORMAL

    def _select_structure(
        self, signal: Signal, iv_regime: IVRegime
    ) -> OptionsStructure:
        """Core mapping logic: signal profile + IV regime → structure."""
        strength = signal.strength
        direction = signal.direction
        cfg = self.config

        is_strong = strength >= cfg.strong_signal_threshold
        is_weak = strength < cfg.weak_signal_threshold

        # Check signal metadata for strategy type hints
        strategy_type = signal.metadata.get("strategy_type", "")

        # Breakout strategies → straddle in low IV (expecting vol expansion)
        if "breakout" in strategy_type and iv_regime == IVRegime.LOW:
            return OptionsStructure.LONG_STRADDLE

        # Mean-reversion strategies → sell premium
        if "mean_reversion" in strategy_type and iv_regime == IVRegime.HIGH:
            return OptionsStructure.SHORT_STRANGLE

        # Strong directional signal
        if is_strong:
            if iv_regime == IVRegime.LOW:
                # Cheap options → buy outright
                return (
                    OptionsStructure.LONG_CALL
                    if direction > 0
                    else OptionsStructure.LONG_PUT
                )
            elif iv_regime == IVRegime.HIGH:
                # Expensive options → debit spread (cap cost)
                return (
                    OptionsStructure.BULL_CALL_SPREAD
                    if direction > 0
                    else OptionsStructure.BEAR_PUT_SPREAD
                )
            else:
                # Normal IV → debit spread for defined risk
                return (
                    OptionsStructure.BULL_CALL_SPREAD
                    if direction > 0
                    else OptionsStructure.BEAR_PUT_SPREAD
                )

        # Weak/no directional signal
        if is_weak:
            if iv_regime == IVRegime.HIGH:
                # Sell premium: iron condor or short strangle
                return OptionsStructure.IRON_CONDOR
            elif iv_regime == IVRegime.LOW:
                # Buy vol: long straddle/strangle
                return OptionsStructure.LONG_STRANGLE
            else:
                return OptionsStructure.IRON_CONDOR

        # Medium conviction → credit spread (collect premium, directional lean)
        if iv_regime in (IVRegime.HIGH, IVRegime.NORMAL):
            return (
                OptionsStructure.BULL_PUT_SPREAD
                if direction > 0
                else OptionsStructure.BEAR_CALL_SPREAD
            )

        # Default: directional debit spread
        return (
            OptionsStructure.BULL_CALL_SPREAD
            if direction > 0
            else OptionsStructure.BEAR_PUT_SPREAD
        )

    def _select_dte(self, structure: OptionsStructure) -> int:
        cfg = self.config
        if structure in (OptionsStructure.LONG_CALL, OptionsStructure.LONG_PUT):
            return cfg.directional_dte
        elif structure in (
            OptionsStructure.BULL_CALL_SPREAD,
            OptionsStructure.BEAR_PUT_SPREAD,
            OptionsStructure.BULL_PUT_SPREAD,
            OptionsStructure.BEAR_CALL_SPREAD,
        ):
            return cfg.spread_dte
        elif structure in (
            OptionsStructure.LONG_STRADDLE,
            OptionsStructure.LONG_STRANGLE,
        ):
            return cfg.straddle_dte
        else:
            return cfg.premium_dte

    def _build_legs(
        self,
        structure: OptionsStructure,
        signal: Signal,
        spot: float,
        dte: int,
    ) -> list[OptionsLeg]:
        """Build placeholder legs (symbols filled in during execution)."""
        cfg = self.config
        width = spot * cfg.spread_width_pct
        otm_offset = spot * cfg.otm_delta_target * 0.5
        exp = ""  # filled during execution

        if structure == OptionsStructure.LONG_CALL:
            return [OptionsLeg(symbol="", side="buy", option_type="call",
                               strike=round(spot, 2), expiration=exp)]

        elif structure == OptionsStructure.LONG_PUT:
            return [OptionsLeg(symbol="", side="buy", option_type="put",
                               strike=round(spot, 2), expiration=exp)]

        elif structure == OptionsStructure.BULL_CALL_SPREAD:
            return [
                OptionsLeg(symbol="", side="buy", option_type="call",
                           strike=round(spot, 2), expiration=exp),
                OptionsLeg(symbol="", side="sell", option_type="call",
                           strike=round(spot + width, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.BEAR_PUT_SPREAD:
            return [
                OptionsLeg(symbol="", side="buy", option_type="put",
                           strike=round(spot, 2), expiration=exp),
                OptionsLeg(symbol="", side="sell", option_type="put",
                           strike=round(spot - width, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.BULL_PUT_SPREAD:
            return [
                OptionsLeg(symbol="", side="sell", option_type="put",
                           strike=round(spot - otm_offset * 0.5, 2), expiration=exp),
                OptionsLeg(symbol="", side="buy", option_type="put",
                           strike=round(spot - otm_offset * 0.5 - width, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.BEAR_CALL_SPREAD:
            return [
                OptionsLeg(symbol="", side="sell", option_type="call",
                           strike=round(spot + otm_offset * 0.5, 2), expiration=exp),
                OptionsLeg(symbol="", side="buy", option_type="call",
                           strike=round(spot + otm_offset * 0.5 + width, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.LONG_STRADDLE:
            return [
                OptionsLeg(symbol="", side="buy", option_type="call",
                           strike=round(spot, 2), expiration=exp),
                OptionsLeg(symbol="", side="buy", option_type="put",
                           strike=round(spot, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.SHORT_STRADDLE:
            return [
                OptionsLeg(symbol="", side="sell", option_type="call",
                           strike=round(spot, 2), expiration=exp),
                OptionsLeg(symbol="", side="sell", option_type="put",
                           strike=round(spot, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.LONG_STRANGLE:
            return [
                OptionsLeg(symbol="", side="buy", option_type="call",
                           strike=round(spot + otm_offset, 2), expiration=exp),
                OptionsLeg(symbol="", side="buy", option_type="put",
                           strike=round(spot - otm_offset, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.SHORT_STRANGLE:
            return [
                OptionsLeg(symbol="", side="sell", option_type="call",
                           strike=round(spot + otm_offset, 2), expiration=exp),
                OptionsLeg(symbol="", side="sell", option_type="put",
                           strike=round(spot - otm_offset, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.IRON_CONDOR:
            return [
                OptionsLeg(symbol="", side="sell", option_type="put",
                           strike=round(spot - otm_offset, 2), expiration=exp),
                OptionsLeg(symbol="", side="buy", option_type="put",
                           strike=round(spot - otm_offset - width, 2), expiration=exp),
                OptionsLeg(symbol="", side="sell", option_type="call",
                           strike=round(spot + otm_offset, 2), expiration=exp),
                OptionsLeg(symbol="", side="buy", option_type="call",
                           strike=round(spot + otm_offset + width, 2), expiration=exp),
            ]

        elif structure == OptionsStructure.IRON_BUTTERFLY:
            return [
                OptionsLeg(symbol="", side="sell", option_type="put",
                           strike=round(spot, 2), expiration=exp),
                OptionsLeg(symbol="", side="buy", option_type="put",
                           strike=round(spot - width, 2), expiration=exp),
                OptionsLeg(symbol="", side="sell", option_type="call",
                           strike=round(spot, 2), expiration=exp),
                OptionsLeg(symbol="", side="buy", option_type="call",
                           strike=round(spot + width, 2), expiration=exp),
            ]

        return []

    def _estimate_risk_reward(
        self,
        structure: OptionsStructure,
        spot: float,
        legs: list[OptionsLeg],
    ) -> tuple[float, float]:
        """Estimate max loss and max gain for a structure.

        Returns (max_loss, max_gain) per contract in dollars.
        Uses structure geometry; actual values computed during execution.
        """
        width = spot * self.config.spread_width_pct
        premium_est = spot * 0.03  # rough 3% ATM premium estimate

        if structure in (OptionsStructure.LONG_CALL, OptionsStructure.LONG_PUT):
            return (premium_est * 100, float("inf"))

        elif structure in (
            OptionsStructure.BULL_CALL_SPREAD,
            OptionsStructure.BEAR_PUT_SPREAD,
        ):
            debit = premium_est * 0.6 * 100
            max_gain = width * 100 - debit
            return (debit, max_gain)

        elif structure in (
            OptionsStructure.BULL_PUT_SPREAD,
            OptionsStructure.BEAR_CALL_SPREAD,
        ):
            credit = premium_est * 0.4 * 100
            max_loss = width * 100 - credit
            return (max_loss, credit)

        elif structure == OptionsStructure.LONG_STRADDLE:
            return (premium_est * 2 * 100, float("inf"))

        elif structure == OptionsStructure.SHORT_STRADDLE:
            return (float("inf"), premium_est * 2 * 100)

        elif structure == OptionsStructure.LONG_STRANGLE:
            return (premium_est * 1.5 * 100, float("inf"))

        elif structure == OptionsStructure.SHORT_STRANGLE:
            return (float("inf"), premium_est * 1.5 * 100)

        elif structure == OptionsStructure.IRON_CONDOR:
            credit = premium_est * 0.5 * 100
            max_loss = width * 100 - credit
            return (max_loss, credit)

        elif structure == OptionsStructure.IRON_BUTTERFLY:
            credit = premium_est * 1.5 * 100
            max_loss = width * 100 - credit
            return (max_loss, credit)

        return (0, 0)
