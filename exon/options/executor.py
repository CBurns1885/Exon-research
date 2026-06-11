"""Options execution engine — submits single-leg and multi-leg orders via Alpaca.

Handles:
- Contract resolution (map desired strikes/expiries to actual OCC symbols)
- Single-leg option orders
- Multi-leg (MLeg) orders (spreads, iron condors, straddles)
- Dry-run simulation with Black-Scholes pricing
- Position tracking and P&L
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import pandas as pd

from .strategy_mapper import OptionsTradeRecommendation, OptionsLeg, OptionsStructure

logger = logging.getLogger(__name__)


@dataclass
class OptionsOrderResult:
    """Result of an options order execution."""

    order_id: str
    underlying: str
    structure: str
    legs: list[dict]
    status: str
    net_premium: float  # positive = credit received, negative = debit paid
    max_loss: float
    max_gain: float
    raw: dict = field(default_factory=dict)


class OptionsExecutionEngine:
    """Executes options trades via Alpaca API.

    Supports:
    - Dry-run simulation with Black-Scholes pricing
    - Single-leg orders via submit_order
    - Multi-leg orders via MLEG order class
    - Contract resolution from desired strikes to actual symbols
    """

    def __init__(
        self,
        alpaca_client=None,
        chain_client=None,
        dry_run: bool = True,
        max_contracts: int = 10,
        max_premium_per_trade: float = 5_000.0,
    ):
        self.client = alpaca_client
        self.chain_client = chain_client
        self.dry_run = dry_run
        self.max_contracts = max_contracts
        self.max_premium_per_trade = max_premium_per_trade
        self.order_history: list[OptionsOrderResult] = []

    def execute_recommendation(
        self,
        rec: OptionsTradeRecommendation,
        qty: int = 1,
    ) -> OptionsOrderResult:
        """Execute an options trade recommendation.

        Parameters
        ----------
        rec : the trade recommendation from StrategyMapper
        qty : number of contracts per leg
        """
        qty = min(qty, self.max_contracts)

        if self.dry_run:
            return self._simulate_execution(rec, qty)

        # Resolve contract symbols
        resolved_legs = self._resolve_contracts(rec)
        if not resolved_legs:
            return OptionsOrderResult(
                order_id=str(uuid.uuid4())[:8],
                underlying=rec.underlying,
                structure=rec.structure.value,
                legs=[],
                status="FAILED_NO_CONTRACTS",
                net_premium=0,
                max_loss=rec.max_loss,
                max_gain=rec.max_gain,
            )

        # Submit order
        if len(resolved_legs) == 1:
            return self._submit_single_leg(rec, resolved_legs[0], qty)
        else:
            return self._submit_multi_leg(rec, resolved_legs, qty)

    def _simulate_execution(
        self, rec: OptionsTradeRecommendation, qty: int
    ) -> OptionsOrderResult:
        """Simulate options execution with estimated pricing."""
        from .pricing import black_scholes_price, OptionType

        spot = rec.metadata.get("spot_price", 100)
        iv = rec.metadata.get("current_iv", 0.25)
        tte = rec.target_dte / 365.0

        net_premium = 0.0
        leg_details = []

        for leg in rec.legs:
            opt_type = OptionType.CALL if leg.option_type == "call" else OptionType.PUT
            price = black_scholes_price(spot, leg.strike, tte, iv, 0.05, opt_type)

            if leg.side == "buy":
                net_premium -= price * 100 * qty
            else:
                net_premium += price * 100 * qty

            leg_details.append({
                "side": leg.side,
                "type": leg.option_type,
                "strike": leg.strike,
                "premium": round(price, 2),
                "qty": qty,
            })

        logger.info(
            "[DRY RUN] %s %s on %s | %d legs | net premium: $%.2f",
            rec.structure.value,
            rec.underlying,
            "simulated",
            len(rec.legs),
            net_premium,
        )

        result = OptionsOrderResult(
            order_id=f"sim-{uuid.uuid4().hex[:8]}",
            underlying=rec.underlying,
            structure=rec.structure.value,
            legs=leg_details,
            status="FILLED_SIM",
            net_premium=round(net_premium, 2),
            max_loss=rec.max_loss * qty,
            max_gain=rec.max_gain * qty if rec.max_gain < float("inf") else float("inf"),
        )
        self.order_history.append(result)
        return result

    def _resolve_contracts(
        self, rec: OptionsTradeRecommendation
    ) -> list[dict]:
        """Resolve recommendation legs to actual contract symbols."""
        if not self.chain_client:
            return []

        resolved = []
        spot = rec.metadata.get("spot_price", 0)

        for leg in rec.legs:
            contracts = self.chain_client.get_contracts(
                rec.underlying,
                option_type=leg.option_type,
                min_dte=rec.target_dte - 7,
                max_dte=rec.target_dte + 7,
                strike_range_pct=0.15,
                spot_price=spot,
            )

            if not contracts:
                logger.warning(
                    "No contracts found for %s %s strike=%.2f",
                    rec.underlying, leg.option_type, leg.strike,
                )
                return []

            # Find closest strike
            best = min(contracts, key=lambda c: abs(c.strike - leg.strike))
            resolved.append({
                "symbol": best.symbol,
                "side": leg.side,
                "strike": best.strike,
                "expiration": best.expiration,
                "option_type": leg.option_type,
            })

        return resolved

    def _submit_single_leg(
        self, rec: OptionsTradeRecommendation, leg: dict, qty: int
    ) -> OptionsOrderResult:
        """Submit a single-leg option order."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        try:
            order_data = MarketOrderRequest(
                symbol=leg["symbol"],
                qty=qty,
                side=OrderSide.BUY if leg["side"] == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.trading_client.submit_order(order_data=order_data)

            result = OptionsOrderResult(
                order_id=str(order.id),
                underlying=rec.underlying,
                structure=rec.structure.value,
                legs=[leg],
                status=str(order.status),
                net_premium=0,
                max_loss=rec.max_loss * qty,
                max_gain=rec.max_gain * qty if rec.max_gain < float("inf") else float("inf"),
                raw={"order_id": str(order.id), "status": str(order.status)},
            )
            self.order_history.append(result)
            return result
        except Exception as e:
            logger.error("Single-leg order failed: %s", e)
            return OptionsOrderResult(
                order_id=str(uuid.uuid4())[:8],
                underlying=rec.underlying,
                structure=rec.structure.value,
                legs=[leg],
                status="FAILED",
                net_premium=0,
                max_loss=rec.max_loss,
                max_gain=rec.max_gain,
                raw={"error": str(e)},
            )

    def _submit_multi_leg(
        self, rec: OptionsTradeRecommendation, legs: list[dict], qty: int
    ) -> OptionsOrderResult:
        """Submit a multi-leg (MLEG) option order."""
        from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce

        try:
            order_legs = []
            for leg in legs:
                order_legs.append(OptionLegRequest(
                    symbol=leg["symbol"],
                    side=OrderSide.BUY if leg["side"] == "buy" else OrderSide.SELL,
                    ratio_qty=1,
                ))

            order_data = MarketOrderRequest(
                qty=qty,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=order_legs,
            )
            order = self.client.trading_client.submit_order(order_data=order_data)

            result = OptionsOrderResult(
                order_id=str(order.id),
                underlying=rec.underlying,
                structure=rec.structure.value,
                legs=legs,
                status=str(order.status),
                net_premium=0,
                max_loss=rec.max_loss * qty,
                max_gain=rec.max_gain * qty if rec.max_gain < float("inf") else float("inf"),
                raw={"order_id": str(order.id), "status": str(order.status)},
            )
            self.order_history.append(result)
            return result
        except Exception as e:
            logger.error("Multi-leg order failed: %s", e)
            return OptionsOrderResult(
                order_id=str(uuid.uuid4())[:8],
                underlying=rec.underlying,
                structure=rec.structure.value,
                legs=legs,
                status="FAILED",
                net_premium=0,
                max_loss=rec.max_loss,
                max_gain=rec.max_gain,
                raw={"error": str(e)},
            )

    def get_portfolio_greeks(self) -> dict:
        """Summarise portfolio-level Greeks from open positions."""
        # In production, would fetch live positions and compute aggregate Greeks
        return {
            "net_delta": 0.0,
            "net_gamma": 0.0,
            "net_theta": 0.0,
            "net_vega": 0.0,
            "total_positions": len(self.order_history),
        }
