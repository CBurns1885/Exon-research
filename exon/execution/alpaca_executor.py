"""Execution engine for Alpaca Markets — equities paper/live trading.

Translates portfolio weight targets into orders, manages order lifecycle,
and tracks fills. Uses Alpaca's paper trading sandbox by default.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    qty: float  # number of shares
    limit_price: float | None = None  # None = market order
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    avg_price: float
    status: str
    fees: float
    raw: dict = field(default_factory=dict)


class AlpacaExecutionEngine:
    """Manages order execution against Alpaca Markets.

    Supports:
    - Paper trading (sandbox) and live trading
    - Market and limit orders
    - Position reconciliation via Alpaca positions API
    - Dry-run simulation mode (no API calls)
    """

    def __init__(
        self,
        client,  # AlpacaClient
        dry_run: bool = True,
        max_order_value_usd: float = 50_000.0,
    ):
        self.client = client
        self.dry_run = dry_run
        self.max_order_value_usd = max_order_value_usd
        self.order_history: list[OrderResult] = []

    def execute_target_weights(
        self,
        target_weights: pd.Series,
        portfolio_value: float,
        current_prices: dict[str, float],
        current_positions: dict[str, float],
    ) -> list[OrderResult]:
        """Execute trades to reach target portfolio weights.

        Parameters
        ----------
        target_weights : desired weight per asset (e.g., {"AAPL": 0.10})
        portfolio_value : total portfolio value in USD
        current_prices : current market prices per asset
        current_positions : current holdings in shares
        """
        orders = self._compute_orders(
            target_weights, portfolio_value, current_prices, current_positions
        )
        results = []
        for order in orders:
            result = self._execute_order(order, current_prices)
            results.append(result)
            self.order_history.append(result)
        return results

    def _compute_orders(
        self,
        target_weights: pd.Series,
        portfolio_value: float,
        current_prices: dict[str, float],
        current_positions: dict[str, float],
    ) -> list[OrderRequest]:
        orders = []
        all_assets = set(target_weights.index) | set(current_positions.keys())

        for asset in all_assets:
            target_w = target_weights.get(asset, 0.0)
            price = current_prices.get(asset)
            if price is None or price <= 0:
                continue

            target_value = portfolio_value * target_w
            current_qty = current_positions.get(asset, 0.0)
            current_value = current_qty * price
            delta_value = target_value - current_value

            # Skip small trades (under $50 for equities)
            if abs(delta_value) < 50:
                continue

            # Enforce single-order value limit
            if abs(delta_value) > self.max_order_value_usd:
                logger.warning(
                    "Order for %s ($%.0f) exceeds max ($%.0f), capping",
                    asset, abs(delta_value), self.max_order_value_usd,
                )
                delta_value = self.max_order_value_usd * (1 if delta_value > 0 else -1)

            qty = abs(delta_value / price)
            # Round to whole shares for equities (Alpaca supports fractional but safer this way)
            qty = max(int(qty), 1) if qty >= 0.5 else 0
            if qty == 0:
                continue

            side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL
            orders.append(OrderRequest(symbol=asset, side=side, qty=float(qty)))

        return orders

    def _execute_order(
        self, order: OrderRequest, current_prices: dict[str, float]
    ) -> OrderResult:
        if self.dry_run:
            return self._simulate_order(order, current_prices)

        try:
            if order.limit_price is not None:
                raw = self.client.submit_limit_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=order.side.value,
                    limit_price=order.limit_price,
                )
            else:
                raw = self.client.submit_market_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=order.side.value,
                )

            return OrderResult(
                order_id=raw.get("order_id", order.client_order_id),
                symbol=order.symbol,
                side=order.side.value,
                qty=order.qty,
                filled_qty=float(raw.get("filled_qty", 0)),
                avg_price=float(raw.get("filled_avg_price", 0)),
                status=raw.get("status", "SUBMITTED"),
                fees=0,  # Alpaca is commission-free
                raw=raw,
            )
        except Exception as e:
            logger.error("Order execution failed for %s: %s", order.symbol, e)
            return OrderResult(
                order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side.value,
                qty=order.qty,
                filled_qty=0,
                avg_price=0,
                status="FAILED",
                fees=0,
                raw={"error": str(e)},
            )

    def _simulate_order(
        self, order: OrderRequest, current_prices: dict[str, float]
    ) -> OrderResult:
        """Simulate order execution for dry-run mode."""
        price = current_prices.get(order.symbol, 0)
        slippage = price * 0.0002  # 2 bps for equities (tighter than crypto)
        fill_price = price + slippage if order.side == OrderSide.BUY else price - slippage

        logger.info(
            "[DRY RUN] %s %s %.0f shares @ $%.2f",
            order.side.value,
            order.symbol,
            order.qty,
            fill_price,
        )

        return OrderResult(
            order_id=f"sim-{order.client_order_id[:8]}",
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            filled_qty=order.qty,
            avg_price=fill_price,
            status="FILLED_SIM",
            fees=0,  # commission-free
        )

    def get_current_positions(self) -> dict[str, float]:
        """Fetch current positions from Alpaca."""
        if self.dry_run:
            return {}
        positions = self.client.get_positions()
        return {sym: info["qty"] for sym, info in positions.items()}

    def get_current_prices(self, assets: list[str]) -> dict[str, float]:
        """Fetch current prices from Alpaca positions or latest bars."""
        if self.dry_run:
            return {}
        try:
            # Use latest bar data
            bars = self.client.get_bars(
                assets,
                start=(pd.Timestamp.now("UTC") - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                timeframe="1Day",
            )
            if bars.empty:
                return {}
            prices = {}
            if isinstance(bars.index, pd.MultiIndex):
                for sym in assets:
                    try:
                        prices[sym] = float(bars.loc[sym]["close"].iloc[-1])
                    except (KeyError, IndexError):
                        pass
            return prices
        except Exception:
            logger.warning("Failed to get current prices")
            return {}

    def get_portfolio_value(self) -> float:
        """Get total portfolio value from Alpaca account."""
        if self.dry_run:
            return 0
        acct = self.client.get_account()
        return acct.get("portfolio_value", 0)
