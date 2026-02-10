"""Live execution engine for Coinbase Advanced Trade API.

Translates portfolio weight targets into orders, manages order lifecycle,
and tracks fills. Includes safety checks and dry-run mode.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ..data.coinbase_client import CoinbaseClient

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderRequest:
    product_id: str
    side: OrderSide
    size: float  # base currency quantity
    limit_price: float | None = None  # None = market order
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class OrderResult:
    order_id: str
    product_id: str
    side: str
    size: float
    filled_size: float
    avg_price: float
    status: str
    fees: float
    raw: dict = field(default_factory=dict)


class ExecutionEngine:
    """Manages order execution against Coinbase.

    Supports:
    - Dry-run mode (paper trading)
    - Market and limit orders
    - TWAP execution for larger orders
    - Position reconciliation
    """

    def __init__(
        self,
        client: CoinbaseClient,
        dry_run: bool = True,
        max_order_value_usd: float = 10_000.0,
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
        target_weights : desired weight per asset (e.g., {"BTC-USD": 0.3})
        portfolio_value : total portfolio value in USD
        current_prices : current market prices per asset
        current_positions : current holdings in base currency units
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
        """Compute the orders needed to transition to target weights."""
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

            # Skip small trades (under $10)
            if abs(delta_value) < 10:
                continue

            # Enforce single-order value limit
            if abs(delta_value) > self.max_order_value_usd:
                logger.warning(
                    "Order for %s ($%.0f) exceeds max ($%.0f), capping",
                    asset, abs(delta_value), self.max_order_value_usd,
                )
                delta_value = self.max_order_value_usd * (1 if delta_value > 0 else -1)

            size = abs(delta_value / price)
            side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL

            orders.append(OrderRequest(product_id=asset, side=side, size=size))

        return orders

    def _execute_order(
        self, order: OrderRequest, current_prices: dict[str, float]
    ) -> OrderResult:
        """Execute a single order (live or dry-run)."""
        if self.dry_run:
            return self._simulate_order(order, current_prices)

        order_config = {
            "client_order_id": order.client_order_id,
            "product_id": order.product_id,
            "side": order.side.value,
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": f"{order.size:.8f}",
                }
            },
        }

        if order.limit_price is not None:
            order_config["order_configuration"] = {
                "limit_limit_gtc": {
                    "base_size": f"{order.size:.8f}",
                    "limit_price": f"{order.limit_price:.2f}",
                    "post_only": True,
                }
            }

        try:
            raw = self.client.create_order(order_config)
            order_data = raw.get("success_response", raw)

            # Poll for fill
            fill = self._wait_for_fill(order_data.get("order_id", ""), timeout=30)

            return OrderResult(
                order_id=order_data.get("order_id", order.client_order_id),
                product_id=order.product_id,
                side=order.side.value,
                size=order.size,
                filled_size=float(fill.get("filled_size", 0)),
                avg_price=float(fill.get("average_filled_price", 0)),
                status=fill.get("status", "UNKNOWN"),
                fees=float(fill.get("total_fees", 0)),
                raw=raw,
            )
        except Exception as e:
            logger.error("Order execution failed for %s: %s", order.product_id, e)
            return OrderResult(
                order_id=order.client_order_id,
                product_id=order.product_id,
                side=order.side.value,
                size=order.size,
                filled_size=0,
                avg_price=0,
                status="FAILED",
                fees=0,
                raw={"error": str(e)},
            )

    def _simulate_order(
        self, order: OrderRequest, current_prices: dict[str, float]
    ) -> OrderResult:
        """Simulate order execution for dry-run mode."""
        price = current_prices.get(order.product_id, 0)
        slippage = price * 0.0005  # 5 bps simulated slippage
        fill_price = price + slippage if order.side == OrderSide.BUY else price - slippage
        fees = order.size * fill_price * 0.004  # simulated maker fee

        logger.info(
            "[DRY RUN] %s %s %.6f @ $%.2f (fees: $%.2f)",
            order.side.value,
            order.product_id,
            order.size,
            fill_price,
            fees,
        )

        return OrderResult(
            order_id=f"sim-{order.client_order_id[:8]}",
            product_id=order.product_id,
            side=order.side.value,
            size=order.size,
            filled_size=order.size,
            avg_price=fill_price,
            status="FILLED_SIM",
            fees=fees,
        )

    def _wait_for_fill(self, order_id: str, timeout: int = 30) -> dict:
        """Poll order status until filled or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                order = self.client.get_order(order_id)
                status = order.get("order", {}).get("status", "")
                if status in ("FILLED", "CANCELLED", "EXPIRED", "FAILED"):
                    return order.get("order", {})
            except Exception:
                pass
            time.sleep(1)
        return {"status": "TIMEOUT"}

    def get_current_positions(self) -> dict[str, float]:
        """Fetch current account balances from Coinbase."""
        if self.dry_run:
            return {}

        accounts = self.client.list_accounts()
        positions = {}
        for acct in accounts:
            currency = acct.get("currency", "")
            balance = float(acct.get("available_balance", {}).get("value", 0))
            if balance > 0 and currency != "USD":
                positions[f"{currency}-USD"] = balance
        return positions

    def get_current_prices(self, assets: list[str]) -> dict[str, float]:
        """Fetch current prices for a list of assets."""
        prices = {}
        for asset in assets:
            try:
                product = self.client.get_product(asset)
                prices[asset] = float(product.get("price", 0))
            except Exception:
                logger.warning("Failed to get price for %s", asset)
        return prices
