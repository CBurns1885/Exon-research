"""Coinbase Advanced Trade API client for market data and order execution.

Supports both public (unauthenticated) endpoints for market data
and authenticated endpoints for trading via API key + secret.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any

import requests

BASE_URL = "https://api.coinbase.com"
ADV_TRADE = "/api/v3/brokerage"


@dataclass
class CoinbaseAuth:
    """Holds API credentials for authenticated requests."""

    api_key: str
    api_secret: str

    def sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = f"{timestamp}{method.upper()}{path}{body}"
        return hmac.new(
            self.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

    def headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = str(int(time.time()))
        sig = self.sign(ts, method, path, body)
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": sig,
            "CB-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }


@dataclass
class CoinbaseClient:
    """Wrapper around Coinbase Advanced Trade API.

    Can operate in unauthenticated mode (market data only) or
    authenticated mode (trading + market data).
    """

    auth: CoinbaseAuth | None = None
    session: requests.Session = field(default_factory=requests.Session)
    rate_limit_pause: float = 0.1  # seconds between requests

    # -- low-level ---------------------------------------------------------

    def _request(
        self, method: str, path: str, params: dict | None = None, json_body: dict | None = None
    ) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        body = ""
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if json_body:
            import json as _json
            body = _json.dumps(json_body)

        if self.auth:
            headers.update(self.auth.headers(method, path, body))

        time.sleep(self.rate_limit_pause)

        resp = self.session.request(
            method, url, headers=headers, params=params, data=body if body else None
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def _post(self, path: str, json_body: dict | None = None) -> dict[str, Any]:
        return self._request("POST", path, json_body=json_body)

    # -- public market data ------------------------------------------------

    def list_products(self, product_type: str = "SPOT") -> list[dict]:
        """List all available trading pairs."""
        data = self._get(f"{ADV_TRADE}/products", {"product_type": product_type})
        return data.get("products", [])

    def get_product(self, product_id: str) -> dict:
        """Get details for a single product (e.g. 'BTC-USD')."""
        return self._get(f"{ADV_TRADE}/products/{product_id}")

    def get_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_HOUR",
    ) -> list[dict]:
        """Fetch OHLCV candles for a product.

        Parameters
        ----------
        product_id : e.g. "BTC-USD"
        start, end : UNIX timestamps (seconds)
        granularity : ONE_MINUTE | FIVE_MINUTE | FIFTEEN_MINUTE |
                      THIRTY_MINUTE | ONE_HOUR | TWO_HOUR | SIX_HOUR |
                      ONE_DAY
        """
        params = {"start": str(start), "end": str(end), "granularity": granularity}
        data = self._get(f"{ADV_TRADE}/products/{product_id}/candles", params)
        return data.get("candles", [])

    def get_ticker(self, product_id: str) -> dict:
        """Get current best bid/ask and last trade."""
        return self._get(f"{ADV_TRADE}/products/{product_id}/ticker")

    # -- authenticated: accounts -------------------------------------------

    def list_accounts(self) -> list[dict]:
        data = self._get(f"{ADV_TRADE}/accounts")
        return data.get("accounts", [])

    # -- authenticated: orders ---------------------------------------------

    def create_order(self, order_config: dict) -> dict:
        """Submit an order. See Coinbase docs for order_configuration schema."""
        return self._post(f"{ADV_TRADE}/orders", order_config)

    def cancel_orders(self, order_ids: list[str]) -> dict:
        return self._post(f"{ADV_TRADE}/orders/batch_cancel", {"order_ids": order_ids})

    def list_orders(self, **filters) -> list[dict]:
        data = self._get(f"{ADV_TRADE}/orders/historical/batch", filters)
        return data.get("orders", [])

    def get_order(self, order_id: str) -> dict:
        return self._get(f"{ADV_TRADE}/orders/historical/{order_id}")
