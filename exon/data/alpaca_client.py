"""Alpaca Markets API client for equities market data and order execution.

Uses the official alpaca-py SDK in paper trading (sandbox) mode by default.
Supports both historical data fetching and live trading via the Alpaca API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlpacaConfig:
    """Holds Alpaca API credentials and settings."""

    api_key: str = ""
    api_secret: str = ""
    paper: bool = True  # Use paper trading sandbox
    data_feed: str = "iex"  # "iex" (free) or "sip" (paid, all exchanges)


class AlpacaClient:
    """Wrapper around the Alpaca alpaca-py SDK.

    Provides a unified interface for:
    - Historical stock bar data (OHLCV)
    - Account information and positions
    - Order submission and management
    """

    def __init__(self, config: AlpacaConfig | None = None):
        self.config = config or AlpacaConfig()
        self._trading_client = None
        self._data_client = None

    @property
    def trading_client(self):
        """Lazy-init the TradingClient."""
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient

            self._trading_client = TradingClient(
                self.config.api_key,
                self.config.api_secret,
                paper=self.config.paper,
            )
        return self._trading_client

    @property
    def data_client(self):
        """Lazy-init the StockHistoricalDataClient."""
        if self._data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._data_client = StockHistoricalDataClient(
                self.config.api_key,
                self.config.api_secret,
            )
        return self._data_client

    # -- Market Data -------------------------------------------------------

    def get_bars(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars for one or more symbols.

        Parameters
        ----------
        symbols : list of ticker symbols, e.g. ["AAPL", "MSFT"]
        start : start date (ISO format or datetime)
        end : end date (ISO format or datetime); defaults to now
        timeframe : "1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"

        Returns
        -------
        DataFrame with MultiIndex (symbol, timestamp) and columns
        [open, high, low, close, volume, vwap, trade_count]
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=pd.Timestamp(start).isoformat() if isinstance(start, str) else start.isoformat(),
            end=pd.Timestamp(end).isoformat() if end else None,
        )

        bars = self.data_client.get_stock_bars(request_params)
        return bars.df

    # -- Account & Positions -----------------------------------------------

    def get_account(self) -> dict[str, Any]:
        """Get account information (equity, buying power, etc.)."""
        acct = self.trading_client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "account_number": acct.account_number,
        }

    def get_positions(self) -> dict[str, dict]:
        """Get all open positions."""
        positions = self.trading_client.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "side": pos.side,
            }
        return result

    # -- Orders ------------------------------------------------------------

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Submit a market order."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
        }

        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=tif_map.get(time_in_force, TimeInForce.DAY),
        )
        order = self.trading_client.submit_order(order_data=order_data)
        return self._order_to_dict(order)

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Submit a limit order."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
        }

        order_data = LimitOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            limit_price=limit_price,
            time_in_force=tif_map.get(time_in_force, TimeInForce.DAY),
        )
        order = self.trading_client.submit_order(order_data=order_data)
        return self._order_to_dict(order)

    def cancel_order(self, order_id: str):
        """Cancel an order by ID."""
        self.trading_client.cancel_order_by_id(order_id)

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.trading_client.cancel_orders()

    def list_assets(self, asset_class: str = "us_equity", status: str = "active") -> list[dict]:
        """List tradeable assets."""
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus

        cls_map = {"us_equity": AssetClass.US_EQUITY}
        status_map = {"active": AssetStatus.ACTIVE}

        params = GetAssetsRequest(
            asset_class=cls_map.get(asset_class, AssetClass.US_EQUITY),
            status=status_map.get(status, AssetStatus.ACTIVE),
        )
        assets = self.trading_client.get_all_assets(params)
        return [
            {"symbol": a.symbol, "name": a.name, "exchange": a.exchange, "tradable": a.tradable}
            for a in assets[:100]  # limit for performance
        ]

    @staticmethod
    def _order_to_dict(order) -> dict[str, Any]:
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "side": str(order.side),
            "qty": float(order.qty) if order.qty else 0,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0,
            "status": str(order.status),
            "type": str(order.type),
            "time_in_force": str(order.time_in_force),
        }
