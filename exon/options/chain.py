"""Options chain data fetching and analysis via Alpaca API.

Provides a high-level interface over Alpaca's option chain endpoints
with contract filtering, strike selection, and expiry management.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Normalised option contract data."""

    symbol: str  # OCC symbol e.g. "SPY250321C00580000"
    underlying: str
    expiration: str  # ISO date
    strike: float
    option_type: str  # "call" or "put"
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last_price

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid > 0:
            return self.spread / mid
        return 0.0

    @property
    def days_to_expiry(self) -> int:
        exp = pd.Timestamp(self.expiration)
        now = pd.Timestamp.now("UTC").normalize()
        return max((exp - now).days, 0)

    @property
    def tte(self) -> float:
        """Time to expiry in years."""
        return self.days_to_expiry / 365.0


class OptionsChainClient:
    """High-level options chain interface built on top of AlpacaClient."""

    def __init__(self, alpaca_client):
        """
        Parameters
        ----------
        alpaca_client : AlpacaClient instance with configured API credentials
        """
        self.client = alpaca_client
        self._option_data_client = None

    @property
    def option_data_client(self):
        """Lazy-init the OptionHistoricalDataClient."""
        if self._option_data_client is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self._option_data_client = OptionHistoricalDataClient(
                self.client.config.api_key,
                self.client.config.api_secret,
            )
        return self._option_data_client

    def get_chain(
        self,
        underlying: str,
        expiration_min: str | None = None,
        expiration_max: str | None = None,
        strike_min: float | None = None,
        strike_max: float | None = None,
        option_type: str | None = None,
    ) -> list[OptionContract]:
        """Fetch option chain for an underlying symbol.

        Parameters
        ----------
        underlying : e.g. "SPY", "AAPL"
        expiration_min/max : date range for expirations
        strike_min/max : strike price range
        option_type : "call", "put", or None for both

        Returns
        -------
        List of OptionContract objects with greeks and quotes
        """
        from alpaca.data.requests import OptionChainRequest

        kwargs = {"underlying_symbol": underlying}
        if option_type:
            kwargs["type"] = option_type
        if strike_min is not None:
            kwargs["strike_price_gte"] = str(strike_min)
        if strike_max is not None:
            kwargs["strike_price_lte"] = str(strike_max)
        if expiration_min:
            kwargs["expiration_date_gte"] = expiration_min
        if expiration_max:
            kwargs["expiration_date_lte"] = expiration_max

        req = OptionChainRequest(**kwargs)
        chain = self.option_data_client.get_option_chain(req)

        contracts = []
        for symbol, snapshot in chain.items():
            contract = self._snapshot_to_contract(symbol, underlying, snapshot)
            if contract is not None:
                contracts.append(contract)

        return contracts

    def get_contracts(
        self,
        underlying: str,
        option_type: str = "call",
        min_dte: int = 7,
        max_dte: int = 60,
        strike_range_pct: float = 0.10,
        spot_price: float | None = None,
    ) -> list[OptionContract]:
        """Fetch filtered option contracts via the trading API.

        Parameters
        ----------
        underlying : ticker symbol
        option_type : "call" or "put"
        min_dte / max_dte : days to expiry range
        strike_range_pct : percentage around spot for strike range
        spot_price : current underlying price (fetched if not provided)
        """
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import AssetStatus, ContractType

        now = pd.Timestamp.now("UTC")
        exp_min = (now + pd.Timedelta(days=min_dte)).strftime("%Y-%m-%d")
        exp_max = (now + pd.Timedelta(days=max_dte)).strftime("%Y-%m-%d")

        ct = ContractType.CALL if option_type == "call" else ContractType.PUT

        kwargs = {
            "underlying_symbols": [underlying],
            "status": AssetStatus.ACTIVE,
            "type": ct,
            "expiration_date_gte": exp_min,
            "expiration_date_lte": exp_max,
        }

        if spot_price and strike_range_pct:
            kwargs["strike_price_gte"] = str(spot_price * (1 - strike_range_pct))
            kwargs["strike_price_lte"] = str(spot_price * (1 + strike_range_pct))

        req = GetOptionContractsRequest(**kwargs)
        result = self.client.trading_client.get_option_contracts(req)

        contracts = []
        for c in result.option_contracts or []:
            contracts.append(OptionContract(
                symbol=c.symbol,
                underlying=underlying,
                expiration=str(c.expiration_date),
                strike=float(c.strike_price),
                option_type=option_type,
            ))

        return contracts

    def find_atm_contracts(
        self,
        underlying: str,
        spot_price: float,
        dte_target: int = 30,
        dte_tolerance: int = 7,
    ) -> dict[str, OptionContract | None]:
        """Find the nearest ATM call and put contracts.

        Returns dict with keys "call" and "put".
        """
        calls = self.get_contracts(
            underlying, "call",
            min_dte=dte_target - dte_tolerance,
            max_dte=dte_target + dte_tolerance,
            strike_range_pct=0.03,
            spot_price=spot_price,
        )
        puts = self.get_contracts(
            underlying, "put",
            min_dte=dte_target - dte_tolerance,
            max_dte=dte_target + dte_tolerance,
            strike_range_pct=0.03,
            spot_price=spot_price,
        )

        atm_call = min(calls, key=lambda c: abs(c.strike - spot_price)) if calls else None
        atm_put = min(puts, key=lambda c: abs(c.strike - spot_price)) if puts else None

        return {"call": atm_call, "put": atm_put}

    def find_otm_contracts(
        self,
        underlying: str,
        spot_price: float,
        delta_target: float = 0.30,
        dte_target: int = 30,
        dte_tolerance: int = 7,
    ) -> dict[str, OptionContract | None]:
        """Find OTM contracts near a target delta.

        Uses strike distance as delta proxy when live greeks unavailable.
        """
        # For calls, OTM = strike > spot; for puts, OTM = strike < spot
        # delta_target of 0.30 ≈ ~1 std dev OTM
        from .pricing import compute_greeks, OptionType

        call_strike = spot_price * (1 + delta_target * 0.5)
        put_strike = spot_price * (1 - delta_target * 0.5)

        calls = self.get_contracts(
            underlying, "call",
            min_dte=dte_target - dte_tolerance,
            max_dte=dte_target + dte_tolerance,
            strike_range_pct=0.15,
            spot_price=spot_price,
        )
        puts = self.get_contracts(
            underlying, "put",
            min_dte=dte_target - dte_tolerance,
            max_dte=dte_target + dte_tolerance,
            strike_range_pct=0.15,
            spot_price=spot_price,
        )

        otm_call = min(calls, key=lambda c: abs(c.strike - call_strike)) if calls else None
        otm_put = min(puts, key=lambda c: abs(c.strike - put_strike)) if puts else None

        return {"call": otm_call, "put": otm_put}

    @staticmethod
    def _snapshot_to_contract(
        symbol: str, underlying: str, snapshot
    ) -> OptionContract | None:
        """Convert an Alpaca OptionsSnapshot to our OptionContract."""
        try:
            bid = ask = last = 0.0
            if snapshot.latest_quote:
                bid = float(snapshot.latest_quote.bid_price or 0)
                ask = float(snapshot.latest_quote.ask_price or 0)
            if snapshot.latest_trade:
                last = float(snapshot.latest_trade.price or 0)

            iv = float(snapshot.implied_volatility or 0)
            delta = gamma = theta = vega = 0.0
            if snapshot.greeks:
                delta = float(snapshot.greeks.delta or 0)
                gamma = float(snapshot.greeks.gamma or 0)
                theta = float(snapshot.greeks.theta or 0)
                vega = float(snapshot.greeks.vega or 0)

            # Parse OCC symbol for strike/expiry/type
            # Format: UNDERLYING + YYMMDD + C/P + strike*1000 (8 digits)
            info = _parse_occ_symbol(symbol)

            return OptionContract(
                symbol=symbol,
                underlying=underlying,
                expiration=info.get("expiration", ""),
                strike=info.get("strike", 0.0),
                option_type=info.get("type", "call"),
                bid=bid,
                ask=ask,
                last_price=last,
                implied_volatility=iv,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
            )
        except Exception as e:
            logger.debug("Failed to parse option snapshot %s: %s", symbol, e)
            return None


def _parse_occ_symbol(symbol: str) -> dict:
    """Parse an OCC option symbol like 'AAPL250321C00150000'.

    Returns dict with underlying, expiration, type, strike.
    """
    # Find where the date portion starts (6 digits before C/P)
    # Standard: letters + 6 digits + C/P + 8 digits
    result = {"underlying": "", "expiration": "", "type": "call", "strike": 0.0}

    try:
        # Work backwards: last 8 chars = strike * 1000, char before that = C/P
        strike_str = symbol[-8:]
        cp = symbol[-9]
        date_str = symbol[-15:-9]
        underlying = symbol[:-15]

        result["underlying"] = underlying
        result["type"] = "call" if cp == "C" else "put"
        result["strike"] = int(strike_str) / 1000.0

        # Date is YYMMDD
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        result["expiration"] = f"{year}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        pass

    return result
