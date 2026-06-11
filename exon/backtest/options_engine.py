"""Options backtesting engine — simulates options P&L over historical data.

Handles:
- Multi-leg position tracking with per-leg Greeks
- Daily mark-to-market via Black-Scholes repricing
- Theta decay, IV change impact, delta/gamma P&L attribution
- Expiry settlement (ITM exercise, OTM expire worthless)
- Rolling/management (close before expiry, roll to next month)
- Transaction costs (bid-ask spread simulation)
- Portfolio-level Greeks aggregation and risk monitoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..options.pricing import black_scholes_price, compute_greeks, OptionType
from ..options.strategy_mapper import (
    OptionsStructure,
    OptionsLeg,
    OptionsTradeRecommendation,
    StrategyMapper,
    StrategyMapperConfig,
)
from ..options.vol_surface import (
    analyse_vol_surface,
    estimate_iv_from_returns,
    realised_volatility,
)
from ..strategies.base import Signal

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class OptionsPosition:
    """A live options position (single leg)."""

    leg_id: str
    underlying: str
    option_type: str  # "call" or "put"
    strike: float
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    side: str  # "buy" or "sell" (long or short)
    qty: int  # number of contracts
    entry_price: float  # per-share option premium at entry
    current_price: float = 0.0
    current_delta: float = 0.0
    current_gamma: float = 0.0
    current_theta: float = 0.0
    current_vega: float = 0.0
    structure_id: str = ""  # groups legs of same trade

    @property
    def multiplier(self) -> int:
        """Position sign: +1 for long, -1 for short."""
        return 1 if self.side == "buy" else -1

    @property
    def market_value(self) -> float:
        """Current market value (per contract = 100 shares)."""
        return self.current_price * 100 * self.qty * self.multiplier

    @property
    def entry_value(self) -> float:
        return self.entry_price * 100 * self.qty * self.multiplier

    @property
    def pnl(self) -> float:
        return self.market_value - self.entry_value

    @property
    def days_to_expiry(self) -> float:
        """Approximate DTE from entry context (updated externally)."""
        return 0.0  # overridden by engine


@dataclass
class OptionsTradeRecord:
    """Record of an options trade execution."""

    timestamp: pd.Timestamp
    structure_id: str
    underlying: str
    structure: str
    legs: list[dict]
    action: str  # "open", "close", "expire", "exercise"
    net_premium: float  # positive = credit, negative = debit
    transaction_cost: float


@dataclass
class OptionsBacktestConfig:
    """Configuration for options backtesting."""

    initial_capital: float = 100_000.0
    # Transaction costs
    per_contract_fee: float = 0.65  # typical options commission
    spread_slippage_pct: float = 0.05  # 5% of bid-ask spread lost
    # Position management
    max_positions: int = 20
    max_risk_per_trade_pct: float = 0.02  # 2% of portfolio
    max_portfolio_delta: float = 50.0  # max net delta exposure
    # Expiry management
    close_before_expiry_days: int = 5  # close positions N days before expiry
    # IV simulation
    iv_mean_reversion_speed: float = 0.03  # daily mean reversion of IV
    iv_vol_of_vol: float = 0.01  # daily vol of IV changes
    # Risk-free rate
    risk_free_rate: float = 0.05
    # Bars per year for annualization
    trading_days_per_year: int = 252


@dataclass
class OptionsBacktestResult:
    """Results of an options backtest."""

    equity_curve: pd.Series
    returns: pd.Series
    trades: list[OptionsTradeRecord]
    daily_greeks: pd.DataFrame  # portfolio Greeks over time
    positions_over_time: list[dict]
    config: OptionsBacktestConfig
    strategy_name: str

    @property
    def total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1

    @property
    def annualised_return(self) -> float:
        n_days = len(self.equity_curve)
        years = n_days / self.config.trading_days_per_year
        if years <= 0:
            return 0.0
        return (1 + self.total_return) ** (1 / years) - 1

    @property
    def sharpe_ratio(self) -> float:
        r = self.returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(self.config.trading_days_per_year))

    @property
    def sortino_ratio(self) -> float:
        r = self.returns.dropna()
        downside = r[r < 0]
        if len(downside) < 2 or downside.std() == 0:
            return 0.0
        return float(r.mean() / downside.std() * np.sqrt(self.config.trading_days_per_year))

    @property
    def max_drawdown(self) -> float:
        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak
        return float(dd.min())

    @property
    def win_rate(self) -> float:
        """Win rate based on closed trades (structure-level)."""
        closed = [t for t in self.trades if t.action in ("close", "expire", "exercise")]
        if not closed:
            return 0.0
        # Group by structure_id
        pnl_by_structure: dict[str, float] = {}
        for t in closed:
            pnl_by_structure[t.structure_id] = (
                pnl_by_structure.get(t.structure_id, 0) + t.net_premium
            )
        if not pnl_by_structure:
            return 0.0
        wins = sum(1 for v in pnl_by_structure.values() if v > 0)
        return wins / len(pnl_by_structure)

    @property
    def n_trades(self) -> int:
        return sum(1 for t in self.trades if t.action == "open")

    @property
    def total_fees(self) -> float:
        return sum(t.transaction_cost for t in self.trades)

    @property
    def avg_trade_pnl(self) -> float:
        closed = [t for t in self.trades if t.action in ("close", "expire", "exercise")]
        if not closed:
            return 0.0
        pnl_by_structure: dict[str, float] = {}
        for t in closed:
            pnl_by_structure[t.structure_id] = (
                pnl_by_structure.get(t.structure_id, 0) + t.net_premium
            )
        if not pnl_by_structure:
            return 0.0
        return sum(pnl_by_structure.values()) / len(pnl_by_structure)

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "total_return": f"{self.total_return:.2%}",
            "annualised_return": f"{self.annualised_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.3f}",
            "sortino_ratio": f"{self.sortino_ratio:.3f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "n_trades": self.n_trades,
            "total_fees": f"${self.total_fees:.2f}",
            "avg_trade_pnl": f"${self.avg_trade_pnl:.2f}",
            "initial_capital": self.config.initial_capital,
            "final_equity": f"{self.equity_curve.iloc[-1]:.2f}",
        }


class OptionsBacktestEngine:
    """Event-driven options backtester with Greeks-based P&L.

    Workflow per bar:
    1. Mark-to-market all positions via Black-Scholes
    2. Check for expiries and management triggers
    3. Generate equity signals from underlying strategy
    4. Map signals to options structures via StrategyMapper
    5. Open new positions (if risk budget allows)
    6. Record portfolio value and Greeks
    """

    def __init__(self, config: OptionsBacktestConfig | None = None):
        self.config = config or OptionsBacktestConfig()

    def run(
        self,
        strategy,
        prices: pd.DataFrame,
        mapper_config: StrategyMapperConfig | None = None,
        options_universe: list[str] | None = None,
        rebalance_frequency: int = 1,
    ) -> OptionsBacktestResult:
        """Run options backtest.

        Parameters
        ----------
        strategy : equity Strategy that generates signals
        prices : DataFrame of daily close prices (rows=dates, cols=assets)
        mapper_config : options strategy mapper configuration
        options_universe : subset of assets eligible for options (None = all)
        rebalance_frequency : generate signals every N bars
        """
        cfg = self.config
        mapper = StrategyMapper(mapper_config or StrategyMapperConfig())
        capital = cfg.initial_capital
        positions: list[OptionsPosition] = []
        trades: list[OptionsTradeRecord] = []
        equity_history: list[float] = []
        greeks_history: list[dict] = []
        positions_history: list[dict] = []
        structure_counter = 0

        timestamps = prices.index.tolist()
        eligible = set(options_universe) if options_universe else set(prices.columns)

        # Pre-compute returns for IV estimation
        returns_df = prices.pct_change().fillna(0)

        # Simulate IV paths (IV evolves with mean-reversion + noise)
        rng = np.random.default_rng(42)
        iv_paths = self._simulate_iv_paths(prices, returns_df, rng)

        warmup = min(63, len(timestamps) // 4)

        for i, ts in enumerate(timestamps):
            if i < warmup:
                equity_history.append(capital)
                greeks_history.append(self._zero_greeks(ts))
                positions_history.append({"n_positions": 0})
                continue

            # Current spot prices
            spots = prices.loc[ts].to_dict()

            # 1. Mark-to-market all positions
            self._mark_to_market(positions, spots, iv_paths, ts, timestamps)

            # 2. Manage expiries and close-before-expiry
            close_pnl, close_trades = self._manage_expiries(
                positions, spots, ts, timestamps
            )
            capital += close_pnl
            trades.extend(close_trades)

            # Remove closed positions
            positions = [p for p in positions if self._dte(p, ts, timestamps) > 0]

            # 3. Portfolio value = cash + position mark-to-market
            position_value = sum(p.market_value for p in positions)
            portfolio_value = capital + position_value
            equity_history.append(portfolio_value)

            # 4. Record Greeks
            greeks = self._aggregate_greeks(positions, ts)
            greeks_history.append(greeks)
            positions_history.append({
                "n_positions": len(positions),
                "net_delta": greeks["net_delta"],
                "net_gamma": greeks["net_gamma"],
                "net_theta": greeks["net_theta"],
                "net_vega": greeks["net_vega"],
            })

            # 5. Generate new signals at rebalance frequency
            if i % rebalance_frequency != 0:
                continue

            if len(positions) >= cfg.max_positions:
                continue

            history = prices.iloc[:i + 1]
            try:
                signals = strategy.generate_signals(history)
            except Exception:
                continue

            if not signals:
                continue

            # Filter to options-eligible assets
            signals = [s for s in signals if s.asset in eligible]

            # 6. Map signals to options recommendations
            for sig in signals:
                if sig.asset not in spots or spots[sig.asset] <= 0:
                    continue
                if len(positions) >= cfg.max_positions:
                    break

                spot = spots[sig.asset]
                iv = iv_paths.get(sig.asset, {}).get(i, 0.25)

                # IV percentile from history
                iv_hist = [
                    iv_paths.get(sig.asset, {}).get(j, 0.25)
                    for j in range(max(0, i - 252), i)
                ]
                iv_pctile = (
                    sum(1 for v in iv_hist if v < iv) / len(iv_hist)
                    if iv_hist
                    else 0.5
                )

                # Historical vol
                ret_series = returns_df[sig.asset].iloc[max(0, i - 63):i]
                hvol = float(ret_series.std() * np.sqrt(252)) if len(ret_series) > 5 else 0.20

                rec = mapper.map_signal(
                    sig, spot, iv, iv_pctile, hvol, portfolio_value
                )

                # Risk check: max risk per trade
                if rec.max_loss > portfolio_value * cfg.max_risk_per_trade_pct:
                    continue

                # Open position
                structure_counter += 1
                struct_id = f"S{structure_counter:04d}"
                new_positions, open_trade = self._open_position(
                    rec, struct_id, ts, spot, iv, timestamps, i
                )
                if new_positions:
                    positions.extend(new_positions)
                    trades.append(open_trade)
                    capital += open_trade.net_premium - open_trade.transaction_cost

        # Final mark-to-market
        if not equity_history:
            equity_history.append(capital)

        equity = pd.Series(equity_history, index=timestamps[:len(equity_history)])
        returns = equity.pct_change().fillna(0)
        greeks_df = pd.DataFrame(greeks_history[:len(equity_history)], index=equity.index)

        return OptionsBacktestResult(
            equity_curve=equity,
            returns=returns,
            trades=trades,
            daily_greeks=greeks_df,
            positions_over_time=positions_history[:len(equity_history)],
            config=cfg,
            strategy_name=f"options_{strategy.name}",
        )

    def walk_forward(
        self,
        strategy,
        prices: pd.DataFrame,
        train_size: int = 252,
        test_size: int = 63,
        step: int = 63,
        **kwargs,
    ) -> list[OptionsBacktestResult]:
        """Walk-forward analysis for options strategies."""
        results = []
        n = len(prices)
        for start in range(0, n - train_size - test_size, step):
            train_end = start + train_size
            test_end = min(train_end + test_size, n)
            full_data = prices.iloc[start:test_end]
            result = self.run(strategy, full_data, **kwargs)
            results.append(result)
        return results

    def _simulate_iv_paths(
        self,
        prices: pd.DataFrame,
        returns_df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> dict[str, dict[int, float]]:
        """Simulate realistic IV paths for each asset.

        Uses mean-reverting process anchored to realised vol:
        IV_t = IV_{t-1} + κ(RV - IV_{t-1}) + σ_v * ε
        """
        cfg = self.config
        iv_paths: dict[str, dict[int, float]] = {}

        for col in prices.columns:
            rets = returns_df[col].values
            n = len(rets)
            ivs: dict[int, float] = {}

            # Initial IV from first 21 days of realised vol
            if n > 21:
                rv_init = float(np.std(rets[1:22]) * np.sqrt(252))
            else:
                rv_init = 0.20
            iv = max(rv_init * 1.1, 0.10)  # IV typically > RV (VRP)

            for t in range(n):
                # Realised vol estimate (trailing 21d)
                start_idx = max(0, t - 21)
                window_rets = rets[start_idx:t + 1]
                if len(window_rets) > 5:
                    rv = float(np.std(window_rets) * np.sqrt(252))
                    rv = max(rv, 0.05)
                else:
                    rv = rv_init

                # Mean-revert IV toward RV with a premium
                target = rv * 1.1  # IV premium over RV
                iv += cfg.iv_mean_reversion_speed * (target - iv)
                iv += cfg.iv_vol_of_vol * rng.normal()
                iv = np.clip(iv, 0.05, 2.0)
                ivs[t] = float(iv)

            iv_paths[col] = ivs

        return iv_paths

    def _mark_to_market(
        self,
        positions: list[OptionsPosition],
        spots: dict[str, float],
        iv_paths: dict[str, dict[int, float]],
        ts: pd.Timestamp,
        timestamps: list,
    ) -> None:
        """Reprice all positions using Black-Scholes."""
        bar_idx = timestamps.index(ts) if ts in timestamps else 0

        for pos in positions:
            spot = spots.get(pos.underlying, 0)
            if spot <= 0:
                continue

            iv = iv_paths.get(pos.underlying, {}).get(bar_idx, 0.25)
            dte = self._dte(pos, ts, timestamps)
            tte = max(dte / TRADING_DAYS_PER_YEAR, 1e-6)

            opt_type = OptionType.CALL if pos.option_type == "call" else OptionType.PUT

            price = black_scholes_price(
                spot, pos.strike, tte, iv, self.config.risk_free_rate, opt_type
            )
            greeks = compute_greeks(
                spot, pos.strike, tte, iv, self.config.risk_free_rate, opt_type
            )

            pos.current_price = price
            pos.current_delta = greeks.delta * pos.multiplier * pos.qty
            pos.current_gamma = greeks.gamma * pos.multiplier * pos.qty
            pos.current_theta = greeks.theta * pos.multiplier * pos.qty
            pos.current_vega = greeks.vega * pos.multiplier * pos.qty

    def _dte(
        self,
        pos: OptionsPosition,
        current_ts: pd.Timestamp,
        timestamps: list,
    ) -> int:
        """Days to expiry for a position."""
        current_idx = timestamps.index(current_ts) if current_ts in timestamps else 0
        expiry_idx = timestamps.index(pos.expiry_date) if pos.expiry_date in timestamps else len(timestamps) - 1
        return max(expiry_idx - current_idx, 0)

    def _manage_expiries(
        self,
        positions: list[OptionsPosition],
        spots: dict[str, float],
        ts: pd.Timestamp,
        timestamps: list,
    ) -> tuple[float, list[OptionsTradeRecord]]:
        """Close expiring positions and settle at-expiry positions."""
        cfg = self.config
        total_pnl = 0.0
        trades: list[OptionsTradeRecord] = []

        for pos in positions:
            dte = self._dte(pos, ts, timestamps)

            if dte <= 0:
                # Expiry settlement
                spot = spots.get(pos.underlying, 0)
                settlement = self._settle_at_expiry(pos, spot)
                total_pnl += settlement
                fee = cfg.per_contract_fee * pos.qty
                total_pnl -= fee

                trades.append(OptionsTradeRecord(
                    timestamp=ts,
                    structure_id=pos.structure_id,
                    underlying=pos.underlying,
                    structure="expiry",
                    legs=[{
                        "type": pos.option_type,
                        "strike": pos.strike,
                        "side": pos.side,
                        "settlement": settlement,
                    }],
                    action="expire" if settlement == 0 else "exercise",
                    net_premium=settlement,
                    transaction_cost=fee,
                ))

            elif dte <= cfg.close_before_expiry_days and dte > 0:
                # Close before expiry
                close_value = pos.current_price * 100 * pos.qty
                if pos.side == "sell":
                    # Buy back short position
                    total_pnl -= close_value
                else:
                    # Sell long position
                    total_pnl += close_value

                # Slippage
                slippage = close_value * cfg.spread_slippage_pct
                total_pnl -= slippage
                fee = cfg.per_contract_fee * pos.qty
                total_pnl -= fee

                pnl = close_value * pos.multiplier
                trades.append(OptionsTradeRecord(
                    timestamp=ts,
                    structure_id=pos.structure_id,
                    underlying=pos.underlying,
                    structure="close_before_expiry",
                    legs=[{
                        "type": pos.option_type,
                        "strike": pos.strike,
                        "side": "sell" if pos.side == "buy" else "buy",
                        "price": pos.current_price,
                    }],
                    action="close",
                    net_premium=pnl - slippage,
                    transaction_cost=fee,
                ))

                # Mark for removal by setting expiry to current
                pos.expiry_date = ts

        return total_pnl, trades

    def _settle_at_expiry(self, pos: OptionsPosition, spot: float) -> float:
        """Compute settlement value at expiry."""
        if pos.option_type == "call":
            intrinsic = max(spot - pos.strike, 0)
        else:
            intrinsic = max(pos.strike - spot, 0)

        value = intrinsic * 100 * pos.qty
        return value * pos.multiplier

    def _open_position(
        self,
        rec: OptionsTradeRecommendation,
        struct_id: str,
        ts: pd.Timestamp,
        spot: float,
        iv: float,
        timestamps: list,
        bar_idx: int,
    ) -> tuple[list[OptionsPosition], OptionsTradeRecord]:
        """Open a new options position from a recommendation."""
        cfg = self.config
        dte = rec.target_dte
        # Expiry date: dte bars into the future
        expiry_idx = min(bar_idx + dte, len(timestamps) - 1)
        expiry_date = timestamps[expiry_idx]
        tte = max(dte / TRADING_DAYS_PER_YEAR, 1e-6)

        positions = []
        leg_details = []
        net_premium = 0.0
        total_fee = 0.0

        for leg_num, leg in enumerate(rec.legs):
            opt_type = OptionType.CALL if leg.option_type == "call" else OptionType.PUT
            price = black_scholes_price(
                spot, leg.strike, tte, iv, cfg.risk_free_rate, opt_type
            )

            # Bid-ask slippage
            slippage = price * cfg.spread_slippage_pct
            if leg.side == "buy":
                fill_price = price + slippage  # pay more
                net_premium -= fill_price * 100
            else:
                fill_price = max(price - slippage, 0.01)  # receive less
                net_premium += fill_price * 100

            fee = cfg.per_contract_fee
            total_fee += fee

            pos = OptionsPosition(
                leg_id=f"{struct_id}_L{leg_num}",
                underlying=rec.underlying,
                option_type=leg.option_type,
                strike=leg.strike,
                entry_date=ts,
                expiry_date=expiry_date,
                side=leg.side,
                qty=1,
                entry_price=fill_price,
                current_price=price,
                structure_id=struct_id,
            )
            positions.append(pos)

            leg_details.append({
                "type": leg.option_type,
                "strike": leg.strike,
                "side": leg.side,
                "price": round(fill_price, 4),
            })

        trade = OptionsTradeRecord(
            timestamp=ts,
            structure_id=struct_id,
            underlying=rec.underlying,
            structure=rec.structure.value,
            legs=leg_details,
            action="open",
            net_premium=round(net_premium, 2),
            transaction_cost=round(total_fee, 2),
        )

        return positions, trade

    def _aggregate_greeks(
        self, positions: list[OptionsPosition], ts: pd.Timestamp
    ) -> dict:
        """Compute portfolio-level Greeks."""
        return {
            "timestamp": ts,
            "net_delta": sum(p.current_delta for p in positions),
            "net_gamma": sum(p.current_gamma for p in positions),
            "net_theta": sum(p.current_theta for p in positions),
            "net_vega": sum(p.current_vega for p in positions),
            "n_positions": len(positions),
        }

    def _zero_greeks(self, ts: pd.Timestamp) -> dict:
        return {
            "timestamp": ts,
            "net_delta": 0.0,
            "net_gamma": 0.0,
            "net_theta": 0.0,
            "net_vega": 0.0,
            "n_positions": 0,
        }
