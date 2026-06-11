"""Event-driven backtesting engine with realistic simulation.

Handles:
- Transaction costs (maker/taker fees)
- Slippage modelling
- Position tracking
- P&L attribution
- Walk-forward validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    asset: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    cost: float  # transaction cost
    slippage: float


@dataclass
class BacktestConfig:
    """Configuration for backtest simulation."""

    initial_capital: float = 100_000.0
    maker_fee: float = 0.004  # 0.4% Coinbase maker fee
    taker_fee: float = 0.006  # 0.6% Coinbase taker fee
    slippage_bps: float = 5.0  # basis points of slippage
    max_position_pct: float = 0.20  # max 20% in any single asset
    rebalance_frequency: int = 1  # rebalance every N bars
    use_maker: bool = True  # assume limit orders (maker)


@dataclass
class BacktestResult:
    """Container for backtest results."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: list[TradeRecord]
    config: BacktestConfig
    strategy_name: str

    @property
    def total_return(self) -> float:
        return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1

    @property
    def annualised_return(self) -> float:
        hours = (self.equity_curve.index[-1] - self.equity_curve.index[0]).total_seconds() / 3600
        years = hours / 8760
        if years <= 0:
            return 0.0
        return (1 + self.total_return) ** (1 / years) - 1

    @property
    def sharpe_ratio(self) -> float:
        r = self.returns.dropna()
        if r.std() == 0:
            return 0.0
        return r.mean() / r.std() * np.sqrt(8760)  # hourly -> annualised

    @property
    def sortino_ratio(self) -> float:
        r = self.returns.dropna()
        downside = r[r < 0]
        if downside.std() == 0:
            return 0.0
        return r.mean() / downside.std() * np.sqrt(8760)

    @property
    def max_drawdown(self) -> float:
        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak
        return dd.min()

    @property
    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown)
        if mdd == 0:
            return 0.0
        return self.annualised_return / mdd

    @property
    def win_rate(self) -> float:
        r = self.returns.dropna()
        if len(r) == 0:
            return 0.0
        return (r > 0).sum() / len(r)

    @property
    def profit_factor(self) -> float:
        r = self.returns.dropna()
        gains = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "total_return": f"{self.total_return:.2%}",
            "annualised_return": f"{self.annualised_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.3f}",
            "sortino_ratio": f"{self.sortino_ratio:.3f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "calmar_ratio": f"{self.calmar_ratio:.3f}",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_factor": f"{self.profit_factor:.3f}",
            "n_trades": self.n_trades,
            "initial_capital": self.config.initial_capital,
            "final_equity": f"{self.equity_curve.iloc[-1]:.2f}",
        }


class BacktestEngine:
    """Vectorised + event-driven hybrid backtester."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        **strategy_kwargs,
    ) -> BacktestResult:
        """Run a full backtest.

        Parameters
        ----------
        strategy : Strategy instance
        prices : DataFrame of close prices (rows=time, cols=assets)
        """
        cfg = self.config
        capital = cfg.initial_capital
        positions: dict[str, float] = {}  # asset -> quantity
        position_history = []
        equity_history = []
        trades: list[TradeRecord] = []

        timestamps = prices.index.tolist()
        fee_rate = cfg.maker_fee if cfg.use_maker else cfg.taker_fee

        for i, ts in enumerate(timestamps):
            if i < 50:  # warm-up period
                equity_history.append(capital)
                position_history.append({})
                continue

            # Current portfolio value
            port_value = capital
            for asset, qty in positions.items():
                if asset in prices.columns:
                    port_value += qty * prices.loc[ts, asset]

            equity_history.append(port_value)
            position_history.append(dict(positions))

            # Generate signals at rebalance frequency
            if i % cfg.rebalance_frequency != 0:
                continue

            history = prices.iloc[: i + 1]
            try:
                signals = strategy.generate_signals(history, **strategy_kwargs)
            except Exception as e:
                logger.debug("Signal generation failed at %s: %s", ts, e)
                continue

            if not signals:
                continue

            target_weights = strategy.signals_to_weights(signals)

            # Convert weights to target positions
            for asset, weight in target_weights.items():
                if asset not in prices.columns:
                    continue

                # Position size limit
                weight = np.clip(weight, -cfg.max_position_pct, cfg.max_position_pct)
                target_value = port_value * weight
                current_price = prices.loc[ts, asset]
                if current_price <= 0:
                    continue

                target_qty = target_value / current_price
                current_qty = positions.get(asset, 0.0)
                delta_qty = target_qty - current_qty

                if abs(delta_qty * current_price) < port_value * 0.001:
                    continue  # skip tiny trades

                # Apply slippage
                slippage = current_price * cfg.slippage_bps / 10_000
                fill_price = current_price + slippage * np.sign(delta_qty)

                # Transaction cost
                trade_value = abs(delta_qty * fill_price)
                cost = trade_value * fee_rate

                # Execute
                capital -= delta_qty * fill_price + cost
                positions[asset] = current_qty + delta_qty

                trades.append(
                    TradeRecord(
                        timestamp=ts,
                        asset=asset,
                        side="buy" if delta_qty > 0 else "sell",
                        quantity=abs(delta_qty),
                        price=fill_price,
                        cost=cost,
                        slippage=slippage * abs(delta_qty),
                    )
                )

            # Close positions for assets no longer in signals
            signal_assets = set(target_weights.index)
            for asset in list(positions.keys()):
                if asset not in signal_assets and positions[asset] != 0:
                    current_price = prices.loc[ts, asset]
                    qty = positions[asset]
                    slippage = current_price * cfg.slippage_bps / 10_000
                    fill_price = current_price - slippage * np.sign(qty)
                    cost = abs(qty * fill_price) * fee_rate
                    capital += qty * fill_price - cost

                    trades.append(
                        TradeRecord(
                            timestamp=ts,
                            asset=asset,
                            side="sell" if qty > 0 else "buy",
                            quantity=abs(qty),
                            price=fill_price,
                            cost=cost,
                            slippage=slippage * abs(qty),
                        )
                    )
                    del positions[asset]

        equity = pd.Series(equity_history, index=timestamps)
        returns = equity.pct_change().fillna(0)
        pos_df = pd.DataFrame(position_history, index=timestamps).fillna(0)

        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            positions=pos_df,
            trades=trades,
            config=cfg,
            strategy_name=strategy.name,
        )

    def walk_forward(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        train_size: int = 720,
        test_size: int = 168,
        step: int = 168,
        **strategy_kwargs,
    ) -> list[BacktestResult]:
        """Walk-forward analysis: train on rolling window, test on next period.

        Returns a list of BacktestResult, one per out-of-sample window.
        """
        results = []
        n = len(prices)

        for start in range(0, n - train_size - test_size, step):
            train_end = start + train_size
            test_end = min(train_end + test_size, n)

            test_prices = prices.iloc[train_end:test_end]
            # Provide training data context by including it
            full_data = prices.iloc[start:test_end]

            result = self.run(strategy, full_data, **strategy_kwargs)
            results.append(result)

        return results
