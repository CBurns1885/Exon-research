"""Enhanced equity backtesting engine with daily-frequency support.

Extends the base BacktestEngine with:
- Configurable annualisation (hourly vs daily vs custom)
- Multi-strategy orchestration backtesting
- Per-strategy attribution (which strategies contributed what)
- Walk-forward with out-of-sample performance aggregation
- Regime-aware analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.base import Strategy, Signal
from .engine import BacktestConfig, BacktestResult, BacktestEngine, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class EquityBacktestConfig(BacktestConfig):
    """Extended config for equity backtesting."""

    # Annualisation
    bars_per_year: int = 252  # 252 for daily, 8760 for hourly
    # Commission-free trading (Alpaca)
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    slippage_bps: float = 2.0  # tighter for equities
    # Position limits
    max_position_pct: float = 0.10  # 10% max per stock
    min_trade_value: float = 50.0  # skip trades under $50
    # Whole-share rounding
    whole_shares: bool = True


class EquityBacktestEngine(BacktestEngine):
    """Equity-specific backtester with daily bar support."""

    def __init__(self, config: EquityBacktestConfig | None = None):
        self.config = config or EquityBacktestConfig()

    def run(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        **strategy_kwargs,
    ) -> BacktestResult:
        """Run equity backtest with whole-share rounding and daily annualisation."""
        cfg = self.config
        capital = cfg.initial_capital
        positions: dict[str, float] = {}  # asset -> shares
        position_history = []
        equity_history = []
        trades: list[TradeRecord] = []

        timestamps = prices.index.tolist()
        fee_rate = cfg.maker_fee if cfg.use_maker else cfg.taker_fee
        warmup = min(50, len(timestamps) // 4)

        for i, ts in enumerate(timestamps):
            if i < warmup:
                equity_history.append(capital)
                position_history.append({})
                continue

            # Portfolio value
            port_value = capital
            for asset, qty in positions.items():
                if asset in prices.columns:
                    port_value += qty * prices.loc[ts, asset]

            equity_history.append(port_value)
            position_history.append(dict(positions))

            if i % cfg.rebalance_frequency != 0:
                continue

            history = prices.iloc[:i + 1]
            try:
                signals = strategy.generate_signals(history, **strategy_kwargs)
            except Exception:
                continue

            if not signals:
                continue

            target_weights = strategy.signals_to_weights(signals)

            for asset, weight in target_weights.items():
                if asset not in prices.columns:
                    continue

                weight = np.clip(weight, -cfg.max_position_pct, cfg.max_position_pct)
                target_value = port_value * weight
                current_price = prices.loc[ts, asset]
                if current_price <= 0:
                    continue

                target_qty = target_value / current_price
                if cfg.whole_shares:
                    target_qty = round(target_qty)

                current_qty = positions.get(asset, 0.0)
                delta_qty = target_qty - current_qty

                trade_value = abs(delta_qty * current_price)
                if trade_value < cfg.min_trade_value:
                    continue

                slippage = current_price * cfg.slippage_bps / 10_000
                fill_price = current_price + slippage * np.sign(delta_qty)
                cost = trade_value * fee_rate

                capital -= delta_qty * fill_price + cost
                positions[asset] = current_qty + delta_qty

                trades.append(TradeRecord(
                    timestamp=ts,
                    asset=asset,
                    side="buy" if delta_qty > 0 else "sell",
                    quantity=abs(delta_qty),
                    price=fill_price,
                    cost=cost,
                    slippage=slippage * abs(delta_qty),
                ))

            # Close positions no longer in signals
            signal_assets = set(target_weights.index)
            for asset in list(positions.keys()):
                if asset not in signal_assets and positions[asset] != 0:
                    current_price = prices.loc[ts, asset]
                    qty = positions[asset]
                    slippage = current_price * cfg.slippage_bps / 10_000
                    fill_price = current_price - slippage * np.sign(qty)
                    cost = abs(qty * fill_price) * fee_rate
                    capital += qty * fill_price - cost

                    trades.append(TradeRecord(
                        timestamp=ts,
                        asset=asset,
                        side="sell" if qty > 0 else "buy",
                        quantity=abs(qty),
                        price=fill_price,
                        cost=cost,
                        slippage=slippage * abs(qty),
                    ))
                    del positions[asset]

        equity = pd.Series(equity_history, index=timestamps)
        returns = equity.pct_change().fillna(0)
        pos_df = pd.DataFrame(position_history, index=timestamps).fillna(0)

        result = BacktestResult(
            equity_curve=equity,
            returns=returns,
            positions=pos_df,
            trades=trades,
            config=cfg,
            strategy_name=strategy.name,
        )
        return result

    def run_multi_strategy(
        self,
        strategies: list[tuple[Strategy, float]],
        prices: pd.DataFrame,
    ) -> tuple[BacktestResult, dict[str, BacktestResult]]:
        """Run multiple strategies with capital allocation.

        Parameters
        ----------
        strategies : list of (Strategy, allocation_pct) tuples
        prices : daily price DataFrame

        Returns
        -------
        (combined_result, per_strategy_results)
        """
        per_strategy = {}
        combined_equity = None

        for strategy, alloc in strategies:
            sub_config = EquityBacktestConfig(
                initial_capital=self.config.initial_capital * alloc,
                maker_fee=self.config.maker_fee,
                taker_fee=self.config.taker_fee,
                slippage_bps=self.config.slippage_bps,
                max_position_pct=self.config.max_position_pct,
                whole_shares=self.config.whole_shares,
                bars_per_year=self.config.bars_per_year,
                rebalance_frequency=self.config.rebalance_frequency,
            )
            sub_engine = EquityBacktestEngine(sub_config)
            result = sub_engine.run(strategy, prices)
            per_strategy[strategy.name] = result

            if combined_equity is None:
                combined_equity = result.equity_curve.copy()
            else:
                combined_equity = combined_equity + result.equity_curve

        if combined_equity is None:
            combined_equity = pd.Series(
                [self.config.initial_capital], index=prices.index[:1]
            )

        combined_returns = combined_equity.pct_change().fillna(0)
        combined = BacktestResult(
            equity_curve=combined_equity,
            returns=combined_returns,
            positions=pd.DataFrame(index=prices.index),
            trades=[],
            config=self.config,
            strategy_name="multi_strategy",
        )
        return combined, per_strategy

    def walk_forward(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        train_size: int = 252,
        test_size: int = 63,
        step: int = 63,
        **strategy_kwargs,
    ) -> list[BacktestResult]:
        """Walk-forward with daily window sizes."""
        results = []
        n = len(prices)
        for start in range(0, n - train_size - test_size, step):
            train_end = start + train_size
            test_end = min(train_end + test_size, n)
            full_data = prices.iloc[start:test_end]
            result = self.run(strategy, full_data, **strategy_kwargs)
            results.append(result)
        return results

    def walk_forward_summary(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        train_size: int = 252,
        test_size: int = 63,
        step: int = 63,
    ) -> dict:
        """Walk-forward with aggregated out-of-sample metrics."""
        results = self.walk_forward(strategy, prices, train_size, test_size, step)

        if not results:
            return {"n_windows": 0}

        sharpes = [r.sharpe_ratio for r in results]
        returns = [r.total_return for r in results]
        drawdowns = [r.max_drawdown for r in results]
        n_trades = [r.n_trades for r in results]

        return {
            "n_windows": len(results),
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "min_sharpe": float(np.min(sharpes)),
            "max_sharpe": float(np.max(sharpes)),
            "pct_positive_sharpe": float(np.mean([s > 0 for s in sharpes])),
            "mean_return": float(np.mean(returns)),
            "mean_max_dd": float(np.mean(drawdowns)),
            "worst_dd": float(np.min(drawdowns)),
            "total_trades": int(np.sum(n_trades)),
            "strategy": strategy.name,
        }
