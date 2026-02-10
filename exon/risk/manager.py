"""Risk management system.

Monitors and controls:
- Position sizing and concentration
- Value at Risk (VaR) and Expected Shortfall
- Drawdown limits and circuit breakers
- Correlation-based exposure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk parameters and limits."""

    max_portfolio_var_pct: float = 0.02  # max daily VaR as % of portfolio
    max_position_pct: float = 0.20  # max single position size
    max_sector_pct: float = 0.40  # max sector concentration
    max_drawdown_pct: float = 0.15  # circuit breaker: max drawdown
    max_correlation_exposure: float = 0.80  # max portfolio avg correlation
    max_leverage: float = 1.0  # max gross leverage
    min_cash_pct: float = 0.05  # minimum cash buffer


@dataclass
class RiskReport:
    """Snapshot of current risk state."""

    timestamp: pd.Timestamp
    portfolio_value: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_position_pct: float
    max_position_asset: str
    current_drawdown: float
    breaches: list[str] = field(default_factory=list)

    @property
    def is_within_limits(self) -> bool:
        return len(self.breaches) == 0


class RiskManager:
    """Real-time risk monitoring and position adjustment."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.peak_equity: float = 0.0
        self.risk_history: list[RiskReport] = []

    def check_portfolio(
        self,
        weights: pd.Series,
        portfolio_value: float,
        returns: pd.DataFrame,
        timestamp: pd.Timestamp | None = None,
    ) -> RiskReport:
        """Run full risk check on proposed portfolio weights."""
        if timestamp is None:
            timestamp = pd.Timestamp.now("UTC")

        self.peak_equity = max(self.peak_equity, portfolio_value)
        current_dd = (portfolio_value - self.peak_equity) / self.peak_equity

        breaches = []

        # Position concentration
        max_pos = weights.abs().max() if len(weights) > 0 else 0
        max_pos_asset = weights.abs().idxmax() if len(weights) > 0 else ""
        if max_pos > self.limits.max_position_pct:
            breaches.append(f"position_limit: {max_pos_asset} at {max_pos:.1%}")

        # Leverage
        gross = weights.abs().sum()
        net = weights.sum()
        if gross > self.limits.max_leverage:
            breaches.append(f"leverage: {gross:.2f}x")

        # VaR calculation
        aligned = returns[weights.index.intersection(returns.columns)]
        port_returns = (aligned * weights.reindex(aligned.columns, fill_value=0)).sum(axis=1)
        var_95 = np.percentile(port_returns.dropna(), 5) if len(port_returns) > 50 else 0
        var_99 = np.percentile(port_returns.dropna(), 1) if len(port_returns) > 50 else 0

        if abs(var_95) > self.limits.max_portfolio_var_pct:
            breaches.append(f"var_95: {var_95:.2%}")

        # Expected shortfall (CVaR)
        tail = port_returns[port_returns <= var_95] if len(port_returns) > 50 else pd.Series([0])
        es = tail.mean() if len(tail) > 0 else 0

        # Drawdown circuit breaker
        if current_dd < -self.limits.max_drawdown_pct:
            breaches.append(f"drawdown_circuit_breaker: {current_dd:.2%}")

        # Cash buffer
        cash_pct = 1.0 - gross
        if cash_pct < self.limits.min_cash_pct:
            breaches.append(f"cash_buffer: {cash_pct:.1%}")

        report = RiskReport(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            gross_exposure=gross,
            net_exposure=net,
            leverage=gross,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=es,
            max_position_pct=max_pos,
            max_position_asset=max_pos_asset,
            current_drawdown=current_dd,
            breaches=breaches,
        )
        self.risk_history.append(report)
        return report

    def adjust_weights(
        self,
        weights: pd.Series,
        portfolio_value: float,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Adjust portfolio weights to satisfy risk constraints."""
        adjusted = weights.copy()

        # Clip individual positions
        adjusted = adjusted.clip(
            -self.limits.max_position_pct, self.limits.max_position_pct
        )

        # Scale down if leverage exceeded
        gross = adjusted.abs().sum()
        if gross > self.limits.max_leverage:
            adjusted = adjusted * self.limits.max_leverage / gross

        # Drawdown circuit breaker: reduce exposure
        current_dd = (portfolio_value - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        if current_dd < -self.limits.max_drawdown_pct * 0.75:
            scale = max(0.25, 1.0 + current_dd / self.limits.max_drawdown_pct)
            adjusted = adjusted * scale
            logger.warning("Drawdown scaling applied: %.2f (dd=%.2%%)", scale, current_dd)

        return adjusted

    def position_size_volatility(
        self,
        returns: pd.Series,
        target_vol: float = 0.10,
        lookback: int = 168,
    ) -> float:
        """Size a position to target a specific annualised volatility."""
        recent_vol = returns.iloc[-lookback:].std() * np.sqrt(8760)
        if recent_vol == 0:
            return 0.0
        return target_vol / recent_vol
