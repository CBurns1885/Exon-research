"""Command-line interface for the Exon Research equities trading platform.

Commands:
    exon-eq research   — Run correlation, cointegration, PCA, and regime analysis on equities
    exon-eq backtest   — Backtest strategies on historical equity data via Alpaca
    exon-eq trade      — Run live/paper trading loop via Alpaca
    exon-eq scan-pairs — Scan universe for cointegrated equity pairs
    exon-eq sectors    — Analyse sector rotation signals
"""

from __future__ import annotations

import logging
import sys

import click
import pandas as pd

from .config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("exon.equities")


def _build_alpaca_client(cfg: dict):
    from .data.alpaca_client import AlpacaClient, AlpacaConfig

    alpaca_cfg = cfg.get("alpaca", {})
    config = AlpacaConfig(
        api_key=alpaca_cfg.get("api_key", ""),
        api_secret=alpaca_cfg.get("api_secret", ""),
        paper=alpaca_cfg.get("paper", True),
        data_feed=alpaca_cfg.get("data_feed", "iex"),
    )
    return AlpacaClient(config=config)


def _build_pipeline(cfg: dict):
    from .data.equity_data import EquityDataPipeline

    client = _build_alpaca_client(cfg)
    return EquityDataPipeline(client=client)


def _build_strategies(cfg: dict) -> list:
    from .strategies.momentum import (
        TimeSeriesMomentum,
        CrossSectionalMomentum,
        AdaptiveMomentum,
    )
    from .strategies.mean_reversion import (
        BollingerMeanReversion,
        OUMeanReversion,
    )
    from .strategies.pairs import StatArbPCA
    from .strategies.volatility_breakout import VolatilityBreakout
    from .strategies.sector_rotation import SectorRotation
    from .strategies.earnings_momentum import EarningsMomentum
    from .strategies.market_microstructure import MarketMicrostructure
    from .strategies.kalman_spread import KalmanSpreadScanner
    from .strategies.multifactor import MultiFactor
    from .strategies.hurst_regime import HurstRegimeFilter
    from .strategies.wavelet_momentum import WaveletMomentum
    from .strategies.lead_lag import LeadLagExploitation

    strat_cfg = cfg.get("strategies", {})
    strategies = []

    if strat_cfg.get("ts_momentum", {}).get("enabled"):
        s = strat_cfg["ts_momentum"]
        strategies.append(
            (TimeSeriesMomentum(lookbacks=s.get("lookbacks"), vol_target=s.get("vol_target", 0.15)),
             s.get("allocation", 0.15))
        )

    if strat_cfg.get("xs_momentum", {}).get("enabled"):
        s = strat_cfg["xs_momentum"]
        strategies.append(
            (CrossSectionalMomentum(
                lookback=s.get("lookback", 63),
                top_n=s.get("top_n", 5),
                bottom_n=s.get("bottom_n", 3),
            ), s.get("allocation", 0.10))
        )

    if strat_cfg.get("adaptive_momentum", {}).get("enabled"):
        s = strat_cfg["adaptive_momentum"]
        strategies.append(
            (AdaptiveMomentum(base_lookback=s.get("base_lookback", 63)),
             s.get("allocation", 0.10))
        )

    if strat_cfg.get("bollinger_mr", {}).get("enabled"):
        s = strat_cfg["bollinger_mr"]
        strategies.append(
            (BollingerMeanReversion(window=s.get("window", 21), entry_std=s.get("entry_std", 2.0)),
             s.get("allocation", 0.10))
        )

    if strat_cfg.get("ou_mean_reversion", {}).get("enabled"):
        s = strat_cfg["ou_mean_reversion"]
        strategies.append(
            (OUMeanReversion(calibration_window=s.get("calibration_window", 252)),
             s.get("allocation", 0.05))
        )

    if strat_cfg.get("stat_arb_pca", {}).get("enabled"):
        s = strat_cfg["stat_arb_pca"]
        strategies.append(
            (StatArbPCA(
                n_factors=s.get("n_factors", 5),
                z_entry=s.get("z_entry", 1.5),
            ), s.get("allocation", 0.10))
        )

    if strat_cfg.get("vol_breakout", {}).get("enabled"):
        s = strat_cfg["vol_breakout"]
        strategies.append(
            (VolatilityBreakout(
                channel_window=s.get("channel_window", 21),
                atr_window=s.get("atr_window", 14),
                squeeze_window=s.get("squeeze_window", 63),
                squeeze_threshold=s.get("squeeze_threshold", 0.75),
                confirmation_bars=s.get("confirmation_bars", 2),
            ), s.get("allocation", 0.05))
        )

    if strat_cfg.get("sector_rotation", {}).get("enabled"):
        s = strat_cfg["sector_rotation"]
        strategies.append(
            (SectorRotation(
                momentum_window=s.get("momentum_window", 252),
                skip_window=s.get("skip_window", 21),
                vol_window=s.get("vol_window", 63),
                top_n=s.get("top_n", 3),
                bottom_n=s.get("bottom_n", 2),
                spy_column=s.get("spy_column", "SPY"),
            ), s.get("allocation", 0.10))
        )

    if strat_cfg.get("earnings_momentum", {}).get("enabled"):
        s = strat_cfg["earnings_momentum"]
        strategies.append(
            (EarningsMomentum(
                gap_threshold_std=s.get("gap_threshold_std", 2.5),
                drift_window=s.get("drift_window", 40),
                vol_lookback=s.get("vol_lookback", 63),
                decay_rate=s.get("decay_rate", 0.95),
                max_concurrent=s.get("max_concurrent", 10),
            ), s.get("allocation", 0.10))
        )

    if strat_cfg.get("market_microstructure", {}).get("enabled"):
        s = strat_cfg["market_microstructure"]
        strategies.append(
            (MarketMicrostructure(
                obv_window=s.get("obv_window", 21),
                volume_window=s.get("volume_window", 63),
                rvol_threshold=s.get("rvol_threshold", 1.5),
            ), s.get("allocation", 0.05))
        )

    if strat_cfg.get("kalman_scanner", {}).get("enabled"):
        s = strat_cfg["kalman_scanner"]
        strategies.append(
            (KalmanSpreadScanner(
                max_pairs=s.get("max_pairs", 5),
                delta=s.get("delta", 1e-4),
                z_entry=s.get("z_entry", 2.0),
            ), s.get("allocation", 0.05))
        )

    if strat_cfg.get("multifactor", {}).get("enabled"):
        s = strat_cfg["multifactor"]
        strategies.append(
            (MultiFactor(
                top_n=s.get("top_n", 5),
                bottom_n=s.get("bottom_n", 3),
                use_adaptive_weights=s.get("use_adaptive_weights", True),
                ic_lookback=s.get("ic_lookback", 252),
            ), s.get("allocation", 0.10))
        )

    if strat_cfg.get("hurst_regime", {}).get("enabled"):
        s = strat_cfg["hurst_regime"]
        strategies.append(
            (HurstRegimeFilter(
                hurst_window=s.get("hurst_window", 63),
                threshold_trend=s.get("threshold_trend", 0.55),
                threshold_mr=s.get("threshold_mr", 0.45),
            ), s.get("allocation", 0.05))
        )

    if strat_cfg.get("wavelet_momentum", {}).get("enabled"):
        s = strat_cfg["wavelet_momentum"]
        strategies.append(
            (WaveletMomentum(
                max_level=s.get("max_level", 5),
                lookback=s.get("lookback", 128),
            ), s.get("allocation", 0.05))
        )

    if strat_cfg.get("lead_lag", {}).get("enabled"):
        s = strat_cfg["lead_lag"]
        strategies.append(
            (LeadLagExploitation(
                leaders=s.get("leaders", ["SPY", "QQQ"]),
                max_lag=s.get("max_lag", 3),
                xcorr_window=s.get("xcorr_window", 63),
                min_correlation=s.get("min_correlation", 0.15),
            ), s.get("allocation", 0.05))
        )

    return strategies


@click.group()
@click.option("--config", "-c", default=None, help="Path to config YAML (defaults to config/equities.yaml)")
@click.pass_context
def main(ctx, config):
    """Exon Research — Quantitative equities trading platform (Alpaca)."""
    ctx.ensure_object(dict)
    if config is None:
        from pathlib import Path
        eq_config = Path(__file__).parent.parent / "config" / "equities.yaml"
        if eq_config.exists():
            config = str(eq_config)
    ctx.obj["config"] = load_config(config)


@main.command()
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.pass_context
def research(ctx, start, end):
    """Run research analysis: correlations, cointegration, PCA, regimes on equities."""
    from .research.correlations import (
        correlation_matrix,
        correlation_stability,
        cluster_assets,
        top_correlated_pairs,
    )
    from .research.cointegration import scan_cointegrated_pairs
    from .research.pca import decompose_returns
    from .research.regime import detect_regimes_gmm, regime_statistics

    cfg = ctx.obj["config"]
    pipeline = _build_pipeline(cfg)
    uni = cfg.get("universe", {})
    symbols = uni.get("symbols", [])
    benchmark = uni.get("benchmark", "SPY")
    all_symbols = list(set(symbols + [benchmark]))
    timeframe = uni.get("timeframe", "1Day")

    if not start:
        start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=uni.get("lookback_days", 365))).strftime("%Y-%m-%d")
    if not end:
        end = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    click.echo(f"Fetching data for {len(all_symbols)} equities: {start} -> {end}")
    prices = pipeline.fetch_universe(all_symbols, start, end, timeframe)
    if prices.empty:
        click.echo("No data returned. Check your Alpaca API key and symbols.")
        return

    returns = pipeline.get_returns(prices)

    click.echo("\n=== CORRELATION MATRIX ===")
    corr = correlation_matrix(returns)
    click.echo(corr.to_string())

    click.echo("\n=== TOP CORRELATED / ANTI-CORRELATED PAIRS ===")
    pairs = top_correlated_pairs(returns)
    click.echo(pairs.to_string())

    click.echo("\n=== CORRELATION STABILITY ===")
    stability = correlation_stability(returns)
    click.echo(stability.head(15).to_string())

    click.echo("\n=== ASSET CLUSTERS ===")
    clusters = cluster_assets(returns, n_clusters=min(5, len(all_symbols)))
    click.echo(clusters.to_string())

    click.echo("\n=== COINTEGRATED PAIRS ===")
    coint_pairs = scan_cointegrated_pairs(prices)
    if coint_pairs.empty:
        click.echo("No cointegrated pairs found at p<0.05")
    else:
        click.echo(coint_pairs.to_string())

    click.echo("\n=== PCA DECOMPOSITION ===")
    pca_result = decompose_returns(returns, n_components=min(5, len(all_symbols) - 1))
    click.echo(f"Components: {pca_result.n_components}")
    click.echo(f"Variance explained: {pca_result.cumulative_variance}")
    click.echo("Factor loadings (PC1 — market factor):")
    click.echo(pca_result.components.iloc[0].sort_values(ascending=False).to_string())

    click.echo(f"\n=== REGIME DETECTION ({benchmark}) ===")
    if benchmark in returns.columns:
        regime_df = detect_regimes_gmm(returns[benchmark])
        stats = regime_statistics(regime_df)
        click.echo(stats.to_string())

    pipeline.save(prices, "equity_prices_latest")
    pipeline.save(returns, "equity_returns_latest")
    click.echo("\nData saved to data/")


@main.command()
@click.option("--start", default=None, help="Start date")
@click.option("--end", default=None, help="End date")
@click.option("--strategy", "-s", default=None, help="Single strategy to test (or 'all')")
@click.pass_context
def backtest(ctx, start, end, strategy):
    """Backtest trading strategies on historical equity data."""
    from .backtest.engine import BacktestEngine, BacktestConfig
    from .backtest.metrics import compare_strategies, monte_carlo_sharpe_test
    from .strategies.composite import CompositeStrategy

    cfg = ctx.obj["config"]
    pipeline = _build_pipeline(cfg)
    uni = cfg.get("universe", {})
    symbols = uni.get("symbols", [])
    benchmark = uni.get("benchmark", "SPY")
    all_symbols = list(set(symbols + [benchmark]))

    if not start:
        start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=uni.get("lookback_days", 365))).strftime("%Y-%m-%d")
    if not end:
        end = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    click.echo(f"Fetching data for backtest: {start} -> {end}")
    prices = pipeline.fetch_universe(all_symbols, start, end, uni.get("timeframe", "1Day"))
    if prices.empty:
        click.echo("No data.")
        return

    bt_cfg = cfg.get("backtest", {})
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=bt_cfg.get("initial_capital", 100_000),
            maker_fee=bt_cfg.get("maker_fee", 0.0),
            taker_fee=bt_cfg.get("taker_fee", 0.0),
            slippage_bps=bt_cfg.get("slippage_bps", 2),
            use_maker=bt_cfg.get("use_maker", True),
        )
    )

    strategy_pairs = _build_strategies(cfg)
    results = []

    for strat, alloc in strategy_pairs:
        if strategy and strategy != "all" and strat.name != strategy:
            continue
        click.echo(f"\nBacktesting: {strat.name}")
        result = engine.run(strat, prices)
        results.append(result)
        summary = result.summary()
        for k, v in summary.items():
            click.echo(f"  {k}: {v}")

    if not strategy or strategy == "all":
        composite = CompositeStrategy(strategy_pairs, regime_aware=True)
        click.echo(f"\nBacktesting: composite (all strategies blended)")
        result = engine.run(composite, prices)
        results.append(result)
        summary = result.summary()
        for k, v in summary.items():
            click.echo(f"  {k}: {v}")

        click.echo("\n=== SHARPE RATIO SIGNIFICANCE ===")
        mc = monte_carlo_sharpe_test(result.returns)
        for k, v in mc.items():
            click.echo(f"  {k}: {v}")

    if len(results) > 1:
        click.echo("\n=== STRATEGY COMPARISON ===")
        comp = compare_strategies(results)
        click.echo(comp.to_string())


@main.command()
@click.pass_context
def scan_pairs(ctx):
    """Scan universe for cointegrated equity pairs."""
    from .research.cointegration import scan_cointegrated_pairs

    cfg = ctx.obj["config"]
    pipeline = _build_pipeline(cfg)
    uni = cfg.get("universe", {})
    symbols = uni.get("symbols", [])

    start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=uni.get("lookback_days", 365))).strftime("%Y-%m-%d")
    end = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    click.echo(f"Scanning {len(symbols)} equities for cointegrated pairs...")
    prices = pipeline.fetch_universe(symbols, start, end)
    if prices.empty:
        click.echo("No data.")
        return

    pairs = scan_cointegrated_pairs(prices)
    if pairs.empty:
        click.echo("No cointegrated pairs found.")
    else:
        click.echo(pairs.to_string())


@main.command()
@click.pass_context
def sectors(ctx):
    """Analyse sector rotation signals."""
    from .strategies.sector_rotation import SectorRotation, SECTOR_ETFS

    cfg = ctx.obj["config"]
    pipeline = _build_pipeline(cfg)
    uni = cfg.get("universe", {})
    sector_etfs = uni.get("sector_etfs", list(SECTOR_ETFS.keys()))
    benchmark = uni.get("benchmark", "SPY")
    all_symbols = list(set(sector_etfs + [benchmark]))

    start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=uni.get("lookback_days", 365))).strftime("%Y-%m-%d")
    end = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    click.echo(f"Analysing {len(sector_etfs)} sector ETFs...")
    prices = pipeline.fetch_universe(all_symbols, start, end)
    if prices.empty:
        click.echo("No data.")
        return

    strat_cfg = cfg.get("strategies", {}).get("sector_rotation", {})
    strategy = SectorRotation(
        momentum_window=strat_cfg.get("momentum_window", 252),
        skip_window=strat_cfg.get("skip_window", 21),
        spy_column=benchmark,
    )
    signals = strategy.generate_signals(prices)

    if not signals:
        click.echo("No sector rotation signals generated.")
        return

    click.echo("\n=== SECTOR ROTATION SIGNALS ===")
    for sig in sorted(signals, key=lambda s: s.strength, reverse=True):
        direction = "LONG" if sig.direction > 0 else "SHORT"
        sector = sig.metadata.get("sector", "unknown")
        rank = sig.metadata.get("rank", "?")
        score = sig.metadata.get("composite_score", 0)
        click.echo(f"  {direction:5s} {sig.asset:5s} ({sector:30s}) | rank={rank} score={score:.3f} strength={sig.strength:.3f}")


@main.command()
@click.option("--interval", default=24, help="Hours between rebalances (default: 24 for daily)")
@click.pass_context
def trade(ctx, interval):
    """Run live/paper trading loop via Alpaca."""
    import time as _time
    from .strategies.composite import CompositeStrategy
    from .execution.alpaca_executor import AlpacaExecutionEngine
    from .risk.manager import RiskManager, RiskLimits
    from .portfolio.optimizer import risk_parity, constrain_turnover

    cfg = ctx.obj["config"]
    client = _build_alpaca_client(cfg)
    pipeline = _build_pipeline(cfg)
    uni = cfg.get("universe", {})
    symbols = uni.get("symbols", [])
    benchmark = uni.get("benchmark", "SPY")
    all_symbols = list(set(symbols + [benchmark]))

    exec_cfg = cfg.get("execution", {})
    executor = AlpacaExecutionEngine(
        client=client,
        dry_run=exec_cfg.get("dry_run", True),
        max_order_value_usd=exec_cfg.get("max_order_value_usd", 50_000),
    )

    risk_cfg = cfg.get("risk", {})
    risk_mgr = RiskManager(
        RiskLimits(
            max_position_pct=risk_cfg.get("max_position_pct", 0.10),
            max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.10),
            max_leverage=risk_cfg.get("max_leverage", 1.0),
        )
    )

    strategy_pairs = _build_strategies(cfg)
    composite = CompositeStrategy(strategy_pairs, regime_aware=True)

    mode = "PAPER" if exec_cfg.get("dry_run", True) else "LIVE"
    click.echo(f"Starting {mode} equities trading loop (rebalance every {interval}h)")
    click.echo(f"Universe: {len(symbols)} stocks + {benchmark}")

    current_weights = pd.Series(dtype=float)
    portfolio_value = cfg.get("backtest", {}).get("initial_capital", 100_000)

    while True:
        try:
            click.echo(f"\n--- Rebalance at {pd.Timestamp.now('UTC')} ---")

            end = pd.Timestamp.now("UTC")
            start = end - pd.Timedelta(days=uni.get("lookback_days", 365))
            prices = pipeline.fetch_universe(
                all_symbols, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            )
            if prices.empty:
                click.echo("No data, skipping.")
                _time.sleep(interval * 3600)
                continue

            returns = pipeline.get_returns(prices)

            signals = composite.generate_signals(prices)
            target_weights = composite.signals_to_weights(signals)

            if len(target_weights) > 1:
                cov = returns[target_weights.index.intersection(returns.columns)].cov()
                target_weights = risk_parity(cov)

            target_weights = constrain_turnover(target_weights, current_weights)

            report = risk_mgr.check_portfolio(target_weights, portfolio_value, returns)
            if not report.is_within_limits:
                click.echo(f"Risk breaches: {report.breaches}")
                target_weights = risk_mgr.adjust_weights(target_weights, portfolio_value, returns)

            click.echo(f"Target weights:\n{target_weights.to_string()}")

            current_prices = executor.get_current_prices(symbols)
            if not current_prices:
                current_prices = {s: prices[s].iloc[-1] for s in prices.columns if s in symbols}

            current_positions = executor.get_current_positions()
            results = executor.execute_target_weights(
                target_weights, portfolio_value, current_prices, current_positions
            )

            for r in results:
                click.echo(f"  {r.side} {r.symbol}: {r.filled_qty:.0f} shares @ ${r.avg_price:.2f}")

            current_weights = target_weights

        except KeyboardInterrupt:
            click.echo("\nShutting down.")
            break
        except Exception as e:
            logger.error("Trading loop error: %s", e, exc_info=True)

        click.echo(f"Sleeping {interval} hours...")
        _time.sleep(interval * 3600)


if __name__ == "__main__":
    main()
