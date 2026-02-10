"""Command-line interface for the Exon Research trading platform.

Commands:
    exon research   — Run correlation, cointegration, PCA, and regime analysis
    exon backtest   — Backtest strategies on historical data
    exon trade      — Run live/paper trading loop
    exon scan-pairs — Scan universe for cointegrated pairs
"""

from __future__ import annotations

import json
import logging
import sys

import click
import pandas as pd

from .config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("exon")


def _build_client(cfg: dict):
    from .data.coinbase_client import CoinbaseClient, CoinbaseAuth

    cb_cfg = cfg.get("coinbase", {})
    auth = None
    if cb_cfg.get("api_key") and cb_cfg.get("api_secret"):
        auth = CoinbaseAuth(api_key=cb_cfg["api_key"], api_secret=cb_cfg["api_secret"])
    return CoinbaseClient(auth=auth)


def _build_pipeline(cfg: dict):
    from .data.market_data import MarketDataPipeline

    client = _build_client(cfg)
    return MarketDataPipeline(client=client)


def _build_strategies(cfg: dict) -> list:
    from .strategies.momentum import (
        TimeSeriesMomentum,
        CrossSectionalMomentum,
        AdaptiveMomentum,
    )
    from .strategies.mean_reversion import (
        BollingerMeanReversion,
        OUMeanReversion,
        MultiTimeframeMeanReversion,
    )
    from .strategies.pairs import StatArbPCA

    strat_cfg = cfg.get("strategies", {})
    strategies = []

    if strat_cfg.get("ts_momentum", {}).get("enabled"):
        s = strat_cfg["ts_momentum"]
        strategies.append(
            (TimeSeriesMomentum(lookbacks=s.get("lookbacks"), vol_target=s.get("vol_target", 0.15)),
             s.get("allocation", 0.2))
        )

    if strat_cfg.get("xs_momentum", {}).get("enabled"):
        s = strat_cfg["xs_momentum"]
        strategies.append(
            (CrossSectionalMomentum(
                lookback=s.get("lookback", 168),
                top_n=s.get("top_n", 3),
                bottom_n=s.get("bottom_n", 3),
            ), s.get("allocation", 0.15))
        )

    if strat_cfg.get("adaptive_momentum", {}).get("enabled"):
        s = strat_cfg["adaptive_momentum"]
        strategies.append(
            (AdaptiveMomentum(base_lookback=s.get("base_lookback", 168)),
             s.get("allocation", 0.15))
        )

    if strat_cfg.get("bollinger_mr", {}).get("enabled"):
        s = strat_cfg["bollinger_mr"]
        strategies.append(
            (BollingerMeanReversion(window=s.get("window", 48), entry_std=s.get("entry_std", 2.0)),
             s.get("allocation", 0.15))
        )

    if strat_cfg.get("ou_mean_reversion", {}).get("enabled"):
        s = strat_cfg["ou_mean_reversion"]
        strategies.append(
            (OUMeanReversion(calibration_window=s.get("calibration_window", 720)),
             s.get("allocation", 0.10))
        )

    if strat_cfg.get("stat_arb_pca", {}).get("enabled"):
        s = strat_cfg["stat_arb_pca"]
        strategies.append(
            (StatArbPCA(
                n_factors=s.get("n_factors", 3),
                z_entry=s.get("z_entry", 1.5),
            ), s.get("allocation", 0.20))
        )

    return strategies


@click.group()
@click.option("--config", "-c", default=None, help="Path to config YAML")
@click.pass_context
def main(ctx, config):
    """Exon Research — Quantitative crypto trading platform."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@main.command()
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.pass_context
def research(ctx, start, end):
    """Run research analysis: correlations, cointegration, PCA, regimes."""
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
    symbols = uni.get("symbols", ["BTC-USD", "ETH-USD"])
    granularity = uni.get("granularity", "ONE_HOUR")

    if not start:
        start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=uni.get("lookback_days", 90))).strftime("%Y-%m-%d")
    if not end:
        end = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    click.echo(f"Fetching data for {len(symbols)} assets: {start} -> {end}")
    prices = pipeline.fetch_universe(symbols, start, end, granularity)
    if prices.empty:
        click.echo("No data returned. Check your API key and symbols.")
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
    clusters = cluster_assets(returns, n_clusters=min(4, len(symbols)))
    click.echo(clusters.to_string())

    click.echo("\n=== COINTEGRATED PAIRS ===")
    coint_pairs = scan_cointegrated_pairs(prices)
    if coint_pairs.empty:
        click.echo("No cointegrated pairs found at p<0.05")
    else:
        click.echo(coint_pairs.to_string())

    click.echo("\n=== PCA DECOMPOSITION ===")
    pca_result = decompose_returns(returns, n_components=min(5, len(symbols) - 1))
    click.echo(f"Components: {pca_result.n_components}")
    click.echo(f"Variance explained: {pca_result.cumulative_variance}")
    click.echo("Factor loadings (PC1):")
    click.echo(pca_result.components.iloc[0].sort_values(ascending=False).to_string())

    click.echo("\n=== REGIME DETECTION (BTC) ===")
    if "BTC-USD" in returns.columns:
        regime_df = detect_regimes_gmm(returns["BTC-USD"])
        stats = regime_statistics(regime_df)
        click.echo(stats.to_string())

    pipeline.save(prices, "prices_latest")
    pipeline.save(returns, "returns_latest")
    click.echo("\nData saved to data/")


@main.command()
@click.option("--start", default=None, help="Start date")
@click.option("--end", default=None, help="End date")
@click.option("--strategy", "-s", default=None, help="Single strategy to test (or 'all')")
@click.pass_context
def backtest(ctx, start, end, strategy):
    """Backtest trading strategies on historical data."""
    from .backtest.engine import BacktestEngine, BacktestConfig
    from .backtest.metrics import compare_strategies, monte_carlo_sharpe_test, drawdown_analysis
    from .strategies.composite import CompositeStrategy

    cfg = ctx.obj["config"]
    pipeline = _build_pipeline(cfg)
    uni = cfg.get("universe", {})
    symbols = uni.get("symbols", ["BTC-USD", "ETH-USD"])

    if not start:
        start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=uni.get("lookback_days", 90))).strftime("%Y-%m-%d")
    if not end:
        end = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    click.echo(f"Fetching data for backtest: {start} -> {end}")
    prices = pipeline.fetch_universe(symbols, start, end, uni.get("granularity", "ONE_HOUR"))
    if prices.empty:
        click.echo("No data.")
        return

    bt_cfg = cfg.get("backtest", {})
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=bt_cfg.get("initial_capital", 100_000),
            maker_fee=bt_cfg.get("maker_fee", 0.004),
            taker_fee=bt_cfg.get("taker_fee", 0.006),
            slippage_bps=bt_cfg.get("slippage_bps", 5),
            use_maker=bt_cfg.get("use_maker", True),
        )
    )

    strategy_pairs = _build_strategies(cfg)
    results = []

    # Run individual strategies
    for strat, alloc in strategy_pairs:
        if strategy and strategy != "all" and strat.name != strategy:
            continue
        click.echo(f"\nBacktesting: {strat.name}")
        result = engine.run(strat, prices)
        results.append(result)
        summary = result.summary()
        for k, v in summary.items():
            click.echo(f"  {k}: {v}")

    # Run composite
    if not strategy or strategy == "all":
        composite = CompositeStrategy(strategy_pairs, regime_aware=True)
        click.echo(f"\nBacktesting: composite (all strategies blended)")
        result = engine.run(composite, prices)
        results.append(result)
        summary = result.summary()
        for k, v in summary.items():
            click.echo(f"  {k}: {v}")

        # Statistical significance test
        click.echo("\n=== SHARPE RATIO SIGNIFICANCE ===")
        mc = monte_carlo_sharpe_test(result.returns)
        for k, v in mc.items():
            click.echo(f"  {k}: {v}")

    # Comparison table
    if len(results) > 1:
        click.echo("\n=== STRATEGY COMPARISON ===")
        comp = compare_strategies(results)
        click.echo(comp.to_string())


@main.command()
@click.pass_context
def scan_pairs(ctx):
    """Scan universe for cointegrated pairs."""
    from .research.cointegration import scan_cointegrated_pairs, half_life

    cfg = ctx.obj["config"]
    pipeline = _build_pipeline(cfg)
    uni = cfg.get("universe", {})
    symbols = uni.get("symbols", [])

    start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=uni.get("lookback_days", 90))).strftime("%Y-%m-%d")
    end = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    click.echo(f"Scanning {len(symbols)} assets for cointegrated pairs...")
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
@click.option("--interval", default=4, help="Hours between rebalances")
@click.pass_context
def trade(ctx, interval):
    """Run live/paper trading loop."""
    import time as _time
    from .strategies.composite import CompositeStrategy
    from .execution.executor import ExecutionEngine
    from .risk.manager import RiskManager, RiskLimits
    from .portfolio.optimizer import risk_parity, constrain_turnover

    cfg = ctx.obj["config"]
    client = _build_client(cfg)
    pipeline = _build_pipeline(cfg)
    uni = cfg.get("universe", {})
    symbols = uni.get("symbols", [])

    exec_cfg = cfg.get("execution", {})
    executor = ExecutionEngine(
        client=client,
        dry_run=exec_cfg.get("dry_run", True),
        max_order_value_usd=exec_cfg.get("max_order_value_usd", 10_000),
    )

    risk_cfg = cfg.get("risk", {})
    risk_mgr = RiskManager(
        RiskLimits(
            max_position_pct=risk_cfg.get("max_position_pct", 0.20),
            max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.15),
            max_leverage=risk_cfg.get("max_leverage", 1.0),
        )
    )

    strategy_pairs = _build_strategies(cfg)
    composite = CompositeStrategy(strategy_pairs, regime_aware=True)

    mode = "PAPER" if exec_cfg.get("dry_run", True) else "LIVE"
    click.echo(f"Starting {mode} trading loop (rebalance every {interval}h)")
    click.echo(f"Universe: {symbols}")

    current_weights = pd.Series(dtype=float)
    portfolio_value = cfg.get("backtest", {}).get("initial_capital", 100_000)

    while True:
        try:
            click.echo(f"\n--- Rebalance at {pd.Timestamp.now('UTC')} ---")

            # Fetch recent data
            end = pd.Timestamp.now("UTC")
            start = end - pd.Timedelta(days=uni.get("lookback_days", 90))
            prices = pipeline.fetch_universe(
                symbols, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            )
            if prices.empty:
                click.echo("No data, skipping.")
                _time.sleep(interval * 3600)
                continue

            returns = pipeline.get_returns(prices)

            # Generate signals
            signals = composite.generate_signals(prices)
            target_weights = composite.signals_to_weights(signals)

            # Portfolio optimisation
            if len(target_weights) > 1:
                cov = returns[target_weights.index.intersection(returns.columns)].cov()
                target_weights = risk_parity(cov)

            # Constrain turnover
            target_weights = constrain_turnover(target_weights, current_weights)

            # Risk check
            report = risk_mgr.check_portfolio(
                target_weights, portfolio_value, returns
            )
            if not report.is_within_limits:
                click.echo(f"Risk breaches: {report.breaches}")
                target_weights = risk_mgr.adjust_weights(
                    target_weights, portfolio_value, returns
                )

            click.echo(f"Target weights:\n{target_weights.to_string()}")

            # Execute
            current_prices = executor.get_current_prices(symbols)
            if not current_prices:
                current_prices = {s: prices[s].iloc[-1] for s in prices.columns}

            current_positions = executor.get_current_positions()
            results = executor.execute_target_weights(
                target_weights, portfolio_value, current_prices, current_positions
            )

            for r in results:
                click.echo(f"  {r.side} {r.product_id}: {r.filled_size:.6f} @ ${r.avg_price:.2f}")

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
