"""Tests for the options trading layer: pricing, strategy mapping,
vol surface analysis, and execution simulation.
"""

import math
import numpy as np
import pandas as pd
import pytest

from exon.options.pricing import (
    black_scholes_price,
    compute_greeks,
    full_price,
    implied_volatility,
    put_call_parity_check,
    OptionType,
    Greeks,
)
from exon.options.strategy_mapper import (
    StrategyMapper,
    StrategyMapperConfig,
    OptionsStructure,
    IVRegime,
    OptionsTradeRecommendation,
)
from exon.options.vol_surface import (
    compute_iv_rank,
    compute_iv_percentile,
    realised_volatility,
    variance_risk_premium,
    estimate_iv_from_returns,
    analyse_vol_surface,
)
from exon.options.executor import OptionsExecutionEngine
from exon.options.chain import _parse_occ_symbol, OptionContract
from exon.strategies.base import Signal


# =====================================================================
# BLACK-SCHOLES PRICING TESTS
# =====================================================================

class TestBlackScholes:
    def test_call_price_positive(self):
        price = black_scholes_price(100, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert price > 0
        assert price < 100  # can't exceed spot

    def test_put_price_positive(self):
        price = black_scholes_price(100, 100, 0.25, 0.30, 0.05, OptionType.PUT)
        assert price > 0
        assert price < 100

    def test_deep_itm_call_near_intrinsic(self):
        # Deep ITM call should be close to intrinsic value
        price = black_scholes_price(150, 100, 0.25, 0.20, 0.05, OptionType.CALL)
        assert price > 49  # intrinsic = 50, should be close

    def test_deep_otm_call_near_zero(self):
        price = black_scholes_price(50, 100, 0.10, 0.20, 0.05, OptionType.CALL)
        assert price < 1.0

    def test_higher_vol_higher_price(self):
        low_vol = black_scholes_price(100, 100, 0.25, 0.15, 0.05, OptionType.CALL)
        high_vol = black_scholes_price(100, 100, 0.25, 0.45, 0.05, OptionType.CALL)
        assert high_vol > low_vol

    def test_longer_tte_higher_price(self):
        short = black_scholes_price(100, 100, 0.10, 0.30, 0.05, OptionType.CALL)
        long = black_scholes_price(100, 100, 1.0, 0.30, 0.05, OptionType.CALL)
        assert long > short

    def test_expired_option_call(self):
        price = black_scholes_price(110, 100, 0, 0.30, 0.05, OptionType.CALL)
        assert price == 10.0  # intrinsic only

    def test_expired_option_put(self):
        price = black_scholes_price(90, 100, 0, 0.30, 0.05, OptionType.PUT)
        assert price == 10.0

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)"""
        S, K, T, vol, r = 100, 100, 0.5, 0.25, 0.05
        call = black_scholes_price(S, K, T, vol, r, OptionType.CALL)
        put = black_scholes_price(S, K, T, vol, r, OptionType.PUT)
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 0.01


class TestGreeks:
    def test_atm_call_delta_near_half(self):
        greeks = compute_greeks(100, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert 0.45 < greeks.delta < 0.65

    def test_atm_put_delta_near_neg_half(self):
        greeks = compute_greeks(100, 100, 0.25, 0.30, 0.05, OptionType.PUT)
        assert -0.65 < greeks.delta < -0.40

    def test_gamma_positive(self):
        greeks = compute_greeks(100, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert greeks.gamma > 0

    def test_theta_negative_for_long(self):
        greeks = compute_greeks(100, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert greeks.theta < 0  # time decay

    def test_vega_positive(self):
        greeks = compute_greeks(100, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert greeks.vega > 0

    def test_deep_itm_call_delta_near_one(self):
        greeks = compute_greeks(200, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert greeks.delta > 0.95

    def test_deep_otm_call_delta_near_zero(self):
        greeks = compute_greeks(50, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert greeks.delta < 0.05

    def test_expired_greeks(self):
        greeks = compute_greeks(110, 100, 0, 0.30, 0.05, OptionType.CALL)
        assert greeks.delta == 1.0  # ITM at expiry
        assert greeks.gamma == 0.0

    def test_greeks_as_dict(self):
        greeks = compute_greeks(100, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        d = greeks.as_dict
        assert "delta" in d
        assert "gamma" in d
        assert "theta" in d
        assert "vega" in d
        assert "rho" in d


class TestFullPrice:
    def test_returns_complete_output(self):
        result = full_price(100, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert result.theoretical_price > 0
        assert result.time_value > 0
        assert result.intrinsic_value == 0  # ATM
        assert result.greeks.delta > 0
        assert result.spot == 100
        assert result.strike == 100

    def test_itm_has_intrinsic(self):
        result = full_price(110, 100, 0.25, 0.30, 0.05, OptionType.CALL)
        assert result.intrinsic_value == 10.0
        assert result.time_value > 0


class TestImpliedVolatility:
    def test_round_trip(self):
        """Price → IV → Price should round-trip."""
        true_vol = 0.30
        price = black_scholes_price(100, 100, 0.25, true_vol, 0.05, OptionType.CALL)
        iv = implied_volatility(price, 100, 100, 0.25, 0.05, OptionType.CALL)
        assert abs(iv - true_vol) < 0.001

    def test_put_round_trip(self):
        true_vol = 0.40
        price = black_scholes_price(100, 105, 0.5, true_vol, 0.05, OptionType.PUT)
        iv = implied_volatility(price, 100, 105, 0.5, 0.05, OptionType.PUT)
        assert abs(iv - true_vol) < 0.001

    def test_zero_tte_returns_nan(self):
        iv = implied_volatility(5.0, 100, 95, 0, 0.05, OptionType.CALL)
        assert math.isnan(iv)

    def test_below_intrinsic_returns_nan(self):
        iv = implied_volatility(0.001, 100, 50, 0.25, 0.05, OptionType.CALL)
        assert math.isnan(iv)


class TestPutCallParity:
    def test_parity_holds(self):
        S, K, T, vol, r = 100, 100, 0.5, 0.25, 0.05
        call = black_scholes_price(S, K, T, vol, r, OptionType.CALL)
        put = black_scholes_price(S, K, T, vol, r, OptionType.PUT)
        result = put_call_parity_check(call, put, S, K, T, r)
        assert result["parity_holds"]
        assert abs(result["deviation"]) < 0.01


# =====================================================================
# STRATEGY MAPPER TESTS
# =====================================================================

def _make_signal(direction=1.0, strength=0.7, asset="AAPL", strategy_type=""):
    return Signal(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        asset=asset,
        direction=direction,
        strength=strength,
        metadata={"strategy_type": strategy_type},
    )


class TestStrategyMapper:
    def test_strong_bullish_low_iv_gets_long_call(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.20, 0.15, 0.18)
        assert rec.structure == OptionsStructure.LONG_CALL

    def test_strong_bearish_low_iv_gets_long_put(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=-1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.20, 0.15, 0.18)
        assert rec.structure == OptionsStructure.LONG_PUT

    def test_strong_bullish_high_iv_gets_debit_spread(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.45, 0.80, 0.30)
        assert rec.structure == OptionsStructure.BULL_CALL_SPREAD

    def test_strong_bearish_high_iv_gets_bear_put_spread(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=-1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.45, 0.80, 0.30)
        assert rec.structure == OptionsStructure.BEAR_PUT_SPREAD

    def test_weak_signal_high_iv_gets_iron_condor(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.1)
        rec = mapper.map_signal(sig, 150, 0.45, 0.80, 0.30)
        assert rec.structure == OptionsStructure.IRON_CONDOR

    def test_weak_signal_low_iv_gets_long_strangle(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.1)
        rec = mapper.map_signal(sig, 150, 0.15, 0.15, 0.12)
        assert rec.structure == OptionsStructure.LONG_STRANGLE

    def test_breakout_low_iv_gets_straddle(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.5, strategy_type="breakout")
        rec = mapper.map_signal(sig, 150, 0.15, 0.15, 0.12)
        assert rec.structure == OptionsStructure.LONG_STRADDLE

    def test_mean_reversion_high_iv_gets_short_strangle(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=-1.0, strength=0.5, strategy_type="mean_reversion")
        rec = mapper.map_signal(sig, 150, 0.45, 0.80, 0.30)
        assert rec.structure == OptionsStructure.SHORT_STRANGLE

    def test_medium_conviction_bullish_high_iv_gets_credit_spread(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.4)
        rec = mapper.map_signal(sig, 150, 0.35, 0.75, 0.25)
        assert rec.structure == OptionsStructure.BULL_PUT_SPREAD

    def test_recommendation_has_legs(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.20, 0.15, 0.18)
        assert len(rec.legs) > 0

    def test_iron_condor_has_four_legs(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.1)
        rec = mapper.map_signal(sig, 150, 0.45, 0.80, 0.30)
        assert rec.structure == OptionsStructure.IRON_CONDOR
        assert len(rec.legs) == 4

    def test_spread_has_two_legs(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.45, 0.80, 0.30)
        assert rec.structure == OptionsStructure.BULL_CALL_SPREAD
        assert len(rec.legs) == 2

    def test_metadata_populated(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.20, 0.15, 0.18)
        assert "iv_regime" in rec.metadata
        assert "spot_price" in rec.metadata
        assert rec.conviction == 0.8

    def test_map_signals_batch(self):
        mapper = StrategyMapper()
        signals = [
            _make_signal(1.0, 0.8, "AAPL"),
            _make_signal(-1.0, 0.6, "MSFT"),
        ]
        recs = mapper.map_signals(
            signals,
            {"AAPL": 150, "MSFT": 300},
            {"AAPL": 0.25, "MSFT": 0.40},
            {"AAPL": 0.30, "MSFT": 0.75},
            {"AAPL": 0.20, "MSFT": 0.30},
        )
        assert len(recs) == 2
        assert recs[0].underlying == "AAPL"
        assert recs[1].underlying == "MSFT"

    def test_defined_risk_flag(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.45, 0.80, 0.30)
        # Bull call spread is defined risk
        assert rec.is_defined_risk

    def test_risk_reward_ratio(self):
        mapper = StrategyMapper()
        sig = _make_signal(direction=1.0, strength=0.8)
        rec = mapper.map_signal(sig, 150, 0.45, 0.80, 0.30)
        assert rec.risk_reward_ratio > 0


class TestIVRegime:
    def test_classify_low(self):
        mapper = StrategyMapper()
        assert mapper._classify_iv(0.10) == IVRegime.LOW

    def test_classify_normal(self):
        mapper = StrategyMapper()
        assert mapper._classify_iv(0.50) == IVRegime.NORMAL

    def test_classify_high(self):
        mapper = StrategyMapper()
        assert mapper._classify_iv(0.85) == IVRegime.HIGH


# =====================================================================
# VOL SURFACE TESTS
# =====================================================================

class TestVolSurface:
    def _make_returns(self, n=300, drift=0.0003, vol=0.015, seed=42):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(drift, vol, n))

    def test_iv_rank_at_high(self):
        ivs = pd.Series([0.15, 0.20, 0.25, 0.30, 0.35])
        rank = compute_iv_rank(0.35, ivs)
        assert rank == 1.0

    def test_iv_rank_at_low(self):
        ivs = pd.Series([0.15, 0.20, 0.25, 0.30, 0.35])
        rank = compute_iv_rank(0.15, ivs)
        assert rank == 0.0

    def test_iv_rank_midpoint(self):
        ivs = pd.Series([0.10, 0.20, 0.30, 0.40, 0.50])
        rank = compute_iv_rank(0.30, ivs)
        assert abs(rank - 0.5) < 0.01

    def test_iv_percentile(self):
        ivs = pd.Series([0.10, 0.20, 0.30, 0.40, 0.50])
        pctile = compute_iv_percentile(0.35, ivs)
        assert pctile == 0.6  # 3 out of 5 below 0.35

    def test_realised_vol(self):
        returns = self._make_returns()
        rv = realised_volatility(returns, window=21)
        assert len(rv) == len(returns)
        assert rv.iloc[-1] > 0

    def test_vrp_positive_means_sell(self):
        vrp = variance_risk_premium(0.30, 0.20)
        assert abs(vrp - 0.10) < 1e-9  # IV > RV → sell premium

    def test_vrp_negative_means_buy(self):
        vrp = variance_risk_premium(0.15, 0.25)
        assert vrp == -0.10  # IV < RV → buy premium

    def test_estimate_iv_from_returns(self):
        returns = self._make_returns(n=100)
        est = estimate_iv_from_returns(returns)
        assert "rv_5d" in est
        assert "rv_21d" in est
        assert "ewma_vol" in est
        assert "iv_proxy" in est
        assert est["iv_proxy"] > 0

    def test_analyse_vol_surface(self):
        returns = self._make_returns(n=300)
        snap = analyse_vol_surface("AAPL", returns)
        assert snap.underlying == "AAPL"
        assert snap.atm_iv > 0
        assert 0 <= snap.iv_rank <= 1
        assert 0 <= snap.iv_percentile <= 1
        assert snap.realised_vol > 0

    def test_analyse_with_provided_iv(self):
        returns = self._make_returns(n=300)
        snap = analyse_vol_surface("MSFT", returns, current_iv=0.35)
        assert snap.atm_iv == 0.35


# =====================================================================
# OCC SYMBOL PARSING TESTS
# =====================================================================

class TestOCCParsing:
    def test_parse_call(self):
        result = _parse_occ_symbol("AAPL250321C00150000")
        assert result["underlying"] == "AAPL"
        assert result["type"] == "call"
        assert result["strike"] == 150.0
        assert result["expiration"] == "2025-03-21"

    def test_parse_put(self):
        result = _parse_occ_symbol("SPY260115P00580000")
        assert result["underlying"] == "SPY"
        assert result["type"] == "put"
        assert result["strike"] == 580.0
        assert result["expiration"] == "2026-01-15"

    def test_parse_fractional_strike(self):
        result = _parse_occ_symbol("TSLA250620C00250500")
        assert result["strike"] == 250.5

    def test_option_contract_mid_price(self):
        c = OptionContract(
            symbol="TEST", underlying="TEST",
            expiration="2025-06-20", strike=100,
            option_type="call", bid=2.50, ask=3.00,
        )
        assert c.mid_price == 2.75

    def test_option_contract_spread(self):
        c = OptionContract(
            symbol="TEST", underlying="TEST",
            expiration="2025-06-20", strike=100,
            option_type="call", bid=2.50, ask=3.00,
        )
        assert c.spread == 0.50
        assert abs(c.spread_pct - 0.50 / 2.75) < 0.01


# =====================================================================
# OPTIONS EXECUTION TESTS
# =====================================================================

class TestOptionsExecution:
    def test_dry_run_simulation(self):
        executor = OptionsExecutionEngine(dry_run=True)
        rec = OptionsTradeRecommendation(
            underlying="AAPL",
            structure=OptionsStructure.LONG_CALL,
            legs=[],
            direction=1.0,
            max_loss=500,
            max_gain=float("inf"),
            target_dte=30,
            conviction=0.8,
            metadata={"spot_price": 150, "current_iv": 0.25},
        )
        # Add a leg
        from exon.options.strategy_mapper import OptionsLeg
        rec.legs = [OptionsLeg(symbol="", side="buy", option_type="call", strike=150)]

        result = executor.execute_recommendation(rec, qty=1)
        assert result.status == "FILLED_SIM"
        assert result.net_premium < 0  # buying costs money
        assert result.underlying == "AAPL"

    def test_multi_leg_simulation(self):
        executor = OptionsExecutionEngine(dry_run=True)
        from exon.options.strategy_mapper import OptionsLeg
        rec = OptionsTradeRecommendation(
            underlying="SPY",
            structure=OptionsStructure.IRON_CONDOR,
            legs=[
                OptionsLeg(symbol="", side="sell", option_type="put", strike=570),
                OptionsLeg(symbol="", side="buy", option_type="put", strike=560),
                OptionsLeg(symbol="", side="sell", option_type="call", strike=610),
                OptionsLeg(symbol="", side="buy", option_type="call", strike=620),
            ],
            direction=0.0,
            max_loss=500,
            max_gain=200,
            target_dte=30,
            conviction=0.3,
            metadata={"spot_price": 590, "current_iv": 0.18},
        )
        result = executor.execute_recommendation(rec, qty=2)
        assert result.status == "FILLED_SIM"
        assert len(result.legs) == 4
        # Iron condor = net credit (sell premium)
        assert result.net_premium > 0

    def test_order_history_tracked(self):
        executor = OptionsExecutionEngine(dry_run=True)
        from exon.options.strategy_mapper import OptionsLeg
        rec = OptionsTradeRecommendation(
            underlying="MSFT",
            structure=OptionsStructure.LONG_PUT,
            legs=[OptionsLeg(symbol="", side="buy", option_type="put", strike=400)],
            direction=-1.0,
            max_loss=300,
            max_gain=float("inf"),
            target_dte=30,
            conviction=0.7,
            metadata={"spot_price": 400, "current_iv": 0.30},
        )
        executor.execute_recommendation(rec, qty=1)
        executor.execute_recommendation(rec, qty=1)
        assert len(executor.order_history) == 2

    def test_max_contracts_enforced(self):
        executor = OptionsExecutionEngine(dry_run=True, max_contracts=5)
        from exon.options.strategy_mapper import OptionsLeg
        rec = OptionsTradeRecommendation(
            underlying="AAPL",
            structure=OptionsStructure.LONG_CALL,
            legs=[OptionsLeg(symbol="", side="buy", option_type="call", strike=150)],
            direction=1.0,
            max_loss=500,
            max_gain=float("inf"),
            target_dte=30,
            conviction=0.8,
            metadata={"spot_price": 150, "current_iv": 0.25},
        )
        result = executor.execute_recommendation(rec, qty=100)  # should cap
        assert result.legs[0]["qty"] == 5  # capped at max_contracts

    def test_portfolio_greeks_stub(self):
        executor = OptionsExecutionEngine(dry_run=True)
        greeks = executor.get_portfolio_greeks()
        assert "net_delta" in greeks
        assert "net_gamma" in greeks
        assert "net_theta" in greeks
        assert "net_vega" in greeks


# =====================================================================
# INTEGRATION: EQUITY SIGNAL → OPTIONS TRADE
# =====================================================================

class TestEndToEnd:
    def test_full_pipeline(self):
        """Equity signal → vol surface → strategy mapping → execution."""
        # 1. Generate a signal
        sig = _make_signal(direction=1.0, strength=0.75, asset="AAPL")

        # 2. Analyse vol surface
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0003, 0.015, 300))
        snap = analyse_vol_surface("AAPL", returns)

        # 3. Map to options structure
        mapper = StrategyMapper()
        rec = mapper.map_signal(
            sig,
            spot_price=175.0,
            current_iv=snap.atm_iv,
            iv_percentile=snap.iv_percentile,
            historical_vol=snap.realised_vol,
        )
        assert rec.underlying == "AAPL"
        assert len(rec.legs) > 0

        # 4. Execute (dry run)
        executor = OptionsExecutionEngine(dry_run=True)
        result = executor.execute_recommendation(rec, qty=1)
        assert result.status == "FILLED_SIM"

    def test_batch_pipeline(self):
        """Multiple signals → batch mapping → execution."""
        signals = [
            _make_signal(1.0, 0.8, "AAPL"),
            _make_signal(-1.0, 0.6, "MSFT"),
            _make_signal(0.5, 0.1, "SPY"),  # weak → vol trade
        ]

        mapper = StrategyMapper()
        recs = mapper.map_signals(
            signals,
            {"AAPL": 175, "MSFT": 400, "SPY": 590},
            {"AAPL": 0.22, "MSFT": 0.35, "SPY": 0.15},
            {"AAPL": 0.25, "MSFT": 0.75, "SPY": 0.20},
            {"AAPL": 0.18, "MSFT": 0.28, "SPY": 0.12},
        )
        assert len(recs) == 3

        executor = OptionsExecutionEngine(dry_run=True)
        for rec in recs:
            result = executor.execute_recommendation(rec, qty=1)
            assert result.status == "FILLED_SIM"

    def test_config_loading(self):
        """Verify options config is present and valid."""
        from pathlib import Path
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "equities.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "options" in cfg
        opt = cfg["options"]
        assert opt["enabled"] is True
        assert "options_universe" in opt
        assert len(opt["options_universe"]) >= 5
        assert opt["max_contracts"] == 10
