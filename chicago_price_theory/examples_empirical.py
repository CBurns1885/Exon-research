"""
Empirical Examples - Using Real Economic Data with Models

This script demonstrates how to use both theoretical and empirical modes.

REQUIREMENTS:
- FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
- Set API key in config.json or environment variable
"""

import numpy as np
import pandas as pd
from empirical_models import (
    EmpiricalLaborMarketModel,
    EmpiricalHumanCapitalModel,
    EmpiricalSupplyDemandModel
)


def example_theoretical_mode():
    """
    Example 1: Theoretical Mode (No API Required)

    Works exactly like original models - specify all parameters.
    """
    print("=" * 80)
    print("EXAMPLE 1: THEORETICAL MODE (Default)")
    print("=" * 80)

    print("\nNo API keys needed - specify parameters manually:")

    # Labor market model
    model = EmpiricalLaborMarketModel(
        wage=25.0,
        time_endowment=16.0,
        leisure_preference=0.4
    )

    result = model.labor_supply_individual()
    print(f"\nLabor Supply:")
    print(f"  Wage: ${model.wage}/hour")
    print(f"  Hours worked: {result['hours_worked']:.2f}")
    print(f"  Labor income: ${result['labor_income']:.2f}")

    print("\n✓ Theoretical mode works without any external data!")


def example_empirical_labor_market():
    """
    Example 2: Empirical Labor Market

    Load real wage and unemployment data from FRED.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: EMPIRICAL LABOR MARKET (FRED Data)")
    print("=" * 80)

    # Read API key from config (you need to set this)
    import os
    api_key = os.environ.get('FRED_API_KEY')

    if not api_key:
        print("\nSkipping - FRED API key not found.")
        print("To run this example:")
        print("  1. Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  2. Set environment: export FRED_API_KEY='your_key'")
        return

    # Create model
    model = EmpiricalLaborMarketModel()

    # Load real data
    print("\nLoading real wage and unemployment data from FRED...")
    data = model.load_from_fred(api_key=api_key)

    print(f"\nReal Economic Data (Latest):")
    print(f"  Average hourly earnings: ${data['wage']:.2f}")
    print(f"  Unemployment rate: {data['unemployment_rate']:.1f}%")

    # Run analysis with real data
    result = model.labor_supply_individual()
    print(f"\nLabor Supply (with real wage data):")
    print(f"  Hours worked: {result['hours_worked']:.2f}")
    print(f"  Labor income: ${result['labor_income']:.2f}")

    # Market equilibrium
    eq = model.market_equilibrium(n_workers=100, n_firms=10)
    print(f"\nMarket Equilibrium:")
    print(f"  Equilibrium wage: ${eq['wage']:.2f}/hour")
    print(f"  Total employment: {eq['total_hours']:.0f} hours")

    print("\n✓ Model calibrated with real economic data!")


def example_empirical_human_capital():
    """
    Example 3: Empirical Human Capital

    Estimate returns to education from real wage data.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: EMPIRICAL HUMAN CAPITAL (Education Returns)")
    print("=" * 80)

    # Create model
    model = EmpiricalHumanCapitalModel()

    # Load real education-wage data
    print("\nLoading real education-wage data...")
    try:
        data = model.load_education_wage_data()

        print(f"\nReal Wage Data by Education:")
        for level, weekly_wage in data['wage_by_education'].items():
            annual = weekly_wage * 52
            print(f"  {level:20s}: ${weekly_wage:,.0f}/week (${annual:,.0f}/year)")

        print(f"\nCollege Premium:")
        print(f"  High school wage: ${data['high_school_wage']:,.0f}/year")
        print(f"  College wage: ${data['college_wage']:,.0f}/year")
        print(f"  Premium: {data['college_premium']:.1f}%")
        print(f"  Implied annual return: {data['implied_annual_return']:.1f}% per year")

        # Now use calibrated model
        print(f"\nModel calibrated with return = {model.education_return*100:.1f}%")

        opt = model.optimal_schooling()
        print(f"\nOptimal Education Investment:")
        print(f"  Optimal years: {opt['optimal_years']:.1f}")
        print(f"  NPV: ${opt['npv']:,.0f}")
        print(f"  IRR: {opt['internal_rate_of_return']*100:.1f}%")

        print("\n✓ Model calibrated with real wage-education data!")

    except Exception as e:
        print(f"\nNote: Using approximate education-wage data")
        print(f"(For precise data, query BLS CPS directly)")


def example_empirical_mincer():
    """
    Example 4: Estimate Mincer Equation from Micro Data

    Simulated individual-level data (in practice, use CPS or other survey data).
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: MINCER EQUATION ESTIMATION")
    print("=" * 80)

    print("\nEstimating returns to education from individual data...")
    print("(Using simulated data - in practice, use CPS, PSID, etc.)")

    # Simulate individual-level data (in practice, load real survey data)
    np.random.seed(42)
    n = 5000

    # Generate realistic education-wage data
    education = np.random.choice([12, 14, 16, 18, 20], size=n, p=[0.3, 0.2, 0.3, 0.1, 0.1])
    experience = np.random.randint(0, 40, n)
    ability = np.random.normal(0, 0.3, n)  # Unobserved ability

    # True Mincer equation with ability bias
    log_wage = (2.5 +
                0.10 * education +
                0.05 * experience -
                0.001 * experience**2 +
                0.2 * ability +  # Omitted variable
                np.random.normal(0, 0.2, n))

    wages = np.exp(log_wage)

    # Estimate Mincer equation
    model = EmpiricalHumanCapitalModel()
    results = model.estimate_mincer_equation(education, experience, wages)

    print(f"\nEstimated Mincer Equation:")
    print(f"  log(wage) = {results['intercept']:.3f}")
    print(f"              + {results['return_to_education']:.3f} * education")
    print(f"              + {results['experience_coef']:.3f} * experience")
    print(f"              + {results['experience_sq_coef']:.6f} * experience²")
    print(f"\n  R-squared: {results['r_squared']:.3f}")

    print(f"\nInterpretation:")
    print(f"  • Each year of education increases wages by {results['return_to_education']*100:.1f}%")
    print(f"  • Each year of experience increases wages by {results['experience_coef']*100:.1f}%")
    print(f"    (declining over time due to negative experience² term)")

    # Note on ability bias
    print(f"\nNote: Estimated return ({results['return_to_education']:.3f}) > True return (0.10)")
    print(f"This is ability bias - high ability people get more education AND earn more")
    print(f"Chicago methods: Use IV, twins studies, or natural experiments to address this")

    print("\n✓ Mincer equation estimated from individual data!")


def example_supply_demand_calibration():
    """
    Example 5: Calibrate Supply/Demand from Market Data
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: SUPPLY/DEMAND CALIBRATION")
    print("=" * 80)

    print("\nCalibrating demand curve from observed price-quantity pairs...")

    # Simulated market data (in practice, use real commodity prices)
    np.random.seed(42)
    n_obs = 100

    # True demand: Q = 100 - 2*P
    prices = np.random.uniform(10, 40, n_obs)
    quantities = 100 - 2*prices + np.random.normal(0, 5, n_obs)

    # Calibrate model
    model = EmpiricalSupplyDemandModel()
    params = model.calibrate_from_market_data(prices, quantities, curve_type='demand')

    print(f"\nCalibrated Demand Curve:")
    print(f"  Q = {params['intercept']:.2f} - {params['slope']:.2f} * P")
    print(f"  Price elasticity: {params['elasticity']:.2f}")
    print(f"  Mean price: ${params['mean_price']:.2f}")
    print(f"  Mean quantity: {params['mean_quantity']:.2f}")

    # Use calibrated model for analysis
    eq = model.find_equilibrium()
    print(f"\nMarket Equilibrium (with calibrated demand):")
    print(f"  Price: ${eq['price']:.2f}")
    print(f"  Quantity: {eq['quantity']:.2f}")

    print("\n✓ Supply/Demand calibrated from market data!")


def example_commodity_prices():
    """
    Example 6: Load Real Commodity Price Data
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: REAL COMMODITY PRICE DATA")
    print("=" * 80)

    import os
    api_key = os.environ.get('FRED_API_KEY')

    if not api_key:
        print("\nSkipping - FRED API key not found.")
        return

    # Load oil prices
    model = EmpiricalSupplyDemandModel()

    print("\nLoading crude oil price data from FRED...")
    data = model.load_commodity_data('oil', api_key=api_key)

    print(f"\nCommodity: {data['commodity']}")
    print(f"  Latest price: ${data['latest_price']:.2f}/barrel")
    print(f"  Average price: ${data['mean_price']:.2f}/barrel")
    print(f"  Std deviation: ${data['std_price']:.2f}")

    # Show recent price trend
    recent = data['price_data'].tail(10)
    print(f"\nRecent prices:")
    for idx, row in recent.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row['value']:.2f}")

    print("\n✓ Real commodity price data loaded!")


def main():
    """Run all examples."""
    print("\n" + "=" * 100)
    print(" " * 30 + "EMPIRICAL EXAMPLES")
    print(" " * 20 + "Chicago Price Theory with Real Economic Data")
    print("=" * 100)

    # Always works - no API needed
    example_theoretical_mode()

    # Requires API keys
    try:
        example_empirical_labor_market()
    except:
        print("\n(Skipped - API key needed)")

    try:
        example_empirical_human_capital()
    except:
        print("\n(Skipped - data unavailable)")

    # Works with simulated data
    example_empirical_mincer()
    example_supply_demand_calibration()

    # Requires API key
    try:
        example_commodity_prices()
    except:
        print("\n(Skipped - API key needed)")

    print("\n" + "=" * 100)
    print("SUMMARY: Models work in both theoretical and empirical modes")
    print("=" * 100)
    print("\nTheoretical mode: Always works, specify parameters manually")
    print("Empirical mode: Load real data (requires free API keys)")
    print("\nGet FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
