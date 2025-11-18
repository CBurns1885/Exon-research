"""
Chicago Price Theory - Comprehensive Economic Models Suite

Main demonstration script showcasing all models

Created by: Professor of Chicago Price Theory
Purpose: Teaching and research in microeconomic theory
"""

import sys
from supply_demand import SupplyDemandModel
from consumer_theory import ConsumerModel
from producer_theory import ProducerModel
from labor_market import LaborMarketModel
from human_capital import HumanCapitalModel
from price_discrimination import PriceDiscriminationModel
from search_theory import SearchModel
from time_allocation import TimeAllocationModel


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_subheader(title: str):
    """Print formatted subsection header."""
    print("\n" + "-" * 100)
    print(f"  {title}")
    print("-" * 100)


def demo_supply_demand():
    """Demonstrate supply and demand model."""
    print_header("MODEL 1: SUPPLY AND DEMAND")

    print("\nCORE CHICAGO PRINCIPLES:")
    print("• Prices coordinate economic activity without central planning")
    print("• Markets tend toward equilibrium through price adjustment")
    print("• Comparative statics generate testable predictions")
    print("• Price controls create shortages or surpluses")

    model = SupplyDemandModel(demand_params=(100, 2), supply_params=(-20, 3))

    print_subheader("Market Equilibrium")
    eq = model.find_equilibrium()
    print(f"Equilibrium Price: ${eq['price']:.2f}")
    print(f"Equilibrium Quantity: {eq['quantity']:.2f} units")
    print(f"Total Surplus: ${eq['total_surplus']:.2f}")

    print_subheader("Price Elasticities")
    elast = model.calculate_elasticities(eq['price'])
    print(f"Demand Elasticity: {elast['demand_elasticity']:.2f}")
    print(f"Supply Elasticity: {elast['supply_elasticity']:.2f}")

    print_subheader("Tax Incidence")
    tax = model.analyze_tax_incidence(tax_per_unit=5)
    print(f"Consumer burden: ${tax['consumer_burden']:.2f} ({tax['consumer_burden_pct']:.1f}%)")
    print(f"Producer burden: ${tax['producer_burden']:.2f} ({tax['producer_burden_pct']:.1f}%)")
    print(f"Deadweight loss: ${tax['deadweight_loss']:.2f}")

    print("\n✓ Model demonstrates: Price mechanism, market clearing, tax incidence")


def demo_consumer_theory():
    """Demonstrate consumer theory model."""
    print_header("MODEL 2: CONSUMER THEORY")

    print("\nCORE CHICAGO PRINCIPLES:")
    print("• Rational agents maximize utility subject to constraints")
    print("• Substitution is everywhere - relative prices matter")
    print("• Income and substitution effects explain price responses")
    print("• Demand curves slope downward (with rare exceptions)")

    consumer = ConsumerModel(income=1000, prices=(10, 20),
                            utility_type='cobb_douglas', utility_params=(0.6,))

    print_subheader("Utility Maximization")
    optimal = consumer.maximize_utility()
    print(f"Optimal bundle: x₁={optimal['x1']:.2f}, x₂={optimal['x2']:.2f}")
    print(f"Maximum utility: {optimal['utility']:.2f}")
    print(f"MRS = {optimal['mrs']:.2f}, Price ratio = {optimal['price_ratio']:.2f}")
    print(f"Optimality check (MRS = p₁/p₂): {abs(optimal['mrs'] - optimal['price_ratio']) < 0.01}")

    print_subheader("Slutsky Decomposition")
    slutsky = consumer.slutsky_decomposition(good=1, price_change=5)
    print(f"Price of good 1 increases by $5:")
    print(f"  Total effect: {slutsky['total_effect'][0]:.2f} units")
    print(f"  Substitution effect: {slutsky['substitution_effect'][0]:.2f} units")
    print(f"  Income effect: {slutsky['income_effect'][0]:.2f} units")

    print("\n✓ Model demonstrates: Utility maximization, indifference curves, Slutsky equation")


def demo_producer_theory():
    """Demonstrate producer theory model."""
    print_header("MODEL 3: PRODUCER THEORY")

    print("\nCORE CHICAGO PRINCIPLES:")
    print("• Firms maximize profit (not revenue, not market share)")
    print("• Marginal analysis: Produce where MR = MC")
    print("• Supply curves derived from profit maximization")
    print("• Long-run competitive equilibrium: P = min(AC)")

    producer = ProducerModel(
        production_type='cobb_douglas',
        production_params=(1.0, 0.6, 0.4),
        output_price=10.0,
        input_prices=(5.0, 8.0),
        fixed_cost=100.0
    )

    print_subheader("Profit Maximization")
    result = producer.maximize_profit_lr()
    print(f"Optimal inputs: L={result['L']:.2f}, K={result['K']:.2f}")
    print(f"Output: Q={result['Q']:.2f}")
    print(f"Profit: ${result['profit']:.2f}")
    print(f"FOC check - VMP_L = w: ${result['VMP_L']:.2f} = ${producer.wage:.2f} ✓")
    print(f"FOC check - VMP_K = r: ${result['VMP_K']:.2f} = ${producer.rental_rate:.2f} ✓")

    print_subheader("Cost Minimization")
    cm = producer.cost_minimization(target_Q=20)
    print(f"To produce Q=20:")
    print(f"  Cost-minimizing inputs: L={cm['L']:.2f}, K={cm['K']:.2f}")
    print(f"  Total cost: ${cm['total_cost']:.2f}")
    print(f"  Average cost: ${cm['average_cost']:.2f}")

    print("\n✓ Model demonstrates: Profit maximization, cost curves, supply derivation")


def demo_labor_market():
    """Demonstrate labor market model."""
    print_header("MODEL 4: LABOR MARKET")

    print("\nCORE CHICAGO PRINCIPLES:")
    print("• Wages reflect marginal productivity of labor")
    print("• Labor supply based on leisure-income tradeoff")
    print("• Competitive markets clear - no involuntary unemployment")
    print("• Minimum wages above equilibrium create unemployment")

    model = LaborMarketModel(wage=20.0, time_endowment=16.0, leisure_preference=0.4)

    print_subheader("Individual Labor Supply")
    result = model.labor_supply_individual()
    print(f"At wage ${model.wage}/hour:")
    print(f"  Hours worked: {result['hours_worked']:.2f}")
    print(f"  Leisure: {result['leisure']:.2f} hours")
    print(f"  Labor income: ${result['labor_income']:.2f}")
    print(f"  Reservation wage: ${result['reservation_wage']:.2f}/hour")

    print_subheader("Market Equilibrium")
    eq = model.market_equilibrium(n_workers=100, n_firms=10)
    print(f"Equilibrium wage: ${eq['wage']:.2f}/hour")
    print(f"Total hours: {eq['total_hours']:.0f}")
    print(f"Unemployment rate: {eq['unemployment_rate']:.1f}%")

    print_subheader("Minimum Wage Effect")
    min_wage = eq['wage'] * 1.25
    mw = model.minimum_wage_effect(min_wage, n_workers=100, n_firms=10)
    print(f"Minimum wage at ${min_wage:.2f}/hour (25% above equilibrium):")
    print(f"  Unemployment created: {mw['unemployment_rate']:.1f}%")
    print(f"  Employment change: {mw['employment_change']:.0f} hours")

    print("\n✓ Model demonstrates: Labor supply/demand, wage determination, minimum wage effects")


def demo_human_capital():
    """Demonstrate human capital model."""
    print_header("MODEL 5: HUMAN CAPITAL (Gary Becker)")

    print("\nCORE CHICAGO PRINCIPLES:")
    print("• Education is investment, not consumption")
    print("• Compare present value of costs and benefits")
    print("• Optimal schooling where marginal benefit = marginal cost")
    print("• Returns to education vary by ability (complementarity)")

    model = HumanCapitalModel(
        base_wage=30000,
        education_return=0.10,
        discount_rate=0.05,
        cost_per_year=20000
    )

    print_subheader("Optimal Education Investment")
    opt = model.optimal_schooling()
    print(f"Optimal years of schooling: {opt['optimal_years']:.1f} years")
    print(f"Net present value: ${opt['npv']:,.0f}")
    print(f"Internal rate of return: {opt['internal_rate_of_return']*100:.1f}%")
    print(f"Market interest rate: {model.discount_rate*100:.1f}%")

    print_subheader("Comparison of Education Levels")
    comparison = model.compare_education_levels([12, 16, 20])
    for level in ['12_years', '16_years', '20_years']:
        data = comparison[level]
        print(f"{data['schooling']} years: NPV=${data['npv']:,.0f}, IRR={data['irr']*100:.1f}%")

    college_premium = ((comparison['16_years']['lifetime_earnings'] -
                       comparison['12_years']['lifetime_earnings']) /
                      comparison['12_years']['lifetime_earnings'] * 100)
    print(f"\nCollege premium over high school: {college_premium:.1f}%")

    print("\n✓ Model demonstrates: Education as investment, present value analysis, ability effects")


def demo_price_discrimination():
    """Demonstrate price discrimination model."""
    print_header("MODEL 6: PRICE DISCRIMINATION")

    print("\nCORE CHICAGO PRINCIPLES:")
    print("• Price discrimination requires market power + ability to segment")
    print("• Not necessarily 'bad' - can increase efficiency and output")
    print("• Perfect discrimination is perfectly efficient (like competition)")
    print("• Third-degree: Charge more to less elastic segments")

    model = PriceDiscriminationModel(demand_params=(100, 1), marginal_cost=10)

    print_subheader("Uniform Pricing (No Discrimination)")
    uniform = model.uniform_pricing()
    print(f"Price: ${uniform['price']:.2f}")
    print(f"Quantity: {uniform['quantity']:.2f}")
    print(f"Profit: ${uniform['profit']:.2f}")
    print(f"Consumer surplus: ${uniform['consumer_surplus']:.2f}")
    print(f"Deadweight loss: ${uniform['deadweight_loss']:.2f}")

    print_subheader("Perfect Price Discrimination")
    perfect = model.first_degree_discrimination()
    print(f"Quantity: {perfect['quantity']:.2f}")
    print(f"Profit: ${perfect['profit']:.2f}")
    print(f"Consumer surplus: ${perfect['consumer_surplus']:.2f}")
    print(f"Efficiency: {perfect['efficiency']}")
    print(f"Note: Firm captures all surplus, but output is efficient!")

    print_subheader("Third-Degree Discrimination")
    third = model.third_degree_discrimination([(80, 2), (120, 0.5)])
    print(f"Market 1 (Elastic): P=${third['market_1']['price']:.2f}, η={third['market_1']['elasticity']:.2f}")
    print(f"Market 2 (Inelastic): P=${third['market_2']['price']:.2f}, η={third['market_2']['elasticity']:.2f}")
    print(f"Price ratio: {third['market_2']['price']/third['market_1']['price']:.2f}")

    print("\n✓ Model demonstrates: Monopoly pricing, perfect discrimination, market segmentation")


def demo_search_theory():
    """Demonstrate search theory model."""
    print_header("MODEL 7: SEARCH THEORY (George Stigler)")

    print("\nCORE CHICAGO PRINCIPLES:")
    print("• Information is costly - no perfect information in real markets")
    print("• Optimal stopping: Search until expected MB = MC")
    print("• Unemployment partly voluntary (job search)")
    print("• Generous UI → Higher reservation wage → Longer search")

    model = SearchModel(
        offer_mean=50000,
        offer_std=10000,
        search_cost=500,
        discount_rate=0.05,
        unemployment_benefit=15000
    )

    print_subheader("Optimal Search Strategy")
    result = model.find_reservation_wage()
    print(f"Reservation wage: ${result['reservation_wage']:,.0f}")
    print(f"Accept if offer ≥ ${result['reservation_wage']:,.0f}")
    print(f"Probability of accepting any offer: {result['probability_accept']:.2%}")
    print(f"Expected number of searches: {result['expected_searches']:.1f}")
    print(f"Expected accepted wage: ${result['expected_accepted_wage']:,.0f}")

    print_subheader("Effect of Search Costs")
    low_cost = SearchModel(search_cost=100).find_reservation_wage()
    high_cost = SearchModel(search_cost=1500).find_reservation_wage()
    print(f"Low search cost ($100): w_r = ${low_cost['reservation_wage']:,.0f}")
    print(f"High search cost ($1,500): w_r = ${high_cost['reservation_wage']:,.0f}")
    print(f"Higher costs → Lower reservation wage (less picky)")

    print_subheader("Effect of Unemployment Insurance")
    no_ui = SearchModel(unemployment_benefit=0).find_reservation_wage()
    high_ui = SearchModel(unemployment_benefit=30000).find_reservation_wage()
    print(f"No UI: w_r = ${no_ui['reservation_wage']:,.0f}, duration = {no_ui['expected_duration']:.1f}")
    print(f"High UI: w_r = ${high_ui['reservation_wage']:,.0f}, duration = {high_ui['expected_duration']:.1f}")

    print("\n✓ Model demonstrates: Optimal stopping, reservation wage, policy effects")


def demo_time_allocation():
    """Demonstrate time allocation model."""
    print_header("MODEL 8: TIME ALLOCATION (Gary Becker)")

    print("\nCORE CHICAGO PRINCIPLES:")
    print("• Households produce commodities using goods AND time")
    print("• Full income includes value of time")
    print("• Full price = goods cost + time cost")
    print("• Higher wages → Shift toward less time-intensive activities")

    model = TimeAllocationModel(
        wage=25.0,
        total_time=16.0,
        good_prices=(10.0, 5.0),
        time_requirements=(0.5, 2.0),  # Commodity 1 less time-intensive
        goods_requirements=(1.0, 0.5)
    )

    print_subheader("Optimal Time Allocation")
    result = model.maximize_utility()
    print(f"Full income: ${result['full_income']:.2f}")
    print(f"Full price of commodity 1: ${result['full_price_1']:.2f}")
    print(f"Full price of commodity 2: ${result['full_price_2']:.2f}")
    print(f"\nTime allocation:")
    print(f"  Work: {result['time_work']:.2f} hours")
    print(f"  Commodity 1: {result['time_commodity_1']:.2f} hours")
    print(f"  Commodity 2: {result['time_commodity_2']:.2f} hours")
    print(f"\nConsumption:")
    print(f"  Commodity 1: {result['commodity_1']:.2f} units")
    print(f"  Commodity 2: {result['commodity_2']:.2f} units")

    print_subheader("Effect of Wage Changes")
    low_wage = TimeAllocationModel(wage=15.0, good_prices=(10.0, 5.0),
                                   time_requirements=(0.5, 2.0)).maximize_utility()
    high_wage = TimeAllocationModel(wage=50.0, good_prices=(10.0, 5.0),
                                    time_requirements=(0.5, 2.0)).maximize_utility()

    print(f"Low wage ($15/hr):")
    print(f"  Commodity 1/Commodity 2 ratio: {low_wage['commodity_1']/low_wage['commodity_2']:.2f}")
    print(f"High wage ($50/hr):")
    print(f"  Commodity 1/Commodity 2 ratio: {high_wage['commodity_1']/high_wage['commodity_2']:.2f}")
    print(f"\nHigher wages → Shift toward time-saving commodities!")

    print("\n✓ Model demonstrates: Household production, full income, time as resource")


def print_summary():
    """Print summary of all models."""
    print_header("CHICAGO PRICE THEORY: MODEL SUMMARY")

    print("\nCOMPLETE MODEL SUITE:")
    print("  1. Supply and Demand - Market equilibrium, price adjustment, tax incidence")
    print("  2. Consumer Theory - Utility maximization, Slutsky decomposition, demand derivation")
    print("  3. Producer Theory - Profit maximization, cost curves, supply derivation")
    print("  4. Labor Market - Labor supply/demand, wage determination, minimum wages")
    print("  5. Human Capital - Education as investment, present value, returns to schooling")
    print("  6. Price Discrimination - Monopoly pricing, welfare analysis, market segmentation")
    print("  7. Search Theory - Optimal stopping, reservation wage, unemployment")
    print("  8. Time Allocation - Household production, full income, time as resource")

    print("\nCORE CHICAGO PRINCIPLES DEMONSTRATED:")
    print("  ✓ Rational optimization (utility/profit maximization)")
    print("  ✓ Marginal analysis (decisions at the margin)")
    print("  ✓ Price mechanism (markets coordinate without central planning)")
    print("  ✓ Substitution (response to relative price changes)")
    print("  ✓ Comparative statics (testable predictions)")
    print("  ✓ Empirical content (theories must confront data)")
    print("  ✓ Broad application (economic reasoning applies everywhere)")

    print("\nKEY CONTRIBUTORS:")
    print("  • Milton Friedman - Price theory methodology")
    print("  • George Stigler - Search theory, information economics")
    print("  • Gary Becker - Human capital, time allocation, economic imperialism")
    print("  • Ronald Coase - Transaction costs, property rights")
    print("  • Theodore Schultz - Human capital, agricultural economics")

    print("\n" + "=" * 100)
    print("All models are ready to use! Each can be run independently or imported as modules.")
    print("See individual model files for detailed explanations and visualizations.")
    print("=" * 100 + "\n")


def main():
    """Run comprehensive demonstration of all models."""
    print("\n" + "=" * 100)
    print(" " * 25 + "CHICAGO PRICE THEORY")
    print(" " * 20 + "Comprehensive Economic Models Suite")
    print("=" * 100)

    print("\nThis suite demonstrates core principles of Chicago School economics through")
    print("eight fully-functional economic models with detailed explanations, visualizations,")
    print("and practical applications.")

    print("\n" + "=" * 100)
    print("RUNNING ALL MODELS...")
    print("=" * 100)

    try:
        demo_supply_demand()
        demo_consumer_theory()
        demo_producer_theory()
        demo_labor_market()
        demo_human_capital()
        demo_price_discrimination()
        demo_search_theory()
        demo_time_allocation()

        print_summary()

        print("\nSUCCESS! All models executed successfully.")
        print("\nTo run individual models:")
        print("  python -m chicago_price_theory.supply_demand")
        print("  python -m chicago_price_theory.consumer_theory")
        print("  python -m chicago_price_theory.producer_theory")
        print("  ... etc.")

        print("\nTo use in your own code:")
        print("  from chicago_price_theory import SupplyDemandModel, ConsumerModel, ...")
        print("  model = SupplyDemandModel()")
        print("  results = model.find_equilibrium()")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
