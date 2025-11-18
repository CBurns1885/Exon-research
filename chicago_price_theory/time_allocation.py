"""
Time Allocation Model - Chicago Price Theory (Gary Becker)

THEORETICAL FOUNDATION:
Becker's household production theory revolutionized consumer theory by recognizing
that households produce commodities using both market goods and time. This explains
phenomena traditional theory couldn't: why richer people have less leisure, demand
for time-saving goods, etc.

KEY CONCEPTS:
1. Household Production: Households produce "commodities" (meals, health, entertainment)
2. Inputs: Market goods (X) and time (T)
3. Full Income: Value of time + money income
4. Shadow Prices: Commodity prices include time costs
5. Time Intensity: Different commodities use different time/goods ratios

BECKER'S REVOLUTION:
- Time is a scarce resource like money
- Consumption requires time + goods (not just goods)
- Higher wage → Higher opportunity cost of time
- Explains: Why rich eat out more, hire help, have fewer children

CHICAGO INSIGHTS:
- Value of time rises with wage
- Substitution of goods for time (time-saving appliances)
- Household specialization (comparative advantage)
- Demand for children (time-intensive commodity)
- Gender wage gap implications

CLASSICAL RESULTS:
- Full price of commodity: p_i + w*t_i (goods price + time cost)
- Income effect on time use differs by time-intensity
- Rising wages → Shift toward time-saving activities
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Dict, List


class TimeAllocationModel:
    """
    Household Production Model (Becker)

    Household produces commodities Z_i using:
    - Market goods X_i (purchased at prices p_i)
    - Time T_i (valued at wage w)

    Maximizes utility over commodities subject to:
    - Budget constraint: Σ p_i*X_i = w*T_w + M
    - Time constraint: Σ T_i + T_w = T_total
    """

    def __init__(self,
                 wage: float = 25.0,
                 total_time: float = 16.0,
                 non_labor_income: float = 0.0,
                 good_prices: Tuple[float, float] = (10.0, 5.0),
                 time_requirements: Tuple[float, float] = (0.5, 2.0),
                 goods_requirements: Tuple[float, float] = (1.0, 0.5)):
        """
        Initialize time allocation model.

        Parameters:
        -----------
        wage : float
            Hourly wage rate
        total_time : float
            Total time available per day (hours)
        non_labor_income : float
            Income from non-labor sources
        good_prices : tuple
            Prices (p₁, p₂) of market goods for commodities
        time_requirements : tuple
            Time required (t₁, t₂) per unit of each commodity
        goods_requirements : tuple
            Market goods required (x₁, x₂) per unit of each commodity
        """
        self.wage = wage
        self.total_time = total_time
        self.non_labor_income = non_labor_income
        self.good_prices = np.array(good_prices)
        self.time_requirements = np.array(time_requirements)
        self.goods_requirements = np.array(goods_requirements)

    def production_function(self, z1: float, z2: float) -> Tuple[float, float]:
        """
        Household production functions.

        For each commodity i:
        Z_i = f_i(X_i, T_i)

        Simple fixed-proportions (Leontief) technology:
        Z_i = min(X_i / x_i, T_i / t_i)

        where x_i, t_i are input requirements per unit of commodity.

        Returns required market goods and time.
        """
        # Market goods required
        X1 = z1 * self.goods_requirements[0]
        X2 = z2 * self.goods_requirements[1]

        # Time required
        T1 = z1 * self.time_requirements[0]
        T2 = z2 * self.time_requirements[1]

        return (X1, X2), (T1, T2)

    def full_price(self) -> np.ndarray:
        """
        Full price of each commodity.

        BECKER'S KEY INSIGHT:
        π_i = p_i*x_i + w*t_i

        Full price includes:
        - Goods cost: p_i*x_i
        - Time cost: w*t_i (opportunity cost of time)

        This explains why:
        - High-wage people eat out more (time-intensive cooking has high full price)
        - Poor people watch more TV (time-intensive but low goods cost)
        - Rich people have fewer children (time-intensive commodity)

        Returns:
        --------
        Array of full prices [π₁, π₂]
        """
        goods_cost = self.good_prices * self.goods_requirements
        time_cost = self.wage * self.time_requirements

        return goods_cost + time_cost

    def full_income(self) -> float:
        """
        Full income: Maximum potential income if all time spent working.

        FULL INCOME = w*T_total + M

        This is the "budget" for purchasing commodities (in full price terms).

        BECKER'S INSIGHT:
        Traditional theory only considers money income. But time is valuable!
        Full income accounts for the total value of resources.

        Returns:
        --------
        Full income
        """
        return self.wage * self.total_time + self.non_labor_income

    def utility(self, z1: float, z2: float) -> float:
        """
        Utility over commodities U(Z₁, Z₂).

        Using Cobb-Douglas: U = Z₁^α * Z₂^(1-α)
        """
        if z1 <= 0 or z2 <= 0:
            return -1e10

        alpha = 0.5  # Equal weights for simplicity
        return (z1 ** alpha) * (z2 ** (1 - alpha))

    def maximize_utility(self) -> Dict:
        """
        Household's optimization problem:

        max U(Z₁, Z₂)
        s.t. π₁*Z₁ + π₂*Z₂ ≤ w*T + M  (full income constraint)

        where π_i = p_i*x_i + w*t_i is full price.

        SOLUTION:
        Same as standard consumer problem, but with full prices and full income!

        For Cobb-Douglas with α = 0.5:
        Z₁* = (w*T + M) / (2*π₁)
        Z₂* = (w*T + M) / (2*π₂)

        Returns:
        --------
        dict with optimal commodities, time allocation, expenditures
        """
        full_prices = self.full_price()
        full_inc = self.full_income()

        # Analytical solution for Cobb-Douglas
        alpha = 0.5
        z1_star = (alpha * full_inc) / full_prices[0]
        z2_star = ((1 - alpha) * full_inc) / full_prices[1]

        # Calculate implied inputs
        (X1, X2), (T1, T2) = self.production_function(z1_star, z2_star)

        # Work time
        T_work = self.total_time - T1 - T2

        # Check feasibility
        if T_work < 0:
            # Time constraint violated - corner solution
            T_work = 0
            available_time = self.total_time
            # Re-optimize with time constraint binding
            z1_star = available_time / (2 * self.time_requirements[0])
            z2_star = available_time / (2 * self.time_requirements[1])

        # Money expenditure
        money_expenditure = self.good_prices[0] * X1 + self.good_prices[1] * X2

        # Labor income
        labor_income = self.wage * T_work

        # Utility
        u_star = self.utility(z1_star, z2_star)

        return {
            'commodity_1': z1_star,
            'commodity_2': z2_star,
            'utility': u_star,
            'time_commodity_1': T1,
            'time_commodity_2': T2,
            'time_work': T_work,
            'goods_commodity_1': X1,
            'goods_commodity_2': X2,
            'money_expenditure': money_expenditure,
            'labor_income': labor_income,
            'full_income': full_inc,
            'full_price_1': full_prices[0],
            'full_price_2': full_prices[1],
            'time_intensity_1': self.time_requirements[0] / self.goods_requirements[0],
            'time_intensity_2': self.time_requirements[1] / self.goods_requirements[1]
        }

    def wage_effect_on_time_use(self,
                                wage_range: Tuple[float, float] = (10, 50),
                                n_points: int = 30) -> Dict:
        """
        How time allocation changes with wage.

        BECKER'S PREDICTIONS:
        1. Higher wage → More work (substitution effect)
        2. Higher wage → Shift toward less time-intensive commodities
        3. Higher wage → Purchase more time-saving goods

        Example: As wages rise:
        - Less time cooking (time-intensive) → More eating out
        - Hire cleaning services, childcare, etc.

        Returns:
        --------
        dict with arrays of time use at different wages
        """
        wages = np.linspace(wage_range[0], wage_range[1], n_points)

        time_work = np.zeros(n_points)
        time_comm1 = np.zeros(n_points)
        time_comm2 = np.zeros(n_points)
        z1 = np.zeros(n_points)
        z2 = np.zeros(n_points)

        original_wage = self.wage

        for i, w in enumerate(wages):
            self.wage = w
            result = self.maximize_utility()

            time_work[i] = result['time_work']
            time_comm1[i] = result['time_commodity_1']
            time_comm2[i] = result['time_commodity_2']
            z1[i] = result['commodity_1']
            z2[i] = result['commodity_2']

        self.wage = original_wage

        return {
            'wages': wages,
            'time_work': time_work,
            'time_commodity_1': time_comm1,
            'time_commodity_2': time_comm2,
            'commodity_1': z1,
            'commodity_2': z2
        }

    def goods_time_substitution(self, tech_improvement_1: float = 0.5) -> Dict:
        """
        Effect of technology that reduces time requirement.

        BECKER APPLICATION:
        Technology (washing machines, microwaves, etc.) substitutes goods for time.
        Time-saving tech has bigger impact on high-wage individuals.

        Parameters:
        -----------
        tech_improvement_1 : float
            Reduction in time requirement for commodity 1

        Returns:
        --------
        dict comparing before and after
        """
        # Before technology
        before = self.maximize_utility()

        # After technology: Reduce time requirement
        original_time_req = self.time_requirements.copy()
        self.time_requirements[0] -= tech_improvement_1

        after = self.maximize_utility()

        # Restore
        self.time_requirements = original_time_req

        return {
            'before': before,
            'after': after,
            'change_commodity_1': after['commodity_1'] - before['commodity_1'],
            'change_work_time': after['time_work'] - before['time_work'],
            'change_utility': after['utility'] - before['utility']
        }

    def plot_time_allocation(self) -> None:
        """Visualize time allocation and wage effects."""
        result = self.maximize_utility()
        wage_effects = self.wage_effect_on_time_use()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Time allocation pie chart
        time_categories = [
            result['time_work'],
            result['time_commodity_1'],
            result['time_commodity_2']
        ]
        labels = ['Work', f'Commodity 1\n({result["commodity_1"]:.1f} units)',
                 f'Commodity 2\n({result["commodity_2"]:.1f} units)']
        colors = ['#ff9999', '#66b3ff', '#99ff99']

        ax1.pie(time_categories, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax1.set_title(f'Time Allocation (Wage=${self.wage}/hr)', fontsize=12, fontweight='bold')

        # 2. Full price decomposition
        goods_costs = self.good_prices * self.goods_requirements
        time_costs = self.wage * self.time_requirements

        x = np.arange(2)
        width = 0.35

        ax2.bar(x - width/2, goods_costs, width, label='Goods Cost', color='blue', alpha=0.7)
        ax2.bar(x + width/2, time_costs, width, label='Time Cost', color='red', alpha=0.7)

        # Total bars
        total_prices = goods_costs + time_costs
        for i, (gc, tc, total) in enumerate(zip(goods_costs, time_costs, total_prices)):
            ax2.text(i, total + 0.5, f'${total:.1f}', ha='center', fontweight='bold')

        ax2.set_ylabel('Cost ($)', fontsize=11)
        ax2.set_title('Full Price Decomposition', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Commodity 1', 'Commodity 2'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Time use vs wage
        ax3.plot(wage_effects['wages'], wage_effects['time_work'],
                linewidth=2, label='Work', color='blue')
        ax3.plot(wage_effects['wages'], wage_effects['time_commodity_1'],
                linewidth=2, label='Commodity 1 (less time-intensive)', color='green')
        ax3.plot(wage_effects['wages'], wage_effects['time_commodity_2'],
                linewidth=2, label='Commodity 2 (more time-intensive)', color='red')

        ax3.axvline(x=self.wage, color='gray', linestyle='--', alpha=0.5, label='Current Wage')

        ax3.set_xlabel('Wage ($/hour)', fontsize=11)
        ax3.set_ylabel('Hours per Day', fontsize=11)
        ax3.set_title('Time Allocation vs. Wage', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. Commodity consumption vs wage
        ax4.plot(wage_effects['wages'], wage_effects['commodity_1'],
                linewidth=2, label='Commodity 1', color='green')
        ax4.plot(wage_effects['wages'], wage_effects['commodity_2'],
                linewidth=2, label='Commodity 2', color='red')

        ax4.axvline(x=self.wage, color='gray', linestyle='--', alpha=0.5)

        ax4.set_xlabel('Wage ($/hour)', fontsize=11)
        ax4.set_ylabel('Units Consumed', fontsize=11)
        ax4.set_title('Commodity Consumption vs. Wage', fontsize=12, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/time_allocation.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    print("=" * 80)
    print("TIME ALLOCATION MODEL - CHICAGO PRICE THEORY (GARY BECKER)")
    print("=" * 80)

    # Commodity 1: Less time-intensive (e.g., eating out)
    # Commodity 2: More time-intensive (e.g., home cooking)
    model = TimeAllocationModel(
        wage=25.0,
        total_time=16.0,
        non_labor_income=0.0,
        good_prices=(10.0, 5.0),  # Eating out costs more in goods
        time_requirements=(0.5, 2.0),  # But saves time!
        goods_requirements=(1.0, 0.5)
    )

    # Optimal allocation
    print("\n1. OPTIMAL TIME ALLOCATION")
    print("-" * 80)
    result = model.maximize_utility()

    print(f"Wage rate: ${model.wage}/hour")
    print(f"Total time available: {model.total_time} hours/day")

    print(f"\nFull Prices (goods cost + time cost):")
    print(f"  Commodity 1: ${result['full_price_1']:.2f} (goods: ${model.good_prices[0]*model.goods_requirements[0]:.2f}, "
          f"time: ${model.wage*model.time_requirements[0]:.2f})")
    print(f"  Commodity 2: ${result['full_price_2']:.2f} (goods: ${model.good_prices[1]*model.goods_requirements[1]:.2f}, "
          f"time: ${model.wage*model.time_requirements[1]:.2f})")

    print(f"\nFull income: ${result['full_income']:.2f}")

    print(f"\nOptimal Allocation:")
    print(f"  Commodity 1 consumed: {result['commodity_1']:.2f} units")
    print(f"  Commodity 2 consumed: {result['commodity_2']:.2f} units")
    print(f"  Utility: {result['utility']:.2f}")

    print(f"\nTime Allocation:")
    print(f"  Work: {result['time_work']:.2f} hours ({result['time_work']/model.total_time*100:.1f}%)")
    print(f"  Commodity 1: {result['time_commodity_1']:.2f} hours")
    print(f"  Commodity 2: {result['time_commodity_2']:.2f} hours")
    print(f"  Total: {result['time_work'] + result['time_commodity_1'] + result['time_commodity_2']:.2f} hours")

    print(f"\nMoney Flows:")
    print(f"  Labor income: ${result['labor_income']:.2f}")
    print(f"  Expenditure on goods: ${result['money_expenditure']:.2f}")

    # Wage effects
    print("\n2. EFFECT OF WAGE CHANGES")
    print("-" * 80)

    low_wage_model = TimeAllocationModel(wage=15.0, good_prices=(10.0, 5.0),
                                        time_requirements=(0.5, 2.0))
    high_wage_model = TimeAllocationModel(wage=50.0, good_prices=(10.0, 5.0),
                                         time_requirements=(0.5, 2.0))

    low_result = low_wage_model.maximize_utility()
    high_result = high_wage_model.maximize_utility()

    print(f"Low wage ($15/hr):")
    print(f"  Work time: {low_result['time_work']:.2f} hours")
    print(f"  Commodity 1 (time-saving): {low_result['commodity_1']:.2f} units")
    print(f"  Commodity 2 (time-intensive): {low_result['commodity_2']:.2f} units")

    print(f"\nHigh wage ($50/hr):")
    print(f"  Work time: {high_result['time_work']:.2f} hours")
    print(f"  Commodity 1 (time-saving): {high_result['commodity_1']:.2f} units")
    print(f"  Commodity 2 (time-intensive): {high_result['commodity_2']:.2f} units")

    print(f"\nBecker Insight: Higher wage → Shift toward less time-intensive activities")
    print(f"Commodity 1/Commodity 2 ratio:")
    print(f"  Low wage: {low_result['commodity_1']/low_result['commodity_2']:.2f}")
    print(f"  High wage: {high_result['commodity_1']/high_result['commodity_2']:.2f}")

    # Technology effect
    print("\n3. EFFECT OF TIME-SAVING TECHNOLOGY")
    print("-" * 80)
    print("Scenario: Technology reduces time requirement for Commodity 1 by 0.3 hours")

    tech_effect = model.goods_time_substitution(tech_improvement_1=0.3)

    print(f"\nBefore technology:")
    print(f"  Commodity 1: {tech_effect['before']['commodity_1']:.2f} units")
    print(f"  Work time: {tech_effect['before']['time_work']:.2f} hours")
    print(f"  Utility: {tech_effect['before']['utility']:.2f}")

    print(f"\nAfter technology:")
    print(f"  Commodity 1: {tech_effect['after']['commodity_1']:.2f} units")
    print(f"  Work time: {tech_effect['after']['time_work']:.2f} hours")
    print(f"  Utility: {tech_effect['after']['utility']:.2f}")

    print(f"\nChanges:")
    print(f"  ΔCommodity 1: +{tech_effect['change_commodity_1']:.2f} units")
    print(f"  ΔWork time: {tech_effect['change_work_time']:+.2f} hours")
    print(f"  ΔUtility: +{tech_effect['change_utility']:.2f}")

    print("\nBecker Insight: Time-saving technology allows:")
    print("• More consumption of the commodity (freed up time)")
    print("• Potentially more work (higher productivity)")
    print("• Higher overall utility")

    # Full income concept
    print("\n4. FULL INCOME CONCEPT")
    print("-" * 80)
    money_income = result['labor_income'] + model.non_labor_income
    full_income = model.full_income()

    print(f"Traditional money income: ${money_income:.2f}")
    print(f"Full income (including value of time): ${full_income:.2f}")
    print(f"Difference: ${full_income - money_income:.2f}")
    print(f"\nBecker Revolution: Traditional theory ignores {((full_income-money_income)/full_income*100):.0f}%")
    print(f"of household resources (the value of non-working time)!")

    # Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 80)
    model.plot_time_allocation()
    print("Plot saved to chicago_price_theory/time_allocation.png")
