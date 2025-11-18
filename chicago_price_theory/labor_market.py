"""
Labor Market Model - Chicago Price Theory

THEORETICAL FOUNDATION:
Labor markets are analyzed like any other market - supply and demand determine
wages and employment. Workers supply labor based on leisure-income tradeoffs,
firms demand labor based on marginal productivity.

KEY CONCEPTS:
1. Labor Supply: Workers choose hours based on wage rate (leisure-income tradeoff)
2. Labor Demand: Firms hire where wage = value of marginal product
3. Market Equilibrium: Wage adjusts to clear market
4. Labor-Leisure Choice: Substitution and income effects
5. Human Capital: Education/training as investment

CHICAGO EMPHASIS:
- Wages reflect productivity (marginal productivity theory)
- Discrimination is costly in competitive markets
- Minimum wages create unemployment if above equilibrium
- Compensating differentials for job characteristics
- Dynamic aspects: search, matching, human capital investment

CLASSICAL RESULTS:
- Labor supply slopes up (substitution effect dominates)
- Labor demand slopes down (diminishing marginal product)
- Equilibrium: w* where Ls = Ld
- Policy effects: taxes, minimum wages, unions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from typing import Tuple, Dict, Optional


class LaborMarketModel:
    """
    Labor Market Equilibrium Model

    Individual level:
    - Worker: max U(c, l) s.t. c = w*h, h + l = T (c=consumption, l=leisure, h=hours)
    - Firm: hire where w = P * MPL

    Market level:
    - Aggregate labor supply and demand
    - Wage determination
    - Employment level
    """

    def __init__(self,
                 wage: float = 20.0,
                 time_endowment: float = 16.0,
                 non_labor_income: float = 0.0,
                 leisure_preference: float = 0.5,
                 output_price: float = 10.0,
                 production_params: Tuple[float, float] = (1.0, 0.6)):
        """
        Initialize labor market model.

        Parameters:
        -----------
        wage : float
            Wage rate (w)
        time_endowment : float
            Total time available (T hours/day)
        non_labor_income : float
            Income from non-labor sources
        leisure_preference : float
            Weight on leisure in utility (α in Cobb-Douglas)
        output_price : float
            Price of output produced
        production_params : tuple
            (A, α) for production Q = A * L^α
        """
        self.wage = wage
        self.time_endowment = time_endowment
        self.non_labor_income = non_labor_income
        self.leisure_preference = leisure_preference
        self.output_price = output_price
        self.A, self.alpha = production_params

    def utility(self, consumption: float, leisure: float) -> float:
        """
        Worker utility function U(c, l).

        Using Cobb-Douglas: U = c^(1-α) * l^α
        where α is preference for leisure
        """
        if consumption <= 0 or leisure <= 0:
            return -1e10
        return (consumption ** (1 - self.leisure_preference)) * \
               (leisure ** self.leisure_preference)

    def labor_supply_individual(self) -> Dict:
        """
        Individual labor supply decision.

        Worker's problem:
        max U(c, l)
        s.t. c = w*h + m  (budget constraint, m = non-labor income)
             h + l = T    (time constraint)

        Substitute: c = w*(T - l) + m
        max U(w*(T-l) + m, l)

        FOC: MU_l / MU_c = w
        For Cobb-Douglas:
        [α * c / ((1-α) * l)] = w
        →  l* = α(wT + m) / w
        →  h* = T - l* = (1-α)T - αm/w

        CHICAGO INSIGHTS:
        1. Higher wage → Two effects:
           - Substitution: Leisure more expensive → work more
           - Income: Richer → consume more leisure (if normal good)
        2. Non-labor income → Pure income effect → work less
        3. Participation decision: Work if w > reservation wage

        Returns:
        --------
        dict with optimal hours, leisure, consumption, utility
        """
        alpha = self.leisure_preference
        w = self.wage
        T = self.time_endowment
        m = self.non_labor_income

        # Analytical solution for Cobb-Douglas
        l_star = alpha * (w * T + m) / w
        h_star = T - l_star
        c_star = w * h_star + m

        # Handle corner solutions
        if l_star >= T:  # Don't work
            l_star = T
            h_star = 0
            c_star = m
        elif l_star <= 0:  # No leisure
            l_star = 0
            h_star = T
            c_star = w * T + m

        u_star = self.utility(c_star, l_star)

        # Reservation wage (wage at which indifferent about working)
        if m > 0:
            # At reservation wage, utility from not working = utility from working
            # U(m, T) = U(w*h* + m, l*)
            reservation_wage = m / ((1 - alpha) * T)
        else:
            reservation_wage = 0

        return {
            'hours_worked': h_star,
            'leisure': l_star,
            'consumption': c_star,
            'utility': u_star,
            'labor_income': w * h_star,
            'total_income': w * h_star + m,
            'reservation_wage': reservation_wage,
            'participates': h_star > 0
        }

    def labor_supply_curve(self,
                          wage_range: Tuple[float, float] = (5, 50),
                          n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive individual labor supply curve: h(w).

        Shows how hours worked change with wage.

        CHICAGO PREDICTION:
        - At low wages: Substitution effect dominates → upward sloping
        - At high wages: Income effect may dominate → backward bending possible
        - With non-labor income: Parallel shift inward

        Returns:
        --------
        (wages, hours) arrays
        """
        wages = np.linspace(wage_range[0], wage_range[1], n_points)
        hours = np.zeros(n_points)

        original_wage = self.wage

        for i, w in enumerate(wages):
            self.wage = w
            result = self.labor_supply_individual()
            hours[i] = result['hours_worked']

        self.wage = original_wage

        return wages, hours

    def labor_demand_firm(self, total_labor: float) -> float:
        """
        Firm's marginal product of labor.

        Production: Q = A * L^α
        MPL = dQ/dL = α * A * L^(α-1)

        PROFIT MAXIMIZATION:
        Hire where w = P * MPL
        →  L^d = (w / (P * α * A))^(1/(α-1))

        CHICAGO INSIGHT:
        Labor demand is derived demand - depends on productivity and
        output price. Downward sloping due to diminishing returns.
        """
        if total_labor <= 0:
            return float('inf')

        # MPL = α * A * L^(α-1)
        MPL = self.alpha * self.A * (total_labor ** (self.alpha - 1))
        return MPL

    def labor_demand_curve(self,
                          wage_range: Tuple[float, float] = (5, 50),
                          n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive firm labor demand curve: L^d(w).

        From w = P * MPL:
        L^d = ((P * α * A) / w)^(1/(1-α))

        Returns:
        --------
        (wages, labor_demanded) arrays
        """
        wages = np.linspace(wage_range[0], wage_range[1], n_points)
        labor_demand = np.zeros(n_points)

        P = self.output_price
        alpha = self.alpha
        A = self.A

        for i, w in enumerate(wages):
            # Solve w = P * α * A * L^(α-1) for L
            L_d = ((P * alpha * A) / w) ** (1 / (1 - alpha))
            labor_demand[i] = L_d

        return wages, labor_demand

    def market_equilibrium(self,
                          n_workers: int = 100,
                          n_firms: int = 10) -> Dict:
        """
        Find labor market equilibrium.

        MARKET CLEARING:
        Σ h^s_i(w*) = Σ L^d_j(w*)
        Total hours supplied = Total labor demanded

        Wage adjusts to equilibrate market.

        CHICAGO INSIGHT:
        Market wage reflects both worker preferences and firm productivity.
        No involuntary unemployment in competitive equilibrium.

        Parameters:
        -----------
        n_workers : int
            Number of workers
        n_firms : int
            Number of identical firms

        Returns:
        --------
        dict with equilibrium wage, employment, unemployment
        """
        def excess_demand(w):
            """Excess demand for labor at wage w."""
            # Labor supply from workers
            self.wage = w
            worker_result = self.labor_supply_individual()
            total_supply = n_workers * worker_result['hours_worked']

            # Labor demand from firms
            # For each firm: w = P * MPL
            # L_d = ((P * α * A) / w)^(1/(1-α))
            L_d_per_firm = ((self.output_price * self.alpha * self.A) / w) ** \
                          (1 / (1 - self.alpha))
            total_demand = n_firms * L_d_per_firm

            return total_demand - total_supply

        # Find equilibrium wage
        try:
            w_star = fsolve(excess_demand, x0=20.0)[0]
        except:
            w_star = 20.0

        # Calculate equilibrium quantities
        self.wage = w_star
        worker_result = self.labor_supply_individual()
        total_hours_supplied = n_workers * worker_result['hours_worked']

        L_d_per_firm = ((self.output_price * self.alpha * self.A) / w_star) ** \
                      (1 / (1 - self.alpha))
        total_hours_demanded = n_firms * L_d_per_firm

        # Number of workers who participate
        workers_employed = n_workers if worker_result['participates'] else 0

        return {
            'wage': w_star,
            'total_hours': (total_hours_supplied + total_hours_demanded) / 2,
            'hours_per_worker': worker_result['hours_worked'],
            'workers_employed': workers_employed,
            'unemployment_rate': 0,  # No involuntary unemployment in competitive eq.
            'worker_utility': worker_result['utility'],
            'total_supply': total_hours_supplied,
            'total_demand': total_hours_demanded,
            'market_clears': np.isclose(total_hours_supplied, total_hours_demanded, rtol=0.01)
        }

    def minimum_wage_effect(self, min_wage: float, n_workers: int = 100,
                           n_firms: int = 10) -> Dict:
        """
        Analyze effect of minimum wage.

        CHICAGO PREDICTION:
        - If min wage > equilibrium wage: Creates unemployment
        - If min wage < equilibrium wage: No effect (not binding)
        - Unemployment = Labor supply - Labor demand at min wage

        This is a classic application showing how price floors create surpluses.

        Parameters:
        -----------
        min_wage : float
            Minimum wage level
        n_workers : int
            Number of workers
        n_firms : int
            Number of firms

        Returns:
        --------
        dict comparing equilibrium vs minimum wage outcomes
        """
        # Free market equilibrium
        eq = self.market_equilibrium(n_workers, n_firms)

        if min_wage <= eq['wage']:
            return {
                'binding': False,
                'equilibrium': eq,
                'minimum_wage_outcome': eq,
                'unemployment_created': 0,
                'wage_change': 0
            }

        # Under minimum wage
        self.wage = min_wage
        worker_result = self.labor_supply_individual()
        total_supply = n_workers * worker_result['hours_worked']

        L_d_per_firm = ((self.output_price * self.alpha * self.A) / min_wage) ** \
                      (1 / (1 - self.alpha))
        total_demand = n_firms * L_d_per_firm

        # Unemployment (excess supply)
        unemployment_hours = total_supply - total_demand
        workers_unemployed = (unemployment_hours / worker_result['hours_worked']) \
                            if worker_result['hours_worked'] > 0 else 0

        return {
            'binding': True,
            'equilibrium': eq,
            'minimum_wage': min_wage,
            'total_supply': total_supply,
            'total_demand': total_demand,
            'unemployment_hours': unemployment_hours,
            'workers_unemployed': workers_unemployed,
            'unemployment_rate': (workers_unemployed / n_workers) * 100,
            'wage_change': min_wage - eq['wage'],
            'employment_change': total_demand - eq['total_hours']
        }

    def plot_labor_market(self, n_workers: int = 100, n_firms: int = 10,
                         show_minimum_wage: bool = False,
                         min_wage: Optional[float] = None) -> None:
        """
        Visualize labor market equilibrium.

        Parameters:
        -----------
        n_workers : int
            Number of workers
        n_firms : int
            Number of firms
        show_minimum_wage : bool
            Show effect of minimum wage
        min_wage : float
            Minimum wage level
        """
        # Get supply and demand curves
        wages_s, hours_s = self.labor_supply_curve(wage_range=(5, 50))
        total_supply = hours_s * n_workers

        wages_d, hours_d = self.labor_demand_curve(wage_range=(5, 50))
        total_demand = hours_d * n_firms

        # Find equilibrium
        eq = self.market_equilibrium(n_workers, n_firms)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot curves
        ax.plot(total_supply, wages_s, 'b-', linewidth=2, label='Labor Supply')
        ax.plot(total_demand, wages_d, 'r-', linewidth=2, label='Labor Demand')

        # Equilibrium
        ax.plot(eq['total_hours'], eq['wage'], 'go', markersize=12,
               label=f'Equilibrium: w*=${eq["wage"]:.2f}, L*={eq["total_hours"]:.0f}')

        # Minimum wage analysis
        if show_minimum_wage and min_wage is not None:
            mw_result = self.minimum_wage_effect(min_wage, n_workers, n_firms)

            if mw_result['binding']:
                ax.axhline(y=min_wage, color='purple', linestyle='--', linewidth=2,
                          label=f'Minimum Wage = ${min_wage:.2f}')

                # Show unemployment
                ax.plot([mw_result['total_demand'], mw_result['total_supply']],
                       [min_wage, min_wage],
                       'ro-', linewidth=3, markersize=8)

                # Shade unemployment region
                ax.fill_betweenx([min_wage-1, min_wage+1],
                                mw_result['total_demand'],
                                mw_result['total_supply'],
                                alpha=0.3, color='red',
                                label=f'Unemployment: {mw_result["unemployment_hours"]:.0f} hours')

                ax.text((mw_result['total_demand'] + mw_result['total_supply'])/2,
                       min_wage + 2,
                       f'{mw_result["unemployment_rate"]:.1f}% Unemployed',
                       ha='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Total Hours', fontsize=12)
        ax.set_ylabel('Wage Rate ($/hour)', fontsize=12)
        ax.set_title('Labor Market Equilibrium', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(total_supply.max(), total_demand.max()) * 1.1)
        ax.set_ylim(0, 50)

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/labor_market.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    print("=" * 80)
    print("LABOR MARKET MODEL - CHICAGO PRICE THEORY")
    print("=" * 80)

    model = LaborMarketModel(
        wage=20.0,
        time_endowment=16.0,
        non_labor_income=0.0,
        leisure_preference=0.4
    )

    # Individual labor supply
    print("\n1. INDIVIDUAL LABOR SUPPLY")
    print("-" * 80)
    result = model.labor_supply_individual()
    print(f"Wage rate: ${model.wage}/hour")
    print(f"Time endowment: {model.time_endowment} hours/day")
    print(f"\nOptimal Choice:")
    print(f"  Hours worked: {result['hours_worked']:.2f}")
    print(f"  Leisure: {result['leisure']:.2f} hours")
    print(f"  Consumption: ${result['consumption']:.2f}")
    print(f"  Utility: {result['utility']:.2f}")
    print(f"  Labor income: ${result['labor_income']:.2f}")
    print(f"  Reservation wage: ${result['reservation_wage']:.2f}/hour")

    # Market equilibrium
    print("\n2. LABOR MARKET EQUILIBRIUM")
    print("-" * 80)
    eq = model.market_equilibrium(n_workers=100, n_firms=10)
    print(f"Market participants: 100 workers, 10 firms")
    print(f"\nEquilibrium:")
    print(f"  Wage: ${eq['wage']:.2f}/hour")
    print(f"  Total hours: {eq['total_hours']:.0f}")
    print(f"  Hours per worker: {eq['hours_per_worker']:.2f}")
    print(f"  Workers employed: {eq['workers_employed']}")
    print(f"  Unemployment rate: {eq['unemployment_rate']:.1f}%")
    print(f"  Market clears: {eq['market_clears']}")

    # Minimum wage
    print("\n3. MINIMUM WAGE ANALYSIS")
    print("-" * 80)
    min_wage = eq['wage'] * 1.25  # 25% above equilibrium
    print(f"Impose minimum wage of ${min_wage:.2f}/hour (25% above equilibrium)")
    mw = model.minimum_wage_effect(min_wage, n_workers=100, n_firms=10)
    print(f"\nEffect:")
    print(f"  Binding: {mw['binding']}")
    print(f"  Labor supplied: {mw['total_supply']:.0f} hours")
    print(f"  Labor demanded: {mw['total_demand']:.0f} hours")
    print(f"  Unemployment: {mw['unemployment_hours']:.0f} hours ({mw['unemployment_rate']:.1f}%)")
    print(f"  Employment change: {mw['employment_change']:.0f} hours")
    print("\nChicago Insight: Minimum wage above equilibrium creates unemployment!")

    # Effect of non-labor income
    print("\n4. EFFECT OF NON-LABOR INCOME")
    print("-" * 80)
    model_no_income = LaborMarketModel(wage=20, non_labor_income=0)
    model_with_income = LaborMarketModel(wage=20, non_labor_income=100)

    result_no = model_no_income.labor_supply_individual()
    result_yes = model_with_income.labor_supply_individual()

    print(f"Without non-labor income:")
    print(f"  Hours worked: {result_no['hours_worked']:.2f}")
    print(f"With $100 non-labor income:")
    print(f"  Hours worked: {result_yes['hours_worked']:.2f}")
    print(f"  Change: {result_yes['hours_worked'] - result_no['hours_worked']:.2f} hours")
    print("\nChicago Insight: Non-labor income reduces labor supply (pure income effect)")

    # Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 80)
    model.plot_labor_market(n_workers=100, n_firms=10,
                           show_minimum_wage=True, min_wage=min_wage)
    print("Plot saved to chicago_price_theory/labor_market.png")
