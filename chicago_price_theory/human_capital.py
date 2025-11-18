"""
Human Capital Investment Model - Chicago Price Theory (Gary Becker)

THEORETICAL FOUNDATION:
Gary Becker's human capital theory treats education and training as investments
that increase future productivity and earnings. This revolutionized labor economics
by applying capital theory to human skills.

KEY CONCEPTS:
1. Human Capital: Skills, knowledge, health embodied in workers
2. Investment Decision: Compare costs and benefits over lifetime
3. Present Value: Discount future earnings to compare with current costs
4. Rate of Return: Internal rate of return to education
5. Signaling vs Human Capital: Does education increase productivity or just signal ability?

CHICAGO EMPHASIS:
- Education as rational investment, not consumption
- Marginal analysis: Optimal years of schooling where MB = MC
- Discrimination: Wage differences reflect productivity differences
- On-the-job training: General vs firm-specific
- Age-earnings profiles: Concave due to human capital depreciation

BECKER'S INSIGHTS:
- Individuals invest in human capital until marginal return = market interest rate
- Higher ability → More education (complementarity)
- General training paid by worker, firm-specific paid by firm
- Observable implications: Earnings rise with education, then decline with age
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from typing import Tuple, Dict, List, Optional


class HumanCapitalModel:
    """
    Human Capital Investment Model (Becker)

    Models individual's decision to invest in education/training
    by comparing lifetime earnings with and without investment.
    """

    def __init__(self,
                 base_wage: float = 30000,
                 education_return: float = 0.10,
                 discount_rate: float = 0.05,
                 working_years: int = 40,
                 max_schooling: int = 20,
                 cost_per_year: float = 20000,
                 ability: float = 1.0):
        """
        Initialize human capital model.

        Parameters:
        -----------
        base_wage : float
            Annual wage without additional education ($)
        education_return : float
            Return per year of education (10% = 0.10)
        discount_rate : float
            Interest rate for present value calculations
        working_years : int
            Years in labor force after schooling
        max_schooling : int
            Maximum years of schooling to consider
        cost_per_year : float
            Direct cost per year of schooling (tuition, books, etc.)
        ability : float
            Individual ability level (affects returns)
        """
        self.base_wage = base_wage
        self.education_return = education_return
        self.discount_rate = discount_rate
        self.working_years = working_years
        self.max_schooling = max_schooling
        self.cost_per_year = cost_per_year
        self.ability = ability

    def wage_function(self, years_schooling: float, experience: int = 0) -> float:
        """
        Mincer earnings function (standard in labor economics):

        ln(w) = ln(w₀) + r*S + β₁*Exp + β₂*Exp²

        where:
        - w = wage
        - w₀ = base wage
        - r = return to schooling
        - S = years of schooling
        - Exp = years of experience
        - β₁, β₂ capture experience effects

        CHICAGO INSIGHT:
        This log-linear form implies constant percentage returns to education.
        The experience profile is concave (earnings rise then plateau).

        Parameters:
        -----------
        years_schooling : float
            Years of education
        experience : int
            Years of work experience

        Returns:
        --------
        Annual wage in dollars
        """
        # Ability affects both level and return
        adjusted_return = self.education_return * self.ability

        # Experience coefficients (typical values from empirical literature)
        beta1 = 0.05  # Initial experience return
        beta2 = -0.001  # Experience squared (declining returns)

        log_wage = np.log(self.base_wage) + \
                   adjusted_return * years_schooling + \
                   beta1 * experience + \
                   beta2 * (experience ** 2)

        return np.exp(log_wage)

    def total_cost(self, years_schooling: float) -> float:
        """
        Total cost of schooling.

        TOTAL COST = Direct costs + Opportunity costs

        Direct costs: Tuition, books, fees
        Opportunity costs: Foregone earnings while in school

        CHICAGO PRINCIPLE:
        Opportunity cost is often larger than direct cost!
        Time is valuable.

        Parameters:
        -----------
        years_schooling : float
            Years of education

        Returns:
        --------
        Present value of total costs
        """
        direct_costs = 0
        opportunity_costs = 0

        for year in range(int(years_schooling)):
            # Direct cost in year t
            direct_cost_t = self.cost_per_year

            # Opportunity cost: What could have earned working instead
            # If in school for year t, forego wage with (0) years of schooling
            # and t years of experience
            opportunity_cost_t = self.wage_function(0, experience=year)

            # Discount to present value
            pv_factor = 1 / ((1 + self.discount_rate) ** year)

            direct_costs += direct_cost_t * pv_factor
            opportunity_costs += opportunity_cost_t * pv_factor

        return direct_costs + opportunity_costs

    def lifetime_earnings(self, years_schooling: float) -> float:
        """
        Present value of lifetime earnings with given schooling.

        LIFETIME EARNINGS = Σ w(S, t) / (1+r)^t

        Sum over working years, discounted to present value.

        Parameters:
        -----------
        years_schooling : float
            Years of education

        Returns:
        --------
        Present value of lifetime earnings
        """
        pv_earnings = 0

        for t in range(self.working_years):
            # Wage in year t (experience = t)
            wage_t = self.wage_function(years_schooling, experience=t)

            # Discount to present value
            # Time starts after schooling completes
            years_from_now = int(years_schooling) + t
            pv_factor = 1 / ((1 + self.discount_rate) ** years_from_now)

            pv_earnings += wage_t * pv_factor

        return pv_earnings

    def net_present_value(self, years_schooling: float) -> float:
        """
        Net Present Value of education investment.

        NPV = PV(Benefits) - PV(Costs)
            = PV(Lifetime Earnings) - PV(Total Costs)

        DECISION RULE:
        - NPV > 0: Invest (education pays off)
        - NPV < 0: Don't invest (not worth it)

        CHICAGO APPROACH:
        Education is like any other investment - compare discounted
        benefits and costs.

        Parameters:
        -----------
        years_schooling : float
            Years of education

        Returns:
        --------
        Net present value of investment
        """
        benefits = self.lifetime_earnings(years_schooling)
        costs = self.total_cost(years_schooling)

        return benefits - costs

    def optimal_schooling(self) -> Dict:
        """
        Find optimal years of schooling.

        OPTIMIZATION:
        max NPV(S)
         S

        FOC: dNPV/dS = 0
        Marginal benefit = Marginal cost

        CHICAGO INSIGHT:
        Optimal investment where marginal return equals market interest rate.
        This is analogous to firms investing in physical capital.

        Returns:
        --------
        dict with optimal schooling, NPV, IRR, marginal analysis
        """
        # Find years of schooling that maximizes NPV
        result = minimize_scalar(
            lambda S: -self.net_present_value(S),
            bounds=(0, self.max_schooling),
            method='bounded'
        )

        S_optimal = result.x
        npv_optimal = -result.fun

        # Calculate marginal values at optimum
        eps = 0.1
        marginal_benefit = (self.lifetime_earnings(S_optimal + eps) -
                          self.lifetime_earnings(S_optimal)) / eps
        marginal_cost = (self.total_cost(S_optimal + eps) -
                        self.total_cost(S_optimal)) / eps

        # Internal rate of return
        irr = self.internal_rate_of_return(S_optimal)

        return {
            'optimal_years': S_optimal,
            'npv': npv_optimal,
            'total_benefits': self.lifetime_earnings(S_optimal),
            'total_costs': self.total_cost(S_optimal),
            'marginal_benefit': marginal_benefit,
            'marginal_cost': marginal_cost,
            'internal_rate_of_return': irr,
            'mb_equals_mc': np.isclose(marginal_benefit, marginal_cost, rtol=0.1)
        }

    def internal_rate_of_return(self, years_schooling: float) -> float:
        """
        Internal Rate of Return (IRR) to education.

        IRR is the discount rate that makes NPV = 0.
        Solve: NPV(S, r) = 0 for r

        INTERPRETATION:
        IRR is the "interest rate" earned on education investment.
        Compare to market interest rate:
        - IRR > r: Good investment
        - IRR < r: Better to invest money elsewhere

        Parameters:
        -----------
        years_schooling : float
            Years of education

        Returns:
        --------
        Internal rate of return (as decimal, e.g., 0.10 = 10%)
        """
        original_rate = self.discount_rate

        def npv_at_rate(r):
            self.discount_rate = r
            npv = self.net_present_value(years_schooling)
            self.discount_rate = original_rate
            return npv

        try:
            # Find rate where NPV = 0
            irr = fsolve(npv_at_rate, x0=0.10)[0]
            # Ensure reasonable bounds
            if irr < 0 or irr > 1:
                irr = np.nan
        except:
            irr = np.nan

        return irr

    def age_earnings_profile(self, years_schooling: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate age-earnings profile.

        BECKER PREDICTION:
        Earnings rise with experience (human capital accumulation),
        then decline near retirement (depreciation, obsolescence).

        More educated workers:
        - Start earning later (opportunity cost of schooling)
        - Earn more per year (return to education)
        - Have steeper profiles (more on-the-job training)

        Parameters:
        -----------
        years_schooling : float
            Years of education

        Returns:
        --------
        (ages, earnings) arrays
        """
        ages = []
        earnings = []

        # Start working after schooling
        start_age = 18 + int(years_schooling)

        for exp in range(self.working_years):
            age = start_age + exp
            wage = self.wage_function(years_schooling, experience=exp)

            ages.append(age)
            earnings.append(wage)

        return np.array(ages), np.array(earnings)

    def compare_education_levels(self,
                                schooling_levels: List[int]) -> Dict:
        """
        Compare different education levels.

        CHICAGO ANALYSIS:
        Shows how returns vary with education level.
        Used to understand wage inequality and education policy.

        Parameters:
        -----------
        schooling_levels : list
            Years of schooling to compare (e.g., [12, 16, 20] for HS, BA, PhD)

        Returns:
        --------
        dict with comparison data
        """
        results = {}

        for S in schooling_levels:
            npv = self.net_present_value(S)
            lifetime_earn = self.lifetime_earnings(S)
            costs = self.total_cost(S)
            irr = self.internal_rate_of_return(S)

            results[f'{S}_years'] = {
                'schooling': S,
                'npv': npv,
                'lifetime_earnings': lifetime_earn,
                'total_costs': costs,
                'irr': irr,
                'benefit_cost_ratio': lifetime_earn / costs if costs > 0 else float('inf')
            }

        return results

    def plot_age_earnings_profiles(self,
                                   schooling_levels: List[int] = [12, 16, 20]) -> None:
        """
        Plot age-earnings profiles for different education levels.

        Classic Becker diagram showing:
        - Higher education → higher lifetime earnings
        - Delayed entry (opportunity cost)
        - Steeper profiles with more education
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        labels = {12: 'High School (12 years)',
                 16: 'College (16 years)',
                 18: 'Masters (18 years)',
                 20: 'PhD (20 years)'}

        colors = {12: 'blue', 16: 'green', 18: 'orange', 20: 'red'}

        for S in schooling_levels:
            ages, earnings = self.age_earnings_profile(S)
            label = labels.get(S, f'{S} years education')
            color = colors.get(S, 'black')

            ax.plot(ages, earnings, linewidth=2.5, label=label, color=color)

            # Mark starting point
            ax.plot(ages[0], earnings[0], 'o', markersize=8, color=color)

        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Annual Earnings ($)', fontsize=12)
        ax.set_title('Age-Earnings Profiles by Education Level (Becker Model)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/human_capital_profiles.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_npv_by_schooling(self) -> None:
        """Plot NPV as a function of years of schooling."""
        years = np.linspace(0, self.max_schooling, 100)
        npvs = [self.net_present_value(S) for S in years]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, npvs, 'b-', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

        # Mark optimum
        opt = self.optimal_schooling()
        ax.plot(opt['optimal_years'], opt['npv'], 'ro', markersize=12,
               label=f'Optimal: {opt["optimal_years"]:.1f} years, NPV=${opt["npv"]:,.0f}')

        ax.set_xlabel('Years of Schooling', fontsize=12)
        ax.set_ylabel('Net Present Value ($)', fontsize=12)
        ax.set_title('NPV of Education Investment', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/human_capital_npv.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    print("=" * 80)
    print("HUMAN CAPITAL INVESTMENT MODEL - CHICAGO PRICE THEORY (GARY BECKER)")
    print("=" * 80)

    model = HumanCapitalModel(
        base_wage=30000,
        education_return=0.10,  # 10% return per year
        discount_rate=0.05,
        working_years=40,
        cost_per_year=20000
    )

    # Optimal schooling
    print("\n1. OPTIMAL EDUCATION INVESTMENT")
    print("-" * 80)
    opt = model.optimal_schooling()
    print(f"Optimal years of schooling: {opt['optimal_years']:.1f} years")
    print(f"\nPresent Value Analysis:")
    print(f"  Total benefits (PV): ${opt['total_benefits']:,.0f}")
    print(f"  Total costs (PV): ${opt['total_costs']:,.0f}")
    print(f"  Net Present Value: ${opt['npv']:,.0f}")
    print(f"\nMarginal Analysis:")
    print(f"  Marginal benefit: ${opt['marginal_benefit']:,.0f}")
    print(f"  Marginal cost: ${opt['marginal_cost']:,.0f}")
    print(f"  MB = MC? {opt['mb_equals_mc']}")
    print(f"\nInternal Rate of Return: {opt['internal_rate_of_return']*100:.1f}%")
    print(f"  Market interest rate: {model.discount_rate*100:.1f}%")
    print(f"  IRR > r? {opt['internal_rate_of_return'] > model.discount_rate}")

    # Compare education levels
    print("\n2. COMPARISON OF EDUCATION LEVELS")
    print("-" * 80)
    comparison = model.compare_education_levels([12, 16, 20])

    for level, data in comparison.items():
        print(f"\n{level} ({data['schooling']} years of schooling):")
        print(f"  Lifetime earnings (PV): ${data['lifetime_earnings']:,.0f}")
        print(f"  Total costs (PV): ${data['total_costs']:,.0f}")
        print(f"  NPV: ${data['npv']:,.0f}")
        print(f"  IRR: {data['irr']*100:.1f}%")
        print(f"  Benefit/Cost ratio: {data['benefit_cost_ratio']:.2f}")

    # Calculate earnings premium
    hs_earnings = comparison['12_years']['lifetime_earnings']
    college_earnings = comparison['16_years']['lifetime_earnings']
    college_premium = ((college_earnings - hs_earnings) / hs_earnings) * 100
    print(f"\nCollege premium: {college_premium:.1f}% higher lifetime earnings than HS")

    # Effect of ability
    print("\n3. EFFECT OF ABILITY ON EDUCATION INVESTMENT")
    print("-" * 80)
    model_low_ability = HumanCapitalModel(ability=0.8)
    model_high_ability = HumanCapitalModel(ability=1.2)

    opt_low = model_low_ability.optimal_schooling()
    opt_high = model_high_ability.optimal_schooling()

    print(f"Low ability (0.8): Optimal schooling = {opt_low['optimal_years']:.1f} years, NPV = ${opt_low['npv']:,.0f}")
    print(f"High ability (1.2): Optimal schooling = {opt_high['optimal_years']:.1f} years, NPV = ${opt_high['npv']:,.0f}")
    print("\nChicago Insight: Higher ability individuals invest more in education")
    print("(ability and education are complements)")

    # Cost components
    print("\n4. COST DECOMPOSITION")
    print("-" * 80)
    S = 16  # College
    total_cost = model.total_cost(S)

    # Calculate components
    direct_costs = model.cost_per_year * S / (1 + model.discount_rate)**(S/2)  # Rough PV
    opportunity_costs = total_cost - (model.cost_per_year * S)

    print(f"For {S} years of education:")
    print(f"  Direct costs (tuition, etc.): ${model.cost_per_year * S:,.0f}")
    print(f"  Opportunity costs (foregone earnings): ${opportunity_costs:,.0f}")
    print(f"  Total cost: ${total_cost:,.0f}")
    print(f"\nOpportunity cost is {(opportunity_costs/total_cost)*100:.0f}% of total cost!")
    print("Chicago Insight: Opportunity cost often exceeds direct cost")

    # Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 80)
    model.plot_age_earnings_profiles([12, 16, 20])
    model.plot_npv_by_schooling()
    print("Plots saved to chicago_price_theory/")
