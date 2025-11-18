"""
Consumer Theory Model - Chicago Price Theory

THEORETICAL FOUNDATION:
Consumer theory models rational agents maximizing utility subject to budget constraints.
This is the micro foundation for demand curves and demonstrates how prices guide
resource allocation.

KEY CONCEPTS:
1. Utility Function: U(x₁, x₂) represents preferences over goods
2. Budget Constraint: p₁x₁ + p₂x₂ = I (income)
3. Marginal Rate of Substitution (MRS): Rate at which consumer trades goods
4. Optimization: Choose bundle where MRS = price ratio
5. Demand Functions: Optimal quantities as functions of prices and income

CHICAGO EMPHASIS:
- Revealed preference: Choices reveal preferences
- Substitution is everywhere: Relative price changes drive behavior
- Income and substitution effects: Decomposing response to price changes
- Testable implications: Demand curves slope downward

CLASSICAL RESULTS:
- Slutsky Equation: Decomposes price effect into substitution and income effects
- Engel Curves: How consumption varies with income
- Homogeneity: Demand depends on relative prices and real income
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from typing import Tuple, Dict, List, Callable


class ConsumerModel:
    """
    Consumer Utility Maximization Model

    Supports multiple utility function types:
    - Cobb-Douglas: U(x₁, x₂) = x₁^α * x₂^(1-α)
    - CES (Constant Elasticity of Substitution): U = (αx₁^ρ + (1-α)x₂^ρ)^(1/ρ)
    - Perfect Substitutes: U = αx₁ + βx₂
    - Perfect Complements (Leontief): U = min(αx₁, βx₂)
    """

    def __init__(self,
                 income: float,
                 prices: Tuple[float, float],
                 utility_type: str = 'cobb_douglas',
                 utility_params: Tuple[float, ...] = (0.5,)):
        """
        Initialize consumer model.

        Parameters:
        -----------
        income : float
            Consumer's income (I)
        prices : tuple
            Prices (p₁, p₂) of the two goods
        utility_type : str
            Type of utility function
        utility_params : tuple
            Parameters for utility function
            - Cobb-Douglas: (α,) where 0 < α < 1
            - CES: (α, ρ) where σ = 1/(1-ρ) is elasticity of substitution
            - Perfect Substitutes: (α, β)
            - Perfect Complements: (α, β)
        """
        self.income = income
        self.prices = np.array(prices)
        self.utility_type = utility_type
        self.utility_params = utility_params

    def utility(self, x: np.ndarray) -> float:
        """
        Utility function U(x₁, x₂).

        Different specifications capture different substitution possibilities.
        """
        x1, x2 = x[0], x[1]

        if self.utility_type == 'cobb_douglas':
            alpha = self.utility_params[0]
            # U = x₁^α * x₂^(1-α)
            # This has elasticity of substitution σ = 1
            return (x1 ** alpha) * (x2 ** (1 - alpha))

        elif self.utility_type == 'ces':
            alpha, rho = self.utility_params
            # U = (αx₁^ρ + (1-α)x₂^ρ)^(1/ρ)
            # σ = 1/(1-ρ): ρ→1 (perfect substitutes), ρ→-∞ (perfect complements)
            return (alpha * x1**rho + (1 - alpha) * x2**rho) ** (1/rho)

        elif self.utility_type == 'perfect_substitutes':
            alpha, beta = self.utility_params
            # U = αx₁ + βx₂ (σ = ∞)
            return alpha * x1 + beta * x2

        elif self.utility_type == 'perfect_complements':
            alpha, beta = self.utility_params
            # U = min(αx₁, βx₂) (σ = 0, Leontief)
            return min(alpha * x1, beta * x2)

        else:
            raise ValueError(f"Unknown utility type: {self.utility_type}")

    def marginal_utility(self, x: np.ndarray) -> np.ndarray:
        """
        Marginal utilities: ∂U/∂x₁ and ∂U/∂x₂

        CHICAGO PRINCIPLE:
        Decisions are made at the margin. Marginal utility determines
        the value of an additional unit.
        """
        eps = 1e-6
        x1, x2 = x[0], x[1]

        # Numerical derivatives
        mu1 = (self.utility([x1 + eps, x2]) - self.utility(x)) / eps
        mu2 = (self.utility([x1, x2 + eps]) - self.utility(x)) / eps

        return np.array([mu1, mu2])

    def mrs(self, x: np.ndarray) -> float:
        """
        Marginal Rate of Substitution: MRS = MU₁/MU₂

        INTERPRETATION:
        MRS is the rate at which the consumer is willing to trade good 2 for good 1
        while maintaining constant utility.

        OPTIMALITY CONDITION:
        At optimum, MRS = p₁/p₂ (tangency of indifference curve and budget line)
        """
        mu = self.marginal_utility(x)
        if abs(mu[1]) < 1e-10:
            return float('inf')
        return mu[0] / mu[1]

    def budget_constraint(self, x: np.ndarray) -> float:
        """
        Budget constraint: p₁x₁ + p₂x₂ = I

        Returns expenditure - income (should equal 0 on budget line)
        """
        return np.dot(self.prices, x) - self.income

    def maximize_utility(self) -> Dict:
        """
        Solve the consumer's optimization problem:

        max U(x₁, x₂)
        s.t. p₁x₁ + p₂x₂ ≤ I
             x₁, x₂ ≥ 0

        LAGRANGIAN APPROACH:
        L = U(x₁, x₂) + λ(I - p₁x₁ - p₂x₂)

        FOCs:
        ∂L/∂x₁ = MU₁ - λp₁ = 0  →  MU₁/p₁ = λ
        ∂L/∂x₂ = MU₂ - λp₂ = 0  →  MU₂/p₂ = λ
        ∂L/∂λ = I - p₁x₁ - p₂x₂ = 0

        From FOCs: MU₁/MU₂ = p₁/p₂ (MRS = price ratio)

        CHICAGO INTERPRETATION:
        The consumer equates marginal utility per dollar across all goods.
        This is the foundation of rational choice theory.

        Returns:
        --------
        dict with optimal bundle, utility, MRS, and Lagrange multiplier
        """
        p1, p2 = self.prices

        # Analytical solutions for special cases
        if self.utility_type == 'cobb_douglas':
            alpha = self.utility_params[0]
            # Demand functions for Cobb-Douglas:
            # x₁* = αI/p₁
            # x₂* = (1-α)I/p₂
            x1_star = (alpha * self.income) / p1
            x2_star = ((1 - alpha) * self.income) / p2
            x_star = np.array([x1_star, x2_star])

        elif self.utility_type == 'perfect_substitutes':
            alpha, beta = self.utility_params
            # Corner solution: buy only the good with better "bang for buck"
            if alpha/p1 > beta/p2:
                x_star = np.array([self.income/p1, 0])
            elif alpha/p1 < beta/p2:
                x_star = np.array([0, self.income/p2])
            else:
                # Indifferent - any point on budget line works
                x_star = np.array([self.income/(2*p1), self.income/(2*p2)])

        elif self.utility_type == 'perfect_complements':
            alpha, beta = self.utility_params
            # Must consume in fixed proportions: αx₁ = βx₂
            # Budget: p₁x₁ + p₂x₂ = I
            # Solve: x₁ = βx₂/α
            # p₁(βx₂/α) + p₂x₂ = I
            # x₂(p₁β/α + p₂) = I
            x2_star = self.income / (p1*beta/alpha + p2)
            x1_star = beta * x2_star / alpha
            x_star = np.array([x1_star, x2_star])

        else:  # CES or other - use numerical optimization
            # Objective: minimize -U(x) subject to budget constraint
            def objective(x):
                if x[0] < 0 or x[1] < 0:
                    return 1e10
                return -self.utility(x)

            # Constraint: p₁x₁ + p₂x₂ = I
            constraints = {
                'type': 'eq',
                'fun': self.budget_constraint
            }

            # Initial guess: split income equally
            x0 = np.array([self.income/(2*p1), self.income/(2*p2)])

            # Optimize
            result = minimize(objective, x0, method='SLSQP',
                            constraints=constraints,
                            bounds=[(0, None), (0, None)])
            x_star = result.x

        # Calculate results
        u_star = self.utility(x_star)
        mrs_star = self.mrs(x_star)

        # Lagrange multiplier (marginal utility of income)
        mu = self.marginal_utility(x_star)
        lambda_star = mu[0] / p1  # = mu[1] / p2 at optimum

        return {
            'x1': x_star[0],
            'x2': x_star[1],
            'utility': u_star,
            'mrs': mrs_star,
            'price_ratio': p1/p2,
            'lambda': lambda_star,
            'expenditure': np.dot(self.prices, x_star)
        }

    def derive_demand_curve(self,
                           good: int = 1,
                           price_range: Tuple[float, float] = (1, 50),
                           n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive demand curve by varying price of one good.

        CHICAGO METHODOLOGY:
        Demand curves come from solving optimization problems at different prices.
        This generates testable predictions about price-quantity relationships.

        Parameters:
        -----------
        good : int
            Which good's price to vary (1 or 2)
        price_range : tuple
            (min_price, max_price)
        n_points : int
            Number of price points

        Returns:
        --------
        (prices, quantities) arrays
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        quantities = np.zeros(n_points)

        original_prices = self.prices.copy()

        for i, p in enumerate(prices):
            # Update price
            if good == 1:
                self.prices = np.array([p, original_prices[1]])
            else:
                self.prices = np.array([original_prices[0], p])

            # Solve optimization
            result = self.maximize_utility()
            quantities[i] = result['x1'] if good == 1 else result['x2']

        # Restore original prices
        self.prices = original_prices

        return prices, quantities

    def derive_engel_curve(self,
                          good: int = 1,
                          income_range: Tuple[float, float] = (100, 2000),
                          n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive Engel curve: quantity consumed vs. income.

        CHICAGO INSIGHT:
        Engel curves show how consumption patterns change with wealth.
        - Normal good: quantity increases with income (slope > 0)
        - Inferior good: quantity decreases with income (slope < 0)
        - Necessity: income elasticity < 1
        - Luxury: income elasticity > 1

        Parameters:
        -----------
        good : int
            Which good to analyze (1 or 2)
        income_range : tuple
            (min_income, max_income)
        n_points : int
            Number of income points

        Returns:
        --------
        (incomes, quantities) arrays
        """
        incomes = np.linspace(income_range[0], income_range[1], n_points)
        quantities = np.zeros(n_points)

        original_income = self.income

        for i, inc in enumerate(incomes):
            self.income = inc
            result = self.maximize_utility()
            quantities[i] = result['x1'] if good == 1 else result['x2']

        # Restore original income
        self.income = original_income

        return incomes, quantities

    def slutsky_decomposition(self,
                             good: int = 1,
                             price_change: float = 1.0) -> Dict:
        """
        Slutsky decomposition: Total effect = Substitution effect + Income effect

        CHICAGO THEORETICAL CORNERSTONE:
        When price changes, two things happen:
        1. Substitution effect: Good becomes relatively cheaper/expensive (MRS ≠ p₁/p₂)
        2. Income effect: Real purchasing power changes

        THE SLUTSKY EQUATION:
        dx/dp = dx/dp|U=const - x*(dx/dI)
        Total   = Substitution  - Income effect
        effect    effect

        Key results:
        - Substitution effect always opposes price change (law of demand)
        - Income effect depends on whether good is normal or inferior
        - For normal goods, both effects reinforce → demand slopes down
        - For inferior goods, effects oppose, but substitution usually dominates

        Parameters:
        -----------
        good : int
            Which good's price changes (1 or 2)
        price_change : float
            Change in price (can be positive or negative)

        Returns:
        --------
        dict with initial bundle, final bundle, substitution bundle,
        and decomposed effects
        """
        # Initial equilibrium (point A)
        initial = self.maximize_utility()
        x_initial = np.array([initial['x1'], initial['x2']])
        u_initial = initial['utility']

        # Change price (move to point C)
        original_prices = self.prices.copy()
        if good == 1:
            self.prices = np.array([original_prices[0] + price_change, original_prices[1]])
        else:
            self.prices = np.array([original_prices[0], original_prices[1] + price_change])

        final = self.maximize_utility()
        x_final = np.array([final['x1'], final['x2']])

        # Substitution effect (point A to point B):
        # Find bundle at new prices that achieves original utility
        # This requires compensating income change

        def find_compensated_bundle(income_adjust):
            """Find bundle at new prices that achieves original utility."""
            self.income = self.income + income_adjust
            result = self.maximize_utility()
            achieved_utility = result['utility']
            self.income = self.income - income_adjust  # Reset
            return achieved_utility - u_initial

        # Find compensating variation
        from scipy.optimize import brentq
        try:
            # Compensating variation: income change needed to restore original utility
            compensation = brentq(find_compensated_bundle, -self.income*0.9, self.income*2)
        except:
            # If optimization fails, use approximation
            compensation = (original_prices[good-1] - self.prices[good-1]) * x_initial[good-1]

        # Bundle at point B (substitution effect)
        self.income += compensation
        substitution_result = self.maximize_utility()
        x_substitution = np.array([substitution_result['x1'], substitution_result['x2']])
        self.income -= compensation

        # Restore original prices
        self.prices = original_prices

        # Calculate effects
        total_effect = x_final - x_initial
        substitution_effect = x_substitution - x_initial
        income_effect = x_final - x_substitution

        return {
            'initial_bundle': x_initial,
            'final_bundle': x_final,
            'substitution_bundle': x_substitution,
            'total_effect': total_effect,
            'substitution_effect': substitution_effect,
            'income_effect': income_effect,
            'compensating_variation': compensation,
            'good_type': 'normal' if income_effect[good-1] * (-price_change) > 0 else 'inferior'
        }

    def plot_indifference_curves(self,
                                utility_levels: List[float] = None,
                                show_optimal: bool = True,
                                show_budget: bool = True) -> None:
        """
        Visualize indifference curves and optimal choice.

        Parameters:
        -----------
        utility_levels : list
            Utility levels to plot. If None, automatically chosen.
        show_optimal : bool
            Show optimal consumption bundle
        show_budget : bool
            Show budget constraint
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Find optimal bundle
        optimal = self.maximize_utility()
        x1_opt, x2_opt = optimal['x1'], optimal['x2']
        u_opt = optimal['utility']

        # Set up grid
        x1_max = self.income / self.prices[0] * 1.5
        x2_max = self.income / self.prices[1] * 1.5
        x1_grid = np.linspace(0.1, x1_max, 200)

        # Plot indifference curves
        if utility_levels is None:
            utility_levels = [u_opt * 0.5, u_opt * 0.75, u_opt, u_opt * 1.25]

        for u_level in utility_levels:
            x2_values = []
            for x1 in x1_grid:
                # For each x1, solve for x2 that gives utility u_level
                def find_x2(x2):
                    return self.utility(np.array([x1, x2])) - u_level

                try:
                    x2 = fsolve(find_x2, x0=x2_max/2)[0]
                    if x2 > 0 and x2 < x2_max:
                        x2_values.append(x2)
                    else:
                        x2_values.append(np.nan)
                except:
                    x2_values.append(np.nan)

            label = 'Optimal IC' if np.isclose(u_level, u_opt) else None
            color = 'green' if np.isclose(u_level, u_opt) else 'blue'
            linewidth = 2.5 if np.isclose(u_level, u_opt) else 1.5
            ax.plot(x1_grid, x2_values, color=color, linewidth=linewidth,
                   alpha=0.7, label=label)

        # Budget constraint
        if show_budget:
            x1_budget = np.array([0, self.income / self.prices[0]])
            x2_budget = np.array([self.income / self.prices[1], 0])
            ax.plot(x1_budget, x2_budget, 'r-', linewidth=2,
                   label=f'Budget: {self.prices[0]:.1f}x₁ + {self.prices[1]:.1f}x₂ = {self.income:.0f}')

        # Optimal point
        if show_optimal:
            ax.plot(x1_opt, x2_opt, 'ro', markersize=12,
                   label=f'Optimal: ({x1_opt:.2f}, {x2_opt:.2f})\nU = {u_opt:.2f}')

            # Show tangency
            slope = -self.prices[0] / self.prices[1]  # MRS = p1/p2
            x1_tangent = np.array([x1_opt - 5, x1_opt + 5])
            x2_tangent = x2_opt + slope * (x1_tangent - x1_opt)
            ax.plot(x1_tangent, x2_tangent, 'g--', linewidth=1.5, alpha=0.5,
                   label=f'Tangent: MRS = {optimal["mrs"]:.2f} = p₁/p₂')

        ax.set_xlabel('Good 1 (x₁)', fontsize=12)
        ax.set_ylabel('Good 2 (x₂)', fontsize=12)
        ax.set_title(f'Consumer Optimization - {self.utility_type.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x1_max)
        ax.set_ylim(0, x2_max)

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/consumer_indifference_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_demand_curve(self, good: int = 1) -> None:
        """Plot demand curve for specified good."""
        prices, quantities = self.derive_demand_curve(good=good)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(quantities, prices, 'b-', linewidth=2)
        ax.set_xlabel(f'Quantity of Good {good}', fontsize=12)
        ax.set_ylabel(f'Price of Good {good}', fontsize=12)
        ax.set_title(f'Demand Curve - {self.utility_type.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark current equilibrium
        optimal = self.maximize_utility()
        q_current = optimal['x1'] if good == 1 else optimal['x2']
        p_current = self.prices[good-1]
        ax.plot(q_current, p_current, 'ro', markersize=10,
               label=f'Current: P={p_current:.2f}, Q={q_current:.2f}')
        ax.legend()

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/consumer_demand_curve.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    print("=" * 80)
    print("CONSUMER THEORY MODEL - CHICAGO PRICE THEORY")
    print("=" * 80)

    # Cobb-Douglas consumer
    print("\n1. UTILITY MAXIMIZATION (Cobb-Douglas)")
    print("-" * 80)
    consumer = ConsumerModel(income=1000, prices=(10, 20), utility_type='cobb_douglas',
                            utility_params=(0.6,))

    optimal = consumer.maximize_utility()
    print(f"Income: ${consumer.income}")
    print(f"Prices: p₁ = ${consumer.prices[0]}, p₂ = ${consumer.prices[1]}")
    print(f"\nOptimal Bundle:")
    print(f"  x₁* = {optimal['x1']:.2f} units")
    print(f"  x₂* = {optimal['x2']:.2f} units")
    print(f"  Maximum Utility = {optimal['utility']:.2f}")
    print(f"\nOptimality Check:")
    print(f"  MRS = {optimal['mrs']:.2f}")
    print(f"  Price ratio (p₁/p₂) = {optimal['price_ratio']:.2f}")
    print(f"  MRS = p₁/p₂? {np.isclose(optimal['mrs'], optimal['price_ratio'])}")
    print(f"\nMarginal Utility of Income (λ) = {optimal['lambda']:.4f}")
    print(f"  Interpretation: An extra $1 increases utility by {optimal['lambda']:.4f}")

    # Slutsky decomposition
    print("\n2. SLUTSKY DECOMPOSITION")
    print("-" * 80)
    print("Scenario: Price of good 1 increases from $10 to $15")
    slutsky = consumer.slutsky_decomposition(good=1, price_change=5)
    print(f"\nInitial bundle: x₁={slutsky['initial_bundle'][0]:.2f}, x₂={slutsky['initial_bundle'][1]:.2f}")
    print(f"Final bundle: x₁={slutsky['final_bundle'][0]:.2f}, x₂={slutsky['final_bundle'][1]:.2f}")
    print(f"\nDecomposition for good 1:")
    print(f"  Total effect: Δx₁ = {slutsky['total_effect'][0]:.2f}")
    print(f"  Substitution effect: {slutsky['substitution_effect'][0]:.2f}")
    print(f"  Income effect: {slutsky['income_effect'][0]:.2f}")
    print(f"\nGood 1 is a {slutsky['good_type']} good")
    print(f"Compensating variation: ${slutsky['compensating_variation']:.2f}")

    # Demand curve
    print("\n3. DEMAND CURVE DERIVATION")
    print("-" * 80)
    prices, quantities = consumer.derive_demand_curve(good=1, price_range=(5, 30))
    # Calculate price elasticity at current price
    idx = np.argmin(np.abs(prices - 10))
    if idx > 0 and idx < len(prices) - 1:
        dQ = quantities[idx+1] - quantities[idx-1]
        dP = prices[idx+1] - prices[idx-1]
        elasticity = (dQ/dP) * (prices[idx]/quantities[idx])
        print(f"Price elasticity of demand at P=$10: {elasticity:.2f}")
        print(f"Interpretation: 1% increase in price → {abs(elasticity):.2f}% decrease in quantity")

    # Engel curve
    print("\n4. ENGEL CURVE (Income Effects)")
    print("-" * 80)
    incomes, quantities = consumer.derive_engel_curve(good=1, income_range=(500, 2000))
    # Calculate income elasticity
    idx = len(incomes) // 2
    dQ = quantities[idx+1] - quantities[idx-1]
    dI = incomes[idx+1] - incomes[idx-1]
    income_elasticity = (dQ/dI) * (incomes[idx]/quantities[idx])
    print(f"Income elasticity at I=${incomes[idx]:.0f}: {income_elasticity:.2f}")
    if income_elasticity > 1:
        print("Good 1 is a LUXURY good (income elasticity > 1)")
    elif income_elasticity > 0:
        print("Good 1 is a NECESSITY (0 < income elasticity < 1)")
    else:
        print("Good 1 is an INFERIOR good (income elasticity < 0)")

    # Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 80)
    consumer.plot_indifference_curves()
    consumer.plot_demand_curve(good=1)
    print("Plots saved to chicago_price_theory/")
