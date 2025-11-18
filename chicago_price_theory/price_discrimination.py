"""
Price Discrimination Model - Chicago Price Theory

THEORETICAL FOUNDATION:
Price discrimination occurs when a firm charges different prices to different
consumers for the same product. This is profit-maximizing behavior when firms
have market power and can prevent arbitrage.

KEY CONCEPTS:
1. First-degree (Perfect): Charge each consumer their maximum willingness to pay
2. Second-degree: Quantity discounts, self-selection mechanisms
3. Third-degree: Segment markets by observable characteristics
4. Welfare Effects: Captures consumer surplus, increases output

CHICAGO EMPHASIS:
- Price discrimination is not necessarily "bad" - increases efficiency
- May increase total output (compared to single-price monopoly)
- Requires market power AND ability to prevent resale
- Information is valuable - knowing consumer WTP creates profit
- Ubiquitous in practice: airlines, movies, software, etc.

CLASSICAL RESULTS:
- 1st degree: Perfectly efficient (like perfect competition), firm captures all surplus
- 3rd degree: Price inversely related to demand elasticity (P₁/P₂ = (1+1/η₂)/(1+1/η₁))
- Output: PD typically increases total output vs uniform pricing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Dict, List


class PriceDiscriminationModel:
    """
    Price Discrimination Model

    Analyzes monopolist's pricing strategy under different
    information conditions.
    """

    def __init__(self,
                 demand_params: Tuple[float, float] = (100, 1),
                 marginal_cost: float = 10.0):
        """
        Initialize price discrimination model.

        Parameters:
        -----------
        demand_params : tuple
            (a, b) for demand Q = a - b*P
        marginal_cost : float
            Constant marginal cost (MC)
        """
        self.a, self.b = demand_params
        self.mc = marginal_cost

    def demand(self, price: float) -> float:
        """Demand function Q(P) = a - b*P"""
        return max(0, self.a - self.b * price)

    def inverse_demand(self, quantity: float) -> float:
        """Inverse demand P(Q) = (a - Q)/b"""
        return (self.a - quantity) / self.b

    def marginal_revenue(self, quantity: float) -> float:
        """
        Marginal revenue for linear demand.

        For Q = a - b*P, we have P = (a - Q)/b
        Revenue R = P*Q = (a*Q - Q²)/b
        MR = dR/dQ = (a - 2Q)/b

        CHICAGO PRINCIPLE:
        With market power, MR < P (downward sloping demand)
        Monopolist produces where MR = MC, not P = MC
        """
        return (self.a - 2*quantity) / self.b

    def uniform_pricing(self) -> Dict:
        """
        Single-price monopoly (no discrimination).

        PROFIT MAXIMIZATION:
        max π = (P - MC) * Q
        FOC: MR = MC

        For linear demand: (a - 2Q)/b = MC
        → Q* = (a - b*MC) / 2
        → P* = (a + b*MC) / (2b) = (a/b + MC) / 2

        Returns:
        --------
        dict with price, quantity, profit, consumer surplus, deadweight loss
        """
        # Optimal quantity: MR = MC
        Q_star = (self.a - self.b * self.mc) / 2
        P_star = self.inverse_demand(Q_star)

        # Profit
        profit = (P_star - self.mc) * Q_star

        # Consumer surplus (area between demand and price)
        # For linear demand: CS = 0.5 * (P_max - P*) * Q*
        P_max = self.a / self.b  # Price when Q = 0
        consumer_surplus = 0.5 * (P_max - P_star) * Q_star

        # Deadweight loss (compared to perfect competition)
        # Perfect competition: P = MC
        Q_competitive = self.demand(self.mc)
        dwl = 0.5 * (Q_competitive - Q_star) * (P_star - self.mc)

        return {
            'price': P_star,
            'quantity': Q_star,
            'profit': profit,
            'consumer_surplus': consumer_surplus,
            'producer_surplus': profit,
            'total_surplus': consumer_surplus + profit,
            'deadweight_loss': dwl,
            'markup': P_star - self.mc,
            'lerner_index': (P_star - self.mc) / P_star  # Measure of market power
        }

    def first_degree_discrimination(self) -> Dict:
        """
        First-degree (perfect) price discrimination.

        PERFECT INFORMATION:
        Firm knows each consumer's WTP and charges accordingly.
        Charge P(Q) for each unit - extract entire consumer surplus.

        RESULT:
        - Firm captures all surplus (CS = 0)
        - Produces competitive quantity (P = MC for marginal unit)
        - Perfectly efficient (no deadweight loss)

        CHICAGO PARADOX:
        Perfect discrimination → Perfectly efficient allocation!
        But firm captures all gains from trade.

        Returns:
        --------
        dict with results of perfect price discrimination
        """
        # Produce where P = MC (efficient quantity)
        Q_star = self.demand(self.mc)
        P_marginal = self.mc  # Price for last unit

        # Total revenue: integral from 0 to Q* of P(q) dq
        # For P = (a - Q)/b: integral = (a*Q - Q²/2)/b
        total_revenue = (self.a * Q_star - Q_star**2 / 2) / self.b

        # Total cost
        total_cost = self.mc * Q_star

        # Profit
        profit = total_revenue - total_cost

        # Consumer surplus = 0 (firm captures it all)
        consumer_surplus = 0

        return {
            'quantity': Q_star,
            'marginal_price': P_marginal,
            'average_price': total_revenue / Q_star if Q_star > 0 else 0,
            'total_revenue': total_revenue,
            'profit': profit,
            'consumer_surplus': consumer_surplus,
            'producer_surplus': profit,
            'total_surplus': profit,
            'deadweight_loss': 0,
            'efficiency': 'Perfect'
        }

    def third_degree_discrimination(self,
                                   demand_params_groups: List[Tuple[float, float]]) -> Dict:
        """
        Third-degree price discrimination: Segment markets.

        MARKET SEGMENTATION:
        Separate markets by observable characteristics (age, location, etc.)
        Charge different prices in each market.

        PRICING RULE:
        MR₁ = MR₂ = ... = MC

        For linear demands, leads to:
        (P₁ - MC)/P₁ = -1/η₁  (inverse elasticity rule)

        CHICAGO INSIGHT:
        Charge higher prices to less elastic groups.
        Example: Students (elastic) get discounts, business travelers (inelastic) pay more.

        Parameters:
        -----------
        demand_params_groups : list of tuples
            Demand parameters (a, b) for each market segment

        Returns:
        --------
        dict with prices and quantities for each segment
        """
        results = {}
        total_profit = 0
        total_quantity = 0
        total_cs = 0

        for i, (a_i, b_i) in enumerate(demand_params_groups):
            # For market i: Q_i = a_i - b_i * P_i
            # MR_i = (a_i - 2*Q_i) / b_i
            # Set MR_i = MC:
            Q_i = (a_i - b_i * self.mc) / 2
            P_i = (a_i - Q_i) / b_i

            # Profit from market i
            profit_i = (P_i - self.mc) * Q_i

            # Consumer surplus in market i
            P_max_i = a_i / b_i
            cs_i = 0.5 * (P_max_i - P_i) * Q_i

            # Elasticity at (P_i, Q_i)
            # η = (dQ/dP) * (P/Q) = -b_i * (P_i / Q_i)
            elasticity_i = -b_i * (P_i / Q_i) if Q_i > 0 else float('-inf')

            results[f'market_{i+1}'] = {
                'price': P_i,
                'quantity': Q_i,
                'profit': profit_i,
                'consumer_surplus': cs_i,
                'elasticity': elasticity_i,
                'markup': P_i - self.mc
            }

            total_profit += profit_i
            total_quantity += Q_i
            total_cs += cs_i

        results['total'] = {
            'total_profit': total_profit,
            'total_quantity': total_quantity,
            'total_consumer_surplus': total_cs,
            'total_surplus': total_profit + total_cs
        }

        return results

    def compare_strategies(self) -> Dict:
        """
        Compare uniform pricing, perfect discrimination, and third-degree discrimination.

        CHICAGO ANALYSIS:
        Shows trade-offs between:
        - Firm profit
        - Consumer surplus
        - Total surplus (efficiency)
        - Distribution of surplus

        Returns:
        --------
        dict comparing all strategies
        """
        # Uniform pricing
        uniform = self.uniform_pricing()

        # Perfect discrimination
        perfect = self.first_degree_discrimination()

        # Third-degree (two markets: elastic and inelastic)
        # Market 1: More elastic (students)
        # Market 2: Less elastic (business travelers)
        third = self.third_degree_discrimination([
            (80, 2),   # More elastic: a=80, b=2
            (120, 0.5)  # Less elastic: a=120, b=0.5
        ])

        return {
            'uniform_pricing': uniform,
            'perfect_discrimination': perfect,
            'third_degree_discrimination': third
        }

    def plot_discrimination_comparison(self) -> None:
        """Visualize price discrimination strategies."""
        uniform = self.uniform_pricing()
        perfect = self.first_degree_discrimination()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Uniform pricing vs Perfect discrimination
        Q_range = np.linspace(0, self.a/self.b, 200)
        P_demand = np.array([self.inverse_demand(q) for q in Q_range])

        # Uniform pricing
        ax1.plot(Q_range, P_demand, 'b-', linewidth=2, label='Demand')
        ax1.axhline(y=self.mc, color='green', linestyle='--', linewidth=2, label=f'MC = ${self.mc}')

        # Uniform price
        ax1.axhline(y=uniform['price'], color='red', linestyle='-', linewidth=2,
                   label=f'Uniform Price = ${uniform["price"]:.2f}')
        ax1.axvline(x=uniform['quantity'], color='gray', linestyle=':', alpha=0.5)

        # Shade consumer surplus
        q_uni = np.linspace(0, uniform['quantity'], 100)
        p_uni = np.array([self.inverse_demand(q) for q in q_uni])
        ax1.fill_between(q_uni, uniform['price'], p_uni, alpha=0.3, color='blue',
                        label=f'Consumer Surplus = ${uniform["consumer_surplus"]:.0f}')

        # Shade profit
        ax1.fill_between([0, uniform['quantity']], self.mc, uniform['price'],
                        alpha=0.3, color='red',
                        label=f'Producer Surplus = ${uniform["profit"]:.0f}')

        # Shade DWL
        q_dwl = np.linspace(uniform['quantity'], perfect['quantity'], 50)
        p_dwl = np.array([self.inverse_demand(q) for q in q_dwl])
        ax1.fill_between(q_dwl, self.mc, p_dwl, alpha=0.3, color='gray',
                        label=f'Deadweight Loss = ${uniform["deadweight_loss"]:.0f}')

        ax1.set_xlabel('Quantity', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title('Uniform Pricing', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, self.a/self.b)
        ax1.set_ylim(0, self.a/self.b * 1.2)

        # Perfect discrimination
        ax2.plot(Q_range, P_demand, 'b-', linewidth=2, label='Demand = WTP')
        ax2.axhline(y=self.mc, color='green', linestyle='--', linewidth=2, label=f'MC = ${self.mc}')

        # Shade producer surplus (firm captures all surplus)
        q_perfect = np.linspace(0, perfect['quantity'], 100)
        p_perfect = np.array([self.inverse_demand(q) for q in q_perfect])
        ax2.fill_between(q_perfect, self.mc, p_perfect, alpha=0.3, color='red',
                        label=f'Producer Surplus = ${perfect["profit"]:.0f}')

        ax2.axvline(x=perfect['quantity'], color='gray', linestyle=':', alpha=0.5,
                   label=f'Q = {perfect["quantity"]:.1f} (efficient)')

        ax2.set_xlabel('Quantity', fontsize=12)
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.set_title('Perfect Price Discrimination', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.a/self.b)
        ax2.set_ylim(0, self.a/self.b * 1.2)

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/price_discrimination.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    print("=" * 80)
    print("PRICE DISCRIMINATION MODEL - CHICAGO PRICE THEORY")
    print("=" * 80)

    model = PriceDiscriminationModel(demand_params=(100, 1), marginal_cost=10)

    # Uniform pricing
    print("\n1. UNIFORM PRICING (Single Price)")
    print("-" * 80)
    uniform = model.uniform_pricing()
    print(f"Optimal price: P = ${uniform['price']:.2f}")
    print(f"Quantity sold: Q = {uniform['quantity']:.2f}")
    print(f"Firm profit: ${uniform['profit']:.2f}")
    print(f"Consumer surplus: ${uniform['consumer_surplus']:.2f}")
    print(f"Total surplus: ${uniform['total_surplus']:.2f}")
    print(f"Deadweight loss: ${uniform['deadweight_loss']:.2f}")
    print(f"Lerner index (markup/price): {uniform['lerner_index']:.2f}")

    # Perfect price discrimination
    print("\n2. PERFECT PRICE DISCRIMINATION")
    print("-" * 80)
    perfect = model.first_degree_discrimination()
    print(f"Quantity sold: Q = {perfect['quantity']:.2f}")
    print(f"Marginal price (last unit): ${perfect['marginal_price']:.2f}")
    print(f"Average price: ${perfect['average_price']:.2f}")
    print(f"Firm profit: ${perfect['profit']:.2f}")
    print(f"Consumer surplus: ${perfect['consumer_surplus']:.2f}")
    print(f"Deadweight loss: ${perfect['deadweight_loss']:.2f}")
    print(f"Efficiency: {perfect['efficiency']}")
    print("\nChicago Insight: Perfect discrimination is perfectly efficient!")
    print("Firm captures all surplus, but output equals competitive level.")

    # Third-degree discrimination
    print("\n3. THIRD-DEGREE DISCRIMINATION (Market Segmentation)")
    print("-" * 80)
    third = model.third_degree_discrimination([(80, 2), (120, 0.5)])
    print("Market 1 (Elastic - e.g., Students):")
    print(f"  Price: ${third['market_1']['price']:.2f}")
    print(f"  Quantity: {third['market_1']['quantity']:.2f}")
    print(f"  Elasticity: {third['market_1']['elasticity']:.2f}")
    print(f"  Profit: ${third['market_1']['profit']:.2f}")

    print("\nMarket 2 (Inelastic - e.g., Business):")
    print(f"  Price: ${third['market_2']['price']:.2f}")
    print(f"  Quantity: {third['market_2']['quantity']:.2f}")
    print(f"  Elasticity: {third['market_2']['elasticity']:.2f}")
    print(f"  Profit: ${third['market_2']['profit']:.2f}")

    print(f"\nTotal profit: ${third['total']['total_profit']:.2f}")
    print(f"Price ratio: {third['market_2']['price'] / third['market_1']['price']:.2f}")
    print("\nChicago Insight: Higher prices charged to less elastic segment!")

    # Comparison
    print("\n4. STRATEGY COMPARISON")
    print("-" * 80)
    print(f"{'Strategy':<25} {'Profit':<15} {'Cons. Surplus':<18} {'Total Surplus':<15} {'Quantity':<10}")
    print("-" * 80)
    print(f"{'Uniform Pricing':<25} ${uniform['profit']:<14.0f} ${uniform['consumer_surplus']:<17.0f} "
          f"${uniform['total_surplus']:<14.0f} {uniform['quantity']:<10.1f}")
    print(f"{'Perfect Discrimination':<25} ${perfect['profit']:<14.0f} ${perfect['consumer_surplus']:<17.0f} "
          f"${perfect['total_surplus']:<14.0f} {perfect['quantity']:<10.1f}")
    print(f"{'Third Degree':<25} ${third['total']['total_profit']:<14.0f} "
          f"${third['total']['total_consumer_surplus']:<17.0f} "
          f"${third['total']['total_surplus']:<14.0f} {third['total']['total_quantity']:<10.1f}")

    print("\nKey Insights:")
    print("• Perfect discrimination maximizes total surplus (efficient)")
    print("• Firm always prefers discrimination over uniform pricing")
    print("• Price discrimination can increase output above monopoly level")
    print("• Welfare effects ambiguous - depends on market structure")

    # Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 80)
    model.plot_discrimination_comparison()
    print("Plot saved to chicago_price_theory/price_discrimination.png")
