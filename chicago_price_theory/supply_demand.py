"""
Supply and Demand Model - Chicago Price Theory

THEORETICAL FOUNDATION:
The supply and demand model is the cornerstone of Chicago price theory. It demonstrates
how prices serve as signals that coordinate the decisions of independent agents, leading
to market equilibrium without central planning.

KEY CONCEPTS:
1. Demand: Represents consumer willingness to pay at different prices (downward sloping)
2. Supply: Represents producer willingness to sell at different prices (upward sloping)
3. Equilibrium: Price where quantity demanded equals quantity supplied
4. Price Adjustment: Tatonnement process - prices adjust to clear markets
5. Comparative Statics: How equilibrium changes with shifts in supply/demand

CHICAGO EMPHASIS:
- Prices as information aggregators
- Self-correcting nature of markets
- Predictive power through comparative statics
- Empirical testability of predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from typing import Tuple, Dict, Optional


class SupplyDemandModel:
    """
    Supply and Demand Model with Market Equilibrium Analysis

    Default specifications use linear functions:
    Demand: Q_d = a - b*P  (a > 0, b > 0)
    Supply: Q_s = c + d*P  (c can be negative, d > 0)
    """

    def __init__(self,
                 demand_params: Tuple[float, float] = (100, 2),  # (a, b)
                 supply_params: Tuple[float, float] = (-20, 3),  # (c, d)
                 model_type: str = 'linear'):
        """
        Initialize supply and demand model.

        Parameters:
        -----------
        demand_params : tuple
            For linear: (intercept, slope) where Q_d = a - b*P
        supply_params : tuple
            For linear: (intercept, slope) where Q_s = c + d*P
        model_type : str
            'linear' or 'nonlinear' (nonlinear uses isoelastic forms)
        """
        self.demand_params = demand_params
        self.supply_params = supply_params
        self.model_type = model_type

    def demand(self, price: float) -> float:
        """
        Demand function: Q_d(P)

        For linear: Q_d = a - b*P
        This reflects diminishing marginal utility - consumers buy less at higher prices
        """
        if self.model_type == 'linear':
            a, b = self.demand_params
            return max(0, a - b * price)  # Quantity can't be negative
        else:  # isoelastic
            a, b = self.demand_params
            return a * price**(-b)

    def supply(self, price: float) -> float:
        """
        Supply function: Q_s(P)

        For linear: Q_s = c + d*P
        This reflects increasing marginal cost - producers supply more at higher prices
        """
        if self.model_type == 'linear':
            c, d = self.supply_params
            return max(0, c + d * price)
        else:  # isoelastic
            c, d = self.supply_params
            return c * price**d

    def excess_demand(self, price: float) -> float:
        """
        Excess demand function: Q_d(P) - Q_s(P)

        Positive: shortage (Q_d > Q_s) -> price should rise
        Negative: surplus (Q_d < Q_s) -> price should fall
        Zero: equilibrium (Q_d = Q_s)
        """
        return self.demand(price) - self.supply(price)

    def find_equilibrium(self) -> Dict[str, float]:
        """
        Find market equilibrium using optimization.

        CHICAGO INSIGHT:
        Equilibrium is where markets clear. No central planner needed -
        the price mechanism coordinates millions of independent decisions.

        Returns:
        --------
        dict with keys:
            - price: equilibrium price P*
            - quantity: equilibrium quantity Q*
            - consumer_surplus: area between demand curve and price
            - producer_surplus: area between price and supply curve
            - total_surplus: sum of consumer and producer surplus
        """
        # For linear case, solve analytically
        if self.model_type == 'linear':
            a, b = self.demand_params
            c, d = self.supply_params

            # Set Q_d = Q_s: a - b*P = c + d*P
            # Solve: P* = (a - c) / (b + d)
            p_star = (a - c) / (b + d)
            q_star = self.demand(p_star)
        else:
            # Numerical solution for nonlinear
            p_star = fsolve(self.excess_demand, x0=1.0)[0]
            q_star = self.demand(p_star)

        # Calculate welfare measures
        consumer_surplus = self._calculate_consumer_surplus(p_star, q_star)
        producer_surplus = self._calculate_producer_surplus(p_star, q_star)

        return {
            'price': p_star,
            'quantity': q_star,
            'consumer_surplus': consumer_surplus,
            'producer_surplus': producer_surplus,
            'total_surplus': consumer_surplus + producer_surplus
        }

    def _calculate_consumer_surplus(self, p_eq: float, q_eq: float) -> float:
        """
        Consumer surplus: integral from 0 to Q* of [P_demand(Q) - P*] dQ

        Represents the benefit consumers receive from paying less than
        their maximum willingness to pay.
        """
        if self.model_type == 'linear':
            a, b = self.demand_params
            # For linear demand P = (a - Q)/b
            # CS = integral from 0 to Q* of [(a - Q)/b - P*] dQ
            # CS = [aQ/b - Q²/(2b) - P*Q] from 0 to Q*
            cs = (a * q_eq / b) - (q_eq**2 / (2*b)) - (p_eq * q_eq)
            return cs
        else:
            # Numerical integration for nonlinear
            from scipy.integrate import quad
            inverse_demand = lambda q: (q / self.demand_params[0])**(-1/self.demand_params[1])
            cs, _ = quad(lambda q: inverse_demand(q) - p_eq, 0, q_eq)
            return cs

    def _calculate_producer_surplus(self, p_eq: float, q_eq: float) -> float:
        """
        Producer surplus: integral from 0 to Q* of [P* - P_supply(Q)] dQ

        Represents the benefit producers receive from selling at a price
        higher than their minimum acceptable price (marginal cost).
        """
        if self.model_type == 'linear':
            c, d = self.supply_params
            # For linear supply P = (Q - c)/d
            # PS = integral from 0 to Q* of [P* - (Q - c)/d] dQ
            ps = (p_eq * q_eq) - (q_eq**2 / (2*d)) + (c * q_eq / d)
            return ps
        else:
            from scipy.integrate import quad
            inverse_supply = lambda q: (q / self.supply_params[0])**(1/self.supply_params[1])
            ps, _ = quad(lambda q: p_eq - inverse_supply(q), 0, q_eq)
            return ps

    def calculate_elasticities(self, price: float) -> Dict[str, float]:
        """
        Calculate price elasticities of demand and supply at given price.

        Elasticity = (dQ/dP) * (P/Q)

        CHICAGO INSIGHT:
        Elasticities measure responsiveness to price changes - crucial for
        predicting behavioral responses to policy interventions.

        Returns:
        --------
        dict with:
            - demand_elasticity: % change in Q_d per 1% change in P (negative)
            - supply_elasticity: % change in Q_s per 1% change in P (positive)
        """
        if self.model_type == 'linear':
            a, b = self.demand_params
            c, d = self.supply_params

            q_d = self.demand(price)
            q_s = self.supply(price)

            # For linear: dQ_d/dP = -b, dQ_s/dP = d
            ed = -b * (price / q_d) if q_d > 0 else float('-inf')
            es = d * (price / q_s) if q_s > 0 else float('inf')
        else:
            # For isoelastic forms, elasticity is constant
            ed = -self.demand_params[1]
            es = self.supply_params[1]

        return {
            'demand_elasticity': ed,
            'supply_elasticity': es
        }

    def comparative_statics(self,
                           demand_shift: float = 0,
                           supply_shift: float = 0) -> Dict[str, Dict]:
        """
        Analyze how equilibrium changes with shifts in supply/demand.

        CHICAGO METHODOLOGY:
        Comparative statics is the primary tool for generating testable predictions.
        Example: "An increase in demand raises both price and quantity."

        Parameters:
        -----------
        demand_shift : float
            Change in demand intercept (positive = rightward shift)
        supply_shift : float
            Change in supply intercept (positive = rightward shift)

        Returns:
        --------
        dict with 'initial' and 'new' equilibria
        """
        # Initial equilibrium
        initial_eq = self.find_equilibrium()

        # Create new model with shifted parameters
        new_demand_params = (self.demand_params[0] + demand_shift,
                            self.demand_params[1])
        new_supply_params = (self.supply_params[0] + supply_shift,
                            self.supply_params[1])

        shifted_model = SupplyDemandModel(
            demand_params=new_demand_params,
            supply_params=new_supply_params,
            model_type=self.model_type
        )

        new_eq = shifted_model.find_equilibrium()

        return {
            'initial': initial_eq,
            'new': new_eq,
            'changes': {
                'price_change': new_eq['price'] - initial_eq['price'],
                'quantity_change': new_eq['quantity'] - initial_eq['quantity'],
                'surplus_change': new_eq['total_surplus'] - initial_eq['total_surplus']
            }
        }

    def simulate_price_adjustment(self,
                                 initial_price: float,
                                 adjustment_speed: float = 0.1,
                                 periods: int = 50) -> np.ndarray:
        """
        Simulate tatonnement process: price adjustment toward equilibrium.

        WALRASIAN TATONNEMENT:
        P(t+1) = P(t) + λ * [Q_d(P(t)) - Q_s(P(t))]

        If excess demand (Q_d > Q_s): price rises
        If excess supply (Q_d < Q_s): price falls

        This demonstrates the self-correcting nature of markets emphasized
        in Chicago price theory.

        Parameters:
        -----------
        initial_price : float
            Starting price (can be away from equilibrium)
        adjustment_speed : float
            Speed of adjustment (λ), typically 0 < λ < 1
        periods : int
            Number of time periods to simulate

        Returns:
        --------
        Array of prices over time
        """
        prices = np.zeros(periods)
        prices[0] = initial_price

        for t in range(1, periods):
            excess = self.excess_demand(prices[t-1])
            prices[t] = prices[t-1] + adjustment_speed * excess

            # Ensure price stays positive
            prices[t] = max(0.1, prices[t])

        return prices

    def plot_market(self,
                   show_surplus: bool = True,
                   show_adjustment: bool = False,
                   initial_price: Optional[float] = None) -> None:
        """
        Visualize supply, demand, and equilibrium.

        Parameters:
        -----------
        show_surplus : bool
            Shade consumer and producer surplus areas
        show_adjustment : bool
            Show price adjustment dynamics
        initial_price : float
            Starting price for adjustment simulation
        """
        eq = self.find_equilibrium()
        p_star, q_star = eq['price'], eq['quantity']

        # Create price range for plotting
        p_max = p_star * 2
        prices = np.linspace(0.1, p_max, 200)
        q_demand = np.array([self.demand(p) for p in prices])
        q_supply = np.array([self.supply(p) for p in prices])

        fig, axes = plt.subplots(1, 2 if show_adjustment else 1,
                                figsize=(15 if show_adjustment else 8, 6))

        if not show_adjustment:
            axes = [axes]

        # Main supply-demand plot
        ax = axes[0]
        ax.plot(q_demand, prices, 'b-', linewidth=2, label='Demand')
        ax.plot(q_supply, prices, 'r-', linewidth=2, label='Supply')
        ax.plot(q_star, p_star, 'go', markersize=12, label=f'Equilibrium (P*={p_star:.2f}, Q*={q_star:.2f})')

        if show_surplus:
            # Consumer surplus (area above price, below demand)
            q_range = np.linspace(0, q_star, 100)
            p_demand_curve = np.array([
                (self.demand_params[0] - q) / self.demand_params[1]
                for q in q_range
            ])
            ax.fill_between(q_range, p_star, p_demand_curve,
                           alpha=0.3, color='blue',
                           label=f'Consumer Surplus = {eq["consumer_surplus"]:.2f}')

            # Producer surplus (area below price, above supply)
            p_supply_curve = np.array([
                (q - self.supply_params[0]) / self.supply_params[1]
                for q in q_range
            ])
            ax.fill_between(q_range, p_supply_curve, p_star,
                           alpha=0.3, color='red',
                           label=f'Producer Surplus = {eq["producer_surplus"]:.2f}')

        ax.set_xlabel('Quantity', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title('Supply and Demand - Market Equilibrium', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(q_demand[0], q_supply[-1]) * 1.1)
        ax.set_ylim(0, p_max)

        # Price adjustment dynamics
        if show_adjustment and initial_price is not None:
            ax2 = axes[1]
            price_path = self.simulate_price_adjustment(initial_price)
            periods = len(price_path)

            ax2.plot(range(periods), price_path, 'b-', linewidth=2, label='Price Path')
            ax2.axhline(y=p_star, color='g', linestyle='--', linewidth=2, label=f'Equilibrium P* = {p_star:.2f}')
            ax2.set_xlabel('Time Period', fontsize=12)
            ax2.set_ylabel('Price', fontsize=12)
            ax2.set_title('Tatonnement Process: Price Adjustment', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/supply_demand_plot.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_tax_incidence(self, tax_per_unit: float) -> Dict:
        """
        Analyze the incidence of a per-unit tax.

        CHICAGO INSIGHT ON TAXATION:
        The burden of a tax depends on relative elasticities, NOT on who legally
        pays the tax. The more inelastic side of the market bears more burden.

        Key formula: Consumer burden / Producer burden = Es / |Ed|

        Parameters:
        -----------
        tax_per_unit : float
            Tax amount per unit sold

        Returns:
        --------
        dict with pre-tax and post-tax equilibria, tax incidence
        """
        # Pre-tax equilibrium
        pre_tax = self.find_equilibrium()

        # Post-tax: Supply curve shifts up by tax amount
        # New supply: Q_s = c + d*(P - t) = (c - d*t) + d*P
        tax_supply_params = (
            self.supply_params[0] - self.supply_params[1] * tax_per_unit,
            self.supply_params[1]
        )

        tax_model = SupplyDemandModel(
            demand_params=self.demand_params,
            supply_params=tax_supply_params,
            model_type=self.model_type
        )

        post_tax = tax_model.find_equilibrium()

        # Calculate incidence
        price_increase = post_tax['price'] - pre_tax['price']
        consumer_burden = price_increase
        producer_burden = tax_per_unit - price_increase

        # Elasticities at pre-tax equilibrium
        elasticities = self.calculate_elasticities(pre_tax['price'])

        return {
            'pre_tax': pre_tax,
            'post_tax': post_tax,
            'tax_per_unit': tax_per_unit,
            'consumer_price': post_tax['price'],
            'producer_price': post_tax['price'] - tax_per_unit,
            'consumer_burden': consumer_burden,
            'producer_burden': producer_burden,
            'consumer_burden_pct': (consumer_burden / tax_per_unit) * 100,
            'producer_burden_pct': (producer_burden / tax_per_unit) * 100,
            'deadweight_loss': pre_tax['total_surplus'] - post_tax['total_surplus'] -
                              (tax_per_unit * post_tax['quantity']),
            'elasticities': elasticities,
            'quantity_change': post_tax['quantity'] - pre_tax['quantity']
        }


if __name__ == '__main__':
    print("=" * 80)
    print("SUPPLY AND DEMAND MODEL - CHICAGO PRICE THEORY")
    print("=" * 80)

    # Create model
    model = SupplyDemandModel()

    # Find equilibrium
    print("\n1. MARKET EQUILIBRIUM")
    print("-" * 80)
    eq = model.find_equilibrium()
    print(f"Equilibrium Price: ${eq['price']:.2f}")
    print(f"Equilibrium Quantity: {eq['quantity']:.2f} units")
    print(f"Consumer Surplus: ${eq['consumer_surplus']:.2f}")
    print(f"Producer Surplus: ${eq['producer_surplus']:.2f}")
    print(f"Total Surplus (Social Welfare): ${eq['total_surplus']:.2f}")

    # Elasticities
    print("\n2. PRICE ELASTICITIES AT EQUILIBRIUM")
    print("-" * 80)
    elast = model.calculate_elasticities(eq['price'])
    print(f"Demand Elasticity: {elast['demand_elasticity']:.2f}")
    print(f"Supply Elasticity: {elast['supply_elasticity']:.2f}")
    print("\nInterpretation:")
    print(f"- A 1% increase in price → {abs(elast['demand_elasticity']):.2f}% decrease in quantity demanded")
    print(f"- A 1% increase in price → {elast['supply_elasticity']:.2f}% increase in quantity supplied")

    # Comparative statics
    print("\n3. COMPARATIVE STATICS: DEMAND INCREASE")
    print("-" * 80)
    print("Scenario: Demand increases by 20 units (e.g., population growth)")
    comp_stat = model.comparative_statics(demand_shift=20)
    print(f"Initial: P = ${comp_stat['initial']['price']:.2f}, Q = {comp_stat['initial']['quantity']:.2f}")
    print(f"New: P = ${comp_stat['new']['price']:.2f}, Q = {comp_stat['new']['quantity']:.2f}")
    print(f"Changes: ΔP = ${comp_stat['changes']['price_change']:.2f}, ΔQ = {comp_stat['changes']['quantity_change']:.2f}")
    print("\nChicago Prediction: Demand increase → Both P and Q rise ✓")

    # Tax incidence
    print("\n4. TAX INCIDENCE ANALYSIS")
    print("-" * 80)
    print("Scenario: $5 per-unit tax imposed")
    tax_analysis = model.analyze_tax_incidence(tax_per_unit=5)
    print(f"Consumer burden: ${tax_analysis['consumer_burden']:.2f} ({tax_analysis['consumer_burden_pct']:.1f}%)")
    print(f"Producer burden: ${tax_analysis['producer_burden']:.2f} ({tax_analysis['producer_burden_pct']:.1f}%)")
    print(f"Deadweight loss: ${tax_analysis['deadweight_loss']:.2f}")
    print(f"Quantity reduction: {abs(tax_analysis['quantity_change']):.2f} units")
    print("\nChicago Insight: Tax burden split depends on relative elasticities,")
    print("not on who legally pays the tax!")

    # Plot
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 80)
    model.plot_market(show_surplus=True, show_adjustment=True, initial_price=15)
    print("Plot saved to: chicago_price_theory/supply_demand_plot.png")
