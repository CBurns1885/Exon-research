"""
Producer Theory Model - Chicago Price Theory

THEORETICAL FOUNDATION:
Producer theory models firms as profit-maximizing agents subject to technological
constraints. This is the foundation for supply curves and demonstrates how costs
determine production decisions.

KEY CONCEPTS:
1. Production Function: Q = f(K, L) - technology constraint
2. Cost Functions: TC(Q), AC(Q), MC(Q)
3. Profit Maximization: Choose Q where MR = MC
4. Cost Minimization: Choose inputs where MPL/w = MPK/r
5. Short Run vs Long Run: Fixed vs variable inputs

CHICAGO EMPHASIS:
- Marginal analysis: Decisions at the margin
- Supply curves derived from profit maximization
- Entry/exit driven by profits → zero economic profit in long run
- Efficiency: Competitive firms produce where P = MC
- Empirical content: Cost curves predict firm behavior

CLASSICAL RESULTS:
- Firm supply: P = MC (in competitive markets)
- Shutdown rule: P < AVC in short run
- Exit rule: P < AC in long run
- Long-run competitive equilibrium: P = min(AC)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from typing import Tuple, Dict, Callable, Optional


class ProducerModel:
    """
    Producer Profit Maximization Model

    Supports multiple production function types:
    - Cobb-Douglas: Q = A * L^α * K^β
    - CES: Q = A * (αL^ρ + (1-α)K^ρ)^(1/ρ)
    - Linear: Q = aL + bK
    - Leontief: Q = min(L/a, K/b)
    """

    def __init__(self,
                 production_type: str = 'cobb_douglas',
                 production_params: Tuple[float, ...] = (1.0, 0.5, 0.5),
                 output_price: float = 10.0,
                 input_prices: Tuple[float, float] = (5.0, 8.0),
                 fixed_cost: float = 100.0):
        """
        Initialize producer model.

        Parameters:
        -----------
        production_type : str
            Type of production function
        production_params : tuple
            - Cobb-Douglas: (A, α, β)
            - CES: (A, α, ρ)
            - Linear: (a, b)
            - Leontief: (a, b)
        output_price : float
            Price of output (P)
        input_prices : tuple
            (wage, rental rate) = (w, r)
        fixed_cost : float
            Fixed cost in short run (FC)
        """
        self.production_type = production_type
        self.production_params = production_params
        self.output_price = output_price
        self.wage, self.rental_rate = input_prices
        self.fixed_cost = fixed_cost

    def production_function(self, L: float, K: float) -> float:
        """
        Production function Q = f(L, K).

        CHICAGO INSIGHT:
        Technology determines what's feasible. Prices determine what's optimal.
        """
        if self.production_type == 'cobb_douglas':
            A, alpha, beta = self.production_params
            # Q = A * L^α * K^β
            # Returns to scale: α + β > 1 (increasing), = 1 (constant), < 1 (decreasing)
            return A * (L ** alpha) * (K ** beta)

        elif self.production_type == 'ces':
            A, alpha, rho = self.production_params
            # Q = A * (αL^ρ + (1-α)K^ρ)^(1/ρ)
            # σ = 1/(1-ρ) is elasticity of substitution
            return A * (alpha * L**rho + (1 - alpha) * K**rho) ** (1/rho)

        elif self.production_type == 'linear':
            a, b = self.production_params
            # Q = aL + bK (perfect substitutes)
            return a * L + b * K

        elif self.production_type == 'leontief':
            a, b = self.production_params
            # Q = min(L/a, K/b) (perfect complements)
            return min(L/a, K/b)

        else:
            raise ValueError(f"Unknown production type: {self.production_type}")

    def marginal_product(self, L: float, K: float) -> Tuple[float, float]:
        """
        Marginal products: MPL = ∂Q/∂L, MPK = ∂Q/∂K

        CHICAGO PRINCIPLE:
        Marginal product determines the value of an additional unit of input.
        Profit max requires: w = P * MPL and r = P * MPK
        """
        eps = 1e-6

        # Numerical derivatives
        MPL = (self.production_function(L + eps, K) -
               self.production_function(L, K)) / eps
        MPK = (self.production_function(L, K + eps) -
               self.production_function(L, K)) / eps

        return MPL, MPK

    def total_cost(self, L: float, K: float) -> float:
        """
        Total cost: TC = wL + rK + FC

        Short run: K is fixed, TC = wL + rK_fixed + FC = VC(L) + FC
        Long run: Both L and K variable, FC = 0
        """
        variable_cost = self.wage * L + self.rental_rate * K
        return variable_cost + self.fixed_cost

    def profit(self, L: float, K: float) -> float:
        """
        Profit: π = P*Q - TC = P*f(L,K) - wL - rK - FC

        CHICAGO OBJECTIVE:
        Firms maximize profit. This generates supply behavior and
        factor demand.
        """
        revenue = self.output_price * self.production_function(L, K)
        cost = self.total_cost(L, K)
        return revenue - cost

    def maximize_profit_lr(self) -> Dict:
        """
        Long-run profit maximization: Choose L and K to maximize π.

        FOCs:
        ∂π/∂L = P * MPL - w = 0  →  P * MPL = w
        ∂π/∂K = P * MPK - r = 0  →  P * MPK = r

        Equivalently: MPL/w = MPK/r = 1/P (cost minimization)

        CHICAGO INSIGHT:
        Firm equates value of marginal product to input price for each input.
        This determines factor demands.

        Returns:
        --------
        dict with optimal L*, K*, Q*, profit, costs
        """
        # Objective: minimize -π
        def objective(inputs):
            L, K = inputs
            if L < 0 or K < 0:
                return 1e10
            return -self.profit(L, K)

        # Initial guess
        x0 = np.array([10.0, 10.0])

        # Optimize
        result = minimize(objective, x0, method='Nelder-Mead',
                         bounds=[(0, None), (0, None)])

        L_star, K_star = result.x
        Q_star = self.production_function(L_star, K_star)
        profit_star = self.profit(L_star, K_star)
        MPL, MPK = self.marginal_product(L_star, K_star)

        # Cost measures
        TC = self.total_cost(L_star, K_star)
        AC = TC / Q_star if Q_star > 0 else float('inf')
        MC = self.wage / MPL if MPL > 0 else float('inf')  # MC = w/MPL

        return {
            'L': L_star,
            'K': K_star,
            'Q': Q_star,
            'profit': profit_star,
            'total_cost': TC,
            'average_cost': AC,
            'marginal_cost': MC,
            'MPL': MPL,
            'MPK': MPK,
            'VMP_L': self.output_price * MPL,  # Value of marginal product
            'VMP_K': self.output_price * MPK,
            'check_FOC_L': np.isclose(self.output_price * MPL, self.wage, rtol=0.1),
            'check_FOC_K': np.isclose(self.output_price * MPK, self.rental_rate, rtol=0.1)
        }

    def cost_minimization(self, target_Q: float, K_fixed: Optional[float] = None) -> Dict:
        """
        Cost minimization for given output level.

        Long run: min wL + rK s.t. f(L,K) = Q
        Short run: min wL + rK_fixed s.t. f(L,K_fixed) = Q

        LAGRANGIAN:
        L = wL + rK + λ(Q - f(L,K))

        FOCs:
        w = λ * MPL
        r = λ * MPK
        →  MPL/w = MPK/r  (isocost tangent to isoquant)

        Parameters:
        -----------
        target_Q : float
            Target output level
        K_fixed : float
            If provided, short-run optimization with fixed K

        Returns:
        --------
        dict with cost-minimizing inputs and costs
        """
        if K_fixed is not None:
            # SHORT RUN: K is fixed, choose L to produce Q
            def find_L(L):
                return self.production_function(L, K_fixed) - target_Q

            try:
                L_star = fsolve(find_L, x0=10.0)[0]
                if L_star < 0:
                    return {'feasible': False}
                K_star = K_fixed
            except:
                return {'feasible': False}

        else:
            # LONG RUN: Choose both L and K
            def objective(inputs):
                L, K = inputs
                if L < 0 or K < 0:
                    return 1e10
                return self.wage * L + self.rental_rate * K

            def constraint(inputs):
                L, K = inputs
                return self.production_function(L, K) - target_Q

            x0 = np.array([10.0, 10.0])

            result = minimize(objective, x0, method='SLSQP',
                            constraints={'type': 'eq', 'fun': constraint},
                            bounds=[(0, None), (0, None)])

            if not result.success:
                return {'feasible': False}

            L_star, K_star = result.x

        # Calculate costs
        TC = self.total_cost(L_star, K_star)
        VC = self.wage * L_star + self.rental_rate * K_star
        AC = TC / target_Q
        AVC = VC / target_Q
        MPL, MPK = self.marginal_product(L_star, K_star)

        return {
            'feasible': True,
            'L': L_star,
            'K': K_star,
            'Q': target_Q,
            'total_cost': TC,
            'variable_cost': VC,
            'fixed_cost': self.fixed_cost,
            'average_cost': AC,
            'average_variable_cost': AVC,
            'MPL': MPL,
            'MPK': MPK,
            'MRTS': MPL / MPK if MPK > 0 else float('inf'),  # Marginal rate of technical substitution
            'input_price_ratio': self.wage / self.rental_rate,
            'check_optimality': np.isclose(MPL/self.wage, MPK/self.rental_rate, rtol=0.1) if K_fixed is None else True
        }

    def derive_cost_curves(self,
                          q_range: Tuple[float, float] = (1, 100),
                          n_points: int = 50,
                          short_run_K: Optional[float] = None) -> Dict:
        """
        Derive cost curves: TC(Q), AC(Q), MC(Q), AVC(Q).

        CHICAGO METHODOLOGY:
        Cost curves come from solving cost minimization at each output level.
        These generate testable predictions about firm behavior.

        Parameters:
        -----------
        q_range : tuple
            Range of output levels
        n_points : int
            Number of points
        short_run_K : float
            If provided, derive short-run cost curves with K fixed

        Returns:
        --------
        dict with arrays of Q, TC, AC, MC, AVC
        """
        quantities = np.linspace(q_range[0], q_range[1], n_points)
        TC = np.zeros(n_points)
        VC = np.zeros(n_points)

        for i, q in enumerate(quantities):
            result = self.cost_minimization(q, K_fixed=short_run_K)
            if result['feasible']:
                TC[i] = result['total_cost']
                VC[i] = result['variable_cost']
            else:
                TC[i] = np.nan
                VC[i] = np.nan

        # Calculate derived measures
        AC = TC / quantities
        AVC = VC / quantities

        # MC = dTC/dQ (numerical derivative)
        MC = np.gradient(TC, quantities)

        return {
            'Q': quantities,
            'TC': TC,
            'VC': VC,
            'FC': np.ones(n_points) * self.fixed_cost,
            'AC': AC,
            'AVC': AVC,
            'MC': MC
        }

    def derive_supply_curve(self,
                           price_range: Tuple[float, float] = (1, 50),
                           n_points: int = 50,
                           short_run_K: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive firm's supply curve: Q(P).

        SUPPLY RULE:
        - Firm produces where P = MC (if profitable)
        - Short run: Shutdown if P < AVC
        - Long run: Exit if P < AC

        This is the foundation of market supply in Chicago price theory.

        Parameters:
        -----------
        price_range : tuple
            Range of output prices
        n_points : int
            Number of price points
        short_run_K : float
            If provided, short-run supply with K fixed

        Returns:
        --------
        (prices, quantities) arrays
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        quantities = np.zeros(n_points)

        for i, p in enumerate(prices):
            # Temporary change price
            original_price = self.output_price
            self.output_price = p

            if short_run_K is not None:
                # Short run: maximize profit with fixed K
                def objective_sr(L):
                    if L < 0:
                        return 1e10
                    return -self.profit(L, short_run_K)

                result = minimize(objective_sr, x0=10.0, bounds=[(0, None)])
                L_star = result.x[0]
                Q_star = self.production_function(L_star, short_run_K)

                # Check shutdown condition
                VC = self.wage * L_star + self.rental_rate * short_run_K
                AVC = VC / Q_star if Q_star > 0 else float('inf')
                if p < AVC:
                    Q_star = 0  # Shutdown

            else:
                # Long run: maximize profit choosing both L and K
                result = self.maximize_profit_lr()
                Q_star = result['Q']

                # Check exit condition
                if result['profit'] < 0:
                    Q_star = 0  # Exit

            quantities[i] = Q_star

            # Restore price
            self.output_price = original_price

        return prices, quantities

    def plot_cost_curves(self, short_run_K: Optional[float] = None) -> None:
        """
        Visualize cost curves.

        Parameters:
        -----------
        short_run_K : float
            If provided, show short-run curves with K fixed
        """
        cost_data = self.derive_cost_curves(q_range=(1, 50), n_points=100,
                                           short_run_K=short_run_K)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Total cost curves
        ax1.plot(cost_data['Q'], cost_data['TC'], 'b-', linewidth=2, label='Total Cost (TC)')
        ax1.plot(cost_data['Q'], cost_data['VC'], 'g-', linewidth=2, label='Variable Cost (VC)')
        ax1.plot(cost_data['Q'], cost_data['FC'], 'r--', linewidth=2, label='Fixed Cost (FC)')
        ax1.set_xlabel('Quantity (Q)', fontsize=12)
        ax1.set_ylabel('Cost ($)', fontsize=12)
        ax1.set_title('Total Cost Curves', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Average and marginal cost curves
        # Filter out infinities and NaNs for better visualization
        valid_idx = np.isfinite(cost_data['AC']) & np.isfinite(cost_data['MC'])
        Q_valid = cost_data['Q'][valid_idx]
        AC_valid = cost_data['AC'][valid_idx]
        AVC_valid = cost_data['AVC'][valid_idx]
        MC_valid = cost_data['MC'][valid_idx]

        # Cap extremely high values for visualization
        AC_valid = np.minimum(AC_valid, 50)
        AVC_valid = np.minimum(AVC_valid, 50)
        MC_valid = np.minimum(MC_valid, 50)

        ax2.plot(Q_valid, AC_valid, 'b-', linewidth=2, label='Average Cost (AC)')
        ax2.plot(Q_valid, AVC_valid, 'g-', linewidth=2, label='Average Variable Cost (AVC)')
        ax2.plot(Q_valid, MC_valid, 'r-', linewidth=2, label='Marginal Cost (MC)')

        # Show efficient scale (min AC in long run)
        if short_run_K is None:
            min_ac_idx = np.argmin(AC_valid)
            ax2.plot(Q_valid[min_ac_idx], AC_valid[min_ac_idx], 'ko', markersize=10,
                    label=f'Efficient Scale: Q={Q_valid[min_ac_idx]:.1f}, AC={AC_valid[min_ac_idx]:.2f}')

        ax2.set_xlabel('Quantity (Q)', fontsize=12)
        ax2.set_ylabel('Cost per Unit ($)', fontsize=12)
        title = 'Short-Run Cost Curves' if short_run_K else 'Long-Run Cost Curves'
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 30)

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/producer_cost_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_supply_curve(self, short_run_K: Optional[float] = None) -> None:
        """Plot firm supply curve."""
        prices, quantities = self.derive_supply_curve(price_range=(1, 30),
                                                      short_run_K=short_run_K)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(quantities, prices, 'b-', linewidth=2,
               label='Supply Curve (P = MC)')
        ax.set_xlabel('Quantity Supplied (Q)', fontsize=12)
        ax.set_ylabel('Price (P)', fontsize=12)
        title = 'Short-Run Supply Curve' if short_run_K else 'Long-Run Supply Curve'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark current equilibrium
        opt = self.maximize_profit_lr()
        ax.plot(opt['Q'], self.output_price, 'ro', markersize=10,
               label=f'Current: P=${self.output_price:.2f}, Q={opt["Q"]:.2f}, π=${opt["profit"]:.2f}')
        ax.legend()

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/producer_supply_curve.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    print("=" * 80)
    print("PRODUCER THEORY MODEL - CHICAGO PRICE THEORY")
    print("=" * 80)

    # Create producer
    producer = ProducerModel(
        production_type='cobb_douglas',
        production_params=(1.0, 0.6, 0.4),  # Q = L^0.6 * K^0.4 (constant returns)
        output_price=10.0,
        input_prices=(5.0, 8.0),  # w = 5, r = 8
        fixed_cost=100.0
    )

    # Long-run profit maximization
    print("\n1. LONG-RUN PROFIT MAXIMIZATION")
    print("-" * 80)
    lr_result = producer.maximize_profit_lr()
    print(f"Output price: P = ${producer.output_price}")
    print(f"Input prices: w = ${producer.wage}, r = ${producer.rental_rate}")
    print(f"\nOptimal Input Choice:")
    print(f"  Labor: L* = {lr_result['L']:.2f}")
    print(f"  Capital: K* = {lr_result['K']:.2f}")
    print(f"  Output: Q* = {lr_result['Q']:.2f}")
    print(f"\nProfit and Costs:")
    print(f"  Total Revenue: ${producer.output_price * lr_result['Q']:.2f}")
    print(f"  Total Cost: ${lr_result['total_cost']:.2f}")
    print(f"  Profit: π* = ${lr_result['profit']:.2f}")
    print(f"\nMarginal Analysis:")
    print(f"  MPL = {lr_result['MPL']:.3f}, VMP_L = ${lr_result['VMP_L']:.2f}, w = ${producer.wage:.2f}")
    print(f"  MPK = {lr_result['MPK']:.3f}, VMP_K = ${lr_result['VMP_K']:.2f}, r = ${producer.rental_rate:.2f}")
    print(f"  P = ${producer.output_price:.2f}, MC = ${lr_result['marginal_cost']:.2f}")
    print(f"\nOptimality Checks:")
    print(f"  P * MPL = w? {lr_result['check_FOC_L']}")
    print(f"  P * MPK = r? {lr_result['check_FOC_K']}")

    # Cost minimization
    print("\n2. COST MINIMIZATION")
    print("-" * 80)
    target_output = 20.0
    cm_result = producer.cost_minimization(target_Q=target_output)
    print(f"Target output: Q = {target_output}")
    print(f"Cost-minimizing inputs: L = {cm_result['L']:.2f}, K = {cm_result['K']:.2f}")
    print(f"Total cost: TC = ${cm_result['total_cost']:.2f}")
    print(f"Average cost: AC = ${cm_result['average_cost']:.2f}")
    print(f"MRTS = MPL/MPK = {cm_result['MRTS']:.3f}")
    print(f"w/r = {cm_result['input_price_ratio']:.3f}")
    print(f"Optimality (MRTS = w/r)? {cm_result['check_optimality']}")

    # Short run vs long run
    print("\n3. SHORT RUN VS LONG RUN")
    print("-" * 80)
    K_fixed = lr_result['K']
    print(f"Fix capital at K = {K_fixed:.2f}")

    sr_result = producer.cost_minimization(target_Q=target_output, K_fixed=K_fixed)
    lr_result_target = producer.cost_minimization(target_Q=target_output)

    print(f"\nProducing Q = {target_output}:")
    print(f"Short run: L = {sr_result['L']:.2f}, K = {K_fixed:.2f}, TC = ${sr_result['total_cost']:.2f}")
    print(f"Long run: L = {lr_result_target['L']:.2f}, K = {lr_result_target['K']:.2f}, TC = ${lr_result_target['total_cost']:.2f}")
    print(f"\nLong-run cost ≤ Short-run cost: ${lr_result_target['total_cost']:.2f} ≤ ${sr_result['total_cost']:.2f}")

    # Supply curve analysis
    print("\n4. SUPPLY CURVE ANALYSIS")
    print("-" * 80)
    prices, quantities = producer.derive_supply_curve(price_range=(5, 20))

    # Find price elasticity of supply
    idx = len(prices) // 2
    dQ = quantities[idx+1] - quantities[idx-1]
    dP = prices[idx+1] - prices[idx-1]
    if quantities[idx] > 0:
        elasticity = (dQ/dP) * (prices[idx]/quantities[idx])
        print(f"Price elasticity of supply at P=${prices[idx]:.2f}: {elasticity:.2f}")
        print(f"Interpretation: 1% increase in price → {elasticity:.2f}% increase in quantity supplied")

    # Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 80)
    producer.plot_cost_curves(short_run_K=None)  # Long run
    producer.plot_cost_curves(short_run_K=K_fixed)  # Short run
    producer.plot_supply_curve(short_run_K=None)
    print("Plots saved to chicago_price_theory/")
