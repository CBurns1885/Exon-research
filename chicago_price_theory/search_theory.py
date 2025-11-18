"""
Search Theory Model - Chicago Price Theory (Stigler)

THEORETICAL FOUNDATION:
George Stigler's search theory revolutionized understanding of unemployment and
price dispersion. Key insight: Information is costly, so rational agents search
until expected marginal benefit equals marginal cost.

KEY CONCEPTS:
1. Costly Search: Gathering information takes time/money
2. Reservation Wage: Minimum acceptable wage (optimal stopping rule)
3. Sequential Search: Search one option at a time
4. Optimal Stopping: Continue searching while E[benefit] > cost
5. Duration of Search: How long until acceptable offer found

CHICAGO EMPHASIS:
- Unemployment is partly voluntary (job search)
- Information is valuable but costly to acquire
- Dispersion of prices/wages persists in equilibrium
- No such thing as perfect information in real markets
- Search intensity responds to incentives

CLASSICAL RESULTS:
- Reservation wage increases with: search cost reduction, better distribution of offers
- Higher unemployment benefits → Higher reservation wage → Longer search
- Paradox: Even identical goods have price dispersion (search costs)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from typing import Tuple, Dict, Callable


class SearchModel:
    """
    Job Search Model with Optimal Stopping

    Worker searches sequentially for wage offers from a known distribution.
    Decides when to accept an offer based on reservation wage.
    """

    def __init__(self,
                 offer_mean: float = 50000,
                 offer_std: float = 10000,
                 search_cost: float = 500,
                 discount_rate: float = 0.05,
                 unemployment_benefit: float = 15000):
        """
        Initialize search model.

        Parameters:
        -----------
        offer_mean : float
            Mean of wage offer distribution
        offer_std : float
            Standard deviation of wage offers
        search_cost : float
            Cost per search (time, effort, fees)
        discount_rate : float
            Discount rate (time preference)
        unemployment_benefit : float
            Income while searching (unemployment insurance)
        """
        self.offer_mean = offer_mean
        self.offer_std = offer_std
        self.search_cost = search_cost
        self.discount_rate = discount_rate
        self.unemployment_benefit = unemployment_benefit

        # Wage offers assumed normally distributed
        self.offer_distribution = norm(loc=offer_mean, scale=offer_std)

    def present_value_of_wage(self, wage: float) -> float:
        """
        Present value of accepting wage w forever.

        PV = w/r (perpetuity formula)

        CHICAGO INSIGHT:
        Job acceptance is a permanent decision (in this simple model),
        so compare present values, not just current wages.

        Parameters:
        -----------
        wage : float
            Offered wage (annual)

        Returns:
        --------
        Present value of wage stream
        """
        return wage / self.discount_rate

    def expected_value_of_search(self, reservation_wage: float) -> float:
        """
        Expected value of continuing to search given reservation wage.

        E[V|continue] = -c + ∫[w>w_r] PV(w)*f(w)dw + ∫[w≤w_r] E[V|continue]*f(w)dw

        where:
        - c = search cost
        - w_r = reservation wage
        - f(w) = density of wage offers

        OPTIMAL STOPPING RULE:
        Accept offer w if w ≥ w_r
        Continue searching if w < w_r

        Parameters:
        -----------
        reservation_wage : float
            Reservation wage (threshold for acceptance)

        Returns:
        --------
        Expected value of search
        """
        # Probability of accepting offer (w ≥ w_r)
        prob_accept = 1 - self.offer_distribution.cdf(reservation_wage)

        if prob_accept < 0.001:
            # Reservation wage too high - almost never accept
            return -np.inf

        # Expected wage conditional on accepting
        # E[w | w ≥ w_r] for normal distribution
        # Using truncated normal formula
        alpha = (reservation_wage - self.offer_mean) / self.offer_std
        mills_ratio = norm.pdf(alpha) / (1 - norm.cdf(alpha))
        expected_accepted_wage = self.offer_mean + self.offer_std * mills_ratio

        # Expected value if accept
        ev_accept = self.present_value_of_wage(expected_accepted_wage)

        # Value equation (Bellman equation):
        # V = -c + p*E[PV(w)|accept] + (1-p)*V
        # Solving for V:
        # V = [-c + p*E[PV(w)|accept]] / p
        ev_search = (-self.search_cost + prob_accept * ev_accept) / prob_accept

        return ev_search

    def find_reservation_wage(self) -> Dict:
        """
        Find optimal reservation wage.

        OPTIMALITY CONDITION:
        Reservation wage w_r satisfies:
        PV(w_r) = E[V|continue search]

        Indifferent between accepting w_r and continuing to search.

        CHICAGO METHOD:
        This is an optimal stopping problem. Reservation wage balances:
        - Cost: Forgone opportunity (current offer) + search costs
        - Benefit: Possibility of better offer in future

        Returns:
        --------
        dict with reservation wage, search statistics
        """
        # Solve for w_r where worker is indifferent
        # PV(w_r) = E[V|continue]
        # w_r/r = E[V(w_r)]

        # Numerical solution
        def indifference_condition(w_r):
            """Value of accepting w_r minus value of continuing search."""
            pv_accept = self.present_value_of_wage(w_r)
            ev_search = self.expected_value_of_search(w_r)
            return pv_accept - ev_search

        # Search over reasonable range
        w_min = max(0, self.offer_mean - 3 * self.offer_std)
        w_max = self.offer_mean + 3 * self.offer_std
        w_range = np.linspace(w_min, w_max, 1000)

        # Find where indifference condition = 0
        indiff_values = [indifference_condition(w) for w in w_range]

        # Find crossing point
        sign_changes = np.where(np.diff(np.sign(indiff_values)))[0]

        if len(sign_changes) == 0:
            # No crossing - corner solution
            w_reservation = w_min if indiff_values[0] < 0 else w_max
        else:
            idx = sign_changes[0]
            w_reservation = w_range[idx]

        # Calculate implied statistics
        prob_accept = 1 - self.offer_distribution.cdf(w_reservation)

        if prob_accept > 0:
            expected_searches = 1 / prob_accept  # Geometric distribution
            expected_duration = expected_searches  # In periods
        else:
            expected_searches = float('inf')
            expected_duration = float('inf')

        # Expected accepted wage
        alpha = (w_reservation - self.offer_mean) / self.offer_std
        if prob_accept > 0:
            mills_ratio = norm.pdf(alpha) / (1 - norm.cdf(alpha))
            expected_accepted_wage = self.offer_mean + self.offer_std * mills_ratio
        else:
            expected_accepted_wage = self.offer_mean

        return {
            'reservation_wage': w_reservation,
            'probability_accept': prob_accept,
            'expected_searches': expected_searches,
            'expected_duration': expected_duration,
            'expected_accepted_wage': expected_accepted_wage,
            'value_of_search': self.expected_value_of_search(w_reservation)
        }

    def comparative_statics_search_cost(self,
                                       cost_range: Tuple[float, float] = (100, 2000),
                                       n_points: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        How reservation wage changes with search cost.

        CHICAGO PREDICTION:
        Higher search cost → Lower reservation wage → Accept offers sooner
        (Search is more expensive, so less picky)

        Returns:
        --------
        (costs, reservation_wages) arrays
        """
        costs = np.linspace(cost_range[0], cost_range[1], n_points)
        res_wages = np.zeros(n_points)

        original_cost = self.search_cost

        for i, c in enumerate(costs):
            self.search_cost = c
            result = self.find_reservation_wage()
            res_wages[i] = result['reservation_wage']

        self.search_cost = original_cost

        return costs, res_wages

    def comparative_statics_ui_benefits(self,
                                       benefit_range: Tuple[float, float] = (0, 30000),
                                       n_points: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        How reservation wage changes with unemployment insurance.

        CHICAGO POLICY INSIGHT:
        Higher UI benefits → Higher reservation wage → Longer search
        This explains why generous UI may increase unemployment duration.

        Returns:
        --------
        (benefits, reservation_wages) arrays
        """
        benefits = np.linspace(benefit_range[0], benefit_range[1], n_points)
        res_wages = np.zeros(n_points)

        original_benefit = self.unemployment_benefit

        for i, b in enumerate(benefits):
            self.unemployment_benefit = b
            result = self.find_reservation_wage()
            res_wages[i] = result['reservation_wage']

        self.unemployment_benefit = original_benefit

        return benefits, res_wages

    def simulate_search_process(self, n_searchers: int = 1000) -> Dict:
        """
        Simulate search process for multiple individuals.

        MONTE CARLO SIMULATION:
        Each searcher draws offers until finding one ≥ reservation wage.
        Tracks: number of searches, accepted wages, search duration.

        Returns:
        --------
        dict with simulation results
        """
        result = self.find_reservation_wage()
        w_r = result['reservation_wage']

        accepted_wages = []
        num_searches = []

        for _ in range(n_searchers):
            searches = 0
            accepted = False

            while not accepted:
                searches += 1
                # Draw random wage offer
                offer = self.offer_distribution.rvs()

                if offer >= w_r:
                    accepted_wages.append(offer)
                    num_searches.append(searches)
                    accepted = True

                # Safety: max 1000 searches
                if searches > 1000:
                    accepted_wages.append(w_r)
                    num_searches.append(searches)
                    accepted = True

        return {
            'accepted_wages': np.array(accepted_wages),
            'num_searches': np.array(num_searches),
            'mean_accepted_wage': np.mean(accepted_wages),
            'mean_searches': np.mean(num_searches),
            'median_searches': np.median(num_searches)
        }

    def plot_search_results(self) -> None:
        """Visualize search theory results."""
        result = self.find_reservation_wage()
        w_r = result['reservation_wage']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Wage offer distribution and reservation wage
        w_range = np.linspace(self.offer_mean - 3*self.offer_std,
                            self.offer_mean + 3*self.offer_std, 200)
        pdf = self.offer_distribution.pdf(w_range)

        ax1.plot(w_range, pdf, 'b-', linewidth=2, label='Wage Offer Distribution')
        ax1.axvline(x=w_r, color='red', linestyle='--', linewidth=2,
                   label=f'Reservation Wage = ${w_r:,.0f}')
        ax1.axvline(x=self.offer_mean, color='green', linestyle=':', linewidth=2,
                   label=f'Mean Offer = ${self.offer_mean:,.0f}')

        # Shade acceptance region
        accept_region = w_range >= w_r
        ax1.fill_between(w_range, 0, pdf, where=accept_region, alpha=0.3,
                        color='green', label=f'Accept (p={result["probability_accept"]:.2f})')
        ax1.fill_between(w_range, 0, pdf, where=~accept_region, alpha=0.3,
                        color='red', label='Reject & Continue')

        ax1.set_xlabel('Wage Offer ($)', fontsize=11)
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_title('Optimal Stopping Rule', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Search cost effect
        costs, res_wages = self.comparative_statics_search_cost()
        ax2.plot(costs, res_wages, 'b-', linewidth=2)
        ax2.plot(self.search_cost, w_r, 'ro', markersize=10, label='Current')
        ax2.set_xlabel('Search Cost ($)', fontsize=11)
        ax2.set_ylabel('Reservation Wage ($)', fontsize=11)
        ax2.set_title('Effect of Search Cost', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3. UI benefits effect
        benefits, res_wages_ui = self.comparative_statics_ui_benefits()
        ax3.plot(benefits, res_wages_ui, 'g-', linewidth=2)
        ax3.plot(self.unemployment_benefit, w_r, 'ro', markersize=10, label='Current')
        ax3.set_xlabel('Unemployment Benefits ($)', fontsize=11)
        ax3.set_ylabel('Reservation Wage ($)', fontsize=11)
        ax3.set_title('Effect of Unemployment Insurance', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # 4. Simulation results
        sim = self.simulate_search_process(n_searchers=1000)
        ax4.hist(sim['num_searches'], bins=30, density=True, alpha=0.7,
                color='blue', edgecolor='black')
        ax4.axvline(x=sim['mean_searches'], color='red', linestyle='--',
                   linewidth=2, label=f'Mean = {sim["mean_searches"]:.1f} searches')
        ax4.axvline(x=result['expected_searches'], color='green', linestyle=':',
                   linewidth=2, label=f'Predicted = {result["expected_searches"]:.1f}')
        ax4.set_xlabel('Number of Searches', fontsize=11)
        ax4.set_ylabel('Probability Density', fontsize=11)
        ax4.set_title('Search Duration Distribution (Simulated)', fontsize=12, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/Exon-research/chicago_price_theory/search_theory.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    print("=" * 80)
    print("SEARCH THEORY MODEL - CHICAGO PRICE THEORY (STIGLER)")
    print("=" * 80)

    model = SearchModel(
        offer_mean=50000,
        offer_std=10000,
        search_cost=500,
        discount_rate=0.05,
        unemployment_benefit=15000
    )

    # Find optimal reservation wage
    print("\n1. OPTIMAL SEARCH STRATEGY")
    print("-" * 80)
    result = model.find_reservation_wage()
    print(f"Wage offer distribution: N(${model.offer_mean:,.0f}, ${model.offer_std:,.0f})")
    print(f"Search cost per search: ${model.search_cost:,.0f}")
    print(f"Discount rate: {model.discount_rate*100:.1f}%")
    print(f"\nOptimal Strategy:")
    print(f"  Reservation wage: ${result['reservation_wage']:,.0f}")
    print(f"  Accept if offer ≥ ${result['reservation_wage']:,.0f}")
    print(f"  Reject if offer < ${result['reservation_wage']:,.0f}")
    print(f"\nImplied Statistics:")
    print(f"  Probability of accepting any offer: {result['probability_accept']:.2%}")
    print(f"  Expected number of searches: {result['expected_searches']:.1f}")
    print(f"  Expected duration: {result['expected_duration']:.1f} periods")
    print(f"  Expected accepted wage: ${result['expected_accepted_wage']:,.0f}")

    # Comparative statics: search cost
    print("\n2. EFFECT OF SEARCH COST")
    print("-" * 80)
    low_cost_model = SearchModel(search_cost=100)
    high_cost_model = SearchModel(search_cost=1500)

    low_result = low_cost_model.find_reservation_wage()
    high_result = high_cost_model.find_reservation_wage()

    print(f"Low search cost ($100):")
    print(f"  Reservation wage: ${low_result['reservation_wage']:,.0f}")
    print(f"  Expected searches: {low_result['expected_searches']:.1f}")

    print(f"\nHigh search cost ($1,500):")
    print(f"  Reservation wage: ${high_result['reservation_wage']:,.0f}")
    print(f"  Expected searches: {high_result['expected_searches']:.1f}")

    print(f"\nChicago Insight: Higher search costs → Lower reservation wage")
    print(f"(Workers are less picky when search is expensive)")

    # Comparative statics: UI benefits
    print("\n3. EFFECT OF UNEMPLOYMENT INSURANCE")
    print("-" * 80)
    no_ui_model = SearchModel(unemployment_benefit=0)
    high_ui_model = SearchModel(unemployment_benefit=30000)

    no_ui_result = no_ui_model.find_reservation_wage()
    high_ui_result = high_ui_model.find_reservation_wage()

    print(f"No UI benefits:")
    print(f"  Reservation wage: ${no_ui_result['reservation_wage']:,.0f}")
    print(f"  Expected duration: {no_ui_result['expected_duration']:.1f} periods")

    print(f"\nGenerous UI ($30,000/year):")
    print(f"  Reservation wage: ${high_ui_result['reservation_wage']:,.0f}")
    print(f"  Expected duration: {high_ui_result['expected_duration']:.1f} periods")

    print(f"\nChicago Policy Insight: Higher UI → Higher reservation wage → Longer search")
    print(f"Trade-off: Better job matches vs. longer unemployment duration")

    # Simulation
    print("\n4. MONTE CARLO SIMULATION")
    print("-" * 80)
    print("Simulating 1,000 job searchers...")
    sim = model.simulate_search_process(n_searchers=1000)

    print(f"\nSimulation Results:")
    print(f"  Mean accepted wage: ${sim['mean_accepted_wage']:,.0f}")
    print(f"  Mean searches: {sim['mean_searches']:.1f}")
    print(f"  Median searches: {sim['median_searches']:.0f}")

    print(f"\nComparison to Theory:")
    print(f"  Predicted mean searches: {result['expected_searches']:.1f}")
    print(f"  Simulated mean searches: {sim['mean_searches']:.1f}")
    print(f"  Difference: {abs(result['expected_searches'] - sim['mean_searches']):.1f}")

    # Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 80)
    model.plot_search_results()
    print("Plot saved to chicago_price_theory/search_theory.png")
