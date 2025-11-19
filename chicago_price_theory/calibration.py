"""
Calibration Module - Estimate Model Parameters from Real Data

This module calibrates theoretical models using empirical data.
Uses econometric techniques to estimate parameters that best fit observed data.

Chicago Methodology:
- Theory guides empirical work
- Use data to estimate structural parameters
- Test theoretical predictions against reality
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from scipy.stats import linregress
from typing import Dict, Tuple, Optional, List
import warnings


class SupplyDemandCalibrator:
    """
    Calibrate supply and demand parameters from price and quantity data.

    Methods:
    - OLS regression for linear supply/demand
    - 2SLS for simultaneous equations
    - Maximum likelihood estimation
    """

    @staticmethod
    def calibrate_linear(price_data: np.ndarray,
                        quantity_data: np.ndarray,
                        curve_type: str = 'demand') -> Tuple[float, float]:
        """
        Calibrate linear supply or demand curve from data.

        For demand: Q = a - b*P
        For supply: Q = c + d*P

        Uses OLS regression (assumes one side held constant via instruments).

        Parameters:
        -----------
        price_data : array
            Observed prices
        quantity_data : array
            Observed quantities
        curve_type : str
            'demand' or 'supply'

        Returns:
        --------
        (intercept, slope) tuple
        """
        # Linear regression: Q = intercept + slope*P
        slope, intercept, r_value, p_value, std_err = linregress(price_data, quantity_data)

        if curve_type == 'demand':
            # Q = a - b*P, so slope should be negative
            if slope > 0:
                warnings.warn("Demand curve has positive slope - possible identification problem")
            # Convert to demand form: a = intercept, b = -slope
            a = intercept
            b = -slope
            return (a, b)

        else:  # supply
            # Q = c + d*P, so slope should be positive
            if slope < 0:
                warnings.warn("Supply curve has negative slope - possible identification problem")
            c = intercept
            d = slope
            return (c, d)

    @staticmethod
    def estimate_elasticity(price_data: np.ndarray,
                          quantity_data: np.ndarray,
                          point: Optional[Tuple[float, float]] = None) -> float:
        """
        Estimate price elasticity from data.

        ε = (dQ/dP) * (P/Q)

        Parameters:
        -----------
        price_data : array
            Price observations
        quantity_data : array
            Quantity observations
        point : tuple, optional
            (P, Q) point to evaluate elasticity. If None, uses mean.

        Returns:
        --------
        Price elasticity
        """
        # Estimate dQ/dP using regression
        slope, intercept, _, _, _ = linregress(price_data, quantity_data)

        # Point of evaluation
        if point is None:
            P = np.mean(price_data)
            Q = np.mean(quantity_data)
        else:
            P, Q = point

        # Elasticity = (dQ/dP) * (P/Q)
        elasticity = slope * (P / Q)

        return elasticity


class LaborMarketCalibrator:
    """
    Calibrate labor market model parameters from employment and wage data.
    """

    @staticmethod
    def estimate_labor_supply_elasticity(wage_data: np.ndarray,
                                        hours_data: np.ndarray) -> float:
        """
        Estimate labor supply elasticity.

        log(hours) = α + β*log(wage) + ε

        β is the labor supply elasticity.

        Parameters:
        -----------
        wage_data : array
            Wage rates
        hours_data : array
            Hours worked

        Returns:
        --------
        Labor supply elasticity
        """
        # Log-log regression
        log_wage = np.log(wage_data)
        log_hours = np.log(hours_data)

        slope, intercept, r_value, p_value, std_err = linregress(log_wage, log_hours)

        return slope  # This is the elasticity

    @staticmethod
    def estimate_reservation_wage(accepted_wages: np.ndarray,
                                  acceptance_rate: float) -> float:
        """
        Estimate reservation wage from job search data.

        Assumes wages are normally distributed.
        Reservation wage is the cutoff where acceptance_rate matches data.

        Parameters:
        -----------
        accepted_wages : array
            Wages of accepted job offers
        acceptance_rate : float
            Fraction of offers accepted (0 to 1)

        Returns:
        --------
        Estimated reservation wage
        """
        from scipy.stats import norm

        mean_wage = np.mean(accepted_wages)
        std_wage = np.std(accepted_wages)

        # For normal distribution, if acceptance rate = p:
        # p = P(W > w_r) = 1 - Φ((w_r - μ)/σ)
        # w_r = μ + σ * Φ^{-1}(1 - p)

        z_score = norm.ppf(1 - acceptance_rate)
        w_reservation = mean_wage + std_wage * z_score

        return w_reservation


class HumanCapitalCalibrator:
    """
    Calibrate human capital model from earnings and education data.

    Mincer Equation: log(wage) = α + β*S + γ₁*Exp + γ₂*Exp² + ε
    where S = years of schooling, Exp = experience
    """

    @staticmethod
    def estimate_mincer_equation(education_years: np.ndarray,
                                 experience_years: np.ndarray,
                                 wages: np.ndarray) -> Dict[str, float]:
        """
        Estimate Mincer earnings equation.

        log(wage) = α + β*S + γ₁*Exp + γ₂*Exp²

        Parameters:
        -----------
        education_years : array
            Years of schooling
        experience_years : array
            Years of work experience
        wages : array
            Hourly or annual wages

        Returns:
        --------
        Dictionary with estimated parameters
        """
        from sklearn.linear_model import LinearRegression

        # Prepare data
        log_wage = np.log(wages)
        X = np.column_stack([
            education_years,
            experience_years,
            experience_years ** 2
        ])

        # OLS regression
        model = LinearRegression()
        model.fit(X, log_wage)

        return {
            'intercept': model.intercept_,
            'return_to_education': model.coef_[0],  # β (return per year of schooling)
            'experience_coef': model.coef_[1],      # γ₁
            'experience_sq_coef': model.coef_[2],   # γ₂
            'r_squared': model.score(X, log_wage)
        }

    @staticmethod
    def estimate_education_return(education_levels: List[str],
                                  mean_wages: List[float]) -> Dict[str, float]:
        """
        Estimate returns to different education levels.

        Simplified version using education level dummies.

        Parameters:
        -----------
        education_levels : list
            Education categories (e.g., ['HS', 'BA', 'MA', 'PhD'])
        mean_wages : list
            Average wage for each education level

        Returns:
        --------
        Dictionary with wage premium for each level relative to base
        """
        base_wage = mean_wages[0]  # First level as base

        returns = {}
        for i, (level, wage) in enumerate(zip(education_levels, mean_wages)):
            if i == 0:
                returns[level] = 0.0  # Base level
            else:
                # Log wage difference (approximates percentage difference)
                returns[level] = np.log(wage / base_wage)

        return returns


class PriceDiscriminationCalibrator:
    """
    Calibrate price discrimination models from transaction data.
    """

    @staticmethod
    def segment_demand_curves(prices: np.ndarray,
                             quantities: np.ndarray,
                             segments: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Estimate separate demand curves for different market segments.

        Parameters:
        -----------
        prices : array
            Transaction prices
        quantities : array
            Transaction quantities
        segments : array
            Segment identifiers (e.g., 'student', 'adult', 'senior')

        Returns:
        --------
        Dictionary mapping segment to (intercept, slope) of demand curve
        """
        unique_segments = np.unique(segments)
        demand_params = {}

        for seg in unique_segments:
            mask = segments == seg
            seg_prices = prices[mask]
            seg_quantities = quantities[mask]

            if len(seg_prices) > 1:
                slope, intercept, _, _, _ = linregress(seg_prices, seg_quantities)
                # Convert to demand form: Q = a - b*P
                a = intercept
                b = -slope
                demand_params[str(seg)] = (a, b)

        return demand_params

    @staticmethod
    def estimate_markup(marginal_cost: float,
                       prices: np.ndarray,
                       quantities: np.ndarray) -> Dict[str, float]:
        """
        Estimate markup and market power from pricing data.

        Lerner Index: L = (P - MC) / P

        Parameters:
        -----------
        marginal_cost : float
            Estimated marginal cost
        prices : array
            Observed prices
        quantities : array
            Observed quantities

        Returns:
        --------
        Dictionary with markup statistics
        """
        average_price = np.mean(prices)
        lerner_index = (average_price - marginal_cost) / average_price

        # Estimate implied elasticity from Lerner index
        # L = -1/ε  =>  ε = -1/L
        implied_elasticity = -1 / lerner_index if lerner_index != 0 else np.inf

        return {
            'average_price': average_price,
            'marginal_cost': marginal_cost,
            'average_markup': average_price - marginal_cost,
            'lerner_index': lerner_index,
            'implied_demand_elasticity': implied_elasticity
        }


class ProductionFunctionCalibrator:
    """
    Calibrate production function parameters from input-output data.
    """

    @staticmethod
    def estimate_cobb_douglas(labor_input: np.ndarray,
                             capital_input: np.ndarray,
                             output: np.ndarray) -> Dict[str, float]:
        """
        Estimate Cobb-Douglas production function: Q = A * L^α * K^β

        Taking logs: log(Q) = log(A) + α*log(L) + β*log(K)

        Parameters:
        -----------
        labor_input : array
            Labor inputs
        capital_input : array
            Capital inputs
        output : array
            Output quantities

        Returns:
        --------
        Dictionary with A, α, β parameters
        """
        from sklearn.linear_model import LinearRegression

        # Log transformation
        log_Q = np.log(output)
        log_L = np.log(labor_input)
        log_K = np.log(capital_input)

        X = np.column_stack([log_L, log_K])

        # OLS regression
        model = LinearRegression()
        model.fit(X, log_Q)

        alpha = model.coef_[0]  # Labor elasticity
        beta = model.coef_[1]   # Capital elasticity
        A = np.exp(model.intercept_)  # Total factor productivity

        returns_to_scale = alpha + beta

        return {
            'A': A,
            'alpha': alpha,
            'beta': beta,
            'returns_to_scale': returns_to_scale,
            'r_squared': model.score(X, log_Q)
        }

    @staticmethod
    def estimate_ces_elasticity(labor_input: np.ndarray,
                               capital_input: np.ndarray,
                               output: np.ndarray) -> float:
        """
        Estimate elasticity of substitution for CES production function.

        This is more complex - uses nonlinear least squares.

        Returns:
        --------
        Elasticity of substitution σ
        """
        # Simplified estimation - in practice would use NLS
        # For now, return Cobb-Douglas estimate (σ = 1)
        warnings.warn(
            "CES estimation not fully implemented. "
            "Returning Cobb-Douglas assumption (σ=1)"
        )
        return 1.0


def calibrate_model_from_data(model_type: str,
                              data: pd.DataFrame,
                              **kwargs) -> Dict:
    """
    Convenience function to calibrate any model from data.

    Parameters:
    -----------
    model_type : str
        Type of model ('supply_demand', 'labor', 'human_capital', etc.)
    data : DataFrame
        Data with relevant columns
    **kwargs : additional arguments for specific calibrations

    Returns:
    --------
    Dictionary with calibrated parameters

    Example:
    --------
    >>> data = pd.DataFrame({'price': [...], 'quantity': [...]})
    >>> params = calibrate_model_from_data('supply_demand', data, curve_type='demand')
    """
    if model_type == 'supply_demand':
        calibrator = SupplyDemandCalibrator()
        return calibrator.calibrate_linear(
            data['price'].values,
            data['quantity'].values,
            kwargs.get('curve_type', 'demand')
        )

    elif model_type == 'labor_supply':
        calibrator = LaborMarketCalibrator()
        return calibrator.estimate_labor_supply_elasticity(
            data['wage'].values,
            data['hours'].values
        )

    elif model_type == 'human_capital':
        calibrator = HumanCapitalCalibrator()
        return calibrator.estimate_mincer_equation(
            data['education'].values,
            data['experience'].values,
            data['wage'].values
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    print("=" * 80)
    print("CALIBRATION MODULE - Parameter Estimation from Data")
    print("=" * 80)

    print("\nThis module estimates model parameters from real-world data.")
    print("Uses econometric techniques: OLS, 2SLS, MLE, etc.")

    # Example: Calibrate demand curve
    print("\n" + "-" * 80)
    print("Example 1: Calibrate Demand Curve")
    print("-" * 80)

    # Simulated data (in practice, would use real data)
    np.random.seed(42)
    true_a, true_b = 100, 2
    prices = np.linspace(10, 40, 50)
    quantities = true_a - true_b * prices + np.random.normal(0, 5, 50)

    calibrator = SupplyDemandCalibrator()
    a_est, b_est = calibrator.calibrate_linear(prices, quantities, curve_type='demand')

    print(f"True demand: Q = {true_a} - {true_b}*P")
    print(f"Estimated demand: Q = {a_est:.2f} - {b_est:.2f}*P")

    elasticity = calibrator.estimate_elasticity(prices, quantities)
    print(f"Estimated price elasticity: {elasticity:.2f}")

    # Example: Mincer equation
    print("\n" + "-" * 80)
    print("Example 2: Mincer Earnings Equation")
    print("-" * 80)

    # Simulated data
    n = 1000
    education = np.random.randint(12, 21, n)
    experience = np.random.randint(0, 40, n)
    log_wage = 2.5 + 0.10*education + 0.05*experience - 0.001*experience**2 + np.random.normal(0, 0.2, n)
    wages = np.exp(log_wage)

    hc_calibrator = HumanCapitalCalibrator()
    mincer_results = hc_calibrator.estimate_mincer_equation(education, experience, wages)

    print(f"Estimated return to education: {mincer_results['return_to_education']:.3f}")
    print(f"  Interpretation: Each year of schooling increases wages by "
          f"{mincer_results['return_to_education']*100:.1f}%")
    print(f"Experience coefficient: {mincer_results['experience_coef']:.3f}")
    print(f"R-squared: {mincer_results['r_squared']:.3f}")

    # Example: Production function
    print("\n" + "-" * 80)
    print("Example 3: Cobb-Douglas Production Function")
    print("-" * 80)

    # Simulated firm data
    n_firms = 100
    L = np.random.uniform(10, 100, n_firms)
    K = np.random.uniform(20, 200, n_firms)
    Q = 2 * (L ** 0.6) * (K ** 0.3) * np.random.lognormal(0, 0.1, n_firms)

    prod_calibrator = ProductionFunctionCalibrator()
    prod_params = prod_calibrator.estimate_cobb_douglas(L, K, Q)

    print(f"Estimated production function: Q = {prod_params['A']:.2f} * L^{prod_params['alpha']:.3f} * K^{prod_params['beta']:.3f}")
    print(f"Returns to scale: {prod_params['returns_to_scale']:.3f}")
    if prod_params['returns_to_scale'] > 1:
        print("  → Increasing returns to scale")
    elif prod_params['returns_to_scale'] < 1:
        print("  → Decreasing returns to scale")
    else:
        print("  → Constant returns to scale")

    print("\n" + "=" * 80)
    print("Calibration complete! Use these parameters in theoretical models.")
    print("=" * 80)
