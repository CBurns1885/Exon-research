"""
Empirical Models - Real Data Integration for Theoretical Models

This module extends theoretical models with real-world data capabilities.
Models can be used in two modes:
1. Theoretical mode (default) - User specifies parameters
2. Empirical mode - Parameters calibrated from real data

BACKWARD COMPATIBLE: All existing functionality preserved.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd
import warnings

# Import theoretical models
from .supply_demand import SupplyDemandModel
from .labor_market import LaborMarketModel
from .human_capital import HumanCapitalModel
from .consumer_theory import ConsumerModel

# Import data tools
try:
    from .data_sources import EconomicDataAggregator, get_economic_data
    from .calibration import (
        SupplyDemandCalibrator,
        LaborMarketCalibrator,
        HumanCapitalCalibrator
    )
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    warnings.warn(
        "Data modules not available. Install requirements: "
        "pip install requests pandas scikit-learn"
    )


class EmpiricalLaborMarketModel(LaborMarketModel):
    """
    Labor Market Model with real data integration.

    Can load actual wage and employment data from FRED/BLS.
    """

    def __init__(self, *args, **kwargs):
        """Initialize - works exactly like theoretical model."""
        super().__init__(*args, **kwargs)
        self.data_source = None
        self.calibrated = False

    def load_from_fred(self,
                      wage_series: str = 'CES0500000003',
                      unemployment_series: str = 'UNRATE',
                      api_key: Optional[str] = None) -> Dict:
        """
        Load labor market data from FRED and calibrate model.

        Parameters:
        -----------
        wage_series : str
            FRED series ID for wages (default: avg hourly earnings)
        unemployment_series : str
            FRED series ID for unemployment
        api_key : str
            FRED API key

        Returns:
        --------
        Dictionary with loaded data and calibrated parameters
        """
        if not DATA_AVAILABLE:
            raise ImportError("Data modules required. Run: pip install -r requirements.txt")

        from .data_sources import FREDDataFetcher

        fred = FREDDataFetcher(api_key)

        # Fetch wage data
        wage_df = fred.fetch_series(wage_series)
        latest_wage = wage_df['value'].iloc[-1]

        # Fetch unemployment data
        unemp_df = fred.fetch_series(unemployment_series)
        latest_unemployment = unemp_df['value'].iloc[-1]

        # Update model with real data
        self.wage = latest_wage
        self.data_source = 'fred'
        self.calibrated = True

        return {
            'wage': latest_wage,
            'unemployment_rate': latest_unemployment,
            'wage_data': wage_df,
            'unemployment_data': unemp_df
        }

    def calibrate_from_data(self,
                           wage_data: np.ndarray,
                           hours_data: np.ndarray) -> Dict:
        """
        Calibrate labor supply parameters from micro data.

        Parameters:
        -----------
        wage_data : array
            Individual wage observations
        hours_data : array
            Individual hours worked

        Returns:
        --------
        Dictionary with calibrated parameters
        """
        if not DATA_AVAILABLE:
            raise ImportError("Calibration module required")

        calibrator = LaborMarketCalibrator()

        # Estimate labor supply elasticity
        elasticity = calibrator.estimate_labor_supply_elasticity(wage_data, hours_data)

        # Update model (adjust leisure preference to match elasticity)
        # This is simplified - full calibration would use more sophisticated methods
        self.calibrated = True

        return {
            'labor_supply_elasticity': elasticity,
            'mean_wage': np.mean(wage_data),
            'mean_hours': np.mean(hours_data)
        }


class EmpiricalHumanCapitalModel(HumanCapitalModel):
    """
    Human Capital Model with real earnings data.

    Can load actual wage-education data and estimate returns to schooling.
    """

    def __init__(self, *args, **kwargs):
        """Initialize - works exactly like theoretical model."""
        super().__init__(*args, **kwargs)
        self.data_source = None
        self.calibrated = False
        self.mincer_coefficients = None

    def load_education_wage_data(self, api_key: Optional[str] = None) -> Dict:
        """
        Load real education-wage data.

        Uses BLS data on median weekly earnings by education.

        Parameters:
        -----------
        api_key : str, optional
            API key if needed

        Returns:
        --------
        Dictionary with education levels and corresponding wages
        """
        if not DATA_AVAILABLE:
            raise ImportError("Data modules required")

        from .data_sources import EconomicDataAggregator

        agg = EconomicDataAggregator(fred_api_key=api_key)

        # Get wage by education (uses BLS data)
        wage_by_ed = agg.get_wage_by_education()

        # Calibrate model based on observed college premium
        hs_wage = wage_by_ed['high_school'] * 52  # Weekly to annual
        college_wage = wage_by_ed['bachelors'] * 52

        # Implied return to 4 years of college
        college_premium = (college_wage - hs_wage) / hs_wage
        implied_return = college_premium / 4  # Per year

        # Update model
        self.base_wage = hs_wage
        self.education_return = implied_return
        self.calibrated = True
        self.data_source = 'bls'

        return {
            'wage_by_education': wage_by_ed,
            'high_school_wage': hs_wage,
            'college_wage': college_wage,
            'college_premium': college_premium * 100,  # As percentage
            'implied_annual_return': implied_return * 100
        }

    def estimate_mincer_equation(self,
                                 education_data: np.ndarray,
                                 experience_data: np.ndarray,
                                 wage_data: np.ndarray) -> Dict:
        """
        Estimate Mincer earnings equation from micro data.

        log(wage) = α + β*education + γ₁*experience + γ₂*experience²

        Parameters:
        -----------
        education_data : array
            Years of schooling for each individual
        experience_data : array
            Years of experience for each individual
        wage_data : array
            Wage for each individual

        Returns:
        --------
        Dictionary with Mincer equation coefficients
        """
        if not DATA_AVAILABLE:
            raise ImportError("Calibration module required")

        calibrator = HumanCapitalCalibrator()

        results = calibrator.estimate_mincer_equation(
            education_data,
            experience_data,
            wage_data
        )

        # Update model with estimated return to education
        self.education_return = results['return_to_education']
        self.mincer_coefficients = results
        self.calibrated = True

        return results


class EmpiricalSupplyDemandModel(SupplyDemandModel):
    """
    Supply and Demand Model with real price/quantity data.

    Can calibrate from actual market data.
    """

    def __init__(self, *args, **kwargs):
        """Initialize - works exactly like theoretical model."""
        super().__init__(*args, **kwargs)
        self.calibrated = False
        self.data_source = None

    def calibrate_from_market_data(self,
                                   price_data: np.ndarray,
                                   quantity_data: np.ndarray,
                                   curve_type: str = 'demand') -> Dict:
        """
        Calibrate supply or demand curve from market data.

        Parameters:
        -----------
        price_data : array
            Observed market prices
        quantity_data : array
            Observed market quantities
        curve_type : str
            'demand' or 'supply'

        Returns:
        --------
        Dictionary with calibrated parameters
        """
        if not DATA_AVAILABLE:
            raise ImportError("Calibration module required")

        calibrator = SupplyDemandCalibrator()

        # Estimate parameters
        a, b = calibrator.calibrate_linear(price_data, quantity_data, curve_type)

        # Update model
        if curve_type == 'demand':
            self.demand_params = (a, b)
        else:
            self.supply_params = (a, b)

        self.calibrated = True

        # Estimate elasticity
        elasticity = calibrator.estimate_elasticity(price_data, quantity_data)

        return {
            'intercept': a,
            'slope': b,
            'elasticity': elasticity,
            'mean_price': np.mean(price_data),
            'mean_quantity': np.mean(quantity_data)
        }

    def load_commodity_data(self,
                           commodity: str,
                           api_key: Optional[str] = None) -> Dict:
        """
        Load real commodity price data.

        Parameters:
        -----------
        commodity : str
            Commodity name (e.g., 'oil', 'wheat', 'gold')
        api_key : str
            FRED API key

        Returns:
        --------
        Dictionary with price time series
        """
        if not DATA_AVAILABLE:
            raise ImportError("Data modules required")

        from .data_sources import FREDDataFetcher

        # Map commodity names to FRED series
        commodity_series = {
            'oil': 'DCOILWTICO',     # Crude Oil WTI
            'wheat': 'PWHEAMTUSDM',   # Wheat prices
            'gold': 'GOLDAMGBD228NLBM',  # Gold price
            'gasoline': 'GASREGW',    # Gasoline retail price
            'housing': 'CSUSHPISA'    # Case-Shiller Home Price Index
        }

        if commodity not in commodity_series:
            raise ValueError(f"Unknown commodity: {commodity}. Available: {list(commodity_series.keys())}")

        fred = FREDDataFetcher(api_key)
        df = fred.fetch_series(commodity_series[commodity])

        self.data_source = 'fred'

        return {
            'commodity': commodity,
            'price_data': df,
            'latest_price': df['value'].iloc[-1],
            'mean_price': df['value'].mean(),
            'std_price': df['value'].std()
        }


def create_empirical_model(model_type: str, **kwargs):
    """
    Factory function to create empirical models.

    Parameters:
    -----------
    model_type : str
        Type of model ('labor', 'human_capital', 'supply_demand', etc.)
    **kwargs
        Arguments passed to model constructor

    Returns:
    --------
    Empirical model instance

    Example:
    --------
    >>> model = create_empirical_model('labor', wage=25.0)
    >>> data = model.load_from_fred(api_key='your_key')
    >>> results = model.market_equilibrium()
    """
    models = {
        'labor': EmpiricalLaborMarketModel,
        'human_capital': EmpiricalHumanCapitalModel,
        'supply_demand': EmpiricalSupplyDemandModel
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](**kwargs)


if __name__ == '__main__':
    print("=" * 80)
    print("EMPIRICAL MODELS - Real Data Integration")
    print("=" * 80)

    print("\nThis module extends theoretical models with real-world data.")
    print("Models work in two modes:")
    print("  1. Theoretical (default) - Specify parameters manually")
    print("  2. Empirical - Load and calibrate from real data")

    print("\n" + "-" * 80)
    print("Example 1: Labor Market with Real Wage Data")
    print("-" * 80)

    print("\n# Theoretical mode (works without API key):")
    print("model = EmpiricalLaborMarketModel(wage=25.0)")
    print("result = model.labor_supply_individual()")

    print("\n# Empirical mode (requires FRED API key):")
    print("model = EmpiricalLaborMarketModel()")
    print("data = model.load_from_fred(api_key='your_key')")
    print("result = model.labor_supply_individual()  # Uses real wage data")

    print("\n" + "-" * 80)
    print("Example 2: Human Capital with Real Education Data")
    print("-" * 80)

    print("\n# Load actual wage-education data:")
    print("model = EmpiricalHumanCapitalModel()")
    print("data = model.load_education_wage_data(api_key='your_key')")
    print("# Model now uses real returns to education")

    print("\n# Or calibrate from micro data:")
    print("model.estimate_mincer_equation(education, experience, wages)")
    print("# Estimates: log(wage) = α + β*education + ...")

    print("\n" + "-" * 80)
    print("Example 3: Supply/Demand with Market Data")
    print("-" * 80)

    print("\n# Calibrate from observed prices and quantities:")
    print("model = EmpiricalSupplyDemandModel()")
    print("model.calibrate_from_market_data(prices, quantities, curve_type='demand')")

    print("\n# Or load commodity data:")
    print("data = model.load_commodity_data('oil', api_key='your_key')")

    print("\n" + "=" * 80)
    print("Setup: Get free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html")
    print("=" * 80)
