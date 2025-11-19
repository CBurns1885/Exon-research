"""
Chicago Price Theory Economic Models
A collection of economic models implementing core Chicago School principles

Two modes of operation:
1. Theoretical (default): Specify parameters manually
2. Empirical (optional): Load and calibrate from real economic data

Basic usage:
    >>> from chicago_price_theory import SupplyDemandModel
    >>> model = SupplyDemandModel()
    >>> equilibrium = model.find_equilibrium()

With real data (requires API keys):
    >>> from chicago_price_theory.empirical_models import EmpiricalLaborMarketModel
    >>> model = EmpiricalLaborMarketModel()
    >>> data = model.load_from_fred(api_key='your_key')
"""

# Core theoretical models (always available)
from .supply_demand import SupplyDemandModel
from .consumer_theory import ConsumerModel
from .producer_theory import ProducerModel
from .labor_market import LaborMarketModel
from .human_capital import HumanCapitalModel
from .price_discrimination import PriceDiscriminationModel
from .search_theory import SearchModel
from .time_allocation import TimeAllocationModel

# Empirical extensions (optional - require pandas, requests, sklearn)
try:
    from .empirical_models import (
        EmpiricalLaborMarketModel,
        EmpiricalHumanCapitalModel,
        EmpiricalSupplyDemandModel,
        create_empirical_model
    )
    from .data_sources import (
        FREDDataFetcher,
        BLSDataFetcher,
        WorldBankDataFetcher,
        EconomicDataAggregator,
        get_economic_data
    )
    from .calibration import (
        SupplyDemandCalibrator,
        LaborMarketCalibrator,
        HumanCapitalCalibrator,
        PriceDiscriminationCalibrator,
        ProductionFunctionCalibrator,
        calibrate_model_from_data
    )
    EMPIRICAL_AVAILABLE = True
except ImportError:
    EMPIRICAL_AVAILABLE = False

__all__ = [
    # Core theoretical models
    'SupplyDemandModel',
    'ConsumerModel',
    'ProducerModel',
    'LaborMarketModel',
    'HumanCapitalModel',
    'PriceDiscriminationModel',
    'SearchModel',
    'TimeAllocationModel',
]

# Add empirical tools if available
if EMPIRICAL_AVAILABLE:
    __all__.extend([
        # Empirical model extensions
        'EmpiricalLaborMarketModel',
        'EmpiricalHumanCapitalModel',
        'EmpiricalSupplyDemandModel',
        'create_empirical_model',
        # Data fetchers
        'FREDDataFetcher',
        'BLSDataFetcher',
        'WorldBankDataFetcher',
        'EconomicDataAggregator',
        'get_economic_data',
        # Calibration tools
        'SupplyDemandCalibrator',
        'LaborMarketCalibrator',
        'HumanCapitalCalibrator',
        'PriceDiscriminationCalibrator',
        'ProductionFunctionCalibrator',
        'calibrate_model_from_data',
    ])

__version__ = '2.0.0'  # Version 2.0 - Added empirical data integration
