"""
Chicago Price Theory Economic Models
A collection of economic models implementing core Chicago School principles
"""

from .supply_demand import SupplyDemandModel
from .consumer_theory import ConsumerModel
from .producer_theory import ProducerModel
from .labor_market import LaborMarketModel
from .human_capital import HumanCapitalModel
from .price_discrimination import PriceDiscriminationModel
from .search_theory import SearchModel
from .time_allocation import TimeAllocationModel

__all__ = [
    'SupplyDemandModel',
    'ConsumerModel',
    'ProducerModel',
    'LaborMarketModel',
    'HumanCapitalModel',
    'PriceDiscriminationModel',
    'SearchModel',
    'TimeAllocationModel'
]

__version__ = '1.0.0'
