"""
Services package for OdinOracle.
Provides core business logic separated from presentation and data layers.
"""

from services.common import normalize_symbol, infer_market_type, calculate_signal
from services.market_data import MarketDataService
from services.notification import EmailService
from services.portfolio import PortfolioService

__all__ = [
    'normalize_symbol',
    'infer_market_type',
    'calculate_signal',
    'MarketDataService',
    'EmailService',
    'PortfolioService',
]
