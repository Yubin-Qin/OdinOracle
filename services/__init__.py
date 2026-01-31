"""
Services package for OdinOracle.
Provides core business logic separated from presentation and data layers.
"""

from services.common import (
    normalize_symbol,
    infer_market_type,
    calculate_signal,
    calculate_signal_from_input,
    SignalInput,
    SignalResult,
    signal_to_numeric
)
from services.market_data import MarketDataService
from services.notification import EmailService
from services.portfolio import PortfolioService
from services.backtest import (
    SignalBacktester,
    BacktestResult,
    Trade,
    SignalAction,
    backtest_asset,
    format_backtest_report
)

__all__ = [
    # Common utilities
    'normalize_symbol',
    'infer_market_type',
    'calculate_signal',
    'calculate_signal_from_input',
    'SignalInput',
    'SignalResult',
    'signal_to_numeric',
    # Services
    'MarketDataService',
    'EmailService',
    'PortfolioService',
    # Backtesting
    'SignalBacktester',
    'BacktestResult',
    'Trade',
    'SignalAction',
    'backtest_asset',
    'format_backtest_report',
]
