"""
Common utilities and shared functions.
"""

import logging

logger = logging.getLogger(__name__)


def normalize_symbol(symbol: str, market_type: str) -> str:
    """
    Convert a stock symbol to yfinance format based on market type.

    Args:
        symbol: Stock symbol (e.g., "NVDA", "0700", "600519")
        market_type: Market type ("US", "HK", "CN")

    Returns:
        Properly formatted yfinance symbol

    Examples:
        >>> normalize_symbol("NVDA", "US")
        'NVDA'
        >>> normalize_symbol("0700", "HK")
        '0700.HK'
        >>> normalize_symbol("600519", "CN")
        '600519.SS'
    """
    if market_type == "US":
        return symbol
    elif market_type == "HK":
        # HK stocks need .HK suffix
        if not symbol.endswith(".HK"):
            return f"{symbol}.HK"
        return symbol
    elif market_type == "CN":
        # CN stocks need .SS (Shanghai) or .SZ (Shenzhen) suffix
        if symbol.endswith((".SS", ".SZ")):
            return symbol
        # Default to .SS for simplicity (user should specify if different)
        return f"{symbol}.SS"
    else:
        logger.warning(f"Unknown market type: {market_type}, returning symbol as-is")
        return symbol


def infer_market_type(symbol: str) -> str:
    """
    Infer market type from symbol format.

    Args:
        symbol: Stock symbol

    Returns:
        Inferred market type ("US", "HK", or "CN")
    """
    if symbol.endswith((".HK", ".SS", ".SZ")):
        if symbol.endswith(".HK"):
            return "HK"
        return "CN"
    # Check for common Chinese stock patterns
    if len(symbol) == 6 and symbol.isdigit():
        return "CN"
    if len(symbol) == 4 and symbol.isdigit():
        return "HK"
    return "US"
