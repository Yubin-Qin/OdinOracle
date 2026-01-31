"""
Common utilities and shared functions.
Symbol normalization and market type inference.
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
        if not symbol.endswith(".HK"):
            return f"{symbol}.HK"
        return symbol
    elif market_type == "CN":
        if symbol.endswith((".SS", ".SZ")):
            return symbol
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


def calculate_signal(
    rsi: float = None,
    macd_histogram: float = None,
    price_vs_sma20: float = None,  # positive if above
    price_vs_sma200: float = None,  # positive if above
    golden_cross: bool = False,
    bollinger_position: str = None,  # "upper", "middle", "lower"
    volume_squeeze: bool = False
) -> tuple:
    """
    Calculate overall trading signal from multiple indicators.

    Args:
        rsi: RSI value (0-100)
        macd_histogram: MACD histogram value
        price_vs_sma20: Price relative to SMA20 (positive = above)
        price_vs_sma200: Price relative to SMA200 (positive = above)
        golden_cross: True if SMA20 > SMA200
        bollinger_position: Position within Bollinger Bands
        volume_squeeze: True if volume is contracting

    Returns:
        Tuple of (signal: str, confidence: int)
        Signal: "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
        Confidence: 0-10
    """
    score = 5  # Start at neutral
    factors = 0

    # RSI Analysis (weight: 2)
    if rsi is not None:
        factors += 2
        if rsi < 30:
            score += 2  # Strongly oversold
        elif rsi < 40:
            score += 1  # Approaching oversold
        elif rsi > 70:
            score -= 2  # Strongly overbought
        elif rsi > 60:
            score -= 1  # Approaching overbought

    # MACD Histogram (weight: 2)
    if macd_histogram is not None:
        factors += 2
        if macd_histogram > 0:
            if macd_histogram > 0.5:
                score += 2  # Strong bullish momentum
            else:
                score += 1  # Bullish momentum
        else:
            if macd_histogram < -0.5:
                score -= 2  # Strong bearish momentum
            else:
                score -= 1  # Bearish momentum

    # Golden Cross / Death Cross (weight: 2)
    if golden_cross:
        factors += 2
        score += 2  # Bullish
    elif price_vs_sma20 is not None and price_vs_sma200 is not None:
        factors += 2
        if price_vs_sma20 > 0 and price_vs_sma200 < 0:
            score -= 1  # Price above SMA20 but below SMA200 (neutral/bearish)
        elif price_vs_sma20 < 0 and price_vs_sma200 < 0:
            score -= 2  # Death cross confirmed

    # Bollinger Band Position (weight: 1)
    if bollinger_position:
        factors += 1
        if bollinger_position == "lower":
            score += 1  # Near lower band (potential bounce)
        elif bollinger_position == "upper":
            score -= 1  # Near upper band (potential pullback)

    # Volume Squeeze (weight: 1)
    if volume_squeeze:
        factors += 1
        score += 0.5  # Squeeze often precedes breakout

    # Normalize score to 0-10
    score = max(0, min(10, score))

    # Determine signal
    if score >= 8:
        signal = "STRONG_BUY"
    elif score >= 6:
        signal = "BUY"
    elif score <= 2:
        signal = "STRONG_SELL"
    elif score <= 4:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Confidence based on how many factors we had
    confidence = int((factors / 8) * 10)  # Max 8 factors

    return signal, confidence
