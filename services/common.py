"""
Common utilities and shared functions.
Symbol normalization, market type inference, and signal calculation.
Enhanced with decoupled signal calculation for backtesting support.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SignalInput:
    """
    Input data class for signal calculation.
    Encapsulates all indicator values needed for signal generation.
    """
    rsi: Optional[float] = None
    macd_histogram: Optional[float] = None
    price_vs_sma20: Optional[float] = None  # positive if above
    price_vs_sma200: Optional[float] = None  # positive if above
    golden_cross: bool = False
    bollinger_position: Optional[str] = None  # "upper", "middle", "lower"
    volume_squeeze: bool = False
    close_price: Optional[float] = None  # For additional context

    @classmethod
    def from_metric(cls, metric: Any) -> "SignalInput":
        """
        Create SignalInput from an AssetDailyMetric object.

        Args:
            metric: AssetDailyMetric instance or dict with indicator fields

        Returns:
            SignalInput populated from metric data
        """
        if hasattr(metric, '__dict__'):
            data = metric.__dict__
        else:
            data = metric

        # Calculate price vs SMAs if we have close_price
        close_price = data.get('close_price')
        sma_20 = data.get('sma_20')
        sma_200 = data.get('sma_200')

        price_vs_sma20 = None
        price_vs_sma200 = None
        golden_cross = False

        if close_price is not None and sma_20 is not None:
            price_vs_sma20 = close_price - sma_20
        if close_price is not None and sma_200 is not None:
            price_vs_sma200 = close_price - sma_200
        if sma_20 is not None and sma_200 is not None:
            golden_cross = sma_20 > sma_200

        return cls(
            rsi=data.get('rsi_14'),
            macd_histogram=data.get('macd_histogram'),
            price_vs_sma20=price_vs_sma20,
            price_vs_sma200=price_vs_sma200,
            golden_cross=golden_cross,
            bollinger_position=data.get('bollinger_position'),
            volume_squeeze=data.get('volume_ratio', 1.0) < 0.7 if data.get('volume_ratio') else False,
            close_price=close_price
        )


@dataclass
class SignalResult:
    """
    Result data class from signal calculation.
    Contains signal, confidence, and breakdown of factor contributions.
    """
    signal: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence: int  # 0-10
    score: float  # Raw score 0-10
    factors_used: int  # Number of factors considered
    factor_breakdown: Dict[str, Any]  # Detailed breakdown


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

    Legacy function - maintained for backward compatibility.
    New code should use calculate_signal_from_input() for better decoupling.

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
    result = calculate_signal_from_input(
        SignalInput(
            rsi=rsi,
            macd_histogram=macd_histogram,
            price_vs_sma20=price_vs_sma20,
            price_vs_sma200=price_vs_sma200,
            golden_cross=golden_cross,
            bollinger_position=bollinger_position,
            volume_squeeze=volume_squeeze
        )
    )
    return result.signal, result.confidence


def calculate_signal_from_input(input_data: SignalInput) -> SignalResult:
    """
    Calculate trading signal from SignalInput data class.
    Fully decoupled - can be easily run in isolation for backtesting.

    Args:
        input_data: SignalInput containing all indicator values

    Returns:
        SignalResult with signal, confidence, and detailed breakdown
    """
    score = 5.0  # Start at neutral
    factors = 0
    breakdown = {
        'rsi_contribution': 0,
        'macd_contribution': 0,
        'trend_contribution': 0,
        'bollinger_contribution': 0,
        'volume_contribution': 0,
        'details': []
    }

    # RSI Analysis (weight: 2)
    if input_data.rsi is not None:
        factors += 2
        if input_data.rsi < 30:
            score += 2  # Strongly oversold
            breakdown['rsi_contribution'] = 2
            breakdown['details'].append(f"RSI {input_data.rsi:.1f} < 30 (oversold)")
        elif input_data.rsi < 40:
            score += 1  # Approaching oversold
            breakdown['rsi_contribution'] = 1
            breakdown['details'].append(f"RSI {input_data.rsi:.1f} < 40 (approaching oversold)")
        elif input_data.rsi > 70:
            score -= 2  # Strongly overbought
            breakdown['rsi_contribution'] = -2
            breakdown['details'].append(f"RSI {input_data.rsi:.1f} > 70 (overbought)")
        elif input_data.rsi > 60:
            score -= 1  # Approaching overbought
            breakdown['rsi_contribution'] = -1
            breakdown['details'].append(f"RSI {input_data.rsi:.1f} > 60 (approaching overbought)")
        else:
            breakdown['details'].append(f"RSI {input_data.rsi:.1f} neutral")

    # MACD Histogram (weight: 2)
    if input_data.macd_histogram is not None:
        factors += 2
        if input_data.macd_histogram > 0:
            if input_data.macd_histogram > 0.5:
                score += 2  # Strong bullish momentum
                breakdown['macd_contribution'] = 2
                breakdown['details'].append(f"MACD hist {input_data.macd_histogram:.3f} > 0.5 (strong bullish)")
            else:
                score += 1  # Bullish momentum
                breakdown['macd_contribution'] = 1
                breakdown['details'].append(f"MACD hist {input_data.macd_histogram:.3f} > 0 (bullish)")
        else:
            if input_data.macd_histogram < -0.5:
                score -= 2  # Strong bearish momentum
                breakdown['macd_contribution'] = -2
                breakdown['details'].append(f"MACD hist {input_data.macd_histogram:.3f} < -0.5 (strong bearish)")
            else:
                score -= 1  # Bearish momentum
                breakdown['macd_contribution'] = -1
                breakdown['details'].append(f"MACD hist {input_data.macd_histogram:.3f} < 0 (bearish)")

    # Golden Cross / Death Cross (weight: 2)
    if input_data.golden_cross:
        factors += 2
        score += 2  # Bullish
        breakdown['trend_contribution'] = 2
        breakdown['details'].append("Golden Cross (SMA20 > SMA200)")
    elif input_data.price_vs_sma20 is not None and input_data.price_vs_sma200 is not None:
        factors += 2
        if input_data.price_vs_sma20 > 0 and input_data.price_vs_sma200 < 0:
            score -= 1  # Price above SMA20 but below SMA200 (neutral/bearish)
            breakdown['trend_contribution'] = -1
            breakdown['details'].append("Price above SMA20 but below SMA200 (mixed)")
        elif input_data.price_vs_sma20 < 0 and input_data.price_vs_sma200 < 0:
            score -= 2  # Death cross confirmed
            breakdown['trend_contribution'] = -2
            breakdown['details'].append("Death Cross pattern (bearish)")
        elif input_data.price_vs_sma20 > 0 and input_data.price_vs_sma200 > 0:
            score += 1  # Above both SMAs but no golden cross
            breakdown['trend_contribution'] = 1
            breakdown['details'].append("Price above both SMAs (bullish)")
        else:
            breakdown['details'].append("Trend neutral")

    # Bollinger Band Position (weight: 1)
    if input_data.bollinger_position:
        factors += 1
        if input_data.bollinger_position == "lower":
            score += 1  # Near lower band (potential bounce)
            breakdown['bollinger_contribution'] = 1
            breakdown['details'].append("Price at lower Bollinger Band (potential bounce)")
        elif input_data.bollinger_position == "upper":
            score -= 1  # Near upper band (potential pullback)
            breakdown['bollinger_contribution'] = -1
            breakdown['details'].append("Price at upper Bollinger Band (potential pullback)")
        else:
            breakdown['details'].append("Price within Bollinger Bands (neutral)")

    # Volume Squeeze (weight: 1)
    if input_data.volume_squeeze:
        factors += 1
        score += 0.5  # Squeeze often precedes breakout
        breakdown['volume_contribution'] = 0.5
        breakdown['details'].append("Volume squeeze detected (potential breakout)")

    # Normalize score to 0-10
    raw_score = score
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

    breakdown['raw_score'] = raw_score
    breakdown['normalized_score'] = score
    breakdown['max_possible_factors'] = 8

    return SignalResult(
        signal=signal,
        confidence=confidence,
        score=round(score, 2),
        factors_used=factors,
        factor_breakdown=breakdown
    )


def signal_to_numeric(signal: str) -> int:
    """
    Convert signal string to numeric value for calculations.

    Args:
        signal: Signal string ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")

    Returns:
        Numeric value: STRONG_BUY=2, BUY=1, HOLD=0, SELL=-1, STRONG_SELL=-2
    """
    mapping = {
        "STRONG_BUY": 2,
        "BUY": 1,
        "HOLD": 0,
        "SELL": -1,
        "STRONG_SELL": -2
    }
    return mapping.get(signal, 0)
