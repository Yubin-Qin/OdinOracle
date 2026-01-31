"""
Technical Analysis Module for OdinOracle.
Calculates RSI, SMA, and generates buy/sell/hold signals.
Refactored to use services layer.
"""

import pandas as pd
from typing import Optional, Dict, List
import logging

from services.market_data import MarketDataService
from services.common import normalize_symbol

logger = logging.getLogger(__name__)


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame with 'Close' prices
        period: RSI period (default 14)

    Returns:
        Series with RSI values
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        df: DataFrame with 'Close' prices
        period: SMA period

    Returns:
        Series with SMA values
    """
    return df['Close'].rolling(window=period).mean()


def get_technical_indicators(symbol: str, market_type: str = "US",
                             period_months: int = 6) -> Optional[Dict]:
    """
    Fetch historical data and calculate technical indicators.

    Args:
        symbol: Stock symbol
        market_type: Market type (US, HK, CN)
        period_months: Historical period in months (default 6)

    Returns:
        Dictionary with:
            - symbol: str
            - current_price: float
            - rsi: float
            - sma20: float
            - sma200: float
            - signal: str (Buy/Sell/Hold)
            - signal_reason: str
            - history_df: DataFrame with price and SMA data
    """
    try:
        logger.info(f"Fetching technical data for {symbol}")

        # Fetch historical data using MarketDataService (cached)
        hist = MarketDataService.get_historical_data(symbol, market_type, period_months)

        if hist is None or hist.empty:
            logger.warning(f"No historical data for {symbol}")
            return None

        # Calculate indicators
        hist['RSI'] = calculate_rsi(hist, period=14)
        hist['SMA20'] = calculate_sma(hist, period=20)
        hist['SMA200'] = calculate_sma(hist, period=200)

        # Get latest values
        current_price = hist['Close'].iloc[-1]
        current_rsi = hist['RSI'].iloc[-1]
        current_sma20 = hist['SMA20'].iloc[-1]
        current_sma200 = hist['SMA200'].iloc[-1]

        # Generate signal
        signal, reason = generate_signal(current_rsi, current_price, current_sma20, current_sma200)

        # Prepare history DataFrame for plotting (last 60 days)
        plot_df = hist[['Close', 'SMA20']].tail(60).copy()
        plot_df.index = plot_df.index.strftime('%Y-%m-%d')

        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'rsi': round(current_rsi, 2) if not pd.isna(current_rsi) else None,
            'sma20': round(current_sma20, 2) if not pd.isna(current_sma20) else None,
            'sma200': round(current_sma200, 2) if not pd.isna(current_sma200) else None,
            'signal': signal,
            'signal_reason': reason,
            'history_df': plot_df
        }

    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol}: {e}")
        return None


def generate_signal(rsi: Optional[float], price: float,
                   sma20: Optional[float], sma200: Optional[float]) -> tuple:
    """
    Generate buy/sell/hold signal based on indicators.

    Args:
        rsi: RSI value
        price: Current price
        sma20: 20-day SMA
        sma200: 200-day SMA

    Returns:
        Tuple of (signal: str, reason: str)
    """
    signals = []
    reasons = []

    # RSI Analysis
    if rsi is not None:
        if rsi < 30:
            signals.append('BUY')
            reasons.append(f'RSI ({rsi:.1f}) indicates oversold')
        elif rsi > 70:
            signals.append('SELL')
            reasons.append(f'RSI ({rsi:.1f}) indicates overbought')
        elif rsi < 40:
            signals.append('BUY')
            reasons.append(f'RSI ({rsi:.1f}) approaching oversold')
        elif rsi > 60:
            signals.append('SELL')
            reasons.append(f'RSI ({rsi:.1f}) approaching overbought')

    # SMA Cross Analysis (Golden/Death Cross)
    if sma20 is not None and sma200 is not None:
        if sma20 > sma200:
            if price > sma20:
                signals.append('BUY')
                reasons.append('Golden Cross: Price > SMA20 > SMA200 (bullish)')
            else:
                signals.append('HOLD')
                reasons.append('Price pulled back to SMA20 support')
        else:
            if price < sma20:
                signals.append('SELL')
                reasons.append('Death Cross: Price < SMA20 < SMA200 (bearish)')
            else:
                signals.append('HOLD')
                reasons.append('Price below SMA200 but above SMA20')

    # Determine final signal
    if not signals:
        return 'HOLD', 'Insufficient data for signal'

    buy_count = signals.count('BUY')
    sell_count = signals.count('SELL')

    if buy_count > sell_count:
        return 'BUY', ' | '.join(reasons)
    elif sell_count > buy_count:
        return 'SELL', ' | '.join(reasons)
    else:
        return 'HOLD', ' | '.join(reasons)


def analyze_portfolio(assets: List) -> pd.DataFrame:
    """
    Analyze all assets in portfolio and return summary DataFrame.

    Args:
        assets: List of Asset objects

    Returns:
        DataFrame with columns: Symbol, Name, Price, RSI, Signal, Reason
    """
    results = []

    for asset in assets:
        tech_data = get_technical_indicators(asset.symbol, asset.market_type)

        if tech_data:
            results.append({
                'Symbol': asset.symbol,
                'Name': asset.name,
                'Price': f"${tech_data['current_price']}",
                'RSI': f"{tech_data['rsi']:.1f}" if tech_data['rsi'] else 'N/A',
                'Signal': tech_data['signal'],
                'Reason': tech_data['signal_reason']
            })

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()


if __name__ == "__main__":
    # Test the technical analysis
    print("Testing Technical Analysis...")
    print("=" * 60)

    # Test with a US stock
    result = get_technical_indicators("NVDA", "US")
    if result:
        print(f"\nSymbol: {result['symbol']}")
        print(f"Current Price: ${result['current_price']}")
        print(f"RSI: {result['rsi']}")
        print(f"SMA20: ${result['sma20']}")
        print(f"SMA200: ${result['sma200']}")
        print(f"Signal: {result['signal']}")
        print(f"Reason: {result['signal_reason']}")
        print(f"\nHistory DataFrame:\n{result['history_df'].tail()}")
    else:
        print("Failed to fetch data")
