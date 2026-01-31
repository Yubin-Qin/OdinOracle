"""
Market data service for fetching stock information.
Provides cached access to yfinance data.
"""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, Dict

from services.common import normalize_symbol

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Service for fetching market data with caching.
    Uses LRU cache to prevent redundant API calls.
    """

    @staticmethod
    @lru_cache(maxsize=256)
    def get_current_price(symbol: str, market_type: str) -> Optional[float]:
        """
        Fetch current stock price using yfinance with caching.

        Args:
            symbol: Stock symbol
            market_type: Market type (US, HK, CN)

        Returns:
            Current price as float, or None if unavailable
        """
        try:
            yf_symbol = normalize_symbol(symbol, market_type)
            logger.debug(f"Fetching price for {yf_symbol}")

            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            # Try multiple price fields
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('lastPrice')

            if price is None:
                # Fallback to historical data
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]

            if price:
                logger.debug(f"{symbol}: ${price:.2f}")
                return float(price)

            logger.warning(f"Could not retrieve price for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    @staticmethod
    @lru_cache(maxsize=128)
    def get_historical_data(symbol: str, market_type: str, period_months: int = 6) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data with caching.

        Args:
            symbol: Stock symbol
            market_type: Market type (US, HK, CN)
            period_months: Historical period in months (default 6)

        Returns:
            DataFrame with historical data, or None if unavailable
        """
        try:
            yf_symbol = normalize_symbol(symbol, market_type)
            logger.debug(f"Fetching historical data for {yf_symbol} ({period_months} months)")

            ticker = yf.Ticker(yf_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_months * 30)

            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return None

            logger.debug(f"Retrieved {len(hist)} data points for {symbol}")
            return hist

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    @staticmethod
    def get_stock_info(symbol: str, market_type: str) -> Optional[Dict]:
        """
        Fetch comprehensive stock information.

        Args:
            symbol: Stock symbol
            market_type: Market type (US, HK, CN)

        Returns:
            Dictionary with stock information, or None if unavailable
        """
        try:
            yf_symbol = normalize_symbol(symbol, market_type)
            logger.debug(f"Fetching info for {yf_symbol}")

            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'yf_symbol': yf_symbol,
                'name': info.get('longName') or info.get('shortName'),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice') or info.get('lastPrice'),
                'previous_close': info.get('previousClose'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'pe_ratio': info.get('trailingPE'),
                'eps': info.get('trailingEps'),
                'dividend_yield': info.get('dividendYield'),
                'summary': info.get('longBusinessSummary'),
            }

        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return None

    @staticmethod
    def clear_cache():
        """Clear the LRU cache. Useful for forcing fresh data fetch."""
        MarketDataService.get_current_price.cache_clear()
        MarketDataService.get_historical_data.cache_clear()
        logger.info("Market data cache cleared")
