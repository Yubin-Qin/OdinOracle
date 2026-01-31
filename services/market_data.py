"""
Market data service for fetching stock information and calculating technical indicators.
Implements the Factor Pack with professional indicators.
Uses pandas for calculations and caches results.
Enhanced with tenacity for retry logic and resilience.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
from functools import lru_cache
from typing import Optional, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from services.common import normalize_symbol, calculate_signal

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Service for fetching market data and calculating technical indicators.
    Implements professional-grade factor calculations with retry logic.
    """

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _fetch_ticker_info(yf_symbol: str) -> Dict:
        """Fetch ticker info with retry logic."""
        ticker = yf.Ticker(yf_symbol)
        return ticker.info

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _fetch_ticker_history(yf_symbol: str, period: str = None, start=None, end=None) -> pd.DataFrame:
        """Fetch ticker history with retry logic."""
        ticker = yf.Ticker(yf_symbol)
        if start and end:
            return ticker.history(start=start, end=end)
        return ticker.history(period=period or "1d")

    @staticmethod
    @lru_cache(maxsize=256)
    def get_current_price(symbol: str, market_type: str) -> Optional[float]:
        """Fetch current stock price with caching and retry logic."""
        try:
            yf_symbol = normalize_symbol(symbol, market_type)
            info = MarketDataService._fetch_ticker_info(yf_symbol)

            # Try multiple price fields
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('lastPrice')

            if price is None:
                hist = MarketDataService._fetch_ticker_history(yf_symbol, period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]

            if price:
                return float(price)
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    @staticmethod
    def get_historical_data(symbol: str, market_type: str, period_months: int = 6) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data with retry logic.

        Args:
            symbol: Stock symbol
            market_type: Market type (US, HK, CN)
            period_months: Historical period in months

        Returns:
            DataFrame with OHLCV data, or None
        """
        try:
            yf_symbol = normalize_symbol(symbol, market_type)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_months * 30)

            hist = MarketDataService._fetch_ticker_history(yf_symbol, start=start_date, end=end_date)

            if hist.empty:
                return None

            return hist

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns:
            Dict with 'macd', 'signal', 'histogram'
        """
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return {
            'macd': macd.iloc[-1] if not macd.empty else None,
            'signal': signal_line.iloc[-1] if not signal_line.empty else None,
            'histogram': histogram.iloc[-1] if not histogram.empty else None
        }

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Dict:
        """
        Calculate Bollinger Bands.

        Returns:
            Dict with 'upper', 'middle', 'lower', 'bandwidth'
        """
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        bandwidth = (upper - lower) / sma

        current_price = df['Close'].iloc[-1]

        # Determine position
        if current_price >= upper.iloc[-1]:
            position = "upper"
        elif current_price <= lower.iloc[-1]:
            position = "lower"
        else:
            position = "middle"

        return {
            'upper': upper.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower.iloc[-1],
            'bandwidth': bandwidth.iloc[-1],
            'position': position
        }

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return df['Close'].rolling(window=period).mean()

    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> Dict:
        """
        Calculate volume ratio and detect squeeze.

        Returns:
            Dict with 'current_volume', 'avg_volume', 'ratio', 'squeeze'
        """
        current_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].rolling(window=period).mean().iloc[-1]
        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Squeeze: volume < 0.7 of average
        squeeze = ratio < 0.7

        return {
            'current_volume': int(current_vol) if not pd.isna(current_vol) else None,
            'avg_volume': float(avg_vol) if not pd.isna(avg_vol) else None,
            'ratio': round(ratio, 2),
            'squeeze': squeeze
        }

    @staticmethod
    def fetch_and_store_daily_metrics(asset_id: int, symbol: str, market_type: str) -> bool:
        """
        Fetch historical data, calculate all indicators, and store in database.
        This is the core engine that populates the AssetDailyMetric table.

        Args:
            asset_id: Asset ID in database
            symbol: Stock symbol
            market_type: Market type

        Returns:
            True if successful, False otherwise
        """
        from database import AssetDailyMetric, save_daily_metric, get_session
        from sqlmodel import select

        try:
            # Fetch 6 months of historical data
            hist = MarketDataService.get_historical_data(symbol, market_type, 6)
            if hist is None or hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return False

            # Calculate indicators
            rsi_series = MarketDataService.calculate_rsi(hist)
            sma_20_series = MarketDataService.calculate_sma(hist, 20)
            sma_50_series = MarketDataService.calculate_sma(hist, 50)
            sma_200_series = MarketDataService.calculate_sma(hist, 200)
            macd_data = MarketDataService.calculate_macd(hist)
            bb_data = MarketDataService.calculate_bollinger_bands(hist)
            vol_data = MarketDataService.calculate_volume_ratio(hist)

            # Get latest date
            latest_date = hist.index[-1].date()

            # Check if we already have data for this date
            with get_session() as session:
                existing = session.exec(
                    select(AssetDailyMetric).where(
                        AssetDailyMetric.asset_id == asset_id,
                        AssetDailyMetric.metric_date == latest_date
                    )
                ).first()

            if existing:
                logger.debug(f"Metrics already exist for {symbol} on {latest_date}")
                return True

            # Calculate signal
            price_vs_sma20 = hist['Close'].iloc[-1] - sma_20_series.iloc[-1]
            price_vs_sma200 = hist['Close'].iloc[-1] - sma_200_series.iloc[-1]
            golden_cross = sma_20_series.iloc[-1] > sma_200_series.iloc[-1]

            signal, confidence = calculate_signal(
                rsi=rsi_series.iloc[-1],
                macd_histogram=macd_data.get('histogram'),
                price_vs_sma20=price_vs_sma20,
                price_vs_sma200=price_vs_sma200,
                golden_cross=golden_cross,
                bollinger_position=bb_data.get('position'),
                volume_squeeze=vol_data.get('squeeze')
            )

            # Create metric record
            metric = AssetDailyMetric(
                asset_id=asset_id,
                metric_date=latest_date,
                close_price=float(hist['Close'].iloc[-1]),
                sma_20=float(sma_20_series.iloc[-1]) if not pd.isna(sma_20_series.iloc[-1]) else None,
                sma_50=float(sma_50_series.iloc[-1]) if not pd.isna(sma_50_series.iloc[-1]) else None,
                sma_200=float(sma_200_series.iloc[-1]) if not pd.isna(sma_200_series.iloc[-1]) else None,
                rsi_14=float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None,
                macd=float(macd_data['macd']) if macd_data['macd'] else None,
                macd_signal=float(macd_data['signal']) if macd_data['signal'] else None,
                macd_histogram=float(macd_data['histogram']) if macd_data['histogram'] else None,
                bollinger_upper=float(bb_data['upper']) if bb_data['upper'] else None,
                bollinger_middle=float(bb_data['middle']) if bb_data['middle'] else None,
                bollinger_lower=float(bb_data['lower']) if bb_data['lower'] else None,
                bollinger_bandwidth=float(bb_data['bandwidth']) if bb_data['bandwidth'] else None,
                volume=vol_data.get('current_volume'),
                volume_sma_20=vol_data.get('avg_volume'),
                volume_ratio=vol_data.get('ratio'),
                overall_signal=signal,
                confidence_score=confidence
            )

            save_daily_metric(metric)
            logger.info(f"Saved metrics for {symbol} on {latest_date}: {signal} (confidence: {confidence})")
            return True

        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            return False

    @staticmethod
    def get_stock_info(symbol: str, market_type: str) -> Optional[Dict]:
        """Fetch comprehensive stock information with retry logic."""
        try:
            yf_symbol = normalize_symbol(symbol, market_type)
            info = MarketDataService._fetch_ticker_info(yf_symbol)

            return {
                'symbol': symbol,
                'yf_symbol': yf_symbol,
                'name': info.get('longName') or info.get('shortName'),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
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
    def get_previous_close(symbol: str, market_type: str) -> Optional[float]:
        """
        Fetch the previous day's closing price for daily PnL calculation.

        Args:
            symbol: Stock symbol
            market_type: Market type (US, HK, CN)

        Returns:
            Previous close price as float, or None if unavailable
        """
        try:
            yf_symbol = normalize_symbol(symbol, market_type)
            info = MarketDataService._fetch_ticker_info(yf_symbol)

            # Try to get previous close from info
            prev_close = info.get('previousClose')
            if prev_close:
                return float(prev_close)

            # Fallback: fetch 2 days of history and get the second to last close
            hist = MarketDataService._fetch_ticker_history(yf_symbol, period="2d")
            if len(hist) >= 2:
                return float(hist['Close'].iloc[-2])

            logger.warning(f"Could not get previous close for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching previous close for {symbol}: {e}")
            return None

    @staticmethod
    @lru_cache(maxsize=32)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def get_exchange_rate(from_currency: str, to_currency: str = "USD") -> Optional[float]:
        """
        Fetch real-time exchange rate using yfinance with retry logic.

        Args:
            from_currency: Source currency code (e.g., "USD", "HKD", "CNY")
            to_currency: Target currency code (default: "USD")

        Returns:
            Exchange rate as float, or 1.0 if same currency, or None if unavailable

        Examples:
            get_exchange_rate("HKD", "USD") -> 0.128
            get_exchange_rate("CNY", "USD") -> 0.14
            get_exchange_rate("USD", "CNY") -> 7.2
        """
        if from_currency == to_currency:
            return 1.0

        try:
            # yfinance uses FX ticker format: FROMTO=X or FROM=X
            # For example: USDCNY=X, USDHKD=X
            ticker_symbol = f"{from_currency}{to_currency}=X"
            ticker = yf.Ticker(ticker_symbol)

            # Try to get the last price
            hist = ticker.history(period="1d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1]
                if rate > 0:
                    return float(rate)

            # Fallback: try info
            info = ticker.info
            rate = info.get('previousClose') or info.get('regularMarketPrice') or info.get('lastPrice')
            if rate:
                return float(rate)

            logger.warning(f"Could not get exchange rate for {ticker_symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching exchange rate {from_currency}->{to_currency}: {e}")
            return None

    @staticmethod
    def clear_cache():
        """Clear the LRU cache."""
        MarketDataService.get_current_price.cache_clear()
        MarketDataService.get_exchange_rate.cache_clear()
        logger.info("Market data cache cleared")
