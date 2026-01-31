"""
LangChain tools for stock price fetching and news search.
"""

from langchain_core.tools import tool
import yfinance as yf
from duckduckgo_search import DDGS
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_yfinance_symbol(symbol: str, market_type: str) -> str:
    """
    Convert a symbol to yfinance format based on market type.

    Args:
        symbol: Stock symbol (e.g., "NVDA", "0700", "600519")
        market_type: Market type ("US", "HK", "CN")

    Returns:
        Properly formatted yfinance symbol
    """
    if market_type == "US":
        return symbol  # US stocks don't need suffix
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
        return symbol


@tool
def get_stock_price(symbol: str, market_type: str = "US") -> str:
    """
    Get the current stock price and basic info for a given symbol.

    Args:
        symbol: Stock symbol (e.g., "NVDA", "0700", "600519")
        market_type: Market type - "US", "HK", or "CN"

    Returns:
        String with stock price information
    """
    try:
        yf_symbol = get_yfinance_symbol(symbol, market_type)
        logger.info(f"Fetching data for {yf_symbol}")

        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        # Get current price from different possible fields
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('lastPrice')

        if current_price is None:
            # Try to get from history
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            else:
                return f"Could not retrieve price for {symbol}"

        previous_close = info.get('previousClose')
        day_high = info.get('dayHigh')
        day_low = info.get('dayLow')
        volume = info.get('volume')

        result = f"Stock: {symbol} ({yf_symbol})\n"
        result += f"Current Price: ${current_price:.2f}\n"

        if previous_close:
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
            result += f"Change: ${change:+.2f} ({change_pct:+.2f}%)\n"

        if day_high:
            result += f"Day High: ${day_high:.2f}\n"
        if day_low:
            result += f"Day Low: ${day_low:.2f}\n"
        if volume:
            result += f"Volume: {volume:,}\n"

        return result

    except Exception as e:
        logger.error(f"Error fetching stock price for {symbol}: {e}")
        return f"Error fetching stock price for {symbol}: {str(e)}"


@tool
def search_stock_news(query: str, max_results: int = 5) -> str:
    """
    Search for recent news about stocks or markets using DuckDuckGo.

    Args:
        query: Search query (e.g., "NVDA stock news", "Apple earnings")
        max_results: Maximum number of news results to return (default: 5)

    Returns:
        String with news summaries and links
    """
    try:
        logger.info(f"Searching news for query: {query}")
        ddgs = DDGS()

        # Search for news
        results = ddgs.news(query, max_results=max_results)

        if not results:
            return f"No news found for query: {query}"

        output = f"News results for '{query}':\n\n"

        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            source = result.get('source', 'Unknown')
            date = result.get('date', 'Recent')

            output += f"{i}. {title}\n"
            output += f"   Source: {source} | {date}\n"
            output += f"   URL: {url}\n\n"

        return output

    except Exception as e:
        logger.error(f"Error searching news: {e}")
        return f"Error searching news: {str(e)}"


@tool
def get_stock_info(symbol: str, market_type: str = "US") -> str:
    """
    Get detailed company information for a stock.

    Args:
        symbol: Stock symbol (e.g., "NVDA", "0700", "600519")
        market_type: Market type - "US", "HK", or "CN"

    Returns:
        String with detailed company information
    """
    try:
        yf_symbol = get_yfinance_symbol(symbol, market_type)
        logger.info(f"Fetching info for {yf_symbol}")

        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        company_name = info.get('longName') or info.get('shortName') or symbol
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap')
        dividend_yield = info.get('dividendYield')
        pe_ratio = info.get('trailingPE')
        eps = info.get('trailingEps')

        result = f"Company: {company_name}\n"
        result += f"Symbol: {symbol}\n"
        result += f"Sector: {sector}\n"
        result += f"Industry: {industry}\n"

        if market_cap:
            result += f"Market Cap: ${market_cap:,.0f}\n"
        if pe_ratio:
            result += f"P/E Ratio: {pe_ratio:.2f}\n"
        if eps:
            result += f"EPS: ${eps:.2f}\n"
        if dividend_yield:
            result += f"Dividend Yield: {dividend_yield * 100:.2f}%\n"

        # Business summary
        summary = info.get('longBusinessSummary')
        if summary:
            result += f"\nBusiness Summary:\n{summary[:500]}...\n"

        return result

    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {e}")
        return f"Error fetching stock info for {symbol}: {str(e)}"


# Export all tools for LangChain
all_tools = [get_stock_price, search_stock_news, get_stock_info]


if __name__ == "__main__":
    # Test the tools
    print("Testing Stock Price Tool (US):")
    print(get_stock_price.invoke({"symbol": "NVDA", "market_type": "US"}))
    print("\n" + "="*60 + "\n")

    print("Testing Stock Price Tool (HK):")
    print(get_stock_price.invoke({"symbol": "0700", "market_type": "HK"}))
    print("\n" + "="*60 + "\n")

    print("Testing News Search:")
    print(search_stock_news.invoke({"query": "NVDA stock", "max_results": 3}))
