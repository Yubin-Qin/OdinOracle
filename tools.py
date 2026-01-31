"""
LangChain tools for stock price fetching and news search.
Refactored to use services layer.
"""

from langchain_core.tools import tool
from typing import Optional
import logging

from services.market_data import MarketDataService
from services.common import normalize_symbol

logger = logging.getLogger(__name__)


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
        logger.info(f"Fetching price for {symbol}")

        # Get stock info from MarketDataService
        info = MarketDataService.get_stock_info(symbol, market_type)

        if not info:
            return f"Could not retrieve information for {symbol}"

        current_price = info.get('current_price')
        if current_price is None:
            return f"Could not retrieve price for {symbol}"

        previous_close = info.get('previous_close')
        day_high = info.get('day_high')
        day_low = info.get('day_low')
        volume = info.get('volume')

        result = f"Stock: {symbol} ({info.get('yf_symbol', symbol)})\n"
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
        from ddgs import DDGS

        ddgs = DDGS()
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
        logger.info(f"Fetching info for {symbol}")

        # Get stock info from MarketDataService
        info = MarketDataService.get_stock_info(symbol, market_type)

        if not info:
            return f"Could not retrieve information for {symbol}"

        company_name = info.get('name') or symbol
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('market_cap')
        dividend_yield = info.get('dividend_yield')
        pe_ratio = info.get('pe_ratio')
        eps = info.get('eps')

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
        summary = info.get('summary')
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
