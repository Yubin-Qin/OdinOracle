"""
Portfolio service for calculating holdings, PnL, and net worth.
Enhanced with currency conversion and real daily PnL calculation.
"""

import logging
from typing import List, Dict, Optional
from datetime import date

from database import get_all_assets, get_asset_by_id, get_transactions_by_asset, get_all_transactions, get_user_preferences
from services.market_data import MarketDataService

logger = logging.getLogger(__name__)


# Currency mapping for market types
MARKET_CURRENCY_MAP = {
    "US": "USD",
    "HK": "HKD",
    "CN": "CNY"
}


class PortfolioService:
    """
    Service for portfolio calculations and analysis.
    Handles multi-currency portfolios with conversion to base currency.
    """

    @staticmethod
    def _get_currency_rate(from_currency: str, to_currency: str) -> float:
        """
        Get exchange rate between two currencies.
        Returns 1.0 if same currency or if rate unavailable.
        """
        if from_currency == to_currency:
            return 1.0

        rate = MarketDataService.get_exchange_rate(from_currency, to_currency)
        return rate if rate else 1.0

    @staticmethod
    def get_asset_holdings(asset_id: int, base_currency: str = "USD") -> Optional[Dict]:
        """
        Calculate current holdings for a specific asset with currency conversion.

        Args:
            asset_id: Asset ID
            base_currency: Base currency for value calculation (default: "USD")

        Returns:
            Dictionary with holding details including PnL and converted values
        """
        asset = get_asset_by_id(asset_id)
        if not asset:
            return None

        transactions = get_transactions_by_asset(asset_id)
        if not transactions:
            return None

        # Get the asset's currency
        asset_currency = MARKET_CURRENCY_MAP.get(asset.market_type, "USD")

        # Calculate quantity and cost basis (FIFO)
        transactions_sorted = sorted(transactions, key=lambda x: x.transaction_date)

        total_quantity = 0.0
        total_cost = 0.0
        buy_queue = []  # Queue for FIFO: (quantity, price)

        for tx in transactions_sorted:
            if tx.transaction_type == 'buy':
                buy_queue.append((tx.quantity, tx.price))
                total_quantity += tx.quantity
                total_cost += tx.quantity * tx.price
            elif tx.transaction_type == 'sell':
                # Sell from FIFO queue
                remaining_to_sell = tx.quantity
                while remaining_to_sell > 0 and buy_queue:
                    qty, price = buy_queue[0]
                    if qty <= remaining_to_sell:
                        buy_queue.pop(0)
                        total_cost -= qty * price
                        remaining_to_sell -= qty
                    else:
                        buy_queue[0] = (qty - remaining_to_sell, price)
                        total_cost -= remaining_to_sell * price
                        remaining_to_sell = 0
                total_quantity -= tx.quantity

        # Get current and previous prices
        current_price = MarketDataService.get_current_price(asset.symbol, asset.market_type)
        if current_price is None:
            current_price = 0.0

        previous_close = MarketDataService.get_previous_close(asset.symbol, asset.market_type)

        # Calculate average cost (for remaining shares only)
        avg_cost = total_cost / total_quantity if total_quantity > 0 else 0.0

        # Calculate current value and PnL in asset currency
        current_value_asset_ccy = total_quantity * current_price if total_quantity > 0 else 0.0
        pnl_asset_ccy = current_value_asset_ccy - total_cost if total_quantity > 0 else 0.0
        pnl_pct = (pnl_asset_ccy / total_cost * 100) if total_cost > 0 else 0.0

        # Calculate daily PnL (real calculation using previous close)
        daily_pnl_asset_ccy = 0.0
        if previous_close and total_quantity > 0:
            daily_pnl_asset_ccy = (current_price - previous_close) * total_quantity

        # Get exchange rate
        exchange_rate = PortfolioService._get_currency_rate(asset_currency, base_currency)

        # Convert values to base currency
        current_value_base_ccy = current_value_asset_ccy * exchange_rate
        total_cost_base_ccy = total_cost * exchange_rate
        pnl_base_ccy = pnl_asset_ccy * exchange_rate
        daily_pnl_base_ccy = daily_pnl_asset_ccy * exchange_rate

        return {
            'asset_id': asset.id,
            'symbol': asset.symbol,
            'name': asset.name,
            'market_type': asset.market_type,
            'currency': asset_currency,
            'quantity': round(total_quantity, 4),
            'avg_cost': round(avg_cost, 2),
            'total_cost': round(total_cost, 2),  # In asset currency
            'total_cost_base': round(total_cost_base_ccy, 2),  # In base currency
            'current_price': round(current_price, 2) if current_price else 0.0,
            'previous_close': round(previous_close, 2) if previous_close else None,
            'current_value': round(current_value_asset_ccy, 2),  # In asset currency
            'current_value_base': round(current_value_base_ccy, 2),  # In base currency
            'pnl': round(pnl_asset_ccy, 2),  # In asset currency
            'pnl_base': round(pnl_base_ccy, 2),  # In base currency
            'pnl_pct': round(pnl_pct, 2),
            'daily_pnl': round(daily_pnl_asset_ccy, 2),  # In asset currency
            'daily_pnl_base': round(daily_pnl_base_ccy, 2),  # In base currency
            'exchange_rate': round(exchange_rate, 4) if exchange_rate != 1.0 else None,
        }

    @staticmethod
    def calculate_net_worth(base_currency: Optional[str] = None) -> Dict:
        """
        Calculate total portfolio net worth and summary statistics.
        Converts all asset values to base currency for accurate totals.

        Args:
            base_currency: Base currency for calculation (default: from user preferences or "USD")

        Returns:
            Dictionary with portfolio summary in base currency
        """
        # Get base currency from user preferences if not specified
        if base_currency is None:
            prefs = get_user_preferences()
            base_currency = prefs.base_currency if prefs else "USD"

        assets = get_all_assets()
        holdings = []
        total_value_base_ccy = 0.0
        total_cost_base_ccy = 0.0
        total_daily_pnl_base_ccy = 0.0

        for asset in assets:
            holding = PortfolioService.get_asset_holdings(asset.id, base_currency)
            if holding and holding['quantity'] > 0:
                holdings.append(holding)
                total_value_base_ccy += holding['current_value_base']
                total_cost_base_ccy += holding['total_cost_base']
                total_daily_pnl_base_ccy += holding.get('daily_pnl_base', 0.0)

        total_pnl = total_value_base_ccy - total_cost_base_ccy
        total_pnl_pct = (total_pnl / total_cost_base_ccy * 100) if total_cost_base_ccy > 0 else 0.0

        return {
            'base_currency': base_currency,
            'total_value': round(total_value_base_ccy, 2),
            'total_cost': round(total_cost_base_ccy, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'daily_pnl': round(total_daily_pnl_base_ccy, 2),
            'holdings': holdings
        }

    @staticmethod
    def get_top_holdings(limit: int = 5) -> List:
        """Get top holdings by current value (in base currency)."""
        portfolio = PortfolioService.calculate_net_worth()
        holdings = portfolio['holdings']
        holdings.sort(key=lambda x: x['current_value_base'], reverse=True)
        return holdings[:limit]

    @staticmethod
    def get_asset_by_symbol(symbol: str) -> Optional:
        """Find an asset by symbol (case-insensitive)."""
        assets = get_all_assets()
        for asset in assets:
            if asset.symbol.upper() == symbol.upper():
                return asset
        return None

    @staticmethod
    def get_holdings_summary() -> List[Dict]:
        """
        Get a summary of all holdings for display.

        Returns:
            List of holding dictionaries for table display
        """
        portfolio = PortfolioService.calculate_net_worth()
        return portfolio.get('holdings', [])

    @staticmethod
    def calculate_daily_pnl() -> float:
        """
        Calculate daily PnL by comparing current prices to previous close.
        Uses real previous close prices from market data.

        Returns:
            Daily PnL amount in base currency
        """
        portfolio = PortfolioService.calculate_net_worth()
        return portfolio.get('daily_pnl', 0.0)
