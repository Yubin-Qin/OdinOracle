"""
Portfolio service for calculating holdings, PnL, and net worth.
"""

import logging
from typing import List, Dict, Optional
from datetime import date

from database import get_all_assets, get_asset_by_id, get_transactions_by_asset, get_all_transactions
from services.market_data import MarketDataService

logger = logging.getLogger(__name__)


class PortfolioService:
    """
    Service for portfolio calculations and analysis.
    """

    @staticmethod
    def get_asset_holdings(asset_id: int) -> Dict:
        """
        Calculate current holdings for a specific asset.

        Args:
            asset_id: Asset ID

        Returns:
            Dictionary with:
                - asset_id: int
                - symbol: str
                - name: str
                - market_type: str
                - quantity: float (total shares held)
                - avg_cost: float (average cost per share)
                - total_cost: float (total investment)
                - current_price: float
                - current_value: float
                - pnl: float (profit/loss)
                - pnl_pct: float (profit/loss percentage)
        """
        asset = get_asset_by_id(asset_id)
        if not asset:
            return None

        transactions = get_transactions_by_asset(asset_id)
        if not transactions:
            return {
                'asset_id': asset.id,
                'symbol': asset.symbol,
                'name': asset.name,
                'market_type': asset.market_type,
                'quantity': 0.0,
                'avg_cost': 0.0,
                'total_cost': 0.0,
                'current_price': 0.0,
                'current_value': 0.0,
                'pnl': 0.0,
                'pnl_pct': 0.0
            }

        # Calculate quantity and average cost
        total_quantity = 0.0
        total_cost = 0.0

        for tx in transactions:
            if tx.transaction_type == 'buy':
                total_quantity += tx.quantity
                total_cost += tx.quantity * tx.price
            elif tx.transaction_type == 'sell':
                # Reduce quantity (FIFO or average cost - using simplified approach)
                total_quantity -= tx.quantity
                # Cost basis remains for sold shares (simplified)

        # Get current price
        current_price = MarketDataService.get_current_price(asset.symbol, asset.market_type)
        if current_price is None:
            current_price = 0.0

        # Calculate average cost
        avg_cost = total_cost / total_quantity if total_quantity > 0 else 0.0

        # Calculate current value and PnL
        current_value = total_quantity * current_price if total_quantity > 0 else 0.0
        pnl = current_value - total_cost if total_quantity > 0 else 0.0
        pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0.0

        return {
            'asset_id': asset.id,
            'symbol': asset.symbol,
            'name': asset.name,
            'market_type': asset.market_type,
            'quantity': round(total_quantity, 4),
            'avg_cost': round(avg_cost, 2),
            'total_cost': round(total_cost, 2),
            'current_price': round(current_price, 2) if current_price else 0.0,
            'current_value': round(current_value, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2)
        }

    @staticmethod
    def calculate_net_worth() -> Dict:
        """
        Calculate total portfolio net worth and summary statistics.

        Returns:
            Dictionary with:
                - total_value: float (total current value)
                - total_cost: float (total investment)
                - total_pnl: float (total profit/loss)
                - total_pnl_pct: float (total PnL percentage)
                - holdings: List[Dict] (list of asset holdings)
        """
        assets = get_all_assets()
        holdings = []
        total_value = 0.0
        total_cost = 0.0

        for asset in assets:
            holding = PortfolioService.get_asset_holdings(asset.id)
            if holding and holding['quantity'] > 0:
                holdings.append(holding)
                total_value += holding['current_value']
                total_cost += holding['total_cost']

        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

        return {
            'total_value': round(total_value, 2),
            'total_cost': round(total_cost, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'holdings': holdings
        }

    @staticmethod
    def get_top_holdings(limit: int = 5) -> List:
        """
        Get top holdings by current value.

        Args:
            limit: Maximum number of holdings to return

        Returns:
            List of holding dictionaries sorted by value
        """
        portfolio = PortfolioService.calculate_net_worth()
        holdings = portfolio['holdings']
        # Sort by current value descending
        holdings.sort(key=lambda x: x['current_value'], reverse=True)
        return holdings[:limit]

    @staticmethod
    def get_asset_by_symbol(symbol: str) -> Optional:
        """
        Find an asset by symbol (case-insensitive).

        Args:
            symbol: Stock symbol

        Returns:
            Asset object or None
        """
        assets = get_all_assets()
        for asset in assets:
            if asset.symbol.upper() == symbol.upper():
                return asset
        return None
