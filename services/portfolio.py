"""
Portfolio service for calculating holdings, PnL, and net worth.
Enhanced with currency conversion, real daily PnL calculation, and risk metrics.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import date
from dataclasses import dataclass

import numpy as np
import pandas as pd

from database import get_all_assets, get_asset_by_id, get_transactions_by_asset, get_all_transactions, get_user_preferences
from services.market_data import MarketDataService

logger = logging.getLogger(__name__)


# Currency mapping for market types
MARKET_CURRENCY_MAP = {
    "US": "USD",
    "HK": "HKD",
    "CN": "CNY"
}


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics for portfolio analysis."""
    portfolio_sharpe_ratio: float
    portfolio_volatility: float  # Annualized
    portfolio_beta: float  # vs benchmark (would need market data)
    var_95: float  # Value at Risk (95% confidence)
    max_drawdown_pct: float
    correlation_matrix: Optional[pd.DataFrame] = None
    concentration_risk: Dict[str, float] = None  # Position concentration warnings


@dataclass
class PortfolioAnalytics:
    """Comprehensive portfolio analytics including holdings and risk metrics."""
    holdings: List[Dict]
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float
    base_currency: str
    risk_metrics: RiskMetrics


class PortfolioService:
    """
    Service for portfolio calculations and analysis.
    Handles multi-currency portfolios with conversion to base currency.
    Enhanced with risk management metrics (Sharpe Ratio, Correlation Matrix).
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
    def calculate_net_worth(base_currency: Optional[str] = None, use_batch_fetch: bool = True) -> Dict:
        """
        Calculate total portfolio net worth and summary statistics.
        Converts all asset values to base currency for accurate totals.

        Args:
            base_currency: Base currency for calculation (default: from user preferences or "USD")
            use_batch_fetch: If True, use parallel batch fetching for prices (faster)

        Returns:
            Dictionary with portfolio summary in base currency
        """
        # Get base currency from user preferences if not specified
        if base_currency is None:
            prefs = get_user_preferences()
            base_currency = prefs.base_currency if prefs else "USD"

        assets = get_all_assets()

        # Use optimized batch fetching if enabled
        if use_batch_fetch and len(assets) > 1:
            return PortfolioService._calculate_net_worth_batch(assets, base_currency)

        # Fall back to sequential processing
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
    def _calculate_net_worth_batch(assets: List, base_currency: str) -> Dict:
        """
        Optimized portfolio calculation using batch price fetching.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # First, calculate all holdings data except current prices
        holdings_data = []
        asset_price_map = {}

        for asset in assets:
            transactions = get_transactions_by_asset(asset.id)
            if not transactions:
                continue

            # Calculate quantity and cost basis (FIFO)
            transactions_sorted = sorted(transactions, key=lambda x: x.transaction_date)
            total_quantity = 0.0
            total_cost = 0.0
            buy_queue = []

            for tx in transactions_sorted:
                if tx.transaction_type == 'buy':
                    buy_queue.append((tx.quantity, tx.price))
                    total_quantity += tx.quantity
                    total_cost += tx.quantity * tx.price
                elif tx.transaction_type == 'sell':
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

            if total_quantity > 0:
                asset_currency = MARKET_CURRENCY_MAP.get(asset.market_type, "USD")
                holdings_data.append({
                    'asset': asset,
                    'quantity': total_quantity,
                    'total_cost': total_cost,
                    'currency': asset_currency
                })
                asset_price_map[asset.symbol] = (asset.market_type, asset_currency)

        # Batch fetch all prices in parallel
        if holdings_data:
            price_tuples = [(h['asset'].symbol, h['asset'].market_type) for h in holdings_data]
            prices = MarketDataService.get_current_prices_batch(price_tuples, max_workers=5)

            # Also fetch previous closes in batch
            prev_closes = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_symbol = {
                    executor.submit(
                        MarketDataService.get_previous_close,
                        h['asset'].symbol,
                        h['asset'].market_type
                    ): h['asset'].symbol
                    for h in holdings_data
                }
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        prev_closes[symbol] = future.result()
                    except Exception:
                        prev_closes[symbol] = None

            # Calculate final values
            holdings = []
            total_value_base_ccy = 0.0
            total_cost_base_ccy = 0.0
            total_daily_pnl_base_ccy = 0.0

            for h in holdings_data:
                asset = h['asset']
                quantity = h['quantity']
                total_cost = h['total_cost']
                asset_currency = h['currency']

                current_price = prices.get(asset.symbol, 0.0) or 0.0
                previous_close = prev_closes.get(asset.symbol)

                avg_cost = total_cost / quantity if quantity > 0 else 0.0
                current_value_asset_ccy = quantity * current_price
                pnl_asset_ccy = current_value_asset_ccy - total_cost
                pnl_pct = (pnl_asset_ccy / total_cost * 100) if total_cost > 0 else 0.0

                daily_pnl_asset_ccy = 0.0
                if previous_close and quantity > 0:
                    daily_pnl_asset_ccy = (current_price - previous_close) * quantity

                exchange_rate = PortfolioService._get_currency_rate(asset_currency, base_currency)

                current_value_base_ccy = current_value_asset_ccy * exchange_rate
                total_cost_base_ccy_local = total_cost * exchange_rate
                pnl_base_ccy = pnl_asset_ccy * exchange_rate
                daily_pnl_base_ccy = daily_pnl_asset_ccy * exchange_rate

                holdings.append({
                    'asset_id': asset.id,
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'market_type': asset.market_type,
                    'currency': asset_currency,
                    'quantity': round(quantity, 4),
                    'avg_cost': round(avg_cost, 2),
                    'total_cost': round(total_cost, 2),
                    'total_cost_base': round(total_cost_base_ccy_local, 2),
                    'current_price': round(current_price, 2),
                    'previous_close': round(previous_close, 2) if previous_close else None,
                    'current_value': round(current_value_asset_ccy, 2),
                    'current_value_base': round(current_value_base_ccy, 2),
                    'pnl': round(pnl_asset_ccy, 2),
                    'pnl_base': round(pnl_base_ccy, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'daily_pnl': round(daily_pnl_asset_ccy, 2),
                    'daily_pnl_base': round(daily_pnl_base_ccy, 2),
                    'exchange_rate': round(exchange_rate, 4) if exchange_rate != 1.0 else None,
                })

                total_value_base_ccy += current_value_base_ccy
                total_cost_base_ccy += total_cost_base_ccy_local
                total_daily_pnl_base_ccy += daily_pnl_base_ccy

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

        return {
            'base_currency': base_currency,
            'total_value': 0.0,
            'total_cost': 0.0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'daily_pnl': 0.0,
            'holdings': []
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

    # ==================== Risk Management Metrics ====================

    @staticmethod
    def calculate_sharpe_ratio(holdings: List[Dict],
                               lookback_days: int = 30,
                               risk_free_rate: float = 0.02) -> float:
        """
        Calculate portfolio Sharpe Ratio (risk-adjusted return).

        The Sharpe Ratio measures the excess return per unit of risk (volatility).
        Higher is better. Typically:
        - < 1: Sub-optimal
        - 1-2: Good
        - 2-3: Very good
        - > 3: Excellent

        Args:
            holdings: List of holding dictionaries with historical price data
            lookback_days: Number of days for calculation (default: 30)
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Annualized Sharpe ratio
        """
        if len(holdings) < 1:
            return 0.0

        try:
            # For each holding, we need historical returns
            # Since we don't have full historical data in holdings,
            # we'll estimate using the available metrics

            # Collect daily returns from historical metrics if available
            daily_returns = []

            for holding in holdings:
                symbol = holding['symbol']
                market_type = holding['market_type']

                # Fetch historical data
                hist_data = MarketDataService.get_historical_data(
                    symbol, market_type, period_months=lookback_days // 30 + 1
                )

                if hist_data is not None and len(hist_data) > 1:
                    # Calculate daily returns
                    prices = hist_data['Close'].values
                    returns = np.diff(prices) / prices[:-1]

                    # Weight by portfolio allocation
                    weight = holding['current_value_base'] / sum(h['current_value_base'] for h in holdings)
                    weighted_returns = returns * weight

                    if len(daily_returns) == 0:
                        daily_returns = weighted_returns.tolist()
                    else:
                        # Align lengths and add
                        min_len = min(len(daily_returns), len(weighted_returns))
                        daily_returns = [daily_returns[i] + weighted_returns[i] for i in range(min_len)]

            if len(daily_returns) < 2:
                logger.warning("Insufficient historical data for Sharpe ratio calculation")
                return 0.0

            returns_array = np.array(daily_returns)

            # Calculate mean and standard deviation
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return == 0 or np.isnan(std_return):
                return 0.0

            # Daily risk-free rate
            daily_risk_free = risk_free_rate / 252

            # Annualized Sharpe ratio
            sharpe = ((mean_return - daily_risk_free) / std_return) * np.sqrt(252)

            return round(sharpe, 2)

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    @staticmethod
    def calculate_correlation_matrix(holdings: List[Dict],
                                     lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix between portfolio assets.

        Correlation ranges from -1 to 1:
        - 1: Perfect positive correlation (assets move together)
        - 0: No correlation
        - -1: Perfect negative correlation (assets move opposite)

        High correlation (>0.8) between assets indicates concentration risk.

        Args:
            holdings: List of holding dictionaries
            lookback_days: Number of days for calculation (default: 30)

        Returns:
            Correlation matrix DataFrame or None if insufficient data
        """
        if len(holdings) < 2:
            return None

        try:
            # Collect price history for each holding
            price_data = {}

            for holding in holdings:
                symbol = holding['symbol']
                market_type = holding['market_type']

                # Fetch historical data
                hist_data = MarketDataService.get_historical_data(
                    symbol, market_type, period_months=lookback_days // 30 + 1
                )

                if hist_data is not None and len(hist_data) > 5:
                    price_data[symbol] = hist_data['Close']

            if len(price_data) < 2:
                logger.warning("Insufficient price data for correlation calculation")
                return None

            # Create DataFrame with aligned dates
            df = pd.DataFrame(price_data)

            # Calculate returns
            returns_df = df.pct_change().dropna()

            if len(returns_df) < 5:
                return None

            # Calculate correlation matrix
            corr_matrix = returns_df.corr()

            return corr_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None

    @staticmethod
    def check_concentration_risk(holdings: List[Dict],
                                 max_single_position_pct: float = 20.0,
                                 max_sector_pct: float = 40.0) -> Dict[str, any]:
        """
        Check for concentration risk in the portfolio.

        Args:
            holdings: List of holding dictionaries
            max_single_position_pct: Max % for single position (default: 20%)
            max_sector_pct: Max % for single sector (default: 40%)

        Returns:
            Dictionary with concentration warnings and metrics
        """
        if not holdings:
            return {
                'warnings': [],
                'single_position_risk': {},
                'total_positions': 0,
                'largest_position_pct': 0.0
            }

        total_value = sum(h['current_value_base'] for h in holdings)

        if total_value == 0:
            return {
                'warnings': [],
                'single_position_risk': {},
                'total_positions': len(holdings),
                'largest_position_pct': 0.0
            }

        warnings = []
        position_risks = {}

        # Check single position concentration
        sorted_holdings = sorted(holdings, key=lambda x: x['current_value_base'], reverse=True)
        largest_position_pct = 0.0

        for holding in sorted_holdings:
            position_pct = (holding['current_value_base'] / total_value) * 100
            position_risks[holding['symbol']] = {
                'value': holding['current_value_base'],
                'percentage': round(position_pct, 2)
            }

            if position_pct > largest_position_pct:
                largest_position_pct = position_pct

            if position_pct > max_single_position_pct:
                warnings.append({
                    'type': 'single_position',
                    'severity': 'high' if position_pct > 30 else 'medium',
                    'symbol': holding['symbol'],
                    'message': f"{holding['symbol']} is {position_pct:.1f}% of portfolio (max {max_single_position_pct}%)"
                })

        # Check for too few positions (diversification risk)
        if len(holdings) < 3:
            warnings.append({
                'type': 'diversification',
                'severity': 'medium',
                'message': f"Portfolio has only {len(holdings)} position(s). Consider diversifying."
            })

        return {
            'warnings': warnings,
            'single_position_risk': position_risks,
            'total_positions': len(holdings),
            'largest_position_pct': round(largest_position_pct, 2),
            'max_recommended_position_pct': max_single_position_pct
        }

    @staticmethod
    def calculate_risk_metrics(base_currency: Optional[str] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio.

        Args:
            base_currency: Base currency for calculation

        Returns:
            RiskMetrics with Sharpe ratio, volatility, VaR, and correlation data
        """
        portfolio = PortfolioService.calculate_net_worth(base_currency)
        holdings = portfolio.get('holdings', [])

        if not holdings:
            return RiskMetrics(
                portfolio_sharpe_ratio=0.0,
                portfolio_volatility=0.0,
                portfolio_beta=1.0,
                var_95=0.0,
                max_drawdown_pct=0.0,
                correlation_matrix=None,
                concentration_risk=None
            )

        # Calculate Sharpe ratio
        sharpe = PortfolioService.calculate_sharpe_ratio(holdings)

        # Calculate correlation matrix
        corr_matrix = PortfolioService.calculate_correlation_matrix(holdings)

        # Check concentration risk
        concentration = PortfolioService.check_concentration_risk(holdings)

        # Estimate portfolio volatility (simplified)
        volatility = 0.0
        try:
            if corr_matrix is not None:
                # Use average of correlations as rough volatility proxy
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                avg_correlation = np.mean(np.abs(corr_values)) if len(corr_values) > 0 else 0
                volatility = avg_correlation * 20  # Rough annualized estimate
        except:
            pass

        # Estimate VaR (95% confidence)
        # Simplified: 1.645 * volatility * portfolio_value
        var_95 = 1.645 * (volatility / 100) * portfolio['total_value']

        return RiskMetrics(
            portfolio_sharpe_ratio=sharpe,
            portfolio_volatility=round(volatility, 2),
            portfolio_beta=1.0,  # Would need market comparison
            var_95=round(var_95, 2),
            max_drawdown_pct=0.0,  # Would need historical tracking
            correlation_matrix=corr_matrix,
            concentration_risk=concentration
        )

    @staticmethod
    def get_portfolio_analytics(base_currency: Optional[str] = None) -> PortfolioAnalytics:
        """
        Get comprehensive portfolio analytics including risk metrics.

        Args:
            base_currency: Base currency for calculation

        Returns:
            PortfolioAnalytics with holdings, value, and risk metrics
        """
        portfolio = PortfolioService.calculate_net_worth(base_currency)
        risk_metrics = PortfolioService.calculate_risk_metrics(base_currency)

        return PortfolioAnalytics(
            holdings=portfolio['holdings'],
            total_value=portfolio['total_value'],
            total_cost=portfolio['total_cost'],
            total_pnl=portfolio['total_pnl'],
            total_pnl_pct=portfolio['total_pnl_pct'],
            daily_pnl=portfolio['daily_pnl'],
            base_currency=portfolio['base_currency'],
            risk_metrics=risk_metrics
        )

    @staticmethod
    def format_risk_report(analytics: PortfolioAnalytics) -> str:
        """
        Format a risk report for display.

        Args:
            analytics: PortfolioAnalytics object

        Returns:
            Formatted report string
        """
        risk = analytics.risk_metrics

        lines = [
            "=" * 60,
            "PORTFOLIO RISK REPORT",
            "=" * 60,
            "",
            "RISK METRICS",
            "-" * 40,
            f"Sharpe Ratio:              {risk.portfolio_sharpe_ratio:.2f}",
            f"Portfolio Volatility:      {risk.portfolio_volatility:.2f}%",
            f"Value at Risk (95%):       ${risk.var_95:,.2f}",
            "",
            "CONCENTRATION ANALYSIS",
            "-" * 40,
        ]

        if risk.concentration_risk:
            conc = risk.concentration_risk
            lines.append(f"Total Positions:           {conc['total_positions']}")
            lines.append(f"Largest Position:          {conc['largest_position_pct']:.1f}%")
            lines.append(f"Max Recommended:           {conc['max_recommended_position_pct']:.1f}%")

            if conc['warnings']:
                lines.append("")
                lines.append("WARNINGS:")
                for warning in conc['warnings']:
                    emoji = "🔴" if warning['severity'] == 'high' else "🟡"
                    lines.append(f"  {emoji} {warning['message']}")
            else:
                lines.append("  ✓ No concentration risks detected")

        if risk.correlation_matrix is not None:
            lines.append("")
            lines.append("CORRELATION MATRIX")
            lines.append("-" * 40)
            lines.append(risk.correlation_matrix.to_string())
            lines.append("")
            lines.append("Note: High correlation (>0.8) indicates concentration risk")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
