"""
Backtesting module for OdinOracle.
Provides historical signal validation and strategy simulation using AssetDailyMetric data.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass
from enum import Enum

import numpy as np

from models import Asset, AssetDailyMetric
from repositories import MetricRepository
from services.common import SignalInput, calculate_signal_from_input, signal_to_numeric

logger = logging.getLogger(__name__)


class SignalAction(Enum):
    """Trading action based on signal."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class Trade:
    """Represents a single trade in the backtest."""
    date: date
    action: SignalAction
    price: float
    signal: str
    confidence: int
    quantity: float
    value: float
    reason: str


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Configuration
    asset_symbol: str
    start_date: date
    end_date: date
    initial_capital: float

    # Performance Metrics
    total_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float  # Outperformance vs benchmark
    max_drawdown_pct: float
    sharpe_ratio: float

    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_trade_return_pct: float

    # Position Tracking
    final_capital: float
    final_shares: float
    peak_capital: float

    # Trade History
    trades: List[Trade]
    equity_curve: List[Dict]  # Daily equity values

    # Signal Performance
    signal_accuracy: Dict[str, float]  # Accuracy by signal type


class SignalBacktester:
    """
    Backtesting engine for the Factor Pack signal system.
    Simulates trading based on historical signals and calculates performance metrics.
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 position_size_pct: float = 0.95,  # Use 95% of capital per trade
                 commission_rate: float = 0.001):  # 0.1% commission
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting capital for the simulation
            position_size_pct: Percentage of capital to use per trade
            commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission_rate = commission_rate

    def run_backtest(self,
                     asset: Asset,
                     start_date: Optional[date] = None,
                     end_date: Optional[date] = None,
                     use_historical_metrics: bool = True) -> Optional[BacktestResult]:
        """
        Run a backtest for a given asset over a time period.

        Args:
            asset: Asset to backtest
            start_date: Start date (default: 90 days ago)
            end_date: End date (default: today)
            use_historical_metrics: If True, use stored AssetDailyMetric data.
                                   If False, fetch and calculate on the fly.

        Returns:
            BacktestResult with performance metrics, or None if insufficient data
        """
        # Set default dates
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=90)

        # Fetch historical metrics
        metrics = MetricRepository.get_date_range(asset.id, start_date, end_date)

        if len(metrics) < 10:
            logger.warning(f"Insufficient historical data for {asset.symbol}. Found {len(metrics)} days.")
            return None

        # Sort by date ascending
        metrics = sorted(metrics, key=lambda m: m.metric_date)

        # Run simulation
        return self._simulate_strategy(asset, metrics)

    def _simulate_strategy(self, asset: Asset, metrics: List[AssetDailyMetric]) -> BacktestResult:
        """
        Simulate the trading strategy on historical data.

        Strategy:
        - STRONG_BUY: Go all-in (position_size_pct of capital)
        - BUY: Go 50% in
        - HOLD: Maintain current position
        - SELL: Sell 50% of holdings
        - STRONG_SELL: Sell all holdings
        """
        capital = self.initial_capital
        shares = 0.0
        peak_capital = capital
        max_drawdown = 0.0

        trades: List[Trade] = []
        equity_curve: List[Dict] = []

        # Track signal accuracy (predicted vs actual next-day return)
        signal_predictions: List[Tuple[str, float]] = []  # (signal, next_day_return)

        for i, metric in enumerate(metrics):
            current_price = metric.close_price
            current_date = metric.metric_date

            # Calculate signal from metric data
            signal_input = SignalInput.from_metric(metric)
            signal_result = calculate_signal_from_input(signal_input)

            # Store prediction for accuracy calculation (if we have next day)
            if i > 0:
                prev_return = (current_price - metrics[i-1].close_price) / metrics[i-1].close_price
                signal_predictions.append((signal_result.signal, prev_return))

            # Determine action based on signal
            action = self._signal_to_action(signal_result.signal)

            # Execute trade
            trade_value = 0.0
            trade_quantity = 0.0

            if action in (SignalAction.STRONG_BUY, SignalAction.BUY) and capital > 100:
                # Calculate position size
                if action == SignalAction.STRONG_BUY:
                    position_multiplier = 1.0
                else:
                    position_multiplier = 0.5

                invest_amount = capital * self.position_size_pct * position_multiplier
                commission = invest_amount * self.commission_rate
                trade_value = invest_amount - commission
                trade_quantity = trade_value / current_price

                shares += trade_quantity
                capital -= invest_amount

                trade = Trade(
                    date=current_date,
                    action=action,
                    price=current_price,
                    signal=signal_result.signal,
                    confidence=signal_result.confidence,
                    quantity=trade_quantity,
                    value=trade_value,
                    reason=f"Signal: {signal_result.signal} (score: {signal_result.score})"
                )
                trades.append(trade)

            elif action in (SignalAction.STRONG_SELL, SignalAction.SELL) and shares > 0:
                # Calculate sell quantity
                if action == SignalAction.STRONG_SELL:
                    sell_pct = 1.0
                else:
                    sell_pct = 0.5

                trade_quantity = shares * sell_pct
                gross_value = trade_quantity * current_price
                commission = gross_value * self.commission_rate
                trade_value = gross_value - commission

                shares -= trade_quantity
                capital += trade_value

                trade = Trade(
                    date=current_date,
                    action=action,
                    price=current_price,
                    signal=signal_result.signal,
                    confidence=signal_result.confidence,
                    quantity=-trade_quantity,
                    value=trade_value,
                    reason=f"Signal: {signal_result.signal} (score: {signal_result.score})"
                )
                trades.append(trade)

            # Calculate current equity (capital + position value)
            position_value = shares * current_price
            total_equity = capital + position_value

            # Update peak and drawdown
            if total_equity > peak_capital:
                peak_capital = total_equity

            current_drawdown = (peak_capital - total_equity) / peak_capital * 100
            max_drawdown = max(max_drawdown, current_drawdown)

            # Record equity curve
            equity_curve.append({
                'date': current_date,
                'price': current_price,
                'equity': total_equity,
                'cash': capital,
                'shares': shares,
                'position_value': position_value,
                'signal': signal_result.signal,
                'signal_score': signal_result.score,
                'drawdown_pct': current_drawdown
            })

        # Calculate final metrics
        final_equity = equity_curve[-1]['equity'] if equity_curve else self.initial_capital
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # Benchmark: Buy and Hold
        first_price = metrics[0].close_price
        last_price = metrics[-1].close_price
        benchmark_return_pct = ((last_price - first_price) / first_price) * 100

        alpha_pct = total_return_pct - benchmark_return_pct

        # Trade statistics
        winning_trades = sum(1 for t in trades if t.value > 0)
        losing_trades = len(trades) - winning_trades
        win_rate_pct = (winning_trades / len(trades) * 100) if trades else 0

        # Average trade return
        if trades:
            avg_trade_return = sum(t.value for t in trades) / len(trades)
            avg_trade_return_pct = (avg_trade_return / self.initial_capital) * 100
        else:
            avg_trade_return_pct = 0

        # Calculate Sharpe ratio from equity curve
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)

        # Signal accuracy analysis
        signal_accuracy = self._calculate_signal_accuracy(signal_predictions)

        return BacktestResult(
            asset_symbol=asset.symbol,
            start_date=metrics[0].metric_date,
            end_date=metrics[-1].metric_date,
            initial_capital=self.initial_capital,
            total_return_pct=round(total_return_pct, 2),
            benchmark_return_pct=round(benchmark_return_pct, 2),
            alpha_pct=round(alpha_pct, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=round(win_rate_pct, 2),
            avg_trade_return_pct=round(avg_trade_return_pct, 4),
            final_capital=round(capital, 2),
            final_shares=round(shares, 4),
            peak_capital=round(peak_capital, 2),
            trades=trades,
            equity_curve=equity_curve,
            signal_accuracy=signal_accuracy
        )

    def _signal_to_action(self, signal: str) -> SignalAction:
        """Convert signal string to SignalAction enum."""
        mapping = {
            "STRONG_BUY": SignalAction.STRONG_BUY,
            "BUY": SignalAction.BUY,
            "HOLD": SignalAction.HOLD,
            "SELL": SignalAction.SELL,
            "STRONG_SELL": SignalAction.STRONG_SELL
        }
        return mapping.get(signal, SignalAction.HOLD)

    def _calculate_sharpe_ratio(self, equity_curve: List[Dict],
                                risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio from equity curve.

        Args:
            equity_curve: List of daily equity values
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Annualized Sharpe ratio
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate daily returns
        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1]['equity']
            curr_equity = equity_curve[i]['equity']
            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                returns.append(daily_return)

        if not returns or len(returns) < 2:
            return 0.0

        returns = np.array(returns)

        # Calculate mean and std of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Daily risk-free rate (annual / 252 trading days)
        daily_risk_free = risk_free_rate / 252

        # Sharpe ratio (annualized)
        sharpe = (mean_return - daily_risk_free) / std_return * np.sqrt(252)

        return sharpe

    def _calculate_signal_accuracy(self, predictions: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Calculate the accuracy of signals based on subsequent price movements.

        Args:
            predictions: List of (signal, next_day_return) tuples

        Returns:
            Dictionary with accuracy metrics by signal type
        """
        if not predictions:
            return {}

        accuracy = {}

        # Track predictions by signal type
        signal_performance = {
            "STRONG_BUY": {"correct": 0, "total": 0},
            "BUY": {"correct": 0, "total": 0},
            "HOLD": {"correct": 0, "total": 0},
            "SELL": {"correct": 0, "total": 0},
            "STRONG_SELL": {"correct": 0, "total": 0}
        }

        for signal, next_return in predictions:
            numeric_signal = signal_to_numeric(signal)

            # Signal was correct if:
            # - BUY signals followed by positive return
            # - SELL signals followed by negative return
            # - HOLD signals followed by small return (-1% to 1%)

            if numeric_signal > 0:  # BUY signals
                signal_performance[signal]["total"] += 1
                if next_return > 0:
                    signal_performance[signal]["correct"] += 1
            elif numeric_signal < 0:  # SELL signals
                signal_performance[signal]["total"] += 1
                if next_return < 0:
                    signal_performance[signal]["correct"] += 1
            else:  # HOLD
                signal_performance[signal]["total"] += 1
                if abs(next_return) < 0.01:  # Within 1%
                    signal_performance[signal]["correct"] += 1

        # Calculate accuracy percentages
        for signal, stats in signal_performance.items():
            if stats["total"] > 0:
                accuracy[signal] = round(stats["correct"] / stats["total"] * 100, 2)
            else:
                accuracy[signal] = 0.0

        # Overall accuracy
        total_correct = sum(s["correct"] for s in signal_performance.values())
        total_signals = sum(s["total"] for s in signal_performance.values())
        accuracy["OVERALL"] = round(total_correct / total_signals * 100, 2) if total_signals > 0 else 0.0

        return accuracy


def format_backtest_report(result: BacktestResult) -> str:
    """
    Format a BacktestResult into a readable report.

    Args:
        result: BacktestResult to format

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        f"BACKTEST REPORT: {result.asset_symbol}",
        "=" * 60,
        f"Period: {result.start_date} to {result.end_date}",
        f"Initial Capital: ${result.initial_capital:,.2f}",
        "",
        "PERFORMANCE METRICS",
        "-" * 40,
        f"Total Return:        {result.total_return_pct:+.2f}%",
        f"Benchmark (Hold):    {result.benchmark_return_pct:+.2f}%",
        f"Alpha:               {result.alpha_pct:+.2f}%",
        f"Max Drawdown:        {result.max_drawdown_pct:.2f}%",
        f"Sharpe Ratio:        {result.sharpe_ratio:.2f}",
        "",
        "TRADE STATISTICS",
        "-" * 40,
        f"Total Trades:        {result.total_trades}",
        f"Winning Trades:      {result.winning_trades}",
        f"Losing Trades:       {result.losing_trades}",
        f"Win Rate:            {result.win_rate_pct:.1f}%",
        f"Avg Trade Return:    {result.avg_trade_return_pct:.4f}%",
        "",
        "POSITION SUMMARY",
        "-" * 40,
        f"Final Cash:          ${result.final_capital:,.2f}",
        f"Final Shares:        {result.final_shares:.4f}",
        f"Peak Capital:        ${result.peak_capital:,.2f}",
        "",
        "SIGNAL ACCURACY",
        "-" * 40,
    ]

    for signal, accuracy in result.signal_accuracy.items():
        lines.append(f"{signal:20} {accuracy:>6.1f}%")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# Convenience function for easy backtesting
def backtest_asset(asset: Asset,
                   start_date: Optional[date] = None,
                   end_date: Optional[date] = None,
                   initial_capital: float = 10000.0) -> Optional[BacktestResult]:
    """
    Convenience function to run a backtest on a single asset.

    Args:
        asset: Asset to backtest
        start_date: Start date (default: 90 days ago)
        end_date: End date (default: today)
        initial_capital: Starting capital

    Returns:
        BacktestResult or None if insufficient data
    """
    backtester = SignalBacktester(initial_capital=initial_capital)
    return backtester.run_backtest(asset, start_date, end_date)
