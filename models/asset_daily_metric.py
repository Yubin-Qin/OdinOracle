"""
AssetDailyMetric model - stores daily technical indicators for an asset.
"""

from typing import Optional
from datetime import date, datetime
from sqlmodel import SQLModel, Field


class AssetDailyMetric(SQLModel, table=True):
    """
    Daily technical metrics for an asset.
    Stores calculated indicators for historical analysis and backtesting.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    asset_id: int = Field(foreign_key="asset.id", index=True)
    metric_date: date = Field(index=True)

    # Price data
    close_price: float

    # Trend indicators
    sma_20: Optional[float] = Field(default=None)
    sma_50: Optional[float] = Field(default=None)
    sma_200: Optional[float] = Field(default=None)

    # Momentum indicators
    rsi_14: Optional[float] = Field(default=None)
    macd: Optional[float] = Field(default=None)
    macd_signal: Optional[float] = Field(default=None)
    macd_histogram: Optional[float] = Field(default=None)

    # Volatility indicators
    bollinger_upper: Optional[float] = Field(default=None)
    bollinger_middle: Optional[float] = Field(default=None)
    bollinger_lower: Optional[float] = Field(default=None)
    bollinger_bandwidth: Optional[float] = Field(default=None)  # (upper - lower) / middle

    # Volume indicators
    volume: Optional[int] = Field(default=None)
    volume_sma_20: Optional[float] = Field(default=None)
    volume_ratio: Optional[float] = Field(default=None)  # volume / volume_sma_20

    # Signal (calculated from all factors)
    overall_signal: Optional[str] = Field(default=None)  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence_score: Optional[int] = Field(default=None)  # 0-10

    created_at: datetime = Field(default_factory=datetime.now)
