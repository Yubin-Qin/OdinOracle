"""
Asset model - represents a stock/asset in the portfolio.
"""

from typing import Optional
from sqlmodel import SQLModel, Field


class Asset(SQLModel, table=True):
    """Represents a stock/asset in the portfolio."""
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)  # e.g., "NVDA", "0700.HK", "600519.SS"
    name: str = Field(index=True)  # e.g., "NVIDIA Corporation", "Tencent"
    market_type: str  # "US", "HK", "CN"
    alert_price_threshold: Optional[float] = Field(default=None)  # Price threshold for alerts
