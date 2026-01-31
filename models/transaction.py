"""
Transaction model - represents a buy/sell transaction for an asset.
"""

from typing import Optional
from datetime import date
from sqlmodel import SQLModel, Field


class Transaction(SQLModel, table=True):
    """Represents a buy/sell transaction for an asset."""
    id: Optional[int] = Field(default=None, primary_key=True)
    asset_id: int = Field(foreign_key="asset.id")
    transaction_date: date = Field(index=True)
    transaction_type: str  # "buy" or "sell"
    quantity: float
    price: float  # Price per unit at transaction time
