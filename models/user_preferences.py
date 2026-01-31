"""
UserPreferences model - stores user preferences and settings.
"""

from typing import Optional
from sqlmodel import SQLModel, Field


class UserPreferences(SQLModel, table=True):
    """Stores user preferences and settings."""
    id: Optional[int] = Field(default=None, primary_key=True)
    email_address: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default="en")  # "en" or "zh"
    base_currency: Optional[str] = Field(default="USD")  # "USD", "CNY", "HKD", etc.
