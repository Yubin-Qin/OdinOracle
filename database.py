"""
Database models and initialization for OdinOracle.
Uses SQLModel with SQLite for persistent storage.
"""

from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional
from datetime import date
import os

# Database file path
DB_FILE = "odin_oracle.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Create engine
engine = create_engine(DATABASE_URL, echo=False)


class Asset(SQLModel, table=True):
    """Represents a stock/asset in the portfolio."""
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)  # e.g., "NVDA", "0700.HK", "600519.SS"
    name: str = Field(index=True)  # e.g., "NVIDIA Corporation", "Tencent"
    market_type: str  # "US", "HK", "CN"
    alert_price_threshold: Optional[float] = Field(default=None)  # Price threshold for alerts


class Transaction(SQLModel, table=True):
    """Represents a buy/sell transaction for an asset."""
    id: Optional[int] = Field(default=None, primary_key=True)
    asset_id: int = Field(foreign_key="asset.id")
    transaction_date: date = Field(index=True)
    transaction_type: str  # "buy" or "sell"
    quantity: float
    price: float  # Price per unit at transaction time


class UserPreferences(SQLModel, table=True):
    """Stores user preferences and settings."""
    id: Optional[int] = Field(default=None, primary_key=True)
    email_address: Optional[str] = Field(default=None)


def init_db():
    """Initialize the database and create all tables."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get a new database session."""
    return Session(engine)


def add_asset(symbol: str, name: str, market_type: str, alert_price_threshold: Optional[float] = None) -> Asset:
    """Add a new asset to the database."""
    with get_session() as session:
        asset = Asset(
            symbol=symbol,
            name=name,
            market_type=market_type,
            alert_price_threshold=alert_price_threshold
        )
        session.add(asset)
        session.commit()
        session.refresh(asset)
        return asset


def get_all_assets() -> list[Asset]:
    """Retrieve all assets from the database."""
    with get_session() as session:
        statement = select(Asset)
        results = session.exec(statement)
        return list(results.all())


def get_asset_by_id(asset_id: int) -> Optional[Asset]:
    """Retrieve an asset by its ID."""
    with get_session() as session:
        return session.get(Asset, asset_id)


def update_asset_alert_threshold(asset_id: int, alert_threshold: Optional[float]) -> Optional[Asset]:
    """Update the alert price threshold for an asset."""
    with get_session() as session:
        asset = session.get(Asset, asset_id)
        if asset:
            asset.alert_price_threshold = alert_threshold
            session.add(asset)
            session.commit()
            session.refresh(asset)
            return asset
        return None


def add_transaction(asset_id: int, transaction_date: date, transaction_type: str,
                   quantity: float, price: float) -> Transaction:
    """Add a new transaction to the database."""
    with get_session() as session:
        transaction = Transaction(
            asset_id=asset_id,
            transaction_date=transaction_date,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price
        )
        session.add(transaction)
        session.commit()
        session.refresh(transaction)
        return transaction


def get_transactions_by_asset(asset_id: int) -> list[Transaction]:
    """Retrieve all transactions for a specific asset."""
    with get_session() as session:
        statement = select(Transaction).where(Transaction.asset_id == asset_id)
        results = session.exec(statement)
        return list(results.all())


def get_all_transactions() -> list[Transaction]:
    """Retrieve all transactions from the database."""
    with get_session() as session:
        statement = select(Transaction)
        results = session.exec(statement)
        return list(results.all())


def save_user_email(email: str) -> UserPreferences:
    """Save or update user email preferences."""
    with get_session() as session:
        # Check if preferences already exist
        statement = select(UserPreferences)
        results = session.exec(statement)
        prefs = results.first()

        if prefs:
            prefs.email_address = email
            session.add(prefs)
            session.commit()
            session.refresh(prefs)
            return prefs
        else:
            prefs = UserPreferences(email_address=email)
            session.add(prefs)
            session.commit()
            session.refresh(prefs)
            return prefs


def get_user_preferences() -> Optional[UserPreferences]:
    """Retrieve user preferences."""
    with get_session() as session:
        statement = select(UserPreferences)
        results = session.exec(statement)
        return results.first()


if __name__ == "__main__":
    # Initialize database for testing
    init_db()
    print(f"Database initialized at {DB_FILE}")
