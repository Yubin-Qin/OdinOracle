"""
Database models and initialization for OdinOracle.
Uses SQLModel with SQLite for persistent storage.
Enhanced with AssetDailyMetric for technical indicators history.
"""

from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional
from datetime import date, datetime
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
    language: Optional[str] = Field(default="en")  # "en" or "zh"


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


def init_db():
    """Initialize the database and create all tables."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get a new database session."""
    return Session(engine)


# ==================== Asset Operations ====================
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


def get_asset_by_symbol(symbol: str) -> Optional[Asset]:
    """Retrieve an asset by symbol (case-insensitive)."""
    with get_session() as session:
        statement = select(Asset).where(Asset.symbol.ilike(symbol))
        results = session.exec(statement)
        return results.first()


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


# ==================== Transaction Operations ====================
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


# ==================== AssetDailyMetric Operations ====================
def save_daily_metric(metric: AssetDailyMetric) -> AssetDailyMetric:
    """
    Save or update a daily metric record.
    Uses upsert logic: if exists for asset_id + date, update; otherwise insert.
    """
    with get_session() as session:
        # Check if metric already exists
        statement = select(AssetDailyMetric).where(
            AssetDailyMetric.asset_id == metric.asset_id,
            AssetDailyMetric.metric_date == metric.metric_date
        )
        existing = session.exec(statement).first()

        if existing:
            # Update existing record
            existing.close_price = metric.close_price
            existing.sma_20 = metric.sma_20
            existing.sma_50 = metric.sma_50
            existing.sma_200 = metric.sma_200
            existing.rsi_14 = metric.rsi_14
            existing.macd = metric.macd
            existing.macd_signal = metric.macd_signal
            existing.macd_histogram = metric.macd_histogram
            existing.bollinger_upper = metric.bollinger_upper
            existing.bollinger_middle = metric.bollinger_middle
            existing.bollinger_lower = metric.bollinger_lower
            existing.bollinger_bandwidth = metric.bollinger_bandwidth
            existing.volume = metric.volume
            existing.volume_sma_20 = metric.volume_sma_20
            existing.volume_ratio = metric.volume_ratio
            existing.overall_signal = metric.overall_signal
            existing.confidence_score = metric.confidence_score
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return existing
        else:
            # Insert new record
            session.add(metric)
            session.commit()
            session.refresh(metric)
            return metric


def get_latest_metric(asset_id: int) -> Optional[AssetDailyMetric]:
    """Get the most recent daily metric for an asset."""
    with get_session() as session:
        statement = select(AssetDailyMetric).where(
            AssetDailyMetric.asset_id == asset_id
        ).order_by(AssetDailyMetric.metric_date.desc()).limit(1)
        return session.exec(statement).first()


def get_metrics_history(asset_id: int, days: int = 60) -> list[AssetDailyMetric]:
    """Get historical metrics for an asset."""
    with get_session() as session:
        statement = select(AssetDailyMetric).where(
            AssetDailyMetric.asset_id == asset_id
        ).order_by(AssetDailyMetric.metric_date.desc()).limit(days)
        results = session.exec(statement)
        return list(results.all())


def get_metric_by_date(asset_id: int, metric_date: date) -> Optional[AssetDailyMetric]:
    """Get a specific metric by date."""
    with get_session() as session:
        statement = select(AssetDailyMetric).where(
            AssetDailyMetric.asset_id == asset_id,
            AssetDailyMetric.metric_date == metric_date
        )
        return session.exec(statement).first()


# ==================== UserPreferences Operations ====================
def save_user_email(email: str) -> UserPreferences:
    """Save or update user email preferences."""
    with get_session() as session:
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


def save_user_language(language: str) -> UserPreferences:
    """Save or update user language preference."""
    with get_session() as session:
        statement = select(UserPreferences)
        results = session.exec(statement)
        prefs = results.first()

        if prefs:
            prefs.language = language
            session.add(prefs)
            session.commit()
            session.refresh(prefs)
            return prefs
        else:
            prefs = UserPreferences(language=language)
            session.add(prefs)
            session.commit()
            session.refresh(prefs)
            return prefs


if __name__ == "__main__":
    # Initialize database for testing
    init_db()
    print(f"Database initialized at {DB_FILE}")
