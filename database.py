"""
Database initialization and backward-compatible CRUD operations for OdinOracle.

This module re-exports the engine functions from db_engine and provides
backward-compatible CRUD functions that delegate to the repository layer.

New code should use:
- db_engine.get_engine() / db_engine.get_session() for engine/session access
- repositories.* for data access
"""

from typing import Optional, List
from datetime import date

# Re-export engine functions from db_engine (avoids circular imports)
from db_engine import get_engine, get_session, init_db

# Re-export all models for convenience
from models import Asset, Transaction, UserPreferences, AssetDailyMetric

# Import repositories for backward-compatible function delegation
from repositories import (
    AssetRepository,
    TransactionRepository,
    MetricRepository,
    UserPreferencesRepository
)

# ==================== Backward-Compatible CRUD Operations ====================
# These functions delegate to the new repository layer for backward compatibility.
# New code should use the repositories directly.

# Asset Operations
def add_asset(symbol: str, name: str, market_type: str, alert_price_threshold: Optional[float] = None) -> Asset:
    """Add a new asset to the database."""
    return AssetRepository.add(symbol, name, market_type, alert_price_threshold)


def get_all_assets() -> List[Asset]:
    """Retrieve all assets from the database."""
    return AssetRepository.get_all()


def get_asset_by_id(asset_id: int) -> Optional[Asset]:
    """Retrieve an asset by its ID."""
    return AssetRepository.get_by_id(asset_id)


def get_asset_by_symbol(symbol: str) -> Optional[Asset]:
    """Retrieve an asset by symbol (case-insensitive)."""
    return AssetRepository.get_by_symbol(symbol)


def update_asset_alert_threshold(asset_id: int, alert_threshold: Optional[float]) -> Optional[Asset]:
    """Update the alert price threshold for an asset."""
    return AssetRepository.update_alert_threshold(asset_id, alert_threshold)


def delete_asset(asset_id: int) -> bool:
    """Delete an asset and all its transactions."""
    return AssetRepository.delete(asset_id)


# Transaction Operations
def add_transaction(asset_id: int, transaction_date: date, transaction_type: str,
                   quantity: float, price: float) -> Transaction:
    """Add a new transaction to the database."""
    return TransactionRepository.add(asset_id, transaction_date, transaction_type, quantity, price)


def get_transactions_by_asset(asset_id: int) -> List[Transaction]:
    """Retrieve all transactions for a specific asset."""
    return TransactionRepository.get_by_asset(asset_id)


def get_all_transactions() -> List[Transaction]:
    """Retrieve all transactions from the database."""
    return TransactionRepository.get_all()


def get_transaction_by_id(transaction_id: int) -> Optional[Transaction]:
    """Retrieve a transaction by its ID."""
    return TransactionRepository.get_by_id(transaction_id)


def update_transaction(transaction_id: int, transaction_date: Optional[date] = None,
                       transaction_type: Optional[str] = None,
                       quantity: Optional[float] = None,
                       price: Optional[float] = None) -> Optional[Transaction]:
    """Update an existing transaction."""
    return TransactionRepository.update(transaction_id, transaction_date, transaction_type, quantity, price)


def delete_transaction(transaction_id: int) -> bool:
    """Delete a transaction by its ID."""
    return TransactionRepository.delete(transaction_id)


# AssetDailyMetric Operations
def save_daily_metric(metric: AssetDailyMetric) -> AssetDailyMetric:
    """Save or update a daily metric record."""
    return MetricRepository.save(metric)


def get_latest_metric(asset_id: int) -> Optional[AssetDailyMetric]:
    """Get the most recent daily metric for an asset."""
    return MetricRepository.get_latest(asset_id)


def get_metrics_history(asset_id: int, days: int = 60) -> List[AssetDailyMetric]:
    """Get historical metrics for an asset."""
    return MetricRepository.get_history(asset_id, days)


def get_metric_by_date(asset_id: int, metric_date: date) -> Optional[AssetDailyMetric]:
    """Get a specific metric by date."""
    return MetricRepository.get_by_date(asset_id, metric_date)


# UserPreferences Operations
def save_user_email(email: str) -> UserPreferences:
    """Save or update user email preferences."""
    return UserPreferencesRepository.save_email(email)


def get_user_preferences() -> Optional[UserPreferences]:
    """Retrieve user preferences."""
    return UserPreferencesRepository.get()


def save_user_language(language: str) -> UserPreferences:
    """Save or update user language preference."""
    return UserPreferencesRepository.save_language(language)


def save_user_base_currency(base_currency: str) -> UserPreferences:
    """Save or update user base currency preference."""
    return UserPreferencesRepository.save_base_currency(base_currency)


__all__ = [
    # Engine functions
    'get_engine',
    'get_session',
    'init_db',
    # Models
    'Asset',
    'Transaction',
    'UserPreferences',
    'AssetDailyMetric',
    # Asset operations
    'add_asset',
    'get_all_assets',
    'get_asset_by_id',
    'get_asset_by_symbol',
    'update_asset_alert_threshold',
    'delete_asset',
    # Transaction operations
    'add_transaction',
    'get_transactions_by_asset',
    'get_all_transactions',
    'get_transaction_by_id',
    'update_transaction',
    'delete_transaction',
    # Metric operations
    'save_daily_metric',
    'get_latest_metric',
    'get_metrics_history',
    'get_metric_by_date',
    # UserPreferences operations
    'save_user_email',
    'get_user_preferences',
    'save_user_language',
    'save_user_base_currency',
]
