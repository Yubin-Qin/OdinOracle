"""
Asset Repository - data access layer for Asset model.
"""

from typing import Optional, List
from sqlmodel import Session, select

from db_engine import get_engine
from models import Asset


class AssetRepository:
    """Repository for Asset CRUD operations."""

    @staticmethod
    def add(symbol: str, name: str, market_type: str, alert_price_threshold: Optional[float] = None) -> Asset:
        """Add a new asset to the database."""
        with Session(get_engine()) as session:
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

    @staticmethod
    def get_all() -> List[Asset]:
        """Retrieve all assets from the database."""
        with Session(get_engine()) as session:
            statement = select(Asset)
            results = session.exec(statement)
            return list(results.all())

    @staticmethod
    def get_by_id(asset_id: int) -> Optional[Asset]:
        """Retrieve an asset by its ID."""
        with Session(get_engine()) as session:
            return session.get(Asset, asset_id)

    @staticmethod
    def get_by_symbol(symbol: str) -> Optional[Asset]:
        """Retrieve an asset by symbol (case-insensitive)."""
        with Session(get_engine()) as session:
            statement = select(Asset).where(Asset.symbol.ilike(symbol))
            results = session.exec(statement)
            return results.first()

    @staticmethod
    def update_alert_threshold(asset_id: int, alert_threshold: Optional[float]) -> Optional[Asset]:
        """Update the alert price threshold for an asset."""
        with Session(get_engine()) as session:
            asset = session.get(Asset, asset_id)
            if asset:
                asset.alert_price_threshold = alert_threshold
                session.add(asset)
                session.commit()
                session.refresh(asset)
                return asset
            return None

    @staticmethod
    def delete(asset_id: int) -> bool:
        """
        Delete an asset and all its transactions.
        Note: Transactions should be deleted first due to foreign key constraints.
        Returns True if successful, False otherwise.
        """
        from repositories.transaction_repository import TransactionRepository

        with Session(get_engine()) as session:
            try:
                # First, delete all transactions for this asset
                from models import Transaction
                statement = select(Transaction).where(Transaction.asset_id == asset_id)
                transactions = session.exec(statement).all()
                for tx in transactions:
                    session.delete(tx)

                # Then delete the asset
                asset = session.get(Asset, asset_id)
                if asset:
                    session.delete(asset)
                    session.commit()
                    return True
                return False
            except Exception as e:
                session.rollback()
                raise e

    @staticmethod
    def find_by_symbol(symbol: str) -> Optional[Asset]:
        """Find an asset by symbol (case-insensitive exact match)."""
        with Session(get_engine()) as session:
            assets = session.exec(select(Asset)).all()
            for asset in assets:
                if asset.symbol.upper() == symbol.upper():
                    return asset
            return None
