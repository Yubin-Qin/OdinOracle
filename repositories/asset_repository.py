"""
Asset Repository - data access layer for Asset model.
Optimized with optional session parameter for transaction reuse.
"""

from typing import Optional, List
from sqlmodel import Session, select

from db_engine import get_engine
from models import Asset


class AssetRepository:
    """Repository for Asset CRUD operations."""

    @staticmethod
    def add(
        symbol: str,
        name: str,
        market_type: str,
        alert_price_threshold: Optional[float] = None,
        session: Optional[Session] = None
    ) -> Asset:
        """
        Add a new asset to the database.

        Args:
            symbol: Stock symbol
            name: Asset name
            market_type: Market type (US, HK, CN)
            alert_price_threshold: Optional price alert threshold
            session: Optional existing session for transaction reuse

        Returns:
            Created Asset object
        """
        def _create_asset(sess: Session) -> Asset:
            asset = Asset(
                symbol=symbol,
                name=name,
                market_type=market_type,
                alert_price_threshold=alert_price_threshold
            )
            sess.add(asset)
            sess.commit()
            sess.refresh(asset)
            return asset

        if session is not None:
            return _create_asset(session)
        else:
            with Session(get_engine()) as session:
                return _create_asset(session)

    @staticmethod
    def get_all(session: Optional[Session] = None) -> List[Asset]:
        """
        Retrieve all assets from the database.

        Args:
            session: Optional existing session for transaction reuse

        Returns:
            List of all Asset objects
        """
        def _get_all(sess: Session) -> List[Asset]:
            statement = select(Asset)
            results = sess.exec(statement)
            return list(results.all())

        if session is not None:
            return _get_all(session)
        else:
            with Session(get_engine()) as session:
                return _get_all(session)

    @staticmethod
    def get_by_id(asset_id: int, session: Optional[Session] = None) -> Optional[Asset]:
        """
        Retrieve an asset by its ID.

        Args:
            asset_id: Asset ID to look up
            session: Optional existing session for transaction reuse

        Returns:
            Asset object or None if not found
        """
        def _get_by_id(sess: Session) -> Optional[Asset]:
            return sess.get(Asset, asset_id)

        if session is not None:
            return _get_by_id(session)
        else:
            with Session(get_engine()) as session:
                return _get_by_id(session)

    @staticmethod
    def get_by_symbol(symbol: str, session: Optional[Session] = None) -> Optional[Asset]:
        """
        Retrieve an asset by symbol (case-insensitive).

        Args:
            symbol: Symbol to search for
            session: Optional existing session for transaction reuse

        Returns:
            Asset object or None if not found
        """
        def _get_by_symbol(sess: Session) -> Optional[Asset]:
            statement = select(Asset).where(Asset.symbol.ilike(symbol))
            results = sess.exec(statement)
            return results.first()

        if session is not None:
            return _get_by_symbol(session)
        else:
            with Session(get_engine()) as session:
                return _get_by_symbol(session)

    @staticmethod
    def update_alert_threshold(
        asset_id: int,
        alert_threshold: Optional[float],
        session: Optional[Session] = None
    ) -> Optional[Asset]:
        """
        Update the alert price threshold for an asset.

        Args:
            asset_id: Asset ID to update
            alert_threshold: New threshold value (or None to clear)
            session: Optional existing session for transaction reuse

        Returns:
            Updated Asset object or None if not found
        """
        def _update(sess: Session) -> Optional[Asset]:
            asset = sess.get(Asset, asset_id)
            if asset:
                asset.alert_price_threshold = alert_threshold
                sess.add(asset)
                sess.commit()
                sess.refresh(asset)
                return asset
            return None

        if session is not None:
            return _update(session)
        else:
            with Session(get_engine()) as session:
                return _update(session)

    @staticmethod
    def delete(asset_id: int, session: Optional[Session] = None) -> bool:
        """
        Delete an asset and all its transactions.
        Note: Transactions should be deleted first due to foreign key constraints.

        Args:
            asset_id: Asset ID to delete
            session: Optional existing session for transaction reuse

        Returns:
            True if successful, False otherwise
        """
        from models import Transaction

        def _delete(sess: Session) -> bool:
            try:
                # First, delete all transactions for this asset
                statement = select(Transaction).where(Transaction.asset_id == asset_id)
                transactions = sess.exec(statement).all()
                for tx in transactions:
                    sess.delete(tx)

                # Then delete the asset
                asset = sess.get(Asset, asset_id)
                if asset:
                    sess.delete(asset)
                    sess.commit()
                    return True
                return False
            except Exception as e:
                sess.rollback()
                raise e

        if session is not None:
            return _delete(session)
        else:
            with Session(get_engine()) as session:
                return _delete(session)

    @staticmethod
    def find_by_symbol(symbol: str, session: Optional[Session] = None) -> Optional[Asset]:
        """
        Find an asset by symbol (case-insensitive exact match).

        Args:
            symbol: Symbol to search for
            session: Optional existing session for transaction reuse

        Returns:
            Asset object or None if not found
        """
        def _find_by_symbol(sess: Session) -> Optional[Asset]:
            assets = sess.exec(select(Asset)).all()
            for asset in assets:
                if asset.symbol.upper() == symbol.upper():
                    return asset
            return None

        if session is not None:
            return _find_by_symbol(session)
        else:
            with Session(get_engine()) as session:
                return _find_by_symbol(session)
