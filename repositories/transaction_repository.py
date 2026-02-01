"""
Transaction Repository - data access layer for Transaction model.
Optimized with optional session parameter for transaction reuse.
"""

from typing import Optional, List
from datetime import date
from sqlmodel import Session, select

from db_engine import get_engine
from models import Transaction


class TransactionRepository:
    """Repository for Transaction CRUD operations."""

    @staticmethod
    def add(
        asset_id: int,
        transaction_date: date,
        transaction_type: str,
        quantity: float,
        price: float,
        session: Optional[Session] = None
    ) -> Transaction:
        """
        Add a new transaction to the database.

        Args:
            asset_id: Asset ID for the transaction
            transaction_date: Date of the transaction
            transaction_type: 'buy' or 'sell'
            quantity: Number of shares/units
            price: Price per unit
            session: Optional existing session for transaction reuse

        Returns:
            Created Transaction object
        """
        def _create_transaction(sess: Session) -> Transaction:
            transaction = Transaction(
                asset_id=asset_id,
                transaction_date=transaction_date,
                transaction_type=transaction_type,
                quantity=quantity,
                price=price
            )
            sess.add(transaction)
            sess.commit()
            sess.refresh(transaction)
            return transaction

        if session is not None:
            return _create_transaction(session)
        else:
            with Session(get_engine()) as session:
                return _create_transaction(session)

    @staticmethod
    def get_by_asset(asset_id: int, session: Optional[Session] = None) -> List[Transaction]:
        """
        Retrieve all transactions for a specific asset.

        Args:
            asset_id: Asset ID to look up
            session: Optional existing session for transaction reuse

        Returns:
            List of Transaction objects
        """
        def _get_by_asset(sess: Session) -> List[Transaction]:
            statement = select(Transaction).where(Transaction.asset_id == asset_id)
            results = sess.exec(statement)
            return list(results.all())

        if session is not None:
            return _get_by_asset(session)
        else:
            with Session(get_engine()) as session:
                return _get_by_asset(session)

    @staticmethod
    def get_all(session: Optional[Session] = None) -> List[Transaction]:
        """
        Retrieve all transactions from the database.

        Args:
            session: Optional existing session for transaction reuse

        Returns:
            List of all Transaction objects
        """
        def _get_all(sess: Session) -> List[Transaction]:
            statement = select(Transaction)
            results = sess.exec(statement)
            return list(results.all())

        if session is not None:
            return _get_all(session)
        else:
            with Session(get_engine()) as session:
                return _get_all(session)

    @staticmethod
    def get_by_id(transaction_id: int, session: Optional[Session] = None) -> Optional[Transaction]:
        """
        Retrieve a transaction by its ID.

        Args:
            transaction_id: Transaction ID to look up
            session: Optional existing session for transaction reuse

        Returns:
            Transaction object or None if not found
        """
        def _get_by_id(sess: Session) -> Optional[Transaction]:
            return sess.get(Transaction, transaction_id)

        if session is not None:
            return _get_by_id(session)
        else:
            with Session(get_engine()) as session:
                return _get_by_id(session)

    @staticmethod
    def update(
        transaction_id: int,
        transaction_date: Optional[date] = None,
        transaction_type: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        session: Optional[Session] = None
    ) -> Optional[Transaction]:
        """
        Update an existing transaction.
        Only updates fields that are provided (not None).

        Args:
            transaction_id: Transaction ID to update
            transaction_date: New transaction date (optional)
            transaction_type: New transaction type (optional)
            quantity: New quantity (optional)
            price: New price (optional)
            session: Optional existing session for transaction reuse

        Returns:
            Updated Transaction object or None if not found
        """
        def _update(sess: Session) -> Optional[Transaction]:
            transaction = sess.get(Transaction, transaction_id)
            if transaction:
                if transaction_date is not None:
                    transaction.transaction_date = transaction_date
                if transaction_type is not None:
                    transaction.transaction_type = transaction_type
                if quantity is not None:
                    transaction.quantity = quantity
                if price is not None:
                    transaction.price = price
                sess.add(transaction)
                sess.commit()
                sess.refresh(transaction)
                return transaction
            return None

        if session is not None:
            return _update(session)
        else:
            with Session(get_engine()) as session:
                return _update(session)

    @staticmethod
    def delete(transaction_id: int, session: Optional[Session] = None) -> bool:
        """
        Delete a transaction by its ID.

        Args:
            transaction_id: Transaction ID to delete
            session: Optional existing session for transaction reuse

        Returns:
            True if successful, False otherwise
        """
        def _delete(sess: Session) -> bool:
            try:
                transaction = sess.get(Transaction, transaction_id)
                if transaction:
                    sess.delete(transaction)
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
    def delete_by_asset(asset_id: int, session: Optional[Session] = None) -> int:
        """
        Delete all transactions for a specific asset.
        Useful when deleting an asset.

        Args:
            asset_id: Asset ID whose transactions to delete
            session: Optional existing session for transaction reuse

        Returns:
            Number of transactions deleted
        """
        def _delete_by_asset(sess: Session) -> int:
            statement = select(Transaction).where(Transaction.asset_id == asset_id)
            transactions = sess.exec(statement).all()
            count = 0
            for tx in transactions:
                sess.delete(tx)
                count += 1
            sess.commit()
            return count

        if session is not None:
            return _delete_by_asset(session)
        else:
            with Session(get_engine()) as session:
                return _delete_by_asset(session)
