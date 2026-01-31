"""
Transaction Repository - data access layer for Transaction model.
"""

from typing import Optional, List
from datetime import date
from sqlmodel import Session, select

from db_engine import get_engine
from models import Transaction


class TransactionRepository:
    """Repository for Transaction CRUD operations."""

    @staticmethod
    def add(asset_id: int, transaction_date: date, transaction_type: str,
            quantity: float, price: float) -> Transaction:
        """Add a new transaction to the database."""
        with Session(get_engine()) as session:
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

    @staticmethod
    def get_by_asset(asset_id: int) -> List[Transaction]:
        """Retrieve all transactions for a specific asset."""
        with Session(get_engine()) as session:
            statement = select(Transaction).where(Transaction.asset_id == asset_id)
            results = session.exec(statement)
            return list(results.all())

    @staticmethod
    def get_all() -> List[Transaction]:
        """Retrieve all transactions from the database."""
        with Session(get_engine()) as session:
            statement = select(Transaction)
            results = session.exec(statement)
            return list(results.all())

    @staticmethod
    def get_by_id(transaction_id: int) -> Optional[Transaction]:
        """Retrieve a transaction by its ID."""
        with Session(get_engine()) as session:
            return session.get(Transaction, transaction_id)

    @staticmethod
    def update(transaction_id: int, transaction_date: Optional[date] = None,
               transaction_type: Optional[str] = None,
               quantity: Optional[float] = None,
               price: Optional[float] = None) -> Optional[Transaction]:
        """
        Update an existing transaction.
        Only updates fields that are provided (not None).
        """
        with Session(get_engine()) as session:
            transaction = session.get(Transaction, transaction_id)
            if transaction:
                if transaction_date is not None:
                    transaction.transaction_date = transaction_date
                if transaction_type is not None:
                    transaction.transaction_type = transaction_type
                if quantity is not None:
                    transaction.quantity = quantity
                if price is not None:
                    transaction.price = price
                session.add(transaction)
                session.commit()
                session.refresh(transaction)
                return transaction
            return None

    @staticmethod
    def delete(transaction_id: int) -> bool:
        """
        Delete a transaction by its ID.
        Returns True if successful, False otherwise.
        """
        with Session(get_engine()) as session:
            try:
                transaction = session.get(Transaction, transaction_id)
                if transaction:
                    session.delete(transaction)
                    session.commit()
                    return True
                return False
            except Exception as e:
                session.rollback()
                raise e
