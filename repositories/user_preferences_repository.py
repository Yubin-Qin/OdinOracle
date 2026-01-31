"""
UserPreferences Repository - data access layer for UserPreferences model.
"""

from typing import Optional
from sqlmodel import Session, select

from db_engine import get_engine
from models import UserPreferences


class UserPreferencesRepository:
    """Repository for UserPreferences CRUD operations."""

    @staticmethod
    def get() -> Optional[UserPreferences]:
        """Retrieve user preferences (singleton - only one record expected)."""
        with Session(get_engine()) as session:
            statement = select(UserPreferences)
            results = session.exec(statement)
            return results.first()

    @staticmethod
    def save_email(email: str) -> UserPreferences:
        """Save or update user email preferences."""
        with Session(get_engine()) as session:
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

    @staticmethod
    def save_language(language: str) -> UserPreferences:
        """Save or update user language preference."""
        with Session(get_engine()) as session:
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

    @staticmethod
    def save_base_currency(base_currency: str) -> UserPreferences:
        """Save or update user base currency preference."""
        with Session(get_engine()) as session:
            statement = select(UserPreferences)
            results = session.exec(statement)
            prefs = results.first()

            if prefs:
                prefs.base_currency = base_currency
                session.add(prefs)
                session.commit()
                session.refresh(prefs)
                return prefs
            else:
                prefs = UserPreferences(base_currency=base_currency)
                session.add(prefs)
                session.commit()
                session.refresh(prefs)
                return prefs
