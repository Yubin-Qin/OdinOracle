"""
Database engine and session management for OdinOracle.
This module is separate from database.py to avoid circular imports.
Uses SQLModel with SQLite for persistent storage.
Features Write-Ahead Logging (WAL) mode for improved concurrency.
"""

from sqlmodel import SQLModel, Session, create_engine
from typing import Optional
import logging

from config import get_settings

logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[object] = None


def get_engine():
    """Get or create the database engine with WAL mode enabled."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            echo=settings.db_echo,
            connect_args={
                "check_same_thread": False,  # Allow use across threads
            }
        )
        # Enable WAL mode for better concurrency
        _enable_wal_mode()
    return _engine


def _enable_wal_mode():
    """Enable SQLite WAL mode for improved concurrent read/write performance."""
    if _engine is None:
        return

    try:
        with _engine.connect() as conn:
            # Enable WAL mode
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            # Set busy timeout to 5 seconds to handle concurrent access
            conn.exec_driver_sql("PRAGMA busy_timeout=5000")
            logger.info("SQLite WAL mode enabled for concurrent access")
    except Exception as e:
        logger.warning(f"Could not enable WAL mode: {e}")


def init_db():
    """Initialize the database and create all tables."""
    from models import Asset, Transaction, UserPreferences, AssetDailyMetric

    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    logger.info("Database initialized")


def get_session():
    """Get a new database session."""
    return Session(get_engine())
