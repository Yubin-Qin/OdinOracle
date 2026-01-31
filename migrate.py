"""
Database migration script for OdinOracle.
Adds new columns to existing database schema.
"""

import sqlite3
import os

DB_FILE = "odin_oracle.db"


def migrate_userpreferences_add_language():
    """Add language column to userpreferences table if it doesn't exist."""
    if not os.path.exists(DB_FILE):
        print(f"Database {DB_FILE} does not exist. Nothing to migrate.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(userpreferences)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'language' not in columns:
            print("Adding 'language' column to userpreferences table...")
            cursor.execute(
                "ALTER TABLE userpreferences ADD COLUMN language TEXT DEFAULT 'en'"
            )
            conn.commit()
            print("✓ Added 'language' column successfully.")
        else:
            print("✓ Column 'language' already exists in userpreferences table.")

    except sqlite3.OperationalError as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()


def migrate_assetdailymetric_table():
    """
    Create the assetdailymetric table if it doesn't exist.
    This table stores daily technical indicators.
    """
    if not os.path.exists(DB_FILE):
        print(f"Database {DB_FILE} does not exist. Nothing to migrate.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='assetdailymetric'"
        )
        table_exists = cursor.fetchone()

        if not table_exists:
            print("Creating 'assetdailymetric' table...")
            cursor.execute("""
                CREATE TABLE assetdailymetric (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id INTEGER NOT NULL REFERENCES asset(id),
                    metric_date DATE NOT NULL,
                    close_price REAL NOT NULL,
                    sma_20 REAL,
                    sma_50 REAL,
                    sma_200 REAL,
                    rsi_14 REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bollinger_upper REAL,
                    bollinger_middle REAL,
                    bollinger_lower REAL,
                    bollinger_bandwidth REAL,
                    volume INTEGER,
                    volume_sma_20 REAL,
                    volume_ratio REAL,
                    overall_signal TEXT,
                    confidence_score INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            print("✓ Created 'assetdailymetric' table successfully.")
        else:
            print("✓ Table 'assetdailymetric' already exists.")

    except sqlite3.OperationalError as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()


def run_all_migrations():
    """Run all pending migrations."""
    print("=" * 60)
    print("OdinOracle Database Migration")
    print("=" * 60)

    migrate_userpreferences_add_language()
    migrate_assetdailymetric_table()

    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_migrations()
