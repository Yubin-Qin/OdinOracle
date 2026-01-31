"""
Metric Repository - data access layer for AssetDailyMetric model.
"""

from typing import Optional, List
from datetime import date
from sqlmodel import Session, select

from db_engine import get_engine
from models import AssetDailyMetric


class MetricRepository:
    """Repository for AssetDailyMetric CRUD operations."""

    @staticmethod
    def save(metric: AssetDailyMetric) -> AssetDailyMetric:
        """
        Save or update a daily metric record.
        Uses upsert logic: if exists for asset_id + date, update; otherwise insert.
        """
        with Session(get_engine()) as session:
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

    @staticmethod
    def get_latest(asset_id: int) -> Optional[AssetDailyMetric]:
        """Get the most recent daily metric for an asset."""
        with Session(get_engine()) as session:
            statement = select(AssetDailyMetric).where(
                AssetDailyMetric.asset_id == asset_id
            ).order_by(AssetDailyMetric.metric_date.desc()).limit(1)
            return session.exec(statement).first()

    @staticmethod
    def get_history(asset_id: int, days: int = 60) -> List[AssetDailyMetric]:
        """Get historical metrics for an asset."""
        with Session(get_engine()) as session:
            statement = select(AssetDailyMetric).where(
                AssetDailyMetric.asset_id == asset_id
            ).order_by(AssetDailyMetric.metric_date.desc()).limit(days)
            results = session.exec(statement)
            return list(results.all())

    @staticmethod
    def get_by_date(asset_id: int, metric_date: date) -> Optional[AssetDailyMetric]:
        """Get a specific metric by date."""
        with Session(get_engine()) as session:
            statement = select(AssetDailyMetric).where(
                AssetDailyMetric.asset_id == asset_id,
                AssetDailyMetric.metric_date == metric_date
            )
            return session.exec(statement).first()

    @staticmethod
    def get_date_range(asset_id: int, start_date: date, end_date: date) -> List[AssetDailyMetric]:
        """Get metrics for a specific date range."""
        with Session(get_engine()) as session:
            statement = select(AssetDailyMetric).where(
                AssetDailyMetric.asset_id == asset_id,
                AssetDailyMetric.metric_date >= start_date,
                AssetDailyMetric.metric_date <= end_date
            ).order_by(AssetDailyMetric.metric_date.asc())
            results = session.exec(statement)
            return list(results.all())
