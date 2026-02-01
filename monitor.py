"""
Background price monitoring script using APScheduler.
Runs periodically to check stock prices and send email alerts.
Refactored to use services layer.
"""

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from database import get_all_assets, get_user_preferences
from services.market_data import MarketDataService
from services.notification import EmailService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_price_alerts():
    """
    Main job function to check all assets for price alerts.
    Called by the scheduler at configured intervals.
    Optimized to use batch price fetching for improved performance.
    """
    logger.info("=" * 60)
    logger.info("Starting price alert check...")
    logger.info("=" * 60)

    # Initialize email service
    email_service = EmailService()

    # Get user preferences for email
    prefs = get_user_preferences()
    if not prefs or not prefs.email_address:
        logger.warning("No user email configured. Skipping price alerts.")
        return

    user_email = prefs.email_address

    # Check if email service is configured
    if not email_service.is_configured():
        logger.warning("Email service not configured. Please check SMTP settings.")
        return

    # Get all assets
    assets = get_all_assets()
    if not assets:
        logger.info("No assets in database to monitor.")
        return

    # Filter assets that have alert thresholds set
    assets_with_alerts = [a for a in assets if a.alert_price_threshold is not None]

    if not assets_with_alerts:
        logger.info("No assets with alert thresholds configured.")
        return

    logger.info(f"Checking {len(assets_with_alerts)} assets with price alerts...")

    # Prepare asset list for batch fetching
    asset_tuples = [(a.symbol, a.market_type) for a in assets_with_alerts]

    # Fetch all prices in parallel using batch method
    logger.info("Fetching prices in parallel...")
    prices = MarketDataService.get_current_prices_batch(asset_tuples, max_workers=5)

    alerts_triggered = 0
    assets_checked = 0

    # Process results
    for asset in assets_with_alerts:
        current_price = prices.get(asset.symbol)

        if current_price is None:
            logger.warning(f"Could not fetch price for {asset.symbol}, skipping...")
            continue

        assets_checked += 1

        # Check if price is below threshold
        if current_price < asset.alert_price_threshold:
            logger.warning(
                f"ALERT: {asset.symbol} (${current_price:.2f}) is below threshold "
                f"(${asset.alert_price_threshold:.2f})"
            )

            # Send email alert using EmailService
            success = email_service.send_price_alert(
                to_email=user_email,
                asset_name=asset.name,
                symbol=asset.symbol,
                current_price=current_price,
                threshold=asset.alert_price_threshold
            )

            if success:
                alerts_triggered += 1
        else:
            logger.info(
                f"{asset.symbol}: ${current_price:.2f} (threshold: ${asset.alert_price_threshold:.2f}) - OK"
            )

    logger.info("=" * 60)
    logger.info(f"Price alert check complete. Checked: {assets_checked}, Alerts triggered: {alerts_triggered}")
    logger.info("=" * 60)


def start_monitor_scheduler():
    """
    Start the background scheduler for price monitoring.
    Runs every hour during weekdays (market hours).
    """
    scheduler = BackgroundScheduler()

    # Schedule job to run every hour Monday through Friday
    # This runs from 9:30 AM to 4:00 PM EST on weekdays
    scheduler.add_job(
        check_price_alerts,
        trigger=CronTrigger(day_of_week='mon-fri', hour='9-16', minute='0'),
        id='price_alert_check',
        name='Price Alert Check',
        replace_existing=True
    )

    # Also run once at startup for testing
    logger.info("Running initial price check on startup...")
    check_price_alerts()

    # Start the scheduler
    scheduler.start()
    logger.info("Price monitor scheduler started. Running every hour on weekdays.")

    return scheduler


def run_one_time_check():
    """Run a single price alert check (useful for testing)."""
    logger.info("Running one-time price alert check...")
    check_price_alerts()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run once and exit
        run_one_time_check()
    else:
        # Run continuous scheduler
        try:
            scheduler = start_monitor_scheduler()
            print("\n" + "=" * 60)
            print("OdinOracle Price Monitor is running...")
            print("Press Ctrl+C to stop.")
            print("=" * 60 + "\n")

            # Keep the script running
            import time
            while True:
                time.sleep(1)

        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down price monitor...")
            scheduler.shutdown()
            logger.info("Price monitor stopped.")
