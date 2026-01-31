"""
Background price monitoring script using APScheduler.
Runs periodically to check stock prices and send email alerts.
"""

import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import os
from dotenv import load_dotenv

import yfinance as yf
from database import get_session, get_all_assets, get_user_preferences

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Email configuration (from environment variables)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
FROM_EMAIL = os.getenv('FROM_EMAIL', SMTP_USERNAME)


def get_yfinance_symbol(symbol: str, market_type: str) -> str:
    """Convert symbol to yfinance format."""
    if market_type == "US":
        return symbol
    elif market_type == "HK":
        if not symbol.endswith(".HK"):
            return f"{symbol}.HK"
        return symbol
    elif market_type == "CN":
        if symbol.endswith((".SS", ".SZ")):
            return symbol
        return f"{symbol}.SS"
    return symbol


def get_current_price(symbol: str, market_type: str) -> float:
    """
    Fetch current stock price using yfinance.

    Args:
        symbol: Stock symbol
        market_type: Market type (US, HK, CN)

    Returns:
        Current price as float, or None if unavailable
    """
    try:
        yf_symbol = get_yfinance_symbol(symbol, market_type)
        logger.info(f"Fetching price for {yf_symbol}")

        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        # Try multiple price fields
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('lastPrice')

        if price is None:
            # Fallback to historical data
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]

        if price:
            logger.info(f"{symbol}: ${price:.2f}")
            return float(price)

        logger.warning(f"Could not retrieve price for {symbol}")
        return None

    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return None


def send_email_alert(to_email: str, asset_name: str, symbol: str,
                     current_price: float, threshold: float) -> bool:
    """
    Send an email alert for price threshold breach.

    Args:
        to_email: Recipient email address
        asset_name: Name of the asset
        symbol: Stock symbol
        current_price: Current stock price
        threshold: Alert threshold price

    Returns:
        True if email sent successfully, False otherwise
    """
    if not all([SMTP_USERNAME, SMTP_PASSWORD, to_email]):
        logger.error("Missing email configuration. Cannot send alert.")
        return False

    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🚨 Price Alert: {asset_name} ({symbol}) - ${current_price:.2f}"
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email

        # Create email body
        html_content = f"""
        <html>
        <body>
            <h2>🔔 Price Alert Triggered</h2>
            <p>Your price alert for <strong>{asset_name} ({symbol})</strong> has been triggered.</p>

            <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse;">
                <tr>
                    <td><strong>Stock</strong></td>
                    <td>{asset_name} ({symbol})</td>
                </tr>
                <tr>
                    <td><strong>Current Price</strong></td>
                    <td>${current_price:.2f}</td>
                </tr>
                <tr>
                    <td><strong>Your Threshold</strong></td>
                    <td>${threshold:.2f}</td>
                </tr>
                <tr>
                    <td><strong>Status</strong></td>
                    <td style="color: red;"><strong>⚠️ Price Below Threshold</strong></td>
                </tr>
                <tr>
                    <td><strong>Time</strong></td>
                    <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>

            <p><em>This is an automated message from OdinOracle Price Monitor.</em></p>
        </body>
        </html>
        """

        part = MIMEText(html_content, 'html')
        msg.attach(part)

        # Send email
        logger.info(f"Sending email alert to {to_email}")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email alert sent successfully to {to_email}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return False


def check_price_alerts():
    """
    Main job function to check all assets for price alerts.
    Called by the scheduler at configured intervals.
    """
    logger.info("=" * 60)
    logger.info("Starting price alert check...")
    logger.info("=" * 60)

    # Get user preferences for email
    prefs = get_user_preferences()
    if not prefs or not prefs.email_address:
        logger.warning("No user email configured. Skipping price alerts.")
        return

    user_email = prefs.email_address

    # Get all assets
    assets = get_all_assets()
    if not assets:
        logger.info("No assets in database to monitor.")
        return

    logger.info(f"Checking {len(assets)} assets for price alerts...")

    alerts_triggered = 0

    for asset in assets:
        # Skip if no alert threshold set
        if asset.alert_price_threshold is None:
            logger.debug(f"No threshold set for {asset.symbol}, skipping...")
            continue

        # Fetch current price
        current_price = get_current_price(asset.symbol, asset.market_type)

        if current_price is None:
            logger.warning(f"Could not fetch price for {asset.symbol}, skipping...")
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            continue

        # Check if price is below threshold
        if current_price < asset.alert_price_threshold:
            logger.warning(
                f"ALERT: {asset.symbol} (${current_price:.2f}) is below threshold "
                f"(${asset.alert_price_threshold:.2f})"
            )

            # Send email alert
            success = send_email_alert(
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

        # Small delay between requests to be respectful to yfinance
        time.sleep(1)

    logger.info("=" * 60)
    logger.info(f"Price alert check complete. Alerts triggered: {alerts_triggered}")
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
