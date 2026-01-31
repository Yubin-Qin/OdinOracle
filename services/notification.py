"""
Notification service for sending emails.
Centralized SMTP handling for all email communications.
Refactored to use centralized configuration.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)


class EmailService:
    """
    Service for sending emails via SMTP.
    Configuration loaded from centralized config module.
    """

    def __init__(self):
        """Initialize email service with configuration from centralized config."""
        settings = get_settings()
        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password
        self.from_email = settings.email_from

    def is_configured(self) -> bool:
        """Check if email service is properly configured."""
        settings = get_settings()
        return settings.is_email_configured

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        plain_text: Optional[str] = None
    ) -> bool:
        """
        Send an HTML email.

        Args:
            to_email: Recipient email address
            subject: Email subject line
            html_content: HTML body content
            plain_text: Optional plain text fallback

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.is_configured():
            logger.error("Email service not configured. Missing credentials.")
            return False

        if not to_email:
            logger.error("Recipient email address is required.")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email

            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            if plain_text:
                text_part = MIMEText(plain_text, 'plain')
                msg.attach(text_part)

            logger.info(f"Sending email to {to_email}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_price_alert(
        self,
        to_email: str,
        asset_name: str,
        symbol: str,
        current_price: float,
        threshold: float
    ) -> bool:
        """Send a price alert email."""
        from datetime import datetime

        html_content = f"""
        <html>
        <body>
            <h2>🔔 Price Alert Triggered</h2>
            <p>Your price alert for <strong>{asset_name} ({symbol})</strong> has been triggered.</p>
            <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse;">
                <tr><td><strong>Stock</strong></td><td>{asset_name} ({symbol})</td></tr>
                <tr><td><strong>Current Price</strong></td><td>${current_price:.2f}</td></tr>
                <tr><td><strong>Your Threshold</strong></td><td>${threshold:.2f}</td></tr>
                <tr><td><strong>Status</strong></td><td style="color: red;"><strong>⚠️ Price Below Threshold</strong></td></tr>
                <tr><td><strong>Time</strong></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
            <p><em>This is an automated message from OdinOracle Price Monitor.</em></p>
        </body>
        </html>
        """

        subject = f"🚨 Price Alert: {asset_name} ({symbol}) - ${current_price:.2f}"
        return self.send_email(to_email, subject, html_content)

    def send_market_report(self, to_email: str, report: str, report_date: str) -> bool:
        """Send a market intelligence report email."""
        html_content = f"""
        <html>
        <body>
            <h2>📈 OdinOracle Daily Market Briefing</h2>
            <p><strong>Date:</strong> {report_date}</p>
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 20px;">
                <h3>Report Summary:</h3>
                <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6;">{report}</pre>
            </div>
            <p style="margin-top: 30px; color: gray; font-size: 12px;"><em>This is an automated message from OdinOracle Market Intelligence.</em></p>
        </body>
        </html>
        """

        subject = f"📊 OdinOracle Daily Market Briefing - {report_date}"
        return self.send_email(to_email, subject, html_content)

    def send_signal_alert(
        self,
        to_email: str,
        symbol: str,
        signal: str,
        confidence: int,
        price: float,
        reasons: list
    ) -> bool:
        """Send a trading signal alert email."""
        signal_emoji = {
            'STRONG_BUY': '🟢🟢',
            'BUY': '🟢',
            'HOLD': '🟡',
            'SELL': '🔴',
            'STRONG_SELL': '🔴🔴'
        }.get(signal, '📊')

        html_content = f"""
        <html>
        <body>
            <h2>{signal_emoji} Trading Signal Alert</h2>
            <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse;">
                <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
                <tr><td><strong>Signal</strong></td><td><strong>{signal}</strong> (Confidence: {confidence}/10)</td></tr>
                <tr><td><strong>Current Price</strong></td><td>${price:.2f}</td></tr>
            </table>
            <h3>Analysis:</h3>
            <ul>
                {"".join(f"<li>{r}</li>" for r in reasons)}
            </ul>
            <p><em>This is an automated signal from OdinOracle Quant Engine.</em></p>
        </body>
        </html>
        """

        subject = f"📊 Signal Alert: {symbol} - {signal}"
        return self.send_email(to_email, subject, html_content)
