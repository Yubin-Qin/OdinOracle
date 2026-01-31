"""
OdinOracle - Streamlit Application
Portfolio tracking, AI assistant, and price monitoring dashboard.
"""

import streamlit as st
import os
import pandas as pd
import logging
from datetime import date, datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from database import (
    init_db, get_session, add_asset, get_all_assets, get_asset_by_id,
    update_asset_alert_threshold, add_transaction, get_transactions_by_asset,
    get_all_transactions, save_user_email, get_user_preferences
)
from tools import get_stock_price, search_stock_news, get_stock_info
from llm_engine import LLMClient, DEFAULT_SYSTEM_PROMPT
from technical_analysis import get_technical_indicators, analyze_portfolio
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="OdinOracle - AI Investment Assistant",
    page_icon="📈",
    layout="wide"
)

# Initialize database
init_db()


# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_client" not in st.session_state:
    st.session_state.llm_client = None

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False

if "market_report" not in st.session_state:
    st.session_state.market_report = None


# ==================== HELPER FUNCTIONS ====================
def send_test_email(to_address: str) -> bool:
    """Send a test email to verify SMTP settings."""
    try:
        from monitor import send_email_alert
        return send_email_alert(
            to_email=to_address,
            asset_name="Test Asset",
            symbol="TEST",
            current_price=100.0,
            threshold=95.0
        )
    except Exception as e:
        st.error(f"Failed to send test email: {e}")
        return False


def send_market_report_email(to_address: str, report: str) -> bool:
    """Send the daily market report via email."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    FROM_EMAIL = os.getenv('FROM_EMAIL', SMTP_USERNAME)

    if not all([SMTP_USERNAME, SMTP_PASSWORD, to_address]):
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"📊 OdinOracle Daily Market Briefing - {date.today().isoformat()}"
        msg['From'] = FROM_EMAIL
        msg['To'] = to_address

        # Create HTML email body
        html_content = f"""
        <html>
        <body>
            <h2>📈 OdinOracle Daily Market Briefing</h2>
            <p><strong>Date:</strong> {date.today().strftime('%Y-%m-%d')}</p>

            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 20px;">
                <h3>Report Summary:</h3>
                <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6;">{report}</pre>
            </div>

            <p style="margin-top: 30px; color: gray; font-size: 12px;">
                <em>This is an automated message from OdinOracle Market Intelligence.</em>
            </p>
        </body>
        </html>
        """

        part = MIMEText(html_content, 'html')
        msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        return True

    except Exception as e:
        logger.error(f"Failed to send market report email: {e}")
        return False


def auto_initialize_llm():
    """Auto-initialize LLM from .env if configuration exists."""
    if st.session_state.llm_initialized:
        return

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL")

    # Check if all required env vars are set
    if api_key and model_name:
        try:
            # For cloud mode, empty string base_url should be None
            url = base_url if base_url else None

            st.session_state.llm_client = LLMClient(
                mode="cloud",
                model_name=model_name,
                base_url=url,
                api_key=api_key
            )

            # Test with a simple invoke
            test_response = st.session_state.llm_client.invoke("Hello")
            if test_response:
                st.session_state.llm_initialized = True
                st.success("✅ LLM auto-initialized from .env")
        except Exception as e:
            st.session_state.llm_client = None
            st.warning(f"⚠️ Failed to auto-initialize LLM: {e}")


def generate_market_report():
    """Generate a market intelligence report based on portfolio and news."""
    assets = get_all_assets()

    if not assets:
        return "No assets in portfolio to generate report."

    # Get top 5 holdings
    top_assets = assets[:5]

    # Gather information
    context = f"Portfolio Overview (Top {len(top_assets)} holdings):\n\n"

    for asset in top_assets:
        context += f"- {asset.name} ({asset.symbol}, {asset.market_type})\n"

    context += "\n" + "=" * 50 + "\n"
    context += "Fetching latest market news...\n\n"

    # Search news for each top asset
    for asset in top_assets:
        try:
            news = search_stock_news.invoke({
                "query": f"{asset.symbol} stock news",
                "max_results": 2
            })
            context += f"### {asset.symbol} News:\n{news}\n\n"
        except:
            context += f"### {asset.symbol} News:\nUnable to fetch news.\n\n"

    # Add general market news
    try:
        market_news = search_stock_news.invoke({
            "query": "global stock market news today",
            "max_results": 3
        })
        context += f"### Global Market News:\n{market_news}\n\n"
    except:
        context += "### Global Market News:\nUnable to fetch news.\n\n"

    # Generate summary using LLM
    if st.session_state.llm_client is None:
        return context + "\n\n[Note: LLM not configured, showing raw data only]"

    system_prompt = """You are OdinOracle, an AI investment analyst.
Generate a concise daily briefing that:
1. Summarizes key market movements
2. Highlights risks for the portfolio
3. Identifies opportunities
4. Provides actionable insights

Be specific and data-driven. Keep it under 500 words."""

    try:
        report = st.session_state.llm_client.invoke(
            context + "\n\nPlease provide a comprehensive market briefing summarizing risks and opportunities.",
            system_message=system_prompt
        )
        return report
    except Exception as e:
        return f"Error generating report: {e}"


# ==================== SIDEBAR ====================
def render_sidebar():
    """Render the sidebar with settings and forms."""
    st.sidebar.title("⚙️ Settings")

    # --- LLM Settings ---
    st.sidebar.subheader("🤖 LLM Configuration")

    llm_mode = st.sidebar.radio(
        "LLM Mode",
        ["Cloud (OpenAI-Compatible)", "Local (Ollama)"],
        index=0,
        help="Choose between cloud OpenAI-compatible API or local Ollama server"
    )

    mode_value = "cloud" if llm_mode == "Cloud (OpenAI-Compatible)" else "local"

    if mode_value == "cloud":
        model_name = st.sidebar.text_input(
            "Model Name / Endpoint ID",
            value=os.getenv("OPENAI_MODEL", "gpt-4o"),
            help="Model name or endpoint ID (e.g., gpt-4o, ep-2025..., deepseek-chat)"
        )
        api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Your API key (OpenAI, Volcano, DeepSeek, etc.)"
        )
        base_url = st.sidebar.text_input(
            "Base URL (Optional)",
            value=os.getenv("OPENAI_BASE_URL", ""),
            placeholder="Leave empty for official OpenAI, or enter custom URL",
            help="Custom API base URL (e.g., https://ark.cn-beijing.volces.com/api/v3)"
        )
        base_url = base_url if base_url else None
    else:
        model_name = st.sidebar.text_input(
            "Model Name",
            value=os.getenv("LOCAL_MODEL", "qwen2.5:14b"),
            help="Local model name (e.g., qwen2.5:14b, deepseek-r1)"
        )
        base_url = st.sidebar.text_input(
            "Base URL",
            value=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
            help="Local server URL"
        )
        api_key = "ollama"

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random"
    )

    if st.sidebar.button("Update LLM Settings", use_container_width=True):
        try:
            st.session_state.llm_client = LLMClient(
                mode=mode_value,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature
            )
            st.session_state.agent_executor = None
            st.session_state.llm_initialized = True
            st.sidebar.success("✅ LLM settings updated!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

    # --- Email Settings ---
    st.sidebar.subheader("📧 Email Alerts")

    prefs = get_user_preferences()
    current_email = prefs.email_address if prefs else ""

    email_input = st.sidebar.text_input(
        "Email Address",
        value=current_email,
        help="Email address for price alerts"
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Save Email", use_container_width=True):
            if email_input:
                save_user_email(email_input)
                st.sidebar.success("✅ Email saved!")
            else:
                st.sidebar.warning("⚠️ Please enter an email address")
    with col2:
        if st.button("Test Email", use_container_width=True):
            if email_input:
                with st.spinner("Sending test email..."):
                    if send_test_email(email_input):
                        st.sidebar.success("✅ Test email sent!")
                    else:
                        st.sidebar.error("❌ Failed to send")
            else:
                st.sidebar.warning("⚠️ Enter email first")


# ==================== MAIN CONTENT ====================
def render_portfolio_summary():
    """Render portfolio overview section."""
    st.subheader("📊 Portfolio Summary")

    assets = get_all_assets()
    transactions = get_all_transactions()

    if not assets:
        st.info("No assets in portfolio. Go to 'Manage Portfolio' to add holdings!")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Assets", f"{len(assets)}")

    with col2:
        st.metric("Total Transactions", f"{len(transactions)}")

    with col3:
        prefs = get_user_preferences()
        email_status = "✅ Configured" if prefs and prefs.email_address else "❌ Not set"
        st.metric("Email Alerts", email_status)


def render_asset_watchlist():
    """Render asset watchlist with price monitoring and alert configuration."""
    st.subheader("📈 Asset Watchlist")

    assets = get_all_assets()

    if not assets:
        st.info("No assets in watchlist.")
        return

    for asset in assets:
        with st.expander(f"**{asset.name}** ({asset.symbol}) - {asset.market_type}"):
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                if st.button(f"🔄 Refresh Price", key=f"price_{asset.id}"):
                    with st.spinner("Fetching price..."):
                        price_info = get_stock_price.invoke({
                            "symbol": asset.symbol,
                            "market_type": asset.market_type
                        })
                        st.info(price_info)

            with col2:
                current_threshold = asset.alert_price_threshold
                new_threshold = st.number_input(
                    "Alert Price Threshold ($)",
                    min_value=0.0,
                    step=0.01,
                    value=current_threshold if current_threshold else 0.0,
                    key=f"threshold_{asset.id}",
                    help="Email alert sent when price falls below this value"
                )

                if st.button("Set Alert", key=f"alert_{asset.id}"):
                    update_asset_alert_threshold(asset.id, new_threshold if new_threshold > 0 else None)
                    st.success(f"Alert threshold set to ${new_threshold:.2f}")
                    st.rerun()

            with col3:
                if st.button(f"📜 View Transactions", key=f"tx_{asset.id}"):
                    txs = get_transactions_by_asset(asset.id)
                    if txs:
                        st.write("**Transaction History:**")
                        for tx in txs:
                            st.text(
                                f"{tx.transaction_date} | {tx.transaction_type.upper()} | "
                                f"{tx.quantity} shares @ ${tx.price:.2f}"
                            )
                    else:
                        st.info("No transactions recorded.")


def render_add_asset_form():
    """Render form to add new assets."""
    st.subheader("➕ Add New Asset")

    with st.form("add_asset_form"):
        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input("Symbol", placeholder="e.g., NVDA, 0700, 600519", key="new_asset_symbol")
            name = st.text_input("Company Name", placeholder="e.g., NVIDIA", key="new_asset_name")

        with col2:
            market_type = st.selectbox("Market", ["US", "HK", "CN"], key="new_asset_market")
            alert_threshold = st.number_input(
                "Initial Alert Threshold ($)",
                min_value=0.0,
                step=0.01,
                value=0.0,
                key="new_asset_threshold",
                help="Optional: Set price alert threshold"
            )

        submitted = st.form_submit_button("Add Asset", use_container_width=True)

        if submitted:
            if symbol and name:
                threshold = alert_threshold if alert_threshold > 0 else None
                add_asset(symbol.upper(), name, market_type, threshold)
                st.success(f"✅ Added {name} ({symbol.upper()}) to watchlist!")
                st.rerun()
            else:
                st.error("❌ Please fill in symbol and name.")


def render_manage_portfolio():
    """Render portfolio management interface for adding existing positions."""
    st.subheader("📁 Manage Portfolio - Add Existing Position")

    assets = get_all_assets()

    if not assets:
        st.info("No assets available. Add an asset first in the 'Add Asset' tab.")
        return

    with st.form("add_existing_position"):
        col1, col2 = st.columns(2)

        with col1:
            asset_options = {f"{a.name} ({a.symbol})": a.symbol for a in assets}
            selected_symbol = st.selectbox("Select Asset*", options=list(asset_options.keys()))

            quantity = st.number_input(
                "Quantity*",
                min_value=0.0,
                step=0.01,
                value=1.0,
                help="Number of shares you hold"
            )

            purchase_date = st.date_input("Purchase Date", value=date.today())

        with col2:
            avg_cost = st.number_input(
                "Average Cost per Share*",
                min_value=0.0,
                step=0.01,
                value=0.0,
                help="Your average purchase price per share"
            )

            st.info(f"💰 Total Value: ${quantity * avg_cost:.2f}" if avg_cost > 0 else "")

        submitted = st.form_submit_button("Add Position", use_container_width=True)

        if submitted:
            if quantity > 0 and avg_cost > 0:
                # Get asset by symbol
                asset = next((a for a in assets if a.symbol == asset_options[selected_symbol]), None)
                if asset:
                    add_transaction(
                        asset_id=asset.id,
                        transaction_date=purchase_date,
                        transaction_type="buy",
                        quantity=quantity,
                        price=avg_cost
                    )
                    st.success(f"✅ Added {quantity} shares of {asset.symbol} @ ${avg_cost:.2f}!")
                    st.rerun()
            else:
                st.error("❌ Please enter valid quantity and average cost.")

    # Show recent positions
    st.markdown("---")
    st.markdown("### Recent Positions Added")

    transactions = get_all_transactions()[-10:]  # Last 10
    if transactions:
        for tx in reversed(transactions):
            asset = get_asset_by_id(tx.asset_id)
            if asset:
                st.text(
                    f"{tx.transaction_date} | {tx.transaction_type.upper()} "
                    f"{asset.symbol} | {tx.quantity} shares @ ${tx.price:.2f}"
                )
    else:
        st.info("No positions added yet.")


def render_market_intel():
    """Render AI Market Intelligence Dashboard."""
    st.subheader("🌐 AI Market Intelligence")

    st.markdown("""
    Generate a daily briefing based on your portfolio holdings and latest market news.
    The AI will analyze risks and opportunities across your investments.
    """)

    if st.button("📊 Generate Daily Briefing", type="primary", use_container_width=True):
        if st.session_state.llm_client is None:
            st.error("❌ LLM not configured. Please set up LLM settings in the sidebar.")
            return

        with st.spinner("Fetching market data and generating analysis..."):
            report = generate_market_report()
            st.session_state.market_report = report

        st.markdown("---")
        st.markdown("### 📑 Daily Briefing Report")
        st.markdown(report)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"daily_briefing_{date.today().isoformat()}.txt",
                mime="text/plain"
            )
        with col2:
            # Send to Email button
            prefs = get_user_preferences()
            if prefs and prefs.email_address:
                if st.button("📧 Send to Email", use_container_width=True):
                    with st.spinner("Sending email..."):
                        if send_market_report_email(prefs.email_address, report):
                            st.success(f"✅ Report sent to {prefs.email_address}!")
                        else:
                            st.error("❌ Failed to send email. Check SMTP settings.")
            else:
                st.button("📧 Send to Email", use_container_width=True, disabled=True)
                st.info("⚠️ Configure email in sidebar first")


def render_technical_analysis():
    """Render Technical Analysis with RSI and SMA indicators."""
    st.subheader("📉 Technical Analysis & Signals")

    assets = get_all_assets()

    if not assets:
        st.info("No assets in portfolio. Add assets first to analyze.")
        return

    # Asset selector
    asset_options = {f"{a.name} ({a.symbol})": a for a in assets}
    selected = st.selectbox("Select Asset to Analyze", options=list(asset_options.keys()))
    asset = asset_options[selected]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Analyze", use_container_width=True):
            with st.spinner("Calculating indicators..."):
                tech_data = get_technical_indicators(asset.symbol, asset.market_type)

                if tech_data:
                    # Display metrics
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Current Price", f"${tech_data['current_price']}")
                    with m2:
                        rsi = tech_data['rsi']
                        rsi_delta = "🔴" if rsi and rsi > 70 else "🟢" if rsi and rsi < 30 else "🟡"
                        st.metric("RSI (14)", f"{rsi:.1f} {rsi_delta}" if rsi else "N/A")
                    with m3:
                        st.metric("SMA 20", f"${tech_data['sma20']}" if tech_data['sma20'] else "N/A")
                    with m4:
                        signal_emoji = "🟢" if tech_data['signal'] == "BUY" else "🔴" if tech_data['signal'] == "SELL" else "🟡"
                        st.metric("Signal", f"{signal_emoji} {tech_data['signal']}")

                    st.info(f"**Analysis:** {tech_data['signal_reason']}")

                    # Plot chart
                    st.markdown("### Price vs SMA20 (Last 60 Days)")
                    chart_data = tech_data['history_df']
                    st.line_chart(chart_data)
                else:
                    st.error("Failed to fetch technical data. Check symbol and market type.")

    with col2:
        st.markdown("### Portfolio Signals Overview")
        df = analyze_portfolio(assets)
        if not df.empty:
            # Style the Signal column
            def color_signal(val):
                if val == 'BUY':
                    return 'background-color: #90EE90'
                elif val == 'SELL':
                    return 'background-color: #FFB6C1'
                return 'background-color: #FFD700'

            styled = df.style.applymap(color_signal, subset=['Signal'])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No technical data available.")


def render_chat_interface():
    """Render AI chat interface with LangChain agent."""
    st.subheader("🤖 AI Investment Assistant")

    if st.session_state.llm_client is None:
        st.info("⚠️ Please configure LLM settings in the sidebar first.")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about stocks, portfolio, or market analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.agent_executor is None:
                        llm = st.session_state.llm_client.get_llm()
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", DEFAULT_SYSTEM_PROMPT),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                            MessagesPlaceholder(variable_name="agent_scratchpad")
                        ])
                        tools = [get_stock_price, search_stock_news, get_stock_info]
                        agent = create_tool_calling_agent(llm, tools, prompt_template)
                        st.session_state.agent_executor = AgentExecutor(
                            agent=agent,
                            tools=tools,
                            verbose=True,
                            handle_parsing_errors=True
                        )

                    response = st.session_state.agent_executor.invoke({
                        "input": prompt,
                        "chat_history": []
                    })

                    assistant_response = response["output"]
                    st.markdown(assistant_response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# ==================== MAIN APP ====================
def main():
    """Main application entry point."""
    st.title("📈 OdinOracle")
    st.markdown("*AI-Powered Investment Portfolio Tracker & Assistant*")

    # Auto-initialize LLM from .env
    auto_initialize_llm()

    # Render sidebar
    render_sidebar()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard", "📈 Watchlist", "➕ Add Asset",
        "📁 Manage Portfolio", "🌐 Market Intel", "📉 Technical Analysis", "🤖 AI Assistant"
    ])

    with tab1:
        render_portfolio_summary()

    with tab2:
        render_asset_watchlist()

    with tab3:
        render_add_asset_form()

    with tab4:
        render_manage_portfolio()

    with tab5:
        render_market_intel()

    with tab6:
        render_technical_analysis()

    with tab7:
        render_chat_interface()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "⚠️ OdinOracle is for informational purposes only. Not financial advice.</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
