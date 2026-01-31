"""
OdinOracle - Streamlit Application
Portfolio tracking, AI assistant, and price monitoring dashboard.
Refactored with services layer, enhanced dashboard, and bilingual support.
"""

import streamlit as st
import os
import pandas as pd
import logging
from datetime import date, datetime
from dotenv import load_dotenv

from database import (
    init_db, add_asset, get_all_assets, get_asset_by_id,
    update_asset_alert_threshold, add_transaction,
    get_transactions_by_asset, get_all_transactions,
    save_user_email, save_user_language, get_user_preferences,
    get_latest_metric, get_metrics_history
)
from tools import get_stock_price, search_stock_news, get_stock_info
from llm_engine import LLMClient, get_system_prompt
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Services
from services.market_data import MarketDataService
from services.notification import EmailService
from services.portfolio import PortfolioService
from services.common import normalize_symbol, infer_market_type

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

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

if "language" not in st.session_state:
    st.session_state.language = "en"


# ==================== TEXT DICTIONARY ====================
TEXT = {
    "en": {
        "app_title": "OdinOracle",
        "app_subtitle": "AI-Powered Investment Portfolio Tracker & Assistant",
        "dashboard": "Dashboard",
        "watchlist": "Watchlist",
        "add_asset": "Add Asset",
        "manage_portfolio": "Manage Portfolio",
        "market_intel": "Market Intel",
        "technical_analysis": "Technical Analysis",
        "ai_assistant": "AI Assistant",
        "total_net_worth": "Total Net Worth",
        "daily_pnl": "Daily PnL",
        "total_pnl": "Total PnL",
        "pnl_pct": "PnL %",
        "email_alerts": "Email Alerts",
        "configured": "Configured",
        "not_set": "Not set",
        "no_holdings": "No holdings in portfolio. Go to 'Manage Portfolio' to add positions!",
        "holdings_detail": "Holdings Detail",
        "symbol": "Symbol",
        "name": "Name",
        "quantity": "Quantity",
        "avg_cost": "Avg Cost",
        "current_price": "Current Price",
        "value": "Value",
        "pnl": "PnL",
        "r Signal": "Signal",
        "rsi": "RSI",
        "macd": "MACD",
        "bollinger": "Bollinger",
        "volume": "Volume",
        "strong_buy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strong_sell": "Strong Sell",
    },
    "zh": {
        "app_title": "OdinOracle",
        "app_subtitle": "AI驱动的投资组合追踪与助手",
        "dashboard": "仪表盘",
        "watchlist": "观察列表",
        "add_asset": "添加资产",
        "manage_portfolio": "管理投资组合",
        "market_intel": "市场情报",
        "technical_analysis": "技术分析",
        "ai_assistant": "AI助手",
        "total_net_worth": "总净值",
        "daily_pnl": "日盈亏",
        "total_pnl": "总盈亏",
        "pnl_pct": "盈亏%",
        "email_alerts": "邮件提醒",
        "configured": "已配置",
        "not_set": "未设置",
        "no_holdings": "投资组合中没有持仓。请前往'管理投资组合'添加头寸！",
        "holdings_detail": "持仓详情",
        "symbol": "代码",
        "name": "名称",
        "quantity": "数量",
        "avg_cost": "平均成本",
        "current_price": "当前价格",
        "value": "市值",
        "pnl": "盈亏",
        "r Signal": "信号",
        "rsi": "RSI",
        "macd": "MACD",
        "bollinger": "布林带",
        "volume": "成交量",
        "strong_buy": "强烈买入",
        "buy": "买入",
        "hold": "持有",
        "sell": "卖出",
        "strong_sell": "强烈卖出",
    }
}


def T(key: str):
    """Get translated text based on current language."""
    return TEXT[st.session_state.language].get(key, key)


# ==================== HELPER FUNCTIONS ====================
def send_test_email(to_address: str) -> bool:
    """Send a test email to verify SMTP settings."""
    email_service = EmailService()
    return email_service.send_price_alert(
        to_email=to_address,
        asset_name="Test Asset",
        symbol="TEST",
        current_price=100.0,
        threshold=95.0
    )


def auto_initialize_llm():
    """Auto-initialize LLM from .env if configuration exists."""
    if st.session_state.llm_initialized:
        return

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL")

    if api_key and model_name:
        try:
            url = base_url if base_url else None
            st.session_state.llm_client = LLMClient(
                mode="cloud",
                model_name=model_name,
                base_url=url,
                api_key=api_key,
                language=st.session_state.language
            )

            test_response = st.session_state.llm_client.invoke("Hello")
            if test_response:
                st.session_state.llm_initialized = True
                st.success("✅ LLM auto-initialized from .env")
        except Exception as e:
            st.session_state.llm_client = None
            st.warning(f"⚠️ Failed to auto-initialize LLM: {e}")


def ensure_asset_exists(symbol: str, name: str = None) -> bool:
    """Ensure an asset exists in the database. Auto-create if not."""
    asset = PortfolioService.get_asset_by_symbol(symbol)
    if asset:
        return True

    market_type = infer_market_type(symbol)

    try:
        add_asset(
            symbol=symbol.upper(),
            name=name or f"{symbol.upper()} (Auto-created)",
            market_type=market_type,
            alert_price_threshold=None
        )
        logger.info(f"Auto-created asset: {symbol} ({market_type})")
        return True
    except Exception as e:
        st.error(f"Failed to create asset {symbol}: {e}")
        return False


def get_signal_emoji(signal: str) -> str:
    """Get emoji for signal."""
    return {
        'STRONG_BUY': '🟢🟢',
        'BUY': '🟢',
        'HOLD': '🟡',
        'SELL': '🔴',
        'STRONG_SELL': '🔴🔴'
    }.get(signal, '📊')


def generate_market_report():
    """Generate a market intelligence report based on portfolio and news."""
    assets = get_all_assets()
    if not assets:
        return T('no_holdings') if st.session_state.language == 'zh' else "No assets in portfolio to generate report."

    top_assets = PortfolioService.get_top_holdings(limit=5)

    context = f"Portfolio Overview (Top {len(top_assets)} holdings):\n\n"
    for holding in top_assets:
        context += f"- {holding['name']} ({holding['symbol']}, {holding['market_type']})\n"

    context += "\n" + "=" * 50 + "\n"
    context += "Fetching latest market news...\n\n"

    for holding in top_assets:
        try:
            news = search_stock_news.invoke({
                "query": f"{holding['symbol']} stock news",
                "max_results": 2
            })
            context += f"### {holding['symbol']} News:\n{news}\n\n"
        except:
            context += f"### {holding['symbol']} News:\nUnable to fetch news.\n\n"

    try:
        market_news = search_stock_news.invoke({
            "query": "global stock market news today",
            "max_results": 3
        })
        context += f"### Global Market News:\n{market_news}\n\n"
    except:
        context += "### Global Market News:\nUnable to fetch news.\n\n"

    if st.session_state.llm_client is None:
        return context + "\n\n[Note: LLM not configured, showing raw data only]"

    system_prompt = get_system_prompt(st.session_state.language)
    system_prompt += """

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

    # --- Language Selection ---
    st.sidebar.subheader("🌐 Language / 语言")
    lang_options = {"English": "en", "中文": "zh"}
    selected_lang = st.sidebar.selectbox(
        "Select / 选择",
        options=list(lang_options.keys()),
        index=0 if st.session_state.language == "en" else 1
    )
    st.session_state.language = lang_options[selected_lang]

    # --- LLM Settings ---
    st.sidebar.subheader("🤖 LLM Configuration")

    llm_mode = st.sidebar.radio(
        "LLM Mode",
        ["Cloud (OpenAI-Compatible)", "Local (Ollama)"],
        index=0
    )
    mode_value = "cloud" if llm_mode == "Cloud (OpenAI-Compatible)" else "local"

    if mode_value == "cloud":
        model_name = st.sidebar.text_input(
            "Model Name / Endpoint ID",
            value=os.getenv("OPENAI_MODEL", "gpt-4o")
        )
        api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        base_url_input = st.sidebar.text_input(
            "Base URL (Optional)",
            value=os.getenv("OPENAI_BASE_URL", "")
        )
        base_url = base_url_input if base_url_input else None
    else:
        model_name = st.sidebar.text_input(
            "Model Name",
            value=os.getenv("LOCAL_MODEL", "qwen2.5:14b")
        )
        base_url = st.sidebar.text_input(
            "Base URL",
            value=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
        )
        api_key = "ollama"

    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    if st.sidebar.button("Update LLM Settings", use_container_width=True):
        try:
            st.session_state.llm_client = LLMClient(
                mode=mode_value, model_name=model_name,
                base_url=base_url, api_key=api_key, temperature=temperature,
                language=st.session_state.language
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
    email_input = st.sidebar.text_input("Email Address", value=current_email)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Save Email", use_container_width=True):
            if email_input:
                save_user_email(email_input)
                st.sidebar.success("✅ Email saved!")
            else:
                st.sidebar.warning("⚠️ Enter email first")
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
    """Render portfolio overview with net worth, PnL table and metrics grid."""
    st.subheader(f"📊 {T('dashboard')}")

    portfolio = PortfolioService.calculate_net_worth()

    if portfolio['total_value'] == 0:
        st.info(T('no_holdings'))
        return

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(T('total_net_worth'), f"${portfolio['total_value']:,.2f}")
    with col2:
        pnl_color = "🟢" if portfolio['daily_pnl'] >= 0 else "🔴"
        st.metric(f"{pnl_color} {T('daily_pnl')}", f"${portfolio['daily_pnl']:,.2f}")
    with col3:
        pnl_color = "🟢" if portfolio['total_pnl'] >= 0 else "🔴"
        st.metric(f"{pnl_color} {T('total_pnl')}", f"${portfolio['total_pnl']:,.2f}")
    with col4:
        pnl_pct_color = "🟢" if portfolio['total_pnl_pct'] >= 0 else "🔴"
        prefs = get_user_preferences()
        email_status = T('configured') if prefs and prefs.email_address else T('not_set')
        st.metric(f"{'🟢' if prefs and prefs.email_address else '❌'} {T('email_alerts')}", email_status)

    # Holdings table
    st.markdown(f"### {T('holdings_detail')}")

    holdings_data = []
    for h in portfolio['holdings']:
        pnl_emoji = "🟢" if h['pnl'] >= 0 else "🔴"
        holdings_data.append({
            T('symbol'): h['symbol'],
            T('name'): h['name'],
            T('quantity'): h['quantity'],
            T('avg_cost'): f"${h['avg_cost']:.2f}",
            T('current_price'): f"${h['current_price']:.2f}",
            T('value'): f"${h['current_value']:,.2f}",
            T('pnl'): f"{pnl_emoji} ${h['pnl']:,.2f}",
            T('pnl_pct'): f"{h['pnl_pct']:.2f}%"
        })

    if holdings_data:
        df = pd.DataFrame(holdings_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No holdings to display.")


def render_asset_watchlist():
    """Render asset watchlist with price monitoring and alert configuration."""
    st.subheader(f"📈 {T('watchlist')}")

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
                    min_value=0.0, step=0.01,
                    value=current_threshold if current_threshold else 0.0,
                    key=f"threshold_{asset.id}"
                )
                if st.button("Set Alert", key=f"alert_{asset.id}"):
                    update_asset_alert_threshold(asset.id, new_threshold if new_threshold > 0 else None)
                    st.success(f"Alert threshold set to ${new_threshold:.2f}")
                    st.rerun()

            with col3:
                # Show stored metrics if available
                metric = get_latest_metric(asset.id)
                if metric:
                    signal_emoji = get_signal_emoji(metric.overall_signal)
                    st.metric(
                        f"{signal_emoji} {T('r Signal')}",
                        f"{metric.overall_signal or 'N/A'} ({metric.confidence_score or 0}/10)"
                    )


def render_add_asset_form():
    """Render form to add new assets."""
    st.subheader(f"➕ {T('add_asset')}")

    with st.form("add_asset_form"):
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol*", placeholder="e.g., NVDA, 0700, 600519", key="new_asset_symbol")
            name = st.text_input("Company Name", placeholder="e.g., NVIDIA", key="new_asset_name")
        with col2:
            market_type = st.selectbox("Market*", ["US", "HK", "CN"], key="new_asset_market")
            alert_threshold = st.number_input("Initial Alert Threshold ($)", 0.0, 0.01, 0.0, key="new_asset_threshold")

        submitted = st.form_submit_button("Add Asset", use_container_width=True)
        if submitted:
            if symbol and name:
                threshold = alert_threshold if alert_threshold > 0 else None
                add_asset(symbol.upper(), name, market_type, threshold)
                st.success(f"✅ Added {name} ({symbol.upper()}) to watchlist!")
                st.rerun()
            else:
                st.error("❌ Symbol and name are required.")


def render_manage_portfolio():
    """Render portfolio management interface with auto-asset creation."""
    st.subheader(f"📁 {T('manage_portfolio')} - Add Position")

    with st.form("add_position_form"):
        col1, col2 = st.columns(2)
        with col1:
            symbol_input = st.text_input("Symbol*", placeholder="e.g., AAPL, TSLA", key="pos_symbol")
            quantity = st.number_input("Quantity*", min_value=0.0, step=0.01, value=1.0, key="pos_quantity")
            purchase_date = st.date_input("Purchase Date", value=date.today())
        with col2:
            avg_cost = st.number_input("Avg Cost per Share*", min_value=0.0, step=0.01, value=0.0, key="pos_cost")
            asset_name = st.text_input("Company Name (Optional)", placeholder="For new assets only", key="pos_name")
            if avg_cost > 0:
                st.info(f"💰 Total Value: ${quantity * avg_cost:.2f}")

        submitted = st.form_submit_button("Add Position", use_container_width=True)
        if submitted:
            if not symbol_input or quantity <= 0 or avg_cost <= 0:
                st.error("❌ Please fill in all required fields.")
            else:
                symbol = symbol_input.upper().strip()
                if ensure_asset_exists(symbol, asset_name or None):
                    asset = PortfolioService.get_asset_by_symbol(symbol)
                    if asset:
                        add_transaction(
                            asset_id=asset.id,
                            transaction_date=purchase_date,
                            transaction_type="buy",
                            quantity=quantity,
                            price=avg_cost
                        )
                        # Fetch and store metrics
                        MarketDataService.fetch_and_store_daily_metrics(asset.id, symbol, asset.market_type)
                        st.success(f"✅ Added {quantity} shares of {symbol} @ ${avg_cost:.2f}!")
                        st.rerun()

    # Show recent positions
    st.markdown("---")
    st.markdown("### Recent Positions Added")
    transactions = get_all_transactions()[-10:]
    if transactions:
        for tx in reversed(transactions):
            asset = get_asset_by_id(tx.asset_id)
            if asset:
                st.text(f"{tx.transaction_date} | {tx.transaction_type.upper()} | {asset.symbol} | {tx.quantity} shares @ ${tx.price:.2f}")
    else:
        st.info("No positions added yet.")


def render_market_intel():
    """Render AI Market Intelligence Dashboard."""
    st.subheader(f"🌐 {T('market_intel')}")
    st.markdown("Generate a daily briefing based on your portfolio holdings and latest market news.")

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
            prefs = get_user_preferences()
            if prefs and prefs.email_address:
                if st.button("📧 Send to Email", use_container_width=True):
                    email_service = EmailService()
                    with st.spinner("Sending email..."):
                        if email_service.send_market_report(prefs.email_address, report, date.today().isoformat()):
                            st.success(f"✅ Report sent to {prefs.email_address}!")
                        else:
                            st.error("❌ Failed to send email. Check SMTP settings.")
            else:
                st.button("📧 Send to Email", use_container_width=True, disabled=True)
                st.info("⚠️ Configure email in sidebar first")


def render_technical_analysis():
    """Render Technical Analysis with Factor Pack metrics grid."""
    st.subheader(f"📉 {T('technical_analysis')} & Signals")

    assets = get_all_assets()
    if not assets:
        st.info("No assets in portfolio. Add assets first to analyze.")
        return

    asset_options = {f"{a.name} ({a.symbol})": a for a in assets}
    selected = st.selectbox("Select Asset to Analyze", options=list(asset_options.keys()))
    asset = asset_options[selected]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Analyze & Store Metrics", use_container_width=True):
            # Fetch and store metrics first
            with st.spinner("Calculating indicators and storing..."):
                success = MarketDataService.fetch_and_store_daily_metrics(asset.id, asset.symbol, asset.market_type)

            # Get latest metrics from database
            metric = get_latest_metric(asset.id)

            if metric:
                # Display Factor Pack
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    price = metric.close_price
                    st.metric("Current Price", f"${price:.2f}")
                with m2:
                    rsi = metric.rsi_14
                    rsi_delta = "🔴" if rsi and rsi > 70 else "🟢" if rsi and rsi < 30 else "🟡"
                    st.metric(f"{T('rsi')} (14)", f"{rsi:.1f} {rsi_delta}" if rsi else "N/A")
                with m3:
                    signal_emoji = get_signal_emoji(metric.overall_signal)
                    st.metric(f"{signal_emoji} {T('r Signal')}", f"{metric.overall_signal or 'N/A'} ({metric.confidence_score or 0}/10)")
                with m4:
                    macd_hist = metric.macd_histogram
                    macd_status = "🟢" if macd_hist and macd_hist > 0 else "🔴" if macd_hist and macd_hist < 0 else "🟡"
                    st.metric(f"{T('macd')} Hist", f"{macd_hist:.3f} {macd_status}" if macd_hist else "N/A")

                st.info(f"**Analysis**: Signal={metric.overall_signal}, Confidence={metric.confidence_score}/10")

                # Factor Pack Details
                st.markdown("### Factor Pack Details")
                fp_col1, fp_col2 = st.columns(2)

                with fp_col1:
                    st.markdown("**Trend Indicators**")
                    if metric.sma_20:
                        st.text(f"📊 SMA 20: ${metric.sma_20:.2f}")
                    if metric.sma_50:
                        st.text(f"📊 SMA 50: ${metric.sma_50:.2f}")
                    if metric.sma_200:
                        st.text(f"📊 SMA 200: ${metric.sma_200:.2f}")

                    # Golden Cross Check
                    if metric.sma_20 and metric.sma_200:
                        if metric.sma_20 > metric.sma_200:
                            st.success("✅ Golden Cross (Bullish)")
                        else:
                            st.error("❌ Death Cross (Bearish)")

                with fp_col2:
                    st.markdown("**Volatility Indicators**")
                    if metric.bollinger_upper:
                        st.text(f"📈 BB Upper: ${metric.bollinger_upper:.2f}")
                        st.text(f"📊 BB Middle: ${metric.bollinger_middle:.2f}")
                        st.text(f"📉 BB Lower: ${metric.bollinger_lower:.2f}")
                    if metric.bollinger_bandwidth:
                        bbw = metric.bollinger_bandwidth
                        if bbw:
                            if bbw < 0.02:
                                st.info("🔒 Squeeze (potential breakout)")
                            else:
                                st.text(f"📏 Bandwidth: {bbw:.2%}")

                # Volume Indicator
                st.markdown("**Volume Analysis**")
                if metric.volume_ratio:
                    vr = metric.volume_ratio
                    vol_status = "🔥 High" if vr > 1.5 else "📉 Low" if vr < 0.7 else "➡️ Normal"
                    st.text(f"📊 Volume Ratio: {vr:.2f}x {vol_status}")

                # Bollinger Bands Chart
                st.markdown("### Price vs Bollinger Bands (60 Days)")
                hist_metrics = get_metrics_history(asset.id, 60)
                if hist_metrics:
                    chart_data = []
                    for m in reversed(hist_metrics):
                        chart_data.append({
                            'Date': m.metric_date.strftime('%Y-%m-%d'),
                            'Price': m.close_price,
                            'Upper Band': m.bollinger_upper,
                            'Lower Band': m.bollinger_lower
                        })
                    df_chart = pd.DataFrame(chart_data)
                    st.line_chart(df_chart.set_index('Date'))
                else:
                    st.info("No historical data available for charting.")

    with col2:
        st.markdown("### Portfolio Signals Overview")
        # Use stored metrics for all assets
        signals_data = []
        for a in assets:
            m = get_latest_metric(a.id)
            if m and m.overall_signal:
                signals_data.append({
                    'Symbol': a.symbol,
                    'Signal': f"{get_signal_emoji(m.overall_signal)} {m.overall_signal}",
                    'Confidence': f"{m.confidence_score}/10",
                    'RSI': f"{m.rsi_14:.1f}" if m.rsi_14 else 'N/A',
                    'MACD': f"{m.macd_histogram:.3f}" if m.macd_histogram else 'N/A'
                })

        if signals_data:
            df = pd.DataFrame(signals_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Run 'Analyze & Store Metrics' for each asset to see signals.")


def render_chat_interface():
    """Render AI chat interface with Aggressive Hedge Fund Partner persona."""
    st.subheader(f"🤖 {T('ai_assistant')}")

    if st.session_state.llm_client is None:
        st.info("⚠️ Please configure LLM settings in the sidebar first.")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about stocks, portfolio, or market analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    if st.session_state.agent_executor is None:
                        llm = st.session_state.llm_client.get_llm()
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", get_system_prompt(st.session_state.language)),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                            MessagesPlaceholder(variable_name="agent_scratchpad")
                        ])
                        tools = [get_stock_price, search_stock_news, get_stock_info]
                        agent = create_tool_calling_agent(llm, tools, prompt_template)
                        st.session_state.agent_executor = AgentExecutor(
                            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
                        )

                    response = st.session_state.agent_executor.invoke({"input": prompt, "chat_history": []})
                    assistant_response = response["output"]
                    st.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# ==================== MAIN APP ====================
def main():
    """Main application entry point."""
    st.title("📈 OdinOracle")
    st.markdown(f"*{T('app_subtitle')}*")

    # Auto-initialize LLM from .env
    auto_initialize_llm()

    # Render sidebar
    render_sidebar()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        f"📊 {T('dashboard')}", f"📈 {T('watchlist')}", f"➕ {T('add_asset')}",
        f"📁 {T('manage_portfolio')}", f"🌐 {T('market_intel')}", f"📉 {T('technical_analysis')}", f"🤖 {T('ai_assistant')}"
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
