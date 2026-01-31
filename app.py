"""
OdinOracle - Streamlit Application
Portfolio tracking, AI assistant, and price monitoring dashboard.
"""

import streamlit as st
import os
from datetime import date, datetime
from dotenv import load_dotenv

from database import (
    init_db, get_session, add_asset, get_all_assets, get_asset_by_id,
    update_asset_alert_threshold, add_transaction, get_transactions_by_asset,
    get_all_transactions, save_user_email, get_user_preferences
)
from tools import get_stock_price, search_stock_news, get_stock_info
from llm_engine import LLMClient, DEFAULT_SYSTEM_PROMPT
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
        # Treat empty string as None for official OpenAI
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
        api_key = "ollama"  # Dummy key for local

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random"
    )

    # Initialize/Update LLM client
    if st.sidebar.button("Update LLM Settings", use_container_width=True):
        try:
            st.session_state.llm_client = LLMClient(
                mode=mode_value,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature
            )
            st.session_state.agent_executor = None  # Reset agent
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

    if st.sidebar.button("Save Email", use_container_width=True):
        if email_input:
            save_user_email(email_input)
            st.sidebar.success("✅ Email saved!")
        else:
            st.sidebar.warning("⚠️ Please enter an email address")

    # --- Add Transaction ---
    st.sidebar.subheader("📝 Add Transaction")

    assets = get_all_assets()
    if assets:
        asset_options = {f"{a.name} ({a.symbol})": a.id for a in assets}
        selected_asset = st.sidebar.selectbox("Asset", options=list(asset_options.keys()))

        tx_type = st.sidebar.selectbox("Type", ["buy", "sell"])
        tx_date = st.sidebar.date_input("Date", value=date.today())
        tx_quantity = st.sidebar.number_input("Quantity", min_value=0.0, step=0.01, value=1.0)
        tx_price = st.sidebar.number_input("Price per Share", min_value=0.0, step=0.01, value=0.0)

        if st.sidebar.button("Add Transaction", use_container_width=True):
            asset_id = asset_options[selected_asset]
            add_transaction(
                asset_id=asset_id,
                transaction_date=tx_date,
                transaction_type=tx_type,
                quantity=tx_quantity,
                price=tx_price
            )
            st.sidebar.success("✅ Transaction added!")
            st.rerun()
    else:
        st.sidebar.info("Add assets first to record transactions.")


# ==================== MAIN CONTENT ====================
def render_portfolio_summary():
    """Render portfolio overview section."""
    st.subheader("📊 Portfolio Summary")

    assets = get_all_assets()
    transactions = get_all_transactions()

    if not assets:
        st.info("No assets in portfolio. Add your first asset below!")
        return

    # Calculate portfolio value
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
                # Get current price
                if st.button(f"🔄 Refresh Price", key=f"price_{asset.id}"):
                    with st.spinner("Fetching price..."):
                        price_info = get_stock_price.invoke({
                            "symbol": asset.symbol,
                            "market_type": asset.market_type
                        })
                        st.info(price_info)

            with col2:
                # Set alert threshold
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
                # View transactions
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
            symbol = st.text_input("Symbol", placeholder="e.g., NVDA, 0700, 600519")
            name = st.text_input("Company Name", placeholder="e.g., NVIDIA")

        with col2:
            market_type = st.selectbox("Market", ["US", "HK", "CN"])
            alert_threshold = st.number_input(
                "Initial Alert Threshold ($)",
                min_value=0.0,
                step=0.01,
                value=0.0,
                help="Optional: Set price alert threshold"
            )

        submitted = st.form_submit_button("Add Asset", use_container_width=True)

        if submitted:
            if symbol and name:
                threshold = alert_threshold if alert_threshold > 0 else None
                add_asset(symbol, name, market_type, threshold)
                st.success(f"✅ Added {name} ({symbol}) to watchlist!")
                st.rerun()
            else:
                st.error("❌ Please fill in symbol and name.")


def render_chat_interface():
    """Render AI chat interface with LangChain agent."""
    st.subheader("🤖 AI Investment Assistant")

    # Initialize LLM if not already done
    if st.session_state.llm_client is None:
        st.info("⚠️ Please configure LLM settings in the sidebar first.")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about stocks, portfolio, or market analysis..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize agent if needed
                    if st.session_state.agent_executor is None:
                        llm = st.session_state.llm_client.get_llm()

                        # Create prompt template
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", DEFAULT_SYSTEM_PROMPT),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                            MessagesPlaceholder(variable_name="agent_scratchpad")
                        ])

                        # Create tools
                        tools = [get_stock_price, search_stock_news, get_stock_info]

                        # Create agent
                        agent = create_tool_calling_agent(llm, tools, prompt_template)
                        st.session_state.agent_executor = AgentExecutor(
                            agent=agent,
                            tools=tools,
                            verbose=True,
                            handle_parsing_errors=True
                        )

                    # Invoke agent
                    response = st.session_state.agent_executor.invoke({
                        "input": prompt,
                        "chat_history": []  # Can be expanded for history
                    })

                    assistant_response = response["output"]
                    st.markdown(assistant_response)

                    # Add to history
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

    # Render sidebar
    render_sidebar()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", "📈 Watchlist", "➕ Add Asset", "🤖 AI Assistant"
    ])

    with tab1:
        render_portfolio_summary()

    with tab2:
        render_asset_watchlist()

    with tab3:
        render_add_asset_form()

    with tab4:
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
