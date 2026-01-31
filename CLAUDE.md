# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OdinOracle is a local RAG + Agent application for tracking investment portfolios (US, HK, CN stocks), analyzing markets, AI chat assistance, and email price alerts.

## Tech Stack

- **Language**: Python 3.10+
- **Frontend**: Streamlit
- **AI Orchestration**: LangChain (Community & Core)
- **Database**: SQLModel (SQLite)
- **Market Data**: yfinance
- **Search**: duckduckgo-search
- **Scheduling**: APScheduler
- **LLM**: OpenAI-compatible APIs (OpenAI, Volcano/Ark, DeepSeek, etc.) or Ollama (local)

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Run the Web App
```bash
streamlit run app.py
```

### Run Price Monitor (One-time Check)
```bash
python monitor.py --once
```

### Run Price Monitor (Continuous - Schedules Hourly Checks)
```bash
python monitor.py
```

### Database Initialization
The database (`odin_oracle.db`) is auto-initialized on first run. To reinitialize:
```bash
python -c "from database import init_db; init_db()"
```

## Architecture

### Core Components

| File | Purpose |
|------|---------|
| `app.py` | Streamlit frontend with tabs for Dashboard, Watchlist, Add Asset, AI Chat |
| `database.py` | SQLModel models (Asset, Transaction, UserPreferences) and CRUD operations |
| `tools.py` | LangChain tools: `get_stock_price`, `search_stock_news`, `get_stock_info` |
| `llm_engine.py` | `LLMClient` factory for OpenAI-compatible APIs and Ollama backends |
| `monitor.py` | Standalone background scheduler for price alerts via email |

### Database Schema

- **Asset**: symbol, name, market_type (US/HK/CN), alert_price_threshold (optional)
- **Transaction**: asset_id, date, type (buy/sell), quantity, price
- **UserPreferences**: email_address

### LLM Architecture

The app uses a **Model Factory pattern** via `LLMClient`:
- Acts as a **client only** - does NOT load model weights directly
- Connects to external OpenAI-compatible APIs
- Uses `ChatOpenAI` from LangChain for all modes

**Cloud Mode**: Supports OpenAI-compatible APIs including:
- Official OpenAI API (default base_url=None)
- Volcano Engine/Ark (base_url=`https://ark.cn-beijing.volces.com/api/v3`)
- DeepSeek (base_url=`https://api.deepseek.com`)
- Any OpenAI-compatible API

**Base URL Priority**: method argument > `OPENAI_BASE_URL` env var > None (OpenAI official)

**Model Name**: For Volcano Engine, this is typically an Endpoint ID (e.g., `ep-2025...`)

**Local Mode**: Connects to `http://localhost:11434/v1` (Ollama's OpenAI-compatible endpoint)

### Market Symbol Handling

yfinance requires suffixes for non-US markets:
- US: `NVDA` → `NVDA`
- HK: `0700` → `0700.HK`
- CN: `600519` → `600519.SS` (Shanghai) or `.SZ` (Shenzhen)

### Agent Implementation

The AI Assistant uses LangChain's `create_tool_calling_agent` with:
- Tools: stock price, news search, stock info
- Prompt template with system message and chat history
- AgentExecutor for tool execution

### Email Alert System

`monitor.py` uses APScheduler with CronTrigger:
- Runs every hour Monday-Friday during market hours (9 AM - 4 PM)
- Checks if `current_price < alert_price_threshold`
- Sends HTML email via SMTP (configure in `.env`)

## Environment Variables

Required in `.env`:
```
# OpenAI (Cloud mode)
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o

# Ollama (Local mode)
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_MODEL=qwen2.5:14b

# Email alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
```

## Important Notes

- The app deliberately does NOT use `transformers` or `torch` - it's a client-only architecture
- yfinance can be slow with multiple requests - delays are added between requests
- For Gmail SMTP, use an App Password rather than your account password
- The monitor script is designed to run as a standalone background service
