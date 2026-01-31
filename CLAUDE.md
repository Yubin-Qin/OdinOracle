# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OdinOracle is a local RAG + Agent application for tracking investment portfolios (US, HK, CN stocks), analyzing markets, AI chat assistance, and email price alerts.

## Tech Stack

- **Language**: Python 3.10+
- **Frontend**: Streamlit
- **AI Orchestration**: LangChain (Community & Core)
- **Database**: SQLModel (SQLite) with Alembic migrations, WAL mode enabled
- **Configuration**: pydantic-settings for centralized config management
- **Market Data**: yfinance with tenacity retries
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

### Database Migrations
```bash
# Generate new migration after model changes
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Downgrade one revision
alembic downgrade -1
```

### Database Initialization
The database (`odin_oracle.db`) is auto-initialized on first run. To reinitialize:
```bash
python -c "from db_engine import init_db; init_db()"
```

## Architecture

### Directory Structure

```
OdinOracle/
├── app.py                    # Streamlit frontend
├── config.py                 # Centralized pydantic-settings configuration
├── db_engine.py              # Database engine & session management (WAL mode)
├── database.py               # Backward-compatible CRUD exports
├── llm_engine.py             # LLMClient factory
├── monitor.py                # Background price monitoring
├── tools.py                  # LangChain tools
├── models/                   # SQLModel table definitions
│   ├── __init__.py
│   ├── asset.py
│   ├── transaction.py
│   ├── user_preferences.py
│   └── asset_daily_metric.py
├── repositories/             # Data access layer
│   ├── __init__.py
│   ├── asset_repository.py
│   ├── transaction_repository.py
│   ├── metric_repository.py
│   └── user_preferences_repository.py
├── services/                 # Business logic layer
│   ├── common.py             # Signal calculation, symbol normalization
│   ├── market_data.py        # Market data with retry logic
│   ├── notification.py       # Email service
│   ├── portfolio.py          # Portfolio calculations with risk metrics
│   └── backtest.py           # Signal backtesting engine
└── alembic/                  # Database migrations
    ├── env.py
    ├── versions/
    └── alembic.ini
```

### Core Components

| File | Purpose |
|------|---------|
| `app.py` | Streamlit frontend with tabs for Dashboard, Watchlist, Add Asset, AI Chat |
| `config.py` | Centralized configuration using pydantic-settings |
| `db_engine.py` | Database engine with WAL mode, session factory |
| `database.py` | Backward-compatible CRUD exports (delegates to repositories) |
| `models/` | SQLModel table definitions (Asset, Transaction, etc.) |
| `repositories/` | Data access layer with repository pattern |
| `tools.py` | LangChain tools: `get_stock_price`, `search_stock_news`, `get_stock_info` |
| `llm_engine.py` | `LLMClient` factory using centralized config |
| `monitor.py` | Standalone background scheduler for price alerts |

### Database Schema

- **Asset**: symbol, name, market_type (US/HK/CN), alert_price_threshold (optional)
- **Transaction**: asset_id, date, type (buy/sell), quantity, price
- **UserPreferences**: email_address, language (en/zh), base_currency (USD/CNY/HKD)
- **AssetDailyMetric**: Daily technical indicators (sma, rsi, macd, bollinger, signals)

### Configuration Management

All configuration is centralized in `config.py` using pydantic-settings:

```python
from config import get_settings

settings = get_settings()
settings.openai_api_key
settings.smtp_server
settings.database_url
```

Environment variables are loaded from `.env` file automatically.

### Database Layer

**Repository Pattern**: New code should use the repository layer directly:

```python
from repositories import AssetRepository, TransactionRepository

# Create asset
asset = AssetRepository.add('NVDA', 'NVIDIA Corp', 'US', alert_price_threshold=400.0)

# Get all assets
assets = AssetRepository.get_all()

# Get transactions for asset
transactions = TransactionRepository.get_by_asset(asset.id)
```

**Backward Compatibility**: Existing code using `database.py` functions continues to work:

```python
from database import add_asset, get_all_assets

# These delegate to repositories internally
asset = add_asset('NVDA', 'NVIDIA Corp', 'US')
```

**Concurrency**: SQLite WAL mode is enabled in `db_engine.py` to allow concurrent reads/writes between the Streamlit app and Monitor script.

### LLM Architecture

The app uses a **Model Factory pattern** via `LLMClient`:
- Acts as a **client only** - does NOT load model weights directly
- Connects to external OpenAI-compatible APIs
- Uses `ChatOpenAI` from LangChain for all modes
- Configuration loaded from centralized `config.py`

**Cloud Mode**: Supports OpenAI-compatible APIs including:
- Official OpenAI API (default base_url=None)
- Volcano Engine/Ark (base_url=`https://ark.cn-beijing.volces.com/api/v3`)
- DeepSeek (base_url=`https://api.deepseek.com`)
- Any OpenAI-compatible API

**Base URL Priority**: method argument > `OPENAI_BASE_URL` env var > None (OpenAI official)

**Model Name**: For Volcano Engine, this is typically an Endpoint ID (e.g., `ep-2025...`)

**Local Mode**: Connects to `http://localhost:11434/v1` (Ollama's OpenAI-compatible endpoint)

### Market Data Service

The `MarketDataService` in `services/market_data.py` includes:
- **Tenacity retries**: All yfinance calls have exponential backoff retry logic
- **Caching**: LRU cache for current prices and exchange rates
- **Resilience**: 3 retry attempts with exponential backoff (2-10 seconds)

### Backtesting Module

The `SignalBacktester` in `services/backtest.py` validates signal performance historically:

```python
from services.backtest import SignalBacktester, backtest_asset, format_backtest_report

# Run backtest on an asset
result = backtest_asset(asset, start_date, end_date, initial_capital=10000)

# View report
print(format_backtest_report(result))
```

**Backtest Metrics**:
- Total Return vs Benchmark (Buy & Hold)
- Alpha (outperformance)
- Max Drawdown
- Sharpe Ratio
- Win Rate and Trade Statistics
- Signal Accuracy by Type

**Trading Strategy**:
- STRONG_BUY: 100% position
- BUY: 50% position
- HOLD: Maintain current
- SELL: Sell 50%
- STRONG_SELL: Sell 100%

### Risk Management

The `PortfolioService` now includes professional risk metrics:

```python
from services.portfolio import PortfolioService

# Get comprehensive analytics with risk metrics
analytics = PortfolioService.get_portfolio_analytics()

# Access risk metrics
risk = analytics.risk_metrics
print(f"Sharpe Ratio: {risk.portfolio_sharpe_ratio}")
print(f"Volatility: {risk.portfolio_volatility}%")
print(f"VaR (95%): ${risk.var_95}")

# View correlation matrix
if risk.correlation_matrix is not None:
    print(risk.correlation_matrix)

# Check concentration risk
concentration = risk.concentration_risk
for warning in concentration['warnings']:
    print(warning['message'])

# Format full risk report
print(PortfolioService.format_risk_report(analytics))
```

**Risk Metrics**:
- **Sharpe Ratio**: Risk-adjusted return (annualized)
  - < 1: Sub-optimal
  - 1-2: Good
  - 2-3: Very good
  - > 3: Excellent
- **Correlation Matrix**: Shows how assets move together
  - Values > 0.8 indicate concentration risk
- **Concentration Risk**: Warns if single position > 20%
- **Value at Risk (95%)**: Potential portfolio loss at 95% confidence

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
- Sends HTML email via SMTP (configured in `.env`)

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

# Optional: Database URL (default: sqlite:///odin_oracle.db)
DATABASE_URL=sqlite:///odin_oracle.db
```

## Important Notes

- The app deliberately does NOT use `transformers` or `torch` - it's a client-only architecture
- yfinance calls are wrapped with tenacity for automatic retries during API instability
- SQLite WAL mode enables concurrent access between Streamlit and Monitor processes
- For Gmail SMTP, use an App Password rather than your account password
- The monitor script is designed to run as a standalone background service
- All configuration is centralized in `config.py` using pydantic-settings
