# OdinOracle

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Streamlit-Frontend-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangChain-AI-orange.svg" alt="LangChain">
</p>

<p align="center">
  <b>AI-Powered Investment Intelligence Platform</b><br>
  Professional-grade portfolio analytics with Factor Pack signals, risk metrics, and dual-mode LLM support.
</p>

---

## Overview

OdinOracle is a comprehensive investment portfolio tracking and market intelligence platform designed for serious investors. It combines **real-time market data**, **quantitative signal analysis**, **risk management metrics**, and **AI-powered market intelligence** in a professional-grade architecture.

### Key Differentiators

- **Factor Pack Analysis**: Proprietary multi-factor signal system (RSI, MACD, Bollinger Bands, Volume, Trend)
- **Risk Management**: Sharpe Ratio, Value at Risk (VaR), and correlation matrix analysis
- **Signal Backtesting**: Validate Factor Pack signals against historical data
- **Dual-Mode LLM**: Seamlessly switch between Cloud (OpenAI-compatible) and Local (Ollama) AI models
- **Multi-Currency Support**: USD, CNY, HKD with real-time FX conversion
- **Concurrent Data Fetching**: Optimized parallel market data fetching for large portfolios

---

## Features

### Portfolio Management
- **Multi-Market Support**: Track US, HK, and CN stocks in a unified portfolio
- **Real-Time Valuation**: Live PnL calculation with multi-currency conversion
- **Transaction History**: FIFO-based cost basis tracking with full CRUD operations
- **Batch Price Updates**: Concurrent fetching for large portfolios (5x faster)

### Factor Pack Signal System
Professional-grade technical analysis combining 5 factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| RSI (14) | 2x | Mean reversion signals |
| MACD Histogram | 2x | Momentum confirmation |
| Golden/Death Cross | 2x | Trend direction (SMA20 vs SMA200) |
| Bollinger Position | 1x | Volatility-based entry/exit |
| Volume Squeeze | 1x | Breakout detection |

**Signal Output**: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL (with 0-10 confidence)

### Risk Management Metrics
- **Sharpe Ratio**: Risk-adjusted return analysis (annualized)
- **Value at Risk (95%)**: Maximum expected loss at 95% confidence
- **Correlation Matrix**: Cross-asset correlation to detect concentration risk
- **Position Sizing Alerts**: Warnings when single position >20% of portfolio

### AI Market Intelligence
- **Aggressive Hedge Fund Persona**: Direct, data-driven analysis without hedging
- **Bilingual Support**: Full English and Chinese language support
- **Real-Time Tools**: Live stock prices and news search via AI agent
- **Market Briefings**: Automated daily intelligence reports

### Architecture Highlights
- **Repository Pattern**: Clean separation of data access and business logic
- **Alembic Migrations**: Database schema versioning
- **SQLite WAL Mode**: Concurrent access between web app and background monitor
- **Tenacity Retry Logic**: Resilient API calls with exponential backoff
- **Streamlit Caching**: Optimized UI performance with automatic cache invalidation

---

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) Ollama for local LLM mode
- (Optional) SMTP credentials for email alerts

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OdinOracle.git
cd OdinOracle

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head
```

### Configuration

Create a `.env` file:

```bash
# Database
DATABASE_URL=sqlite:///odin_oracle.db

# OpenAI (Cloud Mode)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o
# OPENAI_BASE_URL=https://api.deepseek.com  # Optional: for DeepSeek, Volcano, etc.

# Ollama (Local Mode)
LOCAL_MODEL=qwen2.5:14b
LOCAL_LLM_URL=http://localhost:11434/v1

# Email Alerts (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
```

### Running the Application

**Web Dashboard:**
```bash
streamlit run app.py
```
Access at http://localhost:8501

**Background Price Monitor:**
```bash
# One-time check
python monitor.py --once

# Continuous monitoring (hourly, weekdays)
python monitor.py
```

---

## Architecture

```
OdinOracle/
├── app.py                      # Streamlit web application
├── config.py                   # Pydantic-settings configuration
├── db_engine.py                # Database engine with WAL mode
├── database.py                 # Backward-compatible CRUD exports
├── monitor.py                  # Background price monitoring
├── llm_engine.py               # LLM client factory
├── models/                     # SQLModel table definitions
│   ├── asset.py
│   ├── transaction.py
│   ├── asset_daily_metric.py
│   └── user_preferences.py
├── repositories/               # Data access layer
│   ├── asset_repository.py
│   ├── transaction_repository.py
│   ├── metric_repository.py
│   └── user_preferences_repository.py
├── services/                   # Business logic layer
│   ├── market_data.py          # Price fetching + technical indicators
│   ├── portfolio.py            # Portfolio calculations + risk metrics
│   ├── backtest.py             # Signal backtesting engine
│   ├── common.py               # Signal calculation + utilities
│   └── notification.py         # Email service
├── prompts/                    # LLM system prompts
│   ├── system_prompt_en.txt
│   └── system_prompt_zh.txt
├── alembic/                    # Database migrations
├── tools.py                    # LangChain tools
└── requirements.txt
```

---

## Usage Examples

### Portfolio Analysis with Risk Metrics

```python
from services.portfolio import PortfolioService

# Get comprehensive portfolio analytics
analytics = PortfolioService.get_portfolio_analytics(base_currency="USD")

print(f"Total Value: ${analytics.total_value:,.2f}")
print(f"Sharpe Ratio: {analytics.risk_metrics.portfolio_sharpe_ratio:.2f}")
print(f"VaR (95%): ${analytics.risk_metrics.var_95:,.2f}")

# Print full risk report
print(PortfolioService.format_risk_report(analytics))
```

### Signal Backtesting

```python
from services.backtest import backtest_asset, format_backtest_report
from repositories import AssetRepository

# Get asset
asset = AssetRepository.get_by_symbol("NVDA")

# Run backtest over last 90 days
result = backtest_asset(
    asset=asset,
    start_date=date.today() - timedelta(days=90),
    end_date=date.today(),
    initial_capital=10000
)

# View performance report
print(format_backtest_report(result))
```

### Batch Price Fetching

```python
from services.market_data import MarketDataService

# Fetch multiple prices in parallel
assets = [("NVDA", "US"), ("0700", "HK"), ("600519", "CN")]
prices = MarketDataService.get_current_prices_batch(assets, max_workers=5)

print(prices)  # {'NVDA': 450.50, '0700': 285.40, '600519': 1650.00}
```

---

## Performance Optimizations

| Optimization | Implementation | Benefit |
|--------------|----------------|---------|
| **Concurrent Fetching** | `ThreadPoolExecutor` in `MarketDataService` | 5x faster for large portfolios |
| **Streamlit Caching** | `st.cache_data` with TTL | Reduced redundant calculations |
| **Database WAL Mode** | SQLite WAL + connection pooling | Concurrent app + monitor access |
| **Session Reuse** | Optional session parameter in repositories | Reduced connection overhead |
| **LRU Caching** | `@lru_cache` on price/exchange rate lookups | Reduced API calls |

---

## Signal Methodology

The Factor Pack combines 5 technical indicators into a unified signal:

```
Score = 5 (neutral baseline)
  + RSI contribution (-2 to +2)
  + MACD contribution (-2 to +2)
  + Trend contribution (-2 to +2)
  + Bollinger contribution (-1 to +1)
  + Volume contribution (0 to +0.5)

Signal Mapping:
  Score >= 8: STRONG_BUY
  Score >= 6: BUY
  Score <= 2: STRONG_SELL
  Score <= 4: SELL
  Else: HOLD
```

**Backtesting** validates signals against historical performance to measure:
- Total Return vs Buy-and-Hold (Alpha)
- Maximum Drawdown
- Sharpe Ratio
- Signal Accuracy by Type

---

## Risk Metrics Explained

| Metric | Interpretation | Thresholds |
|--------|---------------|------------|
| **Sharpe Ratio** | Risk-adjusted return | <1: Poor, 1-2: Good, 2-3: Very Good, >3: Excellent |
| **VaR (95%)** | Max loss at 95% confidence | Lower is better |
| **Correlation** | Asset co-movement | >0.8: High concentration risk |
| **Max Drawdown** | Peak-to-trough decline | Lower is better |

---

## LLM Configuration

### Cloud Mode (OpenAI-Compatible)
Supports any OpenAI-compatible API:
- OpenAI GPT-4o
- DeepSeek V3
- Volcano Engine / Ark
- Any compatible endpoint

### Local Mode (Ollama)
```bash
# Install and start Ollama
ollama pull qwen2.5:14b

# Configure in .env
LOCAL_MODEL=qwen2.5:14b
LOCAL_LLM_URL=http://localhost:11434/v1
```

### Customizing Prompts
Edit the prompt files in `prompts/` directory:
- `system_prompt_en.txt` - English persona
- `system_prompt_zh.txt` - Chinese persona

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## License

MIT License - see LICENSE file for details.

---

## Disclaimer

**OdinOracle is for informational purposes only. It is not financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.**

The platform uses real-time market data but does not guarantee accuracy. Past performance of signals does not guarantee future results.
