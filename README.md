# OdinOracle

An AI-powered investment portfolio tracker and assistant with RAG + Agent capabilities.

## Features

- **Portfolio Tracking**: Track US, HK, and CN stocks with transactions
- **Price Monitoring**: Set price thresholds and receive email alerts
- **AI Assistant**: Chat with an AI agent that can fetch real-time stock prices and news
- **Flexible LLM Backend**: Switch between Cloud (OpenAI) and Local (Ollama) models
- **Background Monitoring**: Automated price checks during market hours with email alerts

## Tech Stack

- **Frontend**: Streamlit
- **AI Orchestration**: LangChain
- **Database**: SQLModel (SQLite)
- **Market Data**: yfinance
- **Search**: duckduckgo-search
- **Scheduling**: APScheduler
- **LLM**: OpenAI (cloud) or Ollama (local)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yubin-Qin/OdinOracle.git
cd OdinOracle
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Running the Web App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Running the Price Monitor (Background)

Run once for testing:
```bash
python monitor.py --once
```

Run continuously (schedules hourly checks on weekdays):
```bash
python monitor.py
```

## Configuration

### OpenAI (Cloud Mode)
Set `OPENAI_API_KEY` in your `.env` file.

### Ollama (Local Mode)
1. Install Ollama: https://ollama.com
2. Start Ollama server
3. Pull a model: `ollama pull qwen2.5:14b`
4. Set `LOCAL_LLM_URL=http://localhost:11434/v1` in `.env`

### Email Alerts
Configure SMTP settings in `.env`:
- `SMTP_SERVER`: e.g., `smtp.gmail.com`
- `SMTP_PORT`: e.g., `587`
- `SMTP_USERNAME`: Your email
- `SMTP_PASSWORD`: Your app password (for Gmail, use an App Password)

## Project Structure

```
OdinOracle/
├── app.py           # Streamlit web application
├── database.py      # SQLModel database models and operations
├── tools.py         # LangChain tools (stock price, news search)
├── llm_engine.py    # LLM client factory (OpenAI/Ollama)
├── monitor.py       # Background price monitoring with email alerts
├── requirements.txt # Python dependencies
└── .env.example     # Environment variables template
```

## License

MIT License
