# Crypto Oracle (Simulation)

Crypto Oracle is a desktop app that simulates an AI-driven crypto broker using paper money.
It fetches live market data, asks multiple AI models for trading sentiment, and executes simulated trades into a local wallet.

## Disclaimer
This project is **SIMULATION ONLY**. It is not financial advice and does not place real orders.

## Features
- Live crypto data from CoinGecko for BTC, ETH, and SOL
- AI Jury with multi-provider voting (Gemini + Hugging Face)
- Consensus-driven paper trading engine with configurable risk and investment size
- Trade execution rules with validation and simulated exchange fee (0.1%)
- Persistent wallet storage (`config/wallet.json`)
- Timestamped transaction log (`history.log`)
- Dark desktop dashboard with:
  - Total portfolio value
  - Bot active/idle heartbeat status
  - Coin cards and AI action badges
  - Decision log with model reasoning
  - Portfolio value line chart

## Tech Stack
- Python 3.11+
- CustomTkinter (GUI)
- Matplotlib (embedded chart)
- Requests (HTTP)
- python-dotenv (`.env` management)

## Architecture
- `src/data/market_data.py`
  - `MarketHarvester` fetches live prices, 24h change, and 24h volume
- `src/ai/brain.py`
  - `AIJury` builds prompts, queries AI APIs, and normalizes JSON votes
- `src/sim/engine.py`
  - `ExecutionEngine` validates/executed simulated buy/sell logic and logs transactions
- `src/gui/main_app.py`
  - Threaded GUI loop with non-blocking updates via queue + background worker

## Project Structure
```text
cryptobro/
├─ config/
│  └─ wallet.json
├─ src/
│  ├─ ai/
│  │  └─ brain.py
│  ├─ data/
│  │  └─ market_data.py
│  ├─ gui/
│  │  ├─ app.py
│  │  └─ main_app.py
│  └─ sim/
│     └─ engine.py
├─ .env.example
├─ requirements.txt
└─ history.log (created at runtime)
```

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your local env file:

```bash
copy .env.example .env
```

4. Add keys in `.env`:
- `GEMINI_API_KEY`
- `HUGGINGFACE_API_KEY`

## Run
```bash
python -m src.gui.main_app
```

## Execution Rules (Simulation)
- If signed confidence > `0.7` => buy configured USD amount of coin
- If signed confidence < `-0.7` => sell all units of coin
- Else => hold
- Every trade includes a `0.1%` simulated fee

## UI Theme
- Background: `#0B0E11`
- Text: `#EAECEF`
- Buy: `#0ECB81`
- Sell: `#F6465D`

## Threading / Responsiveness
The app uses a background worker thread for API and AI operations.
The main Tkinter UI thread only applies queued updates, which prevents GUI freezing during model/API latency.

## Screenshots
Add screenshots in `docs/` and reference them here:
- Dashboard
- Trade History
- Settings

## Future Improvements
- Add unit tests for engine and parser edge cases
- Add optional backtesting mode
- Add per-model confidence weighting
- Add portfolio allocation controls
