# Julaba Trading Bot

AI-powered autonomous cryptocurrency trading bot with advanced mathematical analysis.

## Features

- **PhD-Level Math Analysis**: Hurst exponent, GARCH volatility, Kalman filtering, Monte Carlo simulation
- **AI Decision Making**: Google Gemini integration for trade validation
- **Multi-Pair Trading**: Scan and trade multiple pairs simultaneously
- **Real-Time Dashboard**: Web-based monitoring at `http://localhost:5000`
- **Risk Management**: Dynamic position sizing, max drawdown protection
- **Telegram Integration**: Real-time notifications and commands

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the Bot

**Paper Trading (recommended for testing):**
```bash
python bot.py --paper
```

**Live Trading:**
```bash
python bot.py --live
```

**With Dashboard:**
```bash
python bot.py --dashboard --dashboard-port 5000
```

## Configuration

Edit `julaba_config.json` to customize:
- Risk percentage per trade
- Maximum leverage
- TP/SL multipliers
- Balance protection thresholds

## Dashboard

Access the dashboard at `http://localhost:5000` after starting with `--dashboard` flag.

Features:
- Real-time P&L tracking
- Position monitoring
- Math score visualization
- Trade history
- AI decision log

## Security Note

- Never share your API keys
- Use IP whitelisting on Bybit
- Start with paper trading
- Dashboard is localhost-only for security

## License

MIT License - See LICENSE file
