# 🤖 Julaba Trading Bot

AI-powered autonomous cryptocurrency trading bot with PhD-level mathematical analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Exchange](https://img.shields.io/badge/Exchange-Bybit-orange)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧮 **PhD-Level Math** | Hurst exponent, GARCH volatility, Kalman filtering, Monte Carlo simulation |
| 🤖 **AI Decisions** | Google Gemini validates every trade for higher accuracy |
| 📊 **Multi-Pair** | Scans 50+ pairs to find the best opportunities |
| 🖥️ **Dashboard** | Real-time web UI at `localhost:5000` |
| 📱 **Telegram** | Get alerts and control the bot from your phone |
| 🛡️ **Risk Management** | Dynamic sizing, max drawdown protection, position limits |

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone & Setup Environment

```bash
git clone https://github.com/lillybaba1/bot.git
cd bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Get Your API Keys

You need **2 required** keys (both are FREE):

| Service | Link | Time |
|---------|------|------|
| **Bybit** | [bybit.com/app/user/api-management](https://www.bybit.com/app/user/api-management) | 2 min |
| **Gemini AI** | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | 1 min |

#### Bybit API Key Setup:
1. Go to [Bybit API Management](https://www.bybit.com/app/user/api-management)
2. Click "Create New Key" → Select "System-generated API Keys"
3. Name: `Julaba Bot`
4. Permissions needed: ✅ Contract (Read/Write)
5. **IP Restriction**: Add your server IP for security
6. Save the API Key and Secret (shown only once!)

#### Gemini API Key Setup:
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key

### Step 3: Configure

Open the `.env` file and add your keys:

```bash
nano .env
```

Fill in these required fields:
```
BYBIT_API_KEY=your_bybit_key_here
BYBIT_API_SECRET=your_bybit_secret_here
GEMINI_API_KEY=your_gemini_key_here
```

### Step 4: Run the Bot

**Paper Trading (recommended to start):**
```bash
python bot.py --paper --dashboard
```

**Live Trading (real money):**
```bash
python bot.py --live --dashboard
```

Then open your browser: **http://localhost:5000**

---

## 📱 Telegram Setup (Optional)

Get real-time alerts on your phone:

1. **Create Bot:** Message [@BotFather](https://t.me/BotFather) → `/newbot`
2. **Get Chat ID:** Message [@userinfobot](https://t.me/userinfobot) 
3. **Add to .env:**
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
4. **Start your bot** in Telegram (click START)

### Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Current positions & P&L |
| `/balance` | Account balance |
| `/market` | Market analysis |
| `/ai` | AI mode settings |
| `/close` | Close position |
| `/help` | All commands |

---

## ⚙️ Configuration

Edit `julaba_config.json` to customize:

```json
{
  "risk_pct": 0.02,               // 2% risk per trade
  "max_leverage": 10,             // Maximum 10x leverage
  "max_concurrent_positions": 2,  // Max 2 open positions
  "balance_protection": 100       // Stop if balance < $100
}
```

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `risk_pct` | 0.02 | % of balance risked per trade |
| `max_leverage` | 10 | Maximum leverage allowed |
| `tp_multiplier` | 1.5 | Take profit = ATR × multiplier |
| `sl_multiplier` | 1.0 | Stop loss = ATR × multiplier |
| `ai_confidence_threshold` | 0.65 | Minimum AI confidence to trade |
| `balance_protection` | 100 | Stop bot if balance drops below |

---

## 🖥️ Dashboard

Access at `http://localhost:5000` when running with `--dashboard`

Features:
- 📈 Real-time P&L chart
- 📊 Position monitoring with live prices
- 🧮 Math score breakdown
- 📜 Trade history
- 🤖 AI decision log

---

## 🛡️ Security Notes

- ⚠️ **Never share your `.env` file**
- 🔐 Use IP whitelisting on Bybit API
- 📝 Start with paper trading to test
- 💰 Only trade what you can afford to lose
- 🖥️ Dashboard is localhost-only for security

---

## 🐛 Troubleshooting

### Bot won't start
- Check Python version: `python --version` (need 3.10+)
- Verify `.env` file exists and has all keys
- Check API key permissions on Bybit

### "Invalid API key" error
- Make sure no spaces/quotes around your keys in `.env`
- Check if API key has Contract permissions enabled
- Verify IP whitelist includes your server IP

### No trades happening
- **This is normal!** The bot only trades high-quality setups
- AI rejects low-confidence opportunities (this is good!)
- Check the dashboard for scan results
- Market may be choppy or ranging

### Dashboard not loading
- Make sure you used `--dashboard` flag
- Check if port 5000 is free: `lsof -i :5000`
- Try different port: `--dashboard-port 8080`

### Rate limit errors
- Reduce scan frequency in config
- Bot auto-handles rate limits with exponential backoff

---

## 📁 File Structure

```
bot/
├── bot.py              # Main trading engine
├── ai_filter.py        # AI trade validation
├── indicator.py        # Technical indicators
├── risk_manager.py     # Position sizing & risk
├── dashboard.py        # Web interface
├── telegram_bot.py     # Telegram integration
├── julaba_config.json  # Bot configuration
├── .env                # Your API keys (don't share!)
└── requirements.txt    # Python dependencies
```

---

## 📄 License

MIT License - Use at your own risk. 

**Disclaimer:** Trading cryptocurrencies involves substantial risk of loss. This software is provided as-is with no guarantees. Past performance does not indicate future results.

---

## 💬 Support

For issues, open a GitHub issue or contact the developer.
