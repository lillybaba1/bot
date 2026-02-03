# Julaba Trading Bot - Setup Guide

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/lillybaba1/julaba.git
cd julaba
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example file and fill in your credentials:
```bash
cp .env.example .env
nano .env  # or use any text editor
```

---

## 🔑 API Setup

### MEXC Exchange API
1. Go to [MEXC API Management](https://www.mexc.com/user/openapi)
2. Create a new API key
3. Enable **Spot Trading** permissions
4. Copy the API Key and Secret to your `.env` file

### Google Gemini API (AI Filter)
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key to `GEMINI_API_KEY` in `.env`
4. **Free tier** includes 60 requests/minute

---

## 📱 Telegram Bot Setup

### Step 1: Create Your Bot
1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Follow prompts to name your bot
4. Copy the **HTTP API Token** to `TELEGRAM_BOT_TOKEN` in `.env`

### Step 2: Get Your Chat ID
1. Send any message to your new bot
2. Open this URL in browser (replace YOUR_TOKEN):
   ```
   https://api.telegram.org/botYOUR_TOKEN/getUpdates
   ```
3. Find `"chat":{"id": 123456789}` in the response
4. Copy the number to `TELEGRAM_CHAT_ID` in `.env`

### Step 3: Test Connection
```bash
python3 -c "from telegram_bot import TelegramBot; t = TelegramBot(); print('✅ Connected!' if t else '❌ Failed')"
```

---

## 🖥️ Dashboard Setup

### Local Access
The dashboard runs on port 5000 by default:
```bash
python3 bot.py --dashboard --dashboard-port 5000
```
Access at: `http://localhost:5000`

### Remote Access (VPS/Cloud)

#### Option 1: SSH Tunnel (Recommended for security)
```bash
# On your local machine:
ssh -L 5000:localhost:5000 user@your-server-ip
# Then open http://localhost:5000 in your browser
```

#### Option 2: Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### Option 3: Direct Port Access (Not recommended)
```bash
# Open firewall port (Oracle Cloud example)
sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT
```

### Dashboard Features
- **Live P&L tracking** - Real-time profit/loss display
- **Position monitoring** - All open trades with entry prices
- **THE MACHINE** - Visual pipeline of AI decision-making
- **Trade history** - Complete log of all trades
- **Lock/Unlock** - Password protection for controls

---

## 🤖 Running the Bot

### Start Bot with Dashboard
```bash
python3 bot.py --dashboard --dashboard-port 5000
```

### Start in Background (Production)
```bash
nohup python3 bot.py --dashboard --dashboard-port 5000 > /tmp/bot.log 2>&1 &
```

### Quick Commands (j script)
```bash
j s          # Status
j start      # Start bot
j stop       # Stop bot  
j restart    # Restart bot
j log        # View logs
j ml         # Check ML status
```

---

## 📋 Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `API_KEY` | MEXC API Key | ✅ |
| `API_SECRET` | MEXC API Secret | ✅ |
| `GEMINI_API_KEY` | Google Gemini API Key | ✅ |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot Token | Optional |
| `TELEGRAM_CHAT_ID` | Your Telegram Chat ID | Optional |
| `AI_CONFIDENCE_THRESHOLD` | AI approval threshold (0.0-1.0) | Optional |

---

## 🔒 Security Notes

1. **Never commit `.env`** - It's in `.gitignore`
2. **Use SSH tunnel** for remote dashboard access
3. **Rotate API keys** periodically
4. **Enable IP whitelist** on MEXC if possible

---

## 🆘 Troubleshooting

### Bot won't start
```bash
# Check if already running
pgrep -f "python3 bot.py"

# Remove lock file
rm -f julaba.lock
```

### Telegram not working
```bash
# Test bot token
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe
```

### Dashboard not accessible
```bash
# Check if running
curl http://localhost:5000/api/status

# Check port
ss -tlnp | grep 5000
```

---

## 📞 Support

For issues, check logs:
```bash
tail -100 /tmp/bot.log
tail -100 julaba.log
```
