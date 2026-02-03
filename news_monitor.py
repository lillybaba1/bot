#!/usr/bin/env python3
"""
Julaba News Monitor - Real-time crypto news and market sentiment tracking.

Features:
- Multiple news sources (CryptoCompare, CoinGecko, Reddit, Twitter sentiment)
- Major event detection (liquidations, whale movements, regulatory news)
- Market sentiment analysis
- Breaking news alerts via Telegram
- Fear & Greed Index tracking
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class NewsPriority(Enum):
    """News priority levels."""
    CRITICAL = "critical"   # Major market-moving events (500B pullback, exchange hacks)
    HIGH = "high"           # Significant news (ETF decisions, major regulations)
    MEDIUM = "medium"       # Regular crypto news
    LOW = "low"             # Minor updates


class NewsCategory(Enum):
    """News categories."""
    MARKET = "market"           # General market movements
    REGULATORY = "regulatory"   # Government/regulatory actions
    WHALE = "whale"             # Large transactions
    LIQUIDATION = "liquidation" # Mass liquidations
    EXCHANGE = "exchange"       # Exchange news (hacks, listings)
    DEFI = "defi"               # DeFi protocols
    NFT = "nft"                 # NFT market
    MACRO = "macro"             # Macro economic (Fed, inflation)
    TECHNICAL = "technical"     # Network upgrades, forks


@dataclass
class NewsItem:
    """Single news item."""
    id: str
    title: str
    body: str
    source: str
    url: str
    published: datetime
    categories: List[str] = field(default_factory=list)
    priority: str = "medium"
    sentiment: float = 0.0  # -1 (bearish) to +1 (bullish)
    coins: List[str] = field(default_factory=list)  # Mentioned coins
    impact_score: float = 0.0  # 0-100 market impact estimate


@dataclass
class MarketSentiment:
    """Overall market sentiment data."""
    fear_greed_index: int = 50  # 0-100
    fear_greed_label: str = "Neutral"
    btc_dominance: float = 0.0
    total_market_cap: float = 0.0
    market_cap_change_24h: float = 0.0
    total_volume_24h: float = 0.0
    liquidations_24h: float = 0.0
    long_short_ratio: float = 1.0
    updated: str = ""


class NewsMonitor:
    """
    Real-time crypto news monitoring and alerting system.
    """
    
    # Keywords that indicate major market events
    CRITICAL_KEYWORDS = [
        'billion pullback', 'billion outflow', 'mass liquidation',
        'exchange hack', 'rug pull', 'sec lawsuit', 'ban crypto',
        'emergency shutdown', 'flash crash', 'etf rejected',
        'tether depeg', 'usdc depeg', 'stablecoin depeg',
        'mt gox', 'ftx', 'binance halt', 'coinbase down'
    ]
    
    HIGH_KEYWORDS = [
        'etf approved', 'etf decision', 'whale alert', 'whale moves',
        'fed rate', 'interest rate', 'cpi data', 'inflation',
        'regulation', 'regulatory', 'lawsuit', 'investigation',
        'delisting', 'listing', 'partnership', 'acquisition',
        'billion', 'trillion', 'all-time high', 'all-time low'
    ]
    
    BEARISH_KEYWORDS = [
        'crash', 'dump', 'sell', 'bearish', 'decline', 'fall',
        'pullback', 'correction', 'outflow', 'liquidation', 
        'ban', 'hack', 'exploit', 'rug', 'scam', 'fraud',
        'lawsuit', 'investigation', 'sec', 'warning'
    ]
    
    BULLISH_KEYWORDS = [
        'surge', 'pump', 'rally', 'bullish', 'rise', 'gain',
        'inflow', 'accumulation', 'buy', 'adoption', 'approved',
        'partnership', 'integration', 'upgrade', 'milestone',
        'ath', 'breakout', 'institutional'
    ]
    
    def __init__(self, config_path: str = "news_config.json"):
        self.config_path = Path(config_path)
        self.cache_path = Path("news_cache.json")
        
        # Load config
        self.config = self._load_config()
        
        # News cache
        self._news_cache: List[NewsItem] = []
        self._sentiment_cache: Optional[MarketSentiment] = None
        self._last_fetch: Dict[str, float] = {}
        
        # Alert callback (for Telegram notifications)
        self._alert_callback: Optional[Callable] = None
        
        # Seen news IDs to avoid duplicates
        self._seen_ids: set = set()
        
        # Load cached data
        self._load_cache()
    
    def _load_config(self) -> dict:
        """Load configuration."""
        default_config = {
            "enabled": True,
            "fetch_interval_seconds": 300,  # 5 minutes
            "sources": {
                "cryptocompare": True,
                "coingecko": True,
                "fear_greed": True,
                "liquidations": True
            },
            "alert_on_priority": ["critical", "high"],
            "watched_coins": ["BTC", "ETH", "SOL", "LINK"],
            "min_impact_score": 50,
            "telegram_alerts": True
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return {**default_config, **json.load(f)}
            except:
                pass
        
        # Save default config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _load_cache(self):
        """Load cached news data."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                    self._seen_ids = set(data.get('seen_ids', []))
            except:
                pass
    
    def _save_cache(self):
        """Save news cache."""
        try:
            # Only keep last 1000 seen IDs
            seen_list = list(self._seen_ids)[-1000:]
            with open(self.cache_path, 'w') as f:
                json.dump({
                    'seen_ids': seen_list,
                    'last_update': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.error(f"Failed to save news cache: {e}")
    
    def set_alert_callback(self, callback: Callable):
        """Set callback for news alerts (e.g., Telegram notification)."""
        self._alert_callback = callback
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis based on keywords.
        Returns -1 (very bearish) to +1 (very bullish).
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def _calculate_priority(self, text: str) -> str:
        """Determine news priority based on content."""
        text_lower = text.lower()
        
        # Check for critical keywords
        for kw in self.CRITICAL_KEYWORDS:
            if kw in text_lower:
                return NewsPriority.CRITICAL.value
        
        # Check for high priority keywords
        for kw in self.HIGH_KEYWORDS:
            if kw in text_lower:
                return NewsPriority.HIGH.value
        
        return NewsPriority.MEDIUM.value
    
    def _calculate_impact(self, news: NewsItem) -> float:
        """Calculate estimated market impact score (0-100)."""
        score = 30  # Base score
        
        # Priority bonus
        if news.priority == "critical":
            score += 50
        elif news.priority == "high":
            score += 30
        
        # Sentiment magnitude
        score += abs(news.sentiment) * 20
        
        # Source reliability
        reliable_sources = ['bloomberg', 'reuters', 'coindesk', 'cointelegraph']
        if any(s in news.source.lower() for s in reliable_sources):
            score += 10
        
        return min(100, score)
    
    def _extract_coins(self, text: str) -> List[str]:
        """Extract mentioned cryptocurrency symbols from text."""
        # Common coin patterns
        coin_patterns = [
            'BTC', 'ETH', 'SOL', 'LINK', 'XRP', 'ADA', 'DOT', 'AVAX',
            'MATIC', 'DOGE', 'SHIB', 'LTC', 'BCH', 'UNI', 'ATOM',
            'Bitcoin', 'Ethereum', 'Solana', 'Chainlink', 'Ripple'
        ]
        
        found = []
        text_upper = text.upper()
        
        for coin in coin_patterns:
            if coin.upper() in text_upper:
                # Normalize to symbol
                symbol = coin.upper()
                if symbol == 'BITCOIN':
                    symbol = 'BTC'
                elif symbol == 'ETHEREUM':
                    symbol = 'ETH'
                elif symbol == 'SOLANA':
                    symbol = 'SOL'
                elif symbol == 'CHAINLINK':
                    symbol = 'LINK'
                elif symbol == 'RIPPLE':
                    symbol = 'XRP'
                
                if symbol not in found:
                    found.append(symbol)
        
        return found
    
    async def fetch_cryptocompare_news(self) -> List[NewsItem]:
        """Fetch news from CryptoCompare API (free)."""
        cache_key = 'cryptocompare'
        if time.time() - self._last_fetch.get(cache_key, 0) < 120:
            return []
        
        news_items = []
        
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('Data', [])[:30]
                        
                        for article in articles:
                            news_id = f"cc_{article.get('id', '')}"
                            
                            if news_id in self._seen_ids:
                                continue
                            
                            title = article.get('title', '')
                            body = article.get('body', '')[:500]
                            full_text = f"{title} {body}"
                            
                            news_item = NewsItem(
                                id=news_id,
                                title=title,
                                body=body,
                                source=article.get('source', 'CryptoCompare'),
                                url=article.get('url', ''),
                                published=datetime.fromtimestamp(article.get('published_on', time.time())),
                                categories=article.get('categories', '').split('|'),
                                priority=self._calculate_priority(full_text),
                                sentiment=self._analyze_sentiment(full_text),
                                coins=self._extract_coins(full_text)
                            )
                            news_item.impact_score = self._calculate_impact(news_item)
                            
                            news_items.append(news_item)
                            self._seen_ids.add(news_id)
            
            self._last_fetch[cache_key] = time.time()
            
        except Exception as e:
            logger.error(f"CryptoCompare news fetch error: {e}")
        
        return news_items
    
    async def fetch_fear_greed_index(self) -> Optional[MarketSentiment]:
        """Fetch Fear & Greed Index and market sentiment."""
        cache_key = 'fear_greed'
        if time.time() - self._last_fetch.get(cache_key, 0) < 300:
            return self._sentiment_cache
        
        try:
            # Fear & Greed Index API (free)
            fg_url = "https://api.alternative.me/fng/?limit=1"
            
            async with aiohttp.ClientSession() as session:
                # Fetch Fear & Greed
                async with session.get(fg_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        fg_data = data.get('data', [{}])[0]
                        
                        self._sentiment_cache = MarketSentiment(
                            fear_greed_index=int(fg_data.get('value', 50)),
                            fear_greed_label=fg_data.get('value_classification', 'Neutral'),
                            updated=datetime.now().isoformat()
                        )
                
                # Try to get global market data from CoinGecko
                try:
                    global_url = "https://api.coingecko.com/api/v3/global"
                    async with session.get(global_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            global_data = data.get('data', {})
                            
                            if self._sentiment_cache:
                                self._sentiment_cache.total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
                                self._sentiment_cache.market_cap_change_24h = global_data.get('market_cap_change_percentage_24h_usd', 0)
                                self._sentiment_cache.total_volume_24h = global_data.get('total_volume', {}).get('usd', 0)
                                self._sentiment_cache.btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
                except:
                    pass
            
            self._last_fetch[cache_key] = time.time()
            
        except Exception as e:
            logger.error(f"Fear & Greed fetch error: {e}")
        
        return self._sentiment_cache
    
    async def fetch_liquidation_data(self) -> Dict[str, Any]:
        """Fetch recent liquidation data."""
        cache_key = 'liquidations'
        if time.time() - self._last_fetch.get(cache_key, 0) < 60:
            return {}
        
        try:
            # Coinglass API for liquidations (may need API key for full access)
            # Using a free alternative
            url = "https://api.coingecko.com/api/v3/derivatives"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        total_oi = sum(float(d.get('open_interest', 0) or 0) for d in data[:20])
                        
                        self._last_fetch[cache_key] = time.time()
                        
                        return {
                            'total_open_interest': total_oi,
                            'exchanges_count': len(data),
                            'updated': datetime.now().isoformat()
                        }
        
        except Exception as e:
            logger.error(f"Liquidation data fetch error: {e}")
        
        return {}
    
    async def check_for_major_events(self) -> List[NewsItem]:
        """
        Check for major market events that need immediate attention.
        This is the main function to detect things like "$500B pullback".
        """
        major_events = []
        
        # Fetch all news
        news = await self.fetch_cryptocompare_news()
        
        # Filter for critical/high priority
        for item in news:
            if item.priority in ['critical', 'high']:
                major_events.append(item)
                
                # Send alert if callback is set
                if self._alert_callback and self.config.get('telegram_alerts'):
                    await self._send_news_alert(item)
        
        # Check market sentiment
        sentiment = await self.fetch_fear_greed_index()
        
        if sentiment:
            # Alert on extreme fear (potential buying opportunity) or extreme greed (potential top)
            if sentiment.fear_greed_index <= 20:
                logger.warning(f"🔴 EXTREME FEAR: Fear & Greed Index at {sentiment.fear_greed_index}")
            elif sentiment.fear_greed_index >= 80:
                logger.warning(f"🟢 EXTREME GREED: Fear & Greed Index at {sentiment.fear_greed_index}")
            
            # Alert on large market cap changes
            if abs(sentiment.market_cap_change_24h) >= 5:
                direction = "📈" if sentiment.market_cap_change_24h > 0 else "📉"
                logger.warning(f"{direction} MAJOR MARKET MOVE: {sentiment.market_cap_change_24h:.1f}% in 24h")
        
        # Save cache
        self._save_cache()
        
        return major_events
    
    async def _send_news_alert(self, news: NewsItem):
        """Send news alert via callback (Telegram)."""
        if not self._alert_callback:
            return
        
        # Format alert message
        priority_emoji = {
            'critical': '🚨🚨🚨',
            'high': '⚠️',
            'medium': 'ℹ️',
            'low': '📰'
        }
        
        sentiment_emoji = '🐂' if news.sentiment > 0.3 else '🐻' if news.sentiment < -0.3 else '➖'
        
        message = f"""
{priority_emoji.get(news.priority, '📰')} **{news.priority.upper()} PRIORITY NEWS**

**{news.title}**

{news.body[:300]}...

📊 Sentiment: {sentiment_emoji} ({news.sentiment:+.2f})
💥 Impact Score: {news.impact_score:.0f}/100
🪙 Coins: {', '.join(news.coins) if news.coins else 'General'}
📰 Source: {news.source}
🔗 {news.url}
"""
        
        try:
            await self._alert_callback(message)
        except Exception as e:
            logger.error(f"Failed to send news alert: {e}")
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary including news and sentiment."""
        # Fetch latest data
        news = await self.fetch_cryptocompare_news()
        sentiment = await self.fetch_fear_greed_index()
        liquidations = await self.fetch_liquidation_data()
        
        # Categorize recent news
        bullish_news = [n for n in news if n.sentiment > 0.2]
        bearish_news = [n for n in news if n.sentiment < -0.2]
        critical_news = [n for n in news if n.priority == 'critical']
        
        return {
            'sentiment': asdict(sentiment) if sentiment else None,
            'news_summary': {
                'total_articles': len(news),
                'bullish_count': len(bullish_news),
                'bearish_count': len(bearish_news),
                'critical_count': len(critical_news),
                'average_sentiment': sum(n.sentiment for n in news) / len(news) if news else 0
            },
            'critical_news': [asdict(n) for n in critical_news[:5]],
            'recent_news': [asdict(n) for n in news[:10]],
            'liquidations': liquidations,
            'recommendation': self._get_market_recommendation(sentiment, news)
        }
    
    def _get_market_recommendation(self, sentiment: Optional[MarketSentiment], news: List[NewsItem]) -> str:
        """Generate trading recommendation based on news and sentiment."""
        if not sentiment:
            return "NEUTRAL - Insufficient data"
        
        score = 0
        reasons = []
        
        # Fear & Greed contribution
        if sentiment.fear_greed_index <= 25:
            score += 2
            reasons.append(f"Extreme fear ({sentiment.fear_greed_index}) - contrarian buy signal")
        elif sentiment.fear_greed_index <= 40:
            score += 1
            reasons.append(f"Fear ({sentiment.fear_greed_index}) - cautious buying")
        elif sentiment.fear_greed_index >= 75:
            score -= 2
            reasons.append(f"Extreme greed ({sentiment.fear_greed_index}) - contrarian sell signal")
        elif sentiment.fear_greed_index >= 60:
            score -= 1
            reasons.append(f"Greed ({sentiment.fear_greed_index}) - caution advised")
        
        # News sentiment contribution
        if news:
            avg_sentiment = sum(n.sentiment for n in news) / len(news)
            if avg_sentiment > 0.3:
                score += 1
                reasons.append(f"Bullish news sentiment ({avg_sentiment:.2f})")
            elif avg_sentiment < -0.3:
                score -= 1
                reasons.append(f"Bearish news sentiment ({avg_sentiment:.2f})")
            
            # Critical news override
            critical = [n for n in news if n.priority == 'critical']
            if critical:
                for c in critical:
                    if c.sentiment < 0:
                        score -= 2
                        reasons.append(f"CRITICAL BEARISH: {c.title[:50]}")
                    else:
                        score += 1
                        reasons.append(f"CRITICAL BULLISH: {c.title[:50]}")
        
        # Market cap change
        if sentiment.market_cap_change_24h:
            if sentiment.market_cap_change_24h < -5:
                score -= 1
                reasons.append(f"Large market decline ({sentiment.market_cap_change_24h:.1f}%)")
            elif sentiment.market_cap_change_24h > 5:
                score += 1
                reasons.append(f"Strong market rally ({sentiment.market_cap_change_24h:.1f}%)")
        
        # Generate recommendation
        if score >= 2:
            rec = "🟢 BULLISH - Consider long positions"
        elif score <= -2:
            rec = "🔴 BEARISH - Consider reducing exposure"
        else:
            rec = "🟡 NEUTRAL - Wait for clearer signals"
        
        return f"{rec}\n\nFactors:\n" + "\n".join(f"• {r}" for r in reasons)
    
    async def run_continuous_monitor(self, interval_seconds: int = 300):
        """Run continuous news monitoring loop."""
        logger.info(f"📰 News monitor started (interval: {interval_seconds}s)")
        
        while True:
            try:
                events = await self.check_for_major_events()
                
                if events:
                    logger.info(f"📰 Found {len(events)} significant news items")
                    for event in events:
                        logger.info(f"  [{event.priority.upper()}] {event.title[:60]}...")
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("News monitor stopped")
                break
            except Exception as e:
                logger.error(f"News monitor error: {e}")
                await asyncio.sleep(60)


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        monitor = NewsMonitor()
        
        print("\n" + "="*60)
        print("    JULABA NEWS MONITOR - MARKET SUMMARY")
        print("="*60 + "\n")
        
        summary = await monitor.get_market_summary()
        
        # Sentiment
        if summary['sentiment']:
            s = summary['sentiment']
            print(f"📊 MARKET SENTIMENT")
            print(f"   Fear & Greed Index: {s['fear_greed_index']} ({s['fear_greed_label']})")
            if s['total_market_cap']:
                print(f"   Total Market Cap: ${s['total_market_cap']/1e12:.2f}T")
                print(f"   24h Change: {s['market_cap_change_24h']:.2f}%")
            if s['btc_dominance']:
                print(f"   BTC Dominance: {s['btc_dominance']:.1f}%")
        
        print(f"\n📰 NEWS SUMMARY")
        ns = summary['news_summary']
        print(f"   Total Articles: {ns['total_articles']}")
        print(f"   Bullish: {ns['bullish_count']} | Bearish: {ns['bearish_count']}")
        print(f"   Critical Alerts: {ns['critical_count']}")
        print(f"   Avg Sentiment: {ns['average_sentiment']:.2f}")
        
        if summary['critical_news']:
            print(f"\n🚨 CRITICAL NEWS:")
            for n in summary['critical_news'][:3]:
                print(f"   • {n['title'][:70]}...")
        
        print(f"\n📈 RECENT HEADLINES:")
        for n in summary['recent_news'][:5]:
            emoji = '🟢' if n['sentiment'] > 0.2 else '🔴' if n['sentiment'] < -0.2 else '⚪'
            print(f"   {emoji} {n['title'][:65]}...")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   {summary['recommendation']}")
        
        print("\n" + "="*60)
    
    asyncio.run(test())
