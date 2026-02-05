"""
AI Signal Filter Module
Validates trading signals using AI analysis before execution.
Supports Claude (Anthropic) or Google Gemini.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# AI Provider Configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()  # "claude" or "gemini"

# Claude (Anthropic) imports
ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("Anthropic (Claude) package available")
except ImportError:
    logger.debug("Anthropic package not installed")

# Gemini AI imports - try new package first, fall back to old
GENAI_AVAILABLE = False
GENAI_NEW = False

try:
    # Try new google-genai package first
    from google import genai as genai_new
    from google.genai import types
    GENAI_AVAILABLE = True
    GENAI_NEW = True
    logger.info("Using new google-genai package")
except ImportError:
    try:
        # Fall back to old google-generativeai
        import google.generativeai as genai_old
        GENAI_AVAILABLE = True
        GENAI_NEW = False
        logger.info("Using legacy google-generativeai package (deprecated)")
    except ImportError:
        logger.debug("No Gemini AI package installed")

# Persistent history file paths
HISTORY_DIR = Path(__file__).parent
AI_HISTORY_FILE = HISTORY_DIR / "ai_history.json"
TRADE_HISTORY_FILE = HISTORY_DIR / "trade_history.json"
CHAT_HISTORY_FILE = HISTORY_DIR / "chat_history.json"


class AISignalFilter:
    """
    AI-powered signal filter that analyzes market conditions
    and validates trading signals before execution.
    Supports Claude (Anthropic) or Google Gemini.
    
    STRICT MODE: Higher threshold, skeptic prompt, loss cooldown.
    """
    
    def __init__(self, confidence_threshold: float = 0.75, notifier=None):
        """
        Initialize the AI Signal Filter.
        
        Args:
            confidence_threshold: Minimum confidence (0-1) required to approve a trade
                                 (RAISED to 0.75 - AI was rubber-stamping at 0.65)
            notifier: Optional TelegramNotifier instance for notifications
        """
        self.confidence_threshold = confidence_threshold
        self.notifier = notifier
        self.loss_cooldown_threshold = 0.85  # Raised from 0.80 - be more skeptical after losses
        
        # Rate limiting and cooldown tracking
        self.last_ai_call_time = None
        self.ai_call_count = 0
        self.ai_cooldown_until = None  # Set when API is rate limited
        self.ai_failures_in_row = 0
        self.max_failures_before_cooldown = 3
        
        # AI Provider selection
        self.ai_provider = AI_PROVIDER  # "claude" or "gemini"
        
        # Initialize based on provider
        self.api_keys = []
        self.current_key_index = 0
        self.api_key = ""
        self.use_ai = False
        self.trade_history = []
        self.model = None
        self.client = None
        self.anthropic_client = None
        
        if self.ai_provider == "claude":
            # Claude (Anthropic) configuration
            self.model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
            claude_key = os.getenv("ANTHROPIC_API_KEY", "")
            if claude_key and "your_" not in claude_key.lower() and ANTHROPIC_AVAILABLE:
                self.api_key = claude_key
                self.api_keys.append(claude_key)
                try:
                    self.anthropic_client = anthropic.Anthropic(api_key=claude_key)
                    self.use_ai = True
                    logger.info(f"🤖 Claude AI initialized with model: {self.model_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize Claude: {e}")
                    self.use_ai = False
            else:
                if not ANTHROPIC_AVAILABLE:
                    logger.warning("Claude AI disabled - anthropic package not installed")
                elif not claude_key:
                    logger.info("Claude AI disabled - ANTHROPIC_API_KEY not set")
        else:
            # Gemini configuration (primary provider)
            self.model_name = 'gemini-2.0-flash'  # Fast model for quick decisions
            
            # Primary key
            primary_key = os.getenv("GEMINI_API_KEY", "")
            if primary_key and "your_" not in primary_key.lower():
                self.api_keys.append(primary_key)
            
            # Secondary/backup key
            backup_key = os.getenv("GEMINI_API_KEY_2", "")
            if backup_key and "your_" not in backup_key.lower():
                self.api_keys.append(backup_key)
            
            self.api_key = self.api_keys[0] if self.api_keys else ""
            self.use_ai = bool(self.api_key and GENAI_AVAILABLE)
            
            if len(self.api_keys) > 1:
                logger.info(f"🔑 Dual Gemini API keys configured ({len(self.api_keys)} keys)")
            
            # Initialize Gemini model if available
            if self.use_ai:
                try:
                    if GENAI_NEW:
                        # New google-genai package
                        self.client = genai_new.Client(api_key=self.api_key)
                        logger.info(f"Gemini AI initialized (new SDK) with model: {self.model_name}")
                    else:
                        # Legacy google-generativeai package
                        genai_old.configure(api_key=self.api_key)
                        self.model = genai_old.GenerativeModel(self.model_name)
                        logger.info(f"Gemini AI initialized (legacy SDK) with model: {self.model_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini model: {e}")
                    self.use_ai = False
                    self.model = None
                    self.client = None
            else:
                if not GENAI_AVAILABLE:
                    logger.warning("Gemini AI disabled - google-generativeai not installed")
                elif not self.api_key:
                    logger.info("Gemini AI disabled - GEMINI_API_KEY not set")
        
        # Trading performance tracking - load from persistent storage
        self.recent_trades = []  # List of {"result": "win"/"loss", "pnl": float, "time": str, "symbol": str, "side": str}
        self.total_wins = 0
        self.total_losses = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Load historical trades for AI context
        self._load_trade_history()
        
        # Per-symbol loss tracking for cooldown periods
        # Dict of {symbol: {"last_loss_time": datetime, "consecutive_losses": int}}
        self.symbol_loss_tracker = {}
        self.SYMBOL_COOLDOWN_MINUTES = 120  # 2 hours after loss on a symbol (was 60)
        self.MAX_SYMBOL_LOSSES = 2  # After N consecutive losses, longer cooldown
        self.EXTENDED_COOLDOWN_MINUTES = 360  # 6 hours after repeated losses (was 180)
        
        # Proactive scan threshold - lower = more aggressive (PhD-calibrated: 55)
        self.proactive_threshold = 55
        
        # Chat history for conversational context
        self.chat_history = []
        self.max_chat_history = 20
        self._load_chat_history()
    
    def _load_trade_history(self):
        """Load trade history from persistent file for AI context."""
        try:
            trade_file = Path("trade_history.json")
            if trade_file.exists():
                with open(trade_file, 'r') as f:
                    data = json.load(f)
                
                # Handle both formats: list or dict with trades key
                trades = data if isinstance(data, list) else data.get('trades', data.get('recent_trades', []))
                
                # Convert to our format and keep last 25 for direction analysis
                for t in trades[-25:]:
                    pnl = t.get('pnl', 0)
                    is_win = pnl > 0
                    # Handle both symbol formats (BTCUSDT and BTCUSDT:USDT)
                    symbol = t.get('symbol', '').replace(':USDT', '')
                    # Field is 'side' in trade_history.json, not 'direction'
                    side = t.get('side', t.get('direction', '')).upper()
                    
                    self.recent_trades.append({
                        "result": "win" if is_win else "loss",
                        "pnl": pnl,
                        "time": t.get('close_time', t.get('closed_at', t.get('time', ''))),
                        "symbol": symbol,
                        "side": side
                    })
                    
                    if is_win:
                        self.total_wins += 1
                    else:
                        self.total_losses += 1
                
                # Log direction breakdown
                longs = [t for t in self.recent_trades if t.get('side') == 'LONG']
                shorts = [t for t in self.recent_trades if t.get('side') == 'SHORT']
                long_wins = sum(1 for t in longs if t.get('result') == 'win')
                short_wins = sum(1 for t in shorts if t.get('result') == 'win')
                
                logger.info(f"📊 Loaded {len(self.recent_trades)} trades for AI context: LONG {long_wins}W/{len(longs)-long_wins}L, SHORT {short_wins}W/{len(shorts)-short_wins}L")
        except Exception as e:
            logger.warning(f"Could not load trade history for AI: {e}")
    
    def _switch_api_key(self) -> bool:
        """Switch to next available API key. Returns True if switched successfully."""
        if len(self.api_keys) <= 1:
            return False
        
        # Move to next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_index]
        
        # Reinitialize client with new key
        try:
            if GENAI_NEW:
                self.client = genai_new.Client(api_key=self.api_key)
                logger.info(f"🔄 Switched to API key #{self.current_key_index + 1}")
                return True
            else:
                genai_old.configure(api_key=self.api_key)
                self.model = genai_old.GenerativeModel(self.model_name)
                logger.info(f"🔄 Switched to API key #{self.current_key_index + 1}")
                return True
        except Exception as e:
            logger.error(f"Failed to switch API key: {e}")
            return False
    
    def _is_ai_available(self) -> bool:
        """Check if AI is available (not in cooldown, not too many failures)."""
        if not self.use_ai:
            return False
        
        # Check cooldown
        if self.ai_cooldown_until:
            now = datetime.utcnow()
            if now < self.ai_cooldown_until:
                remaining = (self.ai_cooldown_until - now).total_seconds()
                if remaining > 60:  # Only log if > 1 minute
                    logger.debug(f"AI in cooldown for {remaining:.0f}s more")
                return False
            else:
                # Cooldown expired
                self.ai_cooldown_until = None
                self.ai_failures_in_row = 0
                logger.info("🔄 AI cooldown expired, resuming AI calls")
        
        return True
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get current AI status for monitoring/dashboard."""
        now = datetime.utcnow()
        cooldown_remaining = 0
        if self.ai_cooldown_until and now < self.ai_cooldown_until:
            cooldown_remaining = (self.ai_cooldown_until - now).total_seconds()
        
        return {
            "available": self._is_ai_available(),
            "enabled": self.use_ai,
            "provider": self.ai_provider,
            "model": self.model_name if self.use_ai else None,
            "api_keys_count": len(self.api_keys),
            "current_key_index": self.current_key_index + 1,
            "call_count": self.ai_call_count,
            "failures_in_row": self.ai_failures_in_row,
            "in_cooldown": cooldown_remaining > 0,
            "cooldown_remaining_seconds": int(cooldown_remaining),
            "last_call": self.last_ai_call_time.isoformat() if self.last_ai_call_time else None
        }
    
    def _generate_content(self, prompt: str, retry_count: int = 0) -> Optional[str]:
        """Generate content using Claude or Gemini API.
        Automatically switches API key on quota errors.
        Includes rate limiting and cooldown management."""
        
        # Check if we're in cooldown
        if not self._is_ai_available():
            return None
        
        try:
            self.last_ai_call_time = datetime.utcnow()
            self.ai_call_count += 1
            
            # Claude (Anthropic) API
            if self.ai_provider == "claude" and self.anthropic_client:
                message = self.anthropic_client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    timeout=30.0,  # 30 second timeout to prevent hanging
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                # Success - reset failure counter
                self.ai_failures_in_row = 0
                return message.content[0].text
            
            # Gemini (Google) API - new SDK
            elif GENAI_NEW and self.client:
                logger.info(f"🤖 Calling Gemini API (new SDK) with model: {self.model_name}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                # Success - reset failure counter
                self.ai_failures_in_row = 0
                logger.info(f"🤖 Gemini response received ({len(response.text) if response.text else 0} chars)")
                return response.text
            
            # Gemini (Google) API - legacy SDK
            elif self.model:
                response = self.model.generate_content(prompt)
                # Success - reset failure counter
                self.ai_failures_in_row = 0
                return response.text
            else:
                logger.error("No AI model available")
                return None
        except Exception as e:
            error_str = str(e)
            self.ai_failures_in_row += 1
            
            # Check for quota exhaustion (429 error)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower() or "rate_limit" in error_str.lower():
                logger.warning(f"API quota exhausted on key #{self.current_key_index + 1}")
                
                # Try switching to backup key (only once per call)
                if retry_count < len(self.api_keys) - 1 and self._switch_api_key():
                    logger.info("Retrying with backup API key...")
                    return self._generate_content(prompt, retry_count + 1)
                else:
                    # All keys exhausted - NO COOLDOWN, just return None
                    logger.error(f"AI generation error: All API keys exhausted. No cooldown - will retry next call.")
                    return None
            else:
                logger.error(f"AI generation error: {e}")
                
                # Log failures but NO COOLDOWN - always retry next call
                if self.ai_failures_in_row >= self.max_failures_before_cooldown:
                    logger.warning(f"🟡 {self.ai_failures_in_row} consecutive AI failures. Will retry next call (no cooldown).")
                
                return None
    
    def record_trade_result(self, is_win: bool, pnl: float, symbol: str = "", side: str = ""):
        """Record a trade result to inform future AI decisions."""
        result = "win" if is_win else "loss"
        self.recent_trades.append({
            "result": result,
            "pnl": pnl,
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "symbol": symbol,
            "side": side.upper() if side else ""
        })
        # Keep only last 10 trades
        if len(self.recent_trades) > 10:
            self.recent_trades = self.recent_trades[-10:]
        
        if is_win:
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # Win resets the cooldown for this symbol
            if symbol:
                symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                if symbol_key in self.symbol_loss_tracker:
                    del self.symbol_loss_tracker[symbol_key]
                    logger.info(f"✅ {symbol}: Cooldown cleared after WIN")
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            # Track per-symbol losses for cooldown
            if symbol:
                symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                now = datetime.utcnow()
                if symbol_key not in self.symbol_loss_tracker:
                    self.symbol_loss_tracker[symbol_key] = {
                        "last_loss_time": now,
                        "consecutive_losses": 1
                    }
                else:
                    self.symbol_loss_tracker[symbol_key]["last_loss_time"] = now
                    self.symbol_loss_tracker[symbol_key]["consecutive_losses"] += 1
                
                consec = self.symbol_loss_tracker[symbol_key]["consecutive_losses"]
                cooldown = self.EXTENDED_COOLDOWN_MINUTES if consec >= self.MAX_SYMBOL_LOSSES else self.SYMBOL_COOLDOWN_MINUTES
                logger.warning(f"🚫 {symbol}: LOSS #{consec} - Cooldown {cooldown} mins")
        
        logger.info(f"Trade recorded: {result} ${pnl:+.2f} | Streak: {self.consecutive_wins}W / {self.consecutive_losses}L")
        
        # Reset peak profit tracking for closed position
        if hasattr(self, '_peak_profits') and symbol:
            # Clear all peak profits for this symbol (both LONG and SHORT)
            keys_to_remove = [k for k in self._peak_profits.keys() if symbol.upper() in k.upper()]
            for key in keys_to_remove:
                del self._peak_profits[key]
                logger.debug(f"Reset peak profit tracking for {key}")

    def _is_symbol_on_cooldown(self, symbol: str) -> tuple:
        """
        Check if a symbol is on cooldown after recent losses.
        Returns: (is_on_cooldown: bool, reason: str, minutes_remaining: int)
        """
        symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
        
        if symbol_key not in self.symbol_loss_tracker:
            return False, "", 0
        
        tracker = self.symbol_loss_tracker[symbol_key]
        last_loss = tracker.get("last_loss_time")
        consec_losses = tracker.get("consecutive_losses", 1)
        
        if not last_loss:
            return False, "", 0
        
        # Determine cooldown duration based on consecutive losses
        if consec_losses >= self.MAX_SYMBOL_LOSSES:
            cooldown_mins = self.EXTENDED_COOLDOWN_MINUTES
            reason = f"{consec_losses} consecutive losses"
        else:
            cooldown_mins = self.SYMBOL_COOLDOWN_MINUTES
            reason = "recent loss"
        
        now = datetime.utcnow()
        elapsed = (now - last_loss).total_seconds() / 60
        
        if elapsed < cooldown_mins:
            remaining = int(cooldown_mins - elapsed)
            return True, reason, remaining
        
        # Cooldown expired, clean up
        del self.symbol_loss_tracker[symbol_key]
        return False, "", 0

    def _get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific symbol WITH per-direction breakdown.
        Returns recent win/loss history and PnL for this symbol, split by LONG/SHORT.
        This is critical for per-pair direction bias detection.
        """
        symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
        
        # Filter recent trades for this symbol
        symbol_trades = [
            t for t in self.recent_trades 
            if t.get('symbol', '').replace('/', '').replace(':USDT', '').upper() == symbol_key
        ]
        
        if not symbol_trades:
            return {
                "has_history": False,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0,
                "win_rate": "N/A",
                "last_result": "N/A",
                # Per-direction stats (empty)
                "long_wins": 0,
                "long_losses": 0,
                "long_pnl": 0,
                "long_wr": 0,
                "short_wins": 0,
                "short_losses": 0,
                "short_pnl": 0,
                "short_wr": 0,
                "preferred_direction": None,
                "direction_advice": ""
            }
        
        # Overall stats
        wins = sum(1 for t in symbol_trades if t.get('result', '').lower() == 'win')
        losses = len(symbol_trades) - wins
        total_pnl = sum(t.get('pnl', 0) for t in symbol_trades)
        win_rate = wins / len(symbol_trades) * 100 if symbol_trades else 0
        last_result = symbol_trades[-1].get('result', 'unknown').upper() if symbol_trades else "N/A"
        
        # Per-direction breakdown for THIS symbol
        long_trades = [t for t in symbol_trades if t.get('side', '').upper() == 'LONG']
        short_trades = [t for t in symbol_trades if t.get('side', '').upper() == 'SHORT']
        
        long_wins = sum(1 for t in long_trades if t.get('result', '').lower() == 'win')
        long_losses = len(long_trades) - long_wins
        long_pnl = sum(t.get('pnl', 0) for t in long_trades)
        long_wr = (long_wins / len(long_trades) * 100) if long_trades else 0
        
        short_wins = sum(1 for t in short_trades if t.get('result', '').lower() == 'win')
        short_losses = len(short_trades) - short_wins
        short_pnl = sum(t.get('pnl', 0) for t in short_trades)
        short_wr = (short_wins / len(short_trades) * 100) if short_trades else 0
        
        # Determine preferred direction for THIS symbol
        preferred_direction = None
        direction_advice = ""
        
        # Need at least 2 trades in each direction to make judgment
        if len(long_trades) >= 2 and len(short_trades) >= 2:
            if long_wr >= 60 and short_wr < 45:
                preferred_direction = "LONG"
                direction_advice = f"🎯 {symbol_key}: LONG strongly preferred ({long_wins}W/{long_losses}L={long_wr:.0f}% vs SHORT {short_wins}W/{short_losses}L={short_wr:.0f}%)"
            elif short_wr >= 60 and long_wr < 45:
                preferred_direction = "SHORT"
                direction_advice = f"🎯 {symbol_key}: SHORT strongly preferred ({short_wins}W/{short_losses}L={short_wr:.0f}% vs LONG {long_wins}W/{long_losses}L={long_wr:.0f}%)"
            elif long_pnl > 0 and short_pnl < -2:
                preferred_direction = "LONG"
                direction_advice = f"💰 {symbol_key}: LONG profitable (${long_pnl:+.2f}) but SHORT losing (${short_pnl:+.2f})"
            elif short_pnl > 0 and long_pnl < -2:
                preferred_direction = "SHORT"
                direction_advice = f"💰 {symbol_key}: SHORT profitable (${short_pnl:+.2f}) but LONG losing (${long_pnl:+.2f})"
        elif len(long_trades) >= 3 and len(short_trades) == 0:
            if long_wr >= 60:
                preferred_direction = "LONG"
                direction_advice = f"📊 {symbol_key}: Only LONG trades ({long_wins}W/{long_losses}L={long_wr:.0f}%) - no SHORT history"
        elif len(short_trades) >= 3 and len(long_trades) == 0:
            if short_wr >= 60:
                preferred_direction = "SHORT"
                direction_advice = f"📊 {symbol_key}: Only SHORT trades ({short_wins}W/{short_losses}L={short_wr:.0f}%) - no LONG history"
        
        return {
            "has_history": True,
            "wins": wins,
            "losses": losses,
            "total_pnl": total_pnl,
            "win_rate": f"{win_rate:.0f}%",
            "last_result": last_result,
            # Per-direction stats for this symbol
            "long_wins": long_wins,
            "long_losses": long_losses,
            "long_trades": len(long_trades),
            "long_pnl": long_pnl,
            "long_wr": long_wr,
            "short_wins": short_wins,
            "short_losses": short_losses,
            "short_trades": len(short_trades),
            "short_pnl": short_pnl,
            "short_wr": short_wr,
            "preferred_direction": preferred_direction,
            "direction_advice": direction_advice
        }

    def _build_symbol_history_section(self, symbol_perf: Dict, symbol: str) -> str:
        """Build a formatted string showing per-pair direction history for AI prompt.
        NOTE: This is SECONDARY info - math analysis is primary!"""
        if not symbol_perf.get("has_history"):
            return f"No trade history for {symbol} yet - decide based on math analysis (this is fine!)."
        
        lines = []
        lines.append(f"📊 Past trades (for context only, math is primary):")
        lines.append(f"   Total: {symbol_perf['wins']}W/{symbol_perf['losses']}L (${symbol_perf['total_pnl']:+.2f})")
        
        # Per-direction breakdown
        if symbol_perf.get("long_trades", 0) > 0:
            lines.append(f"   LONG: {symbol_perf['long_wins']}W/{symbol_perf['long_losses']}L ({symbol_perf['long_wr']:.0f}%)")
        if symbol_perf.get("short_trades", 0) > 0:
            lines.append(f"   SHORT: {symbol_perf['short_wins']}W/{symbol_perf['short_losses']}L ({symbol_perf['short_wr']:.0f}%)")
        
        # Note that this is secondary
        lines.append(f"⚠️ Note: History is from OLD system - trust math analysis more!")
        
        return "\n".join(lines)

    def _get_direction_performance(self) -> Dict[str, Any]:
        """
        Get performance statistics by trade direction (LONG vs SHORT).
        Critical for understanding which direction is actually profitable.
        """
        long_trades = [t for t in self.recent_trades if t.get('side', '').upper() == 'LONG']
        short_trades = [t for t in self.recent_trades if t.get('side', '').upper() == 'SHORT']
        
        long_wins = sum(1 for t in long_trades if t.get('result', '').upper() == 'WIN')
        long_losses = len(long_trades) - long_wins
        long_pnl = sum(t.get('pnl', 0) for t in long_trades)
        long_wr = (long_wins / len(long_trades) * 100) if long_trades else 0
        
        short_wins = sum(1 for t in short_trades if t.get('result', '').upper() == 'WIN')
        short_losses = len(short_trades) - short_wins
        short_pnl = sum(t.get('pnl', 0) for t in short_trades)
        short_wr = (short_wins / len(short_trades) * 100) if short_trades else 0
        
        # Determine which direction is performing better
        better_direction = None
        direction_warning = ""
        
        if len(long_trades) >= 5 and len(short_trades) >= 5:
            if long_wr >= 55 and short_wr < 45:
                better_direction = "LONG"
                direction_warning = f"🚨 CRITICAL: LONG trades have {long_wr:.0f}% WR (${long_pnl:+.2f}), but SHORT trades only {short_wr:.0f}% WR (${short_pnl:+.2f}). STRONGLY PREFER LONG!"
            elif short_wr >= 55 and long_wr < 45:
                better_direction = "SHORT"
                direction_warning = f"🚨 CRITICAL: SHORT trades have {short_wr:.0f}% WR (${short_pnl:+.2f}), but LONG trades only {long_wr:.0f}% WR (${long_pnl:+.2f}). STRONGLY PREFER SHORT!"
            elif long_pnl > 0 and short_pnl < -5:
                better_direction = "LONG"
                direction_warning = f"⚠️ WARNING: LONG is profitable (${long_pnl:+.2f}), but SHORT is losing (${short_pnl:+.2f}). Favor LONG trades!"
            elif short_pnl > 0 and long_pnl < -5:
                better_direction = "SHORT"
                direction_warning = f"⚠️ WARNING: SHORT is profitable (${short_pnl:+.2f}), but LONG is losing (${long_pnl:+.2f}). Favor SHORT trades!"
        
        return {
            "long_trades": len(long_trades),
            "long_wins": long_wins,
            "long_losses": long_losses,
            "long_pnl": long_pnl,
            "long_wr": long_wr,
            "short_trades": len(short_trades),
            "short_wins": short_wins,
            "short_losses": short_losses,
            "short_pnl": short_pnl,
            "short_wr": short_wr,
            "better_direction": better_direction,
            "direction_warning": direction_warning
        }

    def _get_news_context(self) -> Dict[str, Any]:
        """
        Get current market news and sentiment context for AI decision making.
        Uses cached data to avoid API rate limits during trading.
        """
        try:
            import asyncio
            from news_monitor import NewsMonitor
            
            monitor = NewsMonitor()
            
            # Run async function - handle both sync and async contexts
            try:
                # Check if there's already a running event loop
                loop = asyncio.get_running_loop()
                # Already in async context - create a task and use nest_asyncio or return default
                # For safety, just return cached/default values in async context
                return self._get_default_news_context()
            except RuntimeError:
                # No running loop - safe to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    summary = loop.run_until_complete(monitor.get_market_summary())
                finally:
                    loop.close()
            
            # Extract key info for AI
            sentiment = summary.get('sentiment', {})
            news_summary = summary.get('news_summary', {})
            critical_news = summary.get('critical_news', [])
            
            # Build news context
            news_context = {
                'fear_greed_index': sentiment.get('fear_greed_index', 50),
                'fear_greed_label': sentiment.get('fear_greed_label', 'Neutral'),
                'market_cap_change_24h': sentiment.get('market_cap_change_24h', 0),
                'btc_dominance': sentiment.get('btc_dominance', 0),
                'news_sentiment': news_summary.get('average_sentiment', 0),
                'bullish_news_count': news_summary.get('bullish_count', 0),
                'bearish_news_count': news_summary.get('bearish_count', 0),
                'critical_news_count': news_summary.get('critical_count', 0),
                'critical_headlines': [n.get('title', '')[:80] for n in critical_news[:3]],
                'recommendation': summary.get('recommendation', '').split('\n')[0] if summary.get('recommendation') else ''
            }
            
            # Generate warning based on sentiment
            warning = ""
            fg = news_context['fear_greed_index']
            cap_change = news_context['market_cap_change_24h']
            
            if fg <= 20:
                warning = f"🔴 EXTREME FEAR ({fg}) - Market panic, potential oversold bounce opportunity"
            elif fg <= 35:
                warning = f"🟠 FEAR ({fg}) - Cautious sentiment, watch for reversals"
            elif fg >= 80:
                warning = f"🟢 EXTREME GREED ({fg}) - Euphoria, potential top forming"
            elif fg >= 65:
                warning = f"🟡 GREED ({fg}) - Bullish sentiment, momentum likely continues"
            
            if abs(cap_change) >= 5:
                direction = "📈 SURGING" if cap_change > 0 else "📉 CRASHING"
                warning += f"\n{direction}: Market cap {cap_change:+.1f}% in 24h - MAJOR MOVE"
            
            if news_context['critical_news_count'] > 0:
                warning += f"\n🚨 {news_context['critical_news_count']} CRITICAL NEWS items - check before trading!"
            
            news_context['warning'] = warning
            
            # Log news context for visibility
            logger.info(f"📰 NEWS CONTEXT: Fear&Greed={fg} ({news_context['fear_greed_label']}), Market 24h={cap_change:+.1f}%, Sentiment={news_context['news_sentiment']:+.2f}")
            
            return news_context
            
        except Exception as e:
            logger.debug(f"News context fetch error: {e}")
            return self._get_default_news_context()
    
    def _get_default_news_context(self) -> Dict[str, Any]:
        """Return default news context when async fetch isn't possible."""
        return {
            'fear_greed_index': 50,
            'fear_greed_label': 'Neutral',
            'market_cap_change_24h': 0,
            'news_sentiment': 0,
            'critical_headlines': [],
            'warning': ''
        }

    def record_system_message(self, message: str):
        """Record a system-level event to the chat history."""
        self.chat_history.append({
            "role": "model",
            "content": f"[SYSTEM EVENT]: {message}"
        })
        self._save_chat_history()

    def _ai_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Use Google Gemini API for signal analysis with SKEPTIC MODE. Retries once and notifies via Telegram on fallback."""
        signal_type = "LONG" if signal == 1 else "SHORT"
        perf = self._get_performance_context()
        market = self._get_market_hours_context()
        extra_caution = ""
        if perf["last_trade_was_loss"]:
            extra_caution = f"\n⚠️ CAUTION: Last {perf['consecutive_losses']} trade(s) were losses. Be extra skeptical!"
        if perf["consecutive_losses"] >= 2:
            extra_caution += "\n🛑 LOSING STREAK: Require very high confidence to approve."
        if market["is_weekend"]:
            extra_caution += "\n📅 WEEKEND: Lower liquidity, higher risk of false moves."
        if market["activity_level"] == "low":
            extra_caution += "\n🌙 LOW ACTIVITY HOURS: Increased slippage risk."
        # Build ML insight section for prompt
        ml_insight = context.get('ml_insight', {})
        ml_section = self._build_ml_section(ml_insight)
        
        # Build system score section for prompt
        system_score = context.get('system_score', {})
        system_section = self._build_system_score_section(system_score)
        
        # Build advanced math analysis section for prompt
        math_check = context.get('math_check', {})
        math_section = self._build_math_analysis_section(math_check)
        
        # Log that advanced math is being included in AI prompt
        if math_check.get('detailed_analysis'):
            detailed = math_check['detailed_analysis']
            logger.info(f"🧮 AI MATH INTEGRATION: Score={math_check.get('score', 0):.0f}, "
                       f"Kalman={detailed.get('kalman_momentum', 'N/A')}, "
                       f"POC=${detailed.get('poc_price', 0):,.0f}, "
                       f"RSI_Div={bool(detailed.get('rsi_divergence_mtf', {}).get('regular_bullish_divergence') or detailed.get('rsi_divergence_mtf', {}).get('regular_bearish_divergence'))}")
        
        prompt = (
            f"You are the FINAL DECISION MAKER for an autonomous crypto trading bot.\n"
            f"Act as a PhD Quantitative Analyst and Risk Manager.\n"
            f"Your job is to PROTECT capital by applying rigorous mathematical verification.\n"
            f"The system has already analyzed this signal through multiple layers:\n"
            f"  1. Technical indicators generated this signal\n"
            f"  2. ML model evaluated historical pattern similarity\n"
            f"  3. NOW YOU verify the mathematical probability of success\n\n"
            f"=== SIGNAL ===\n"
            f"Proposed Trade: {signal_type} on {symbol}\n"
            f"=== MARKET DATA ===\n"
            f"Current Price: ${context['current_price']}\n"
            f"1-Hour Price Change: {context['price_change_1h']}%\n"
            f"Volume Ratio (vs avg): {context['volume_ratio']}x\n"
            f"Trend (SMA10 vs SMA20): {context['trend']}\n"
            f"Volatility (ATR%): {context['volatility_pct']}%\n"
            f"{ml_section}"
            f"{system_section}"
            f"{math_section}"
            f"=== TRADING PERFORMANCE ===\n"
            f"Total Trades: {perf['total_trades']}\n"
            f"Win Rate: {perf['win_rate']}%\n"
            f"Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L\n"
            f"Recent P&L (last 5): ${perf['recent_pnl']}\n"
            f"=== MARKET SESSION ===\n"
            f"Session: {market['session']} ({market['hour_utc']}:00 UTC)\n"
            f"Activity Level: {market['activity_level']}\n"
            f"Weekend: {market['is_weekend']}\n"
            f"{extra_caution}\n"
            f"=== YOUR DECISION ===\n"
            f"Use the ADVANCED MATH ANALYSIS above to make your decision:\n"
            f"1. Is the Math Score ({math_check.get('score', 50):.0f}/100) sufficient? (need >= 55 to approve)\n"
            f"2. Review the REASONS FOR and AGAINST from the math analysis\n"
            f"3. Check Kalman momentum direction - does it support this trade?\n"
            f"4. Check Volume Profile (POC) - is price at a good entry level?\n"
            f"5. Check for RSI divergences - any warning signals?\n"
            f"IMPORTANT: Trust the PhD Math Score over technical momentum.\n"
            f"Only approve if Math Score >= 55 AND reasons FOR outweigh reasons AGAINST.\n"
            f"Respond ONLY with this JSON format, no other text:\n"
            f'{{"reasons_against": ["math_reason1", "math_reason2", "math_reason3"], "approved": false, "confidence": 0.65, "reasoning": "quantitative justification referencing the math analysis", "risk_assessment": "low/medium/high", "ml_agreement": "agree/disagree/neutral", "math_score_assessment": "appropriate/too_high/too_low"}}'
        )
        for attempt in range(2):
            try:
                result_text = self._generate_content(prompt)
                if not result_text:
                    continue
                result_text = result_text.strip()
                # Parse JSON from response
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                result_text = result_text.strip()
                result = json.loads(result_text)
                result.setdefault("approved", False)
                result.setdefault("confidence", 0.5)
                result.setdefault("reasoning", "AI analysis")
                result.setdefault("risk_assessment", "medium")
                result.setdefault("reasons_against", [])
                result.setdefault("ml_agreement", "neutral")  # AI's take on ML prediction
                # Apply confidence threshold with LOSS COOLDOWN
                perf = self._get_performance_context()
                if perf["consecutive_losses"] >= 2:
                    required_threshold = self.loss_cooldown_threshold
                    logger.info(f"Loss cooldown active: requiring {required_threshold:.0%} confidence")
                elif perf["last_trade_was_loss"]:
                    required_threshold = 0.85
                else:
                    required_threshold = self.confidence_threshold
                result["approved"] = result["approved"] and result["confidence"] >= required_threshold
                result["threshold_used"] = required_threshold
                if result.get("reasons_against"):
                    logger.info(f"AI reasons against trade: {result['reasons_against']}")
                # Log ML agreement if present
                if result.get("ml_agreement") != "neutral":
                    logger.info(f"AI on ML prediction: {result['ml_agreement']}")
                return result
            except Exception as e:
                logger.warning(f"Gemini analysis attempt {attempt+1} failed: {e}")
        # If both attempts fail, notify via Telegram and fallback
        logger.error(f"Gemini analysis failed twice, falling back to rules")
        if self.notifier and hasattr(self.notifier, 'send_message'):
            try:
                import asyncio
                msg = f"⚠️ Gemini AI analysis failed twice for {symbol} {signal_type}. Falling back to rule-based analysis."
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.notifier.send_message(msg))
                else:
                    loop.run_until_complete(self.notifier.send_message(msg))
            except Exception as notify_err:
                logger.warning(f"Failed to send Telegram notification: {notify_err}")
            if self.notifier and hasattr(self.notifier, 'send_message'):
                try:
                    import asyncio
                    msg = f"⚠️ Gemini AI analysis failed twice for {symbol} {signal_type}. Falling back to rule-based analysis."
                    # If running in an async context, schedule the message
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.notifier.send_message(msg))
                    else:
                        loop.run_until_complete(self.notifier.send_message(msg))
                except Exception as notify_err:
                    logger.warning(f"Failed to send Telegram notification: {notify_err}")
            return self._rule_based_analysis(signal, context, symbol)
    
    def _get_performance_context(self) -> Dict[str, Any]:
        """Get trading performance context for AI prompts."""
        total_trades = self.total_wins + self.total_losses
        win_rate_pct = (self.total_wins / max(1, total_trades)) * 100
        recent_pnl = sum(t.get('pnl', 0) for t in self.recent_trades[-5:]) if self.recent_trades else 0
        return {
            "total_trades": total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "last_trade_was_loss": self.consecutive_losses > 0,
            "recent_trades": self.recent_trades[-5:] if self.recent_trades else [],
            "win_rate": round(win_rate_pct, 1),
            "recent_pnl": round(recent_pnl, 2)
        }
    
    def _get_market_hours_context(self) -> Dict[str, Any]:
        """Get market timing context."""
        now = datetime.utcnow()
        hour = now.hour
        
        # Crypto market sessions (approximate)
        if 0 <= hour < 8:
            session = "Asia"
            activity = "moderate"
        elif 8 <= hour < 14:
            session = "Europe"
            activity = "high"
        elif 14 <= hour < 21:
            session = "US"
            activity = "high"
        else:
            session = "Late US/Early Asia"
            activity = "low"
        
        # Weekend check (lower liquidity)
        is_weekend = now.weekday() >= 5
        
        return {
            "session": session,
            "activity_level": activity,
            "is_weekend": is_weekend,
            "hour_utc": hour
        }
    
    def _build_ml_section(self, ml_insight: Dict[str, Any]) -> str:
        """Build ML insight section for AI prompt."""
        if not ml_insight or not ml_insight.get('ml_available', False):
            return "=== ML MODEL ===\nStatus: Not available (training in progress)\n\n"
        
        win_prob = ml_insight.get('ml_win_probability', 0.5)
        confidence = ml_insight.get('ml_confidence', 'UNKNOWN')
        accuracy = ml_insight.get('ml_accuracy', 0)
        samples = ml_insight.get('ml_samples', 0)
        influence = ml_insight.get('ml_influence', 0)
        
        # Determine ML recommendation
        if win_prob >= 0.60:
            ml_rec = "FAVORABLE - ML suggests this trade type has historically performed well"
        elif win_prob <= 0.40:
            ml_rec = "UNFAVORABLE - ML suggests caution, similar setups had lower win rates"
        else:
            ml_rec = "NEUTRAL - ML has no strong signal for this setup"
        
        # Model maturity assessment
        if samples < 50:
            maturity = "EARLY (limited data, use with caution)"
        elif samples < 200:
            maturity = "DEVELOPING (moderate confidence)"
        else:
            maturity = "MATURE (high confidence in predictions)"
        
        return (
            f"=== ML MODEL INSIGHT ===\n"
            f"Win Probability: {win_prob:.1%}\n"
            f"Confidence Level: {confidence}\n"
            f"Model Accuracy: {accuracy:.1%}\n"
            f"Training Samples: {samples}\n"
            f"Model Maturity: {maturity}\n"
            f"ML Recommendation: {ml_rec}\n"
            f"Influence Weight: {influence:.0%} (0%=advisory only, 100%=full control)\n\n"
        )

    def _build_system_score_section(self, system_score: Dict[str, Any]) -> str:
        """Build system score section for AI prompt."""
        if not system_score or 'combined' not in system_score:
            return "=== SYSTEM SCORE ===\nNot available\n\n"
        
        combined = system_score.get('combined', 50)
        recommendation = system_score.get('recommendation', 'NEUTRAL')
        breakdown = system_score.get('breakdown', 'N/A')
        
        # Interpret the score
        if combined >= 75:
            interpretation = "STRONG - All system components align favorably"
        elif combined >= 60:
            interpretation = "GOOD - Most components favorable, minor concerns"
        elif combined >= 45:
            interpretation = "NEUTRAL - Mixed signals, proceed with caution"
        elif combined >= 30:
            interpretation = "WEAK - Several concerning factors"
        else:
            interpretation = "POOR - System recommends avoiding this trade"
        
        return (
            f"=== SYSTEM SCORE (0-100) ===\n"
            f"Combined Score: {combined:.0f}/100\n"
            f"Recommendation: {recommendation}\n"
            f"Breakdown: {breakdown}\n"
            f"Interpretation: {interpretation}\n\n"
        )

    def _build_math_analysis_section(self, math_check: Dict[str, Any]) -> str:
        """
        Build advanced math analysis section for AI prompt.
        Includes Kalman momentum, Volume Profile (POC), RSI divergence MTF, etc.
        """
        if not math_check:
            return "=== ADVANCED MATH ANALYSIS ===\nNot available\n\n"
        
        score = math_check.get('score', 50)
        approved = math_check.get('approved', False)
        confidence_level = math_check.get('confidence_level', 'unknown')
        reasons_for = math_check.get('reasons_for', [])
        reasons_against = math_check.get('reasons_against', [])
        detailed = math_check.get('detailed_analysis', {})
        
        # Build the section
        lines = [
            "=== ADVANCED MATH ANALYSIS (PhD-Level) ===",
            f"Math Score: {score:.0f}/100 ({'APPROVED' if approved else 'NOT APPROVED'})",
            f"Confidence Level: {confidence_level.upper()}",
            ""
        ]
        
        # Kalman Filter Momentum
        kalman_mom = detailed.get('kalman_momentum')
        if kalman_mom is not None:
            kalman_zscore = detailed.get('kalman_momentum_zscore', 0)
            kalman_trend = detailed.get('kalman_momentum_trend', 'unknown')
            lines.append(f"📊 KALMAN MOMENTUM: {kalman_mom:.2f} (z-score: {kalman_zscore:.2f}, trend: {kalman_trend})")
        
        # Volume Profile / POC
        poc_price = detailed.get('poc_price')
        if poc_price is not None:
            price_vs_poc = detailed.get('price_vs_poc', 0)
            in_va = detailed.get('in_value_area', False)
            va_high = detailed.get('value_area_high', 0)
            va_low = detailed.get('value_area_low', 0)
            lines.append(f"📊 VOLUME PROFILE: POC=${poc_price:,.2f}, Price vs POC: {price_vs_poc:+.2f}%")
            lines.append(f"   Value Area: ${va_low:,.2f}-${va_high:,.2f} ({'IN' if in_va else 'OUT'})")
        
        # RSI Divergence
        rsi_div = detailed.get('rsi_divergence_mtf')
        if rsi_div:
            rsi = rsi_div.get('current_rsi', 50)
            reg_bull = rsi_div.get('regular_bullish_divergence', False)
            reg_bear = rsi_div.get('regular_bearish_divergence', False)
            hid_bull = rsi_div.get('hidden_bullish_divergence', False)
            hid_bear = rsi_div.get('hidden_bearish_divergence', False)
            mtf_conf = rsi_div.get('mtf_confirmation', False)
            
            div_signals = []
            if reg_bull: div_signals.append("REGULAR BULLISH")
            if reg_bear: div_signals.append("REGULAR BEARISH")
            if hid_bull: div_signals.append("HIDDEN BULLISH")
            if hid_bear: div_signals.append("HIDDEN BEARISH")
            
            if div_signals:
                mtf_str = " (MTF CONFIRMED!)" if mtf_conf else ""
                lines.append(f"📊 RSI DIVERGENCE: {', '.join(div_signals)}{mtf_str} (RSI={rsi:.1f})")
            else:
                lines.append(f"📊 RSI: {rsi:.1f} (No divergence detected)")
        
        # Other key metrics
        hurst = detailed.get('hurst_exponent')
        if hurst is not None:
            regime = "TRENDING" if hurst > 0.55 else "MEAN-REVERTING" if hurst < 0.45 else "RANDOM"
            lines.append(f"📊 HURST EXPONENT: {hurst:.3f} ({regime})")
        
        zscore = detailed.get('z_score')
        if zscore is not None:
            lines.append(f"📊 PRICE Z-SCORE: {zscore:.2f}σ")
        
        garch_vol = detailed.get('garch_vol')
        if garch_vol is not None:
            vol_trend = detailed.get('vol_trend', 'stable')
            lines.append(f"📊 GARCH VOLATILITY: {garch_vol:.2%} ({vol_trend})")
        
        sharpe = detailed.get('sharpe_ratio')
        sortino = detailed.get('sortino_ratio')
        if sharpe is not None:
            sortino_str = f"{sortino:.2f}" if sortino else "N/A"
            lines.append(f"📊 RISK-ADJUSTED: Sharpe={sharpe:.2f}, Sortino={sortino_str}")
        
        lines.append("")
        
        # Escape curly braces in reasons to prevent f-string format issues when used in prompts
        def safe_reason(r):
            return str(r).replace('{', '(').replace('}', ')')
        
        # Reasons FOR
        if reasons_for:
            lines.append("✅ REASONS FOR TRADE:")
            for i, reason in enumerate(reasons_for[:5], 1):
                lines.append(f"   {i}. {safe_reason(reason)}")
        else:
            lines.append("✅ REASONS FOR TRADE: None identified")
        
        lines.append("")
        
        # Reasons AGAINST
        if reasons_against:
            lines.append("❌ REASONS AGAINST TRADE:")
            for i, reason in enumerate(reasons_against[:5], 1):
                lines.append(f"   {i}. {safe_reason(reason)}")
        else:
            lines.append("❌ REASONS AGAINST TRADE: None identified")
        
        lines.append("")
        
        return "\n".join(lines) + "\n"

    def analyze_signal(
        self,
        signal: int,  # 1 = long, -1 = short, 0 = none
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str,
        ml_insight: Dict[str, Any] = None,
        system_score: Dict[str, Any] = None,  # Combined system scoring
        market_scanner_context: Dict[str, Any] = None,  # Market scanner recommendation
        tech_score: Dict[str, Any] = None  # Technical score breakdown
    ) -> Dict[str, Any]:
        """
        Analyze a trading signal and return AI validation result.
        AI is the FINAL DECISION MAKER in the autonomous pipeline.
        
        Decision Pipeline: Signal → ML → AI (final)
        
        Args:
            ml_insight: Dict with ML prediction data
            system_score: Dict with combined system scoring:
                - combined: 0-100 overall score
                - recommendation: STRONG_BUY/BUY/NEUTRAL/WEAK/AVOID
                - breakdown: Component scores
            market_scanner_context: Dict with market scanner data:
                - best_pair: Recommended pair from scanner
                - current_pair_rank: Rank of current pair in scanner
            tech_score: Dict with technical score breakdown:
                - score: 0-100 technical quality
                - quality: EXCELLENT/GOOD/MODERATE/WEAK/POOR
                - factors: List of notable factors
                - breakdown: Component scores
        
        Returns:
            Dict with keys: approved, confidence, reasoning, risk_assessment
        """
        if signal == 0:
            return {
                "approved": False,
                "confidence": 0.0,
                "reasoning": "No signal to analyze",
                "risk_assessment": "N/A"
            }
        
        # Gather market context
        context = self._build_market_context(df, current_price, atr)
        
        # Add ML insight and system score to context
        context['ml_insight'] = ml_insight or {'ml_available': False}
        context['system_score'] = system_score or {'combined': 50, 'recommendation': 'NEUTRAL'}
        context['market_scanner'] = market_scanner_context or {}
        context['tech_score'] = tech_score or {'score': 50, 'quality': 'UNKNOWN', 'factors': []}
        
        # === PHASE 1: MATHEMATICAL PRE-CHECK ===
        # Calculate objective math score before AI decision
        math_check = self._comprehensive_math_check(signal, df, current_price, atr, context)
        context['math_check'] = math_check
        
        if self.use_ai:
            result = self._ai_analysis(signal, context, symbol)
            
            # === PHASE 2: MATH VALIDATION OF AI DECISION ===
            # If AI wants to block but math says it's a good trade, override
            result = self._validate_ai_decision_with_math(result, math_check, signal, symbol)
            
            # === PHASE 3: AI POWER RECOMMENDATIONS ===
            # Add AI-powered sizing and aggressiveness recommendations
            result = self._add_ai_power_recommendations(result, math_check, context, signal)
        else:
            result = self._rule_based_analysis(signal, context, symbol)
        
        # Log the analysis
        self._log_analysis(signal, symbol, result)
        
        return result
    
    def _add_ai_power_recommendations(
        self,
        result: Dict[str, Any],
        math_check: Dict[str, Any],
        context: Dict[str, Any],
        signal: int
    ) -> Dict[str, Any]:
        """
        Add AI power recommendations for position sizing and trade aggressiveness.
        This gives the AI more control over the trade execution.
        """
        confidence = result.get('confidence', 0.5)
        math_score = math_check.get('score', 50)
        approved = result.get('approved', False)
        
        if not approved:
            result['ai_power'] = {
                'size_multiplier': 0.0,
                'aggressive_mode': False,
                'conviction': 'BLOCKED'
            }
            return result
        
        # === CONFIDENCE-BASED POSITION SIZING ===
        # High confidence = larger position, low confidence = smaller position
        if confidence >= 0.90 and math_score >= 80:
            size_mult = 1.5  # 50% larger position for high conviction trades
            conviction = 'VERY_HIGH'
            aggressive = True
        elif confidence >= 0.85 and math_score >= 70:
            size_mult = 1.3  # 30% larger
            conviction = 'HIGH'
            aggressive = True
        elif confidence >= 0.80 and math_score >= 65:
            size_mult = 1.15  # 15% larger
            conviction = 'GOOD'
            aggressive = False
        elif confidence >= 0.75:
            size_mult = 1.0  # Standard size
            conviction = 'NORMAL'
            aggressive = False
        elif confidence >= 0.70:
            size_mult = 0.75  # 25% smaller for borderline trades
            conviction = 'LOW'
            aggressive = False
        else:
            size_mult = 0.5  # 50% smaller for low confidence
            conviction = 'MINIMAL'
            aggressive = False
        
        # === ADJUST FOR RECENT PERFORMANCE ===
        # Check consecutive results
        recent_wins = sum(1 for t in self.recent_trades[-5:] if t.get('result', '').lower() == 'win')
        recent_losses = 5 - recent_wins if len(self.recent_trades) >= 5 else 0
        
        # Hot streak boost
        if recent_wins >= 4:
            size_mult = min(size_mult * 1.2, 2.0)  # Cap at 2x
            logger.info(f"🔥 AI HOT STREAK: {recent_wins} wins - size boost to {size_mult:.1f}x")
        
        # Cold streak protection
        elif recent_losses >= 3:
            size_mult = max(size_mult * 0.7, 0.5)  # Floor at 0.5x
            aggressive = False
            logger.info(f"🧊 AI COLD STREAK: {recent_losses} losses - size reduced to {size_mult:.1f}x")
        
        result['ai_power'] = {
            'size_multiplier': round(size_mult, 2),
            'aggressive_mode': aggressive,
            'conviction': conviction,
            'math_score': math_score,
            'confidence': confidence,
            'recent_streak': f"{recent_wins}W/{recent_losses}L"
        }
        
        signal_type = "LONG" if signal == 1 else "SHORT"
        logger.info(f"🤖 AI POWER: {signal_type} | Conviction={conviction} | Size={size_mult:.1f}x | Aggressive={aggressive}")
        
        return result
    
    def ai_strategic_analysis(
        self,
        balance: float,
        recent_trades: List[Dict],
        current_positions: List[Dict],
        market_conditions: Dict[str, Any],
        adaptive_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        AI Strategic Market Analysis - Makes high-level trading decisions.
        
        This gives the AI MORE POWER to:
        1. Recommend parameter adjustments based on market regime
        2. Suggest position sizing changes
        3. Decide aggressive vs conservative mode
        4. Recommend symbol switches
        5. Set overall trading bias (long/short/neutral)
        
        Returns strategic recommendations that the bot can act on.
        """
        if not self.use_ai:
            return {'available': False, 'reason': 'AI not enabled'}
        
        try:
            # Build comprehensive context
            recent_pnl = sum(t.get('pnl', 0) for t in recent_trades[-10:]) if recent_trades else 0
            win_count = sum(1 for t in recent_trades[-10:] if t.get('pnl', 0) > 0) if recent_trades else 0
            loss_count = len(recent_trades[-10:]) - win_count if recent_trades else 0
            win_rate = win_count / max(len(recent_trades[-10:]), 1) * 100
            
            # Format current params
            params_str = "\n".join([
                f"- {name}: {info.get('current', 'N/A')} (range: {info.get('min', 'N/A')}-{info.get('max', 'N/A')})"
                for name, info in adaptive_params.items()
            ])
            
            # Format positions
            positions_str = "No open positions"
            if current_positions:
                positions_str = "\n".join([
                    f"- {p.get('symbol', 'N/A')}: {p.get('side', 'N/A')} @ ${p.get('entry', 0):.4f}, PnL: {p.get('pnl_pct', 0):.2f}%"
                    for p in current_positions
                ])
            
            prompt = f"""You are Julaba's Strategic AI Brain with FULL DECISION-MAKING POWER.

═══════════════════════════════════════════════════════════════════════
ACCOUNT STATUS
═══════════════════════════════════════════════════════════════════════
Balance: ${balance:.2f}
Recent PnL (10 trades): ${recent_pnl:.2f}
Win Rate: {win_rate:.1f}% ({win_count}W / {loss_count}L)

═══════════════════════════════════════════════════════════════════════
CURRENT POSITIONS
═══════════════════════════════════════════════════════════════════════
{positions_str}

═══════════════════════════════════════════════════════════════════════
MARKET CONDITIONS
═══════════════════════════════════════════════════════════════════════
BTC Trend: {market_conditions.get('btc_trend', 'UNKNOWN')}
Market Volatility: {market_conditions.get('volatility', 'UNKNOWN')}
Overall Sentiment: {market_conditions.get('sentiment', 'NEUTRAL')}

═══════════════════════════════════════════════════════════════════════
CURRENT PARAMETERS (YOU CAN ADJUST)
═══════════════════════════════════════════════════════════════════════
{params_str}

═══════════════════════════════════════════════════════════════════════
YOUR STRATEGIC DECISIONS
═══════════════════════════════════════════════════════════════════════
Make strategic decisions to optimize trading performance.

Respond ONLY with JSON:
{{
    "trading_mode": "aggressive" or "normal" or "conservative" or "defensive",
    "bias": "long" or "short" or "neutral",
    "risk_adjustment": -0.02 to +0.02 (change to risk_pct),
    "max_positions": 1 to 4,
    "reasoning": "brief strategic rationale (max 100 chars)",
    "urgent_action": null or {{"action": "close_all" or "reduce_size", "reason": "why"}},
    "param_changes": [
        {{"param": "param_name", "new_value": X, "reason": "why"}}
    ]
}}"""

            result_text = self._generate_content(prompt)
            if not result_text:
                return {'available': False, 'reason': 'AI returned empty response'}
            
            result_text = result_text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            strategy = json.loads(result_text.strip())
            
            logger.info(f"🧠 AI STRATEGIC ANALYSIS:")
            logger.info(f"   Mode: {strategy.get('trading_mode', 'N/A')}")
            logger.info(f"   Bias: {strategy.get('bias', 'N/A')}")
            logger.info(f"   Reasoning: {strategy.get('reasoning', 'N/A')}")
            
            if strategy.get('urgent_action'):
                logger.warning(f"⚠️ AI URGENT ACTION: {strategy['urgent_action']}")
            
            return {
                'available': True,
                'strategy': strategy,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"AI strategic analysis failed: {e}")
            return {'available': False, 'reason': str(e)}
    
    def unified_math_ai_scan(
        self,
        pairs_data: List[Dict[str, Any]],
        current_positions: List[Dict[str, Any]],
        balance: float,
        max_positions: int = 2
    ) -> Dict[str, Any]:
        """
        UNIFIED MATH + AI SCANNER - 3-STAGE PIPELINE
        
        ═══════════════════════════════════════════════════════════════
        STAGE 1: PROACTIVE PRE-FILTER (Quick elimination)
        ═══════════════════════════════════════════════════════════════
        - Fast math check to eliminate obviously bad pairs
        - Check momentum, trend exhaustion, basic thresholds
        - Reduces candidates from 50 → ~5-10
        
        ═══════════════════════════════════════════════════════════════
        STAGE 2: UNIFIED MATH ANALYSIS (Deep analysis)
        ═══════════════════════════════════════════════════════════════
        - Comprehensive PhD math scoring on remaining pairs
        - Both LONG and SHORT directions analyzed
        - Statistical edge, GARCH, Hurst, trend analysis
        - Ranks pairs by math score
        
        ═══════════════════════════════════════════════════════════════
        STAGE 3: AI FINAL DECISION (Strategic validation)
        ═══════════════════════════════════════════════════════════════
        - AI reviews top 3 math candidates
        - Can APPROVE, REVERSE direction, or REJECT
        - Final combined score (Math + AI weighted)
        - Returns BEST opportunity or None
        
        Args:
            pairs_data: List of {symbol, df, price, atr, volume_ratio} for each pair
            current_positions: List of open positions
            balance: Current account balance
            max_positions: Maximum allowed concurrent positions
            
        Returns:
            {
                'has_opportunity': bool,
                'best_pair': {symbol, signal, math_score, ai_score, combined_score, ...},
                'ranked_pairs': [...],  # All pairs ranked by combined score
                'ai_bias': 'long' | 'short' | 'neutral',
                'market_regime': str,
                'recommended_action': str,
                'pipeline_stats': {stage1_input, stage1_output, stage2_output, stage3_output}
            }
        """
        try:
            pipeline_stats = {
                'stage1_input': len(pairs_data),
                'stage1_output': 0,
                'stage2_output': 0,
                'stage3_output': 0
            }
            
            logger.info(f"═══════════════════════════════════════════════════════════════")
            logger.info(f"🔬 UNIFIED SCAN PIPELINE: {len(pairs_data)} pairs starting...")
            logger.info(f"═══════════════════════════════════════════════════════════════")
            
            # Check if we can open more positions
            open_count = len([p for p in current_positions if p])
            if open_count >= max_positions:
                return {
                    'has_opportunity': False,
                    'reason': f'Max positions reached ({open_count}/{max_positions})',
                    'ranked_pairs': [],
                    'pipeline_stats': pipeline_stats
                }
            
            # Symbols we already have positions on
            open_symbols = set()
            for pos in current_positions:
                if pos:
                    # Handle both dict and Position object
                    if hasattr(pos, 'symbol'):
                        sym = pos.symbol  # Position object
                    elif isinstance(pos, dict):
                        sym = pos.get('symbol', '')
                    else:
                        sym = str(pos)
                    open_symbols.add(sym.replace('/', '').replace(':USDT', '').upper())
            
            # ═══════════════════════════════════════════════════════════════
            # STAGE 1: PROACTIVE PRE-FILTER (Quick elimination)
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"📋 STAGE 1: Proactive Pre-filter ({len(pairs_data)} pairs)")
            
            stage1_passed = []
            stage1_rejected = []
            
            for pair in pairs_data:
                symbol = pair.get('symbol', '')
                symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                
                # Skip pairs we already have positions on
                if symbol_key in open_symbols:
                    stage1_rejected.append((symbol, "Already in position"))
                    continue
                
                # ═══════════════════════════════════════════════════════════════
                # PER-SYMBOL COOLDOWN: Skip symbols that recently had losses
                # Prevents repeatedly trading the same losing symbol
                # ═══════════════════════════════════════════════════════════════
                on_cooldown, cooldown_reason, mins_remaining = self._is_symbol_on_cooldown(symbol)
                if on_cooldown:
                    stage1_rejected.append((symbol, f"COOLDOWN: {cooldown_reason} ({mins_remaining}min left)"))
                    logger.info(f"   🚫 {symbol}: On cooldown ({cooldown_reason}, {mins_remaining}min remaining)")
                    continue
                
                df = pair.get('df')
                price = pair.get('price', 0)
                atr = pair.get('atr', 0)
                
                if df is None or len(df) < 20:
                    stage1_rejected.append((symbol, "Insufficient data"))
                    continue
                if price <= 0 or atr <= 0:
                    stage1_rejected.append((symbol, "Invalid price/ATR"))
                    continue
                
                # Quick math scores (use dashboard pre-calculated or calculate fresh)
                dashboard_math_long = pair.get('dashboard_math_long', 0)
                dashboard_math_short = pair.get('dashboard_math_short', 0)
                
                # FIRST: Build context to get momentum info
                context = self._build_market_context(df, price, atr)
                momentum_5 = context.get('roc_5', 0)
                momentum_10 = context.get('roc_10', 0)
                rsi = context.get('rsi', 50)
                adx = context.get('adx', 25)
                
                # ═══════════════════════════════════════════════════════════════
                # MOMENTUM-FIRST SCORING: Simple and effective
                # Forget PhD math when momentum is clear - just follow the trend!
                # ═══════════════════════════════════════════════════════════════
                
                # Calculate simple momentum-based scores
                momentum_long_score = 50  # Base score
                momentum_short_score = 50
                
                # LONG benefits from bullish momentum
                if momentum_5 > 0.3:
                    momentum_long_score += min(30, momentum_5 * 20)  # Up to +30 for strong up
                if momentum_10 > 0.2:
                    momentum_long_score += min(15, momentum_10 * 10)
                if momentum_5 < -0.3:
                    momentum_long_score -= min(30, abs(momentum_5) * 20)  # Penalize counter-trend
                    
                # SHORT benefits from bearish momentum
                if momentum_5 < -0.3:
                    momentum_short_score += min(30, abs(momentum_5) * 20)
                if momentum_10 < -0.2:
                    momentum_short_score += min(15, abs(momentum_10) * 10)
                if momentum_5 > 0.3:
                    momentum_short_score -= min(30, momentum_5 * 20)
                    
                # RSI extremes boost reversal trades
                if rsi < 30:  # Oversold - LONG gets boost
                    momentum_long_score += 15
                elif rsi < 40:
                    momentum_long_score += 8
                if rsi > 70:  # Overbought - SHORT gets boost
                    momentum_short_score += 15
                elif rsi > 60:
                    momentum_short_score += 8
                    
                # ADX confirms trend strength
                if adx > 25:  # Trending market
                    if momentum_5 > 0:
                        momentum_long_score += 10
                    elif momentum_5 < 0:
                        momentum_short_score += 10
                
                # Use momentum scores as primary, PhD math as secondary
                if dashboard_math_long > 0 or dashboard_math_short > 0:
                    # Blend: 60% momentum, 40% PhD math
                    long_score = (momentum_long_score * 0.6) + (dashboard_math_long * 0.4)
                    short_score = (momentum_short_score * 0.6) + (dashboard_math_short * 0.4)
                else:
                    # If no dashboard scores, use momentum-only
                    long_score = momentum_long_score
                    short_score = momentum_short_score
                
                logger.debug(f"   {symbol}: Momentum scores L={momentum_long_score:.0f}/S={momentum_short_score:.0f}, Final L={long_score:.0f}/S={short_score:.0f}")
                
                # PRE-FILTER 1: At least one direction must be viable (score >= 40)
                STAGE1_MIN_SCORE = 40
                best_score = max(long_score, short_score)
                if best_score < STAGE1_MIN_SCORE:
                    stage1_rejected.append((symbol, f"Math too low ({best_score:.0f}<{STAGE1_MIN_SCORE})"))
                    continue
                
                # PRE-FILTER 2: Quick momentum check
                context = self._build_market_context(df, price, atr)
                momentum_5 = context.get('roc_5', 0)  # Use roc_5 (rate of change over 5 bars)
                momentum_10 = context.get('roc_10', 0)  # Use roc_10 (rate of change over 10 bars)
                rsi = context.get('rsi', 50)
                
                # Block LONG into strong bearish momentum (LOWERED thresholds!)
                # Old: 1.0 / 0.5 - too lenient, allowed bad entries
                # New: 0.6 / 0.4 - catch momentum conflicts earlier
                if long_score > short_score and momentum_5 < -0.6 and momentum_10 < -0.4 and rsi > 35:
                    stage1_rejected.append((symbol, f"Bearish momentum (ROC5={momentum_5:.1f}%) blocks LONG"))
                    continue
                # Block SHORT into strong bullish momentum (LOWERED thresholds!)
                if short_score > long_score and momentum_5 > 0.6 and momentum_10 > 0.4 and rsi < 65:
                    stage1_rejected.append((symbol, f"Bullish momentum (ROC5={momentum_5:.1f}%) blocks SHORT"))
                    continue
                
                # PRE-FILTER 3: RSI trend exhaustion check
                if short_score > long_score and rsi <= 30:
                    stage1_rejected.append((symbol, f"RSI oversold ({rsi:.0f}) blocks SHORT"))
                    continue
                if long_score > short_score and rsi >= 70:
                    stage1_rejected.append((symbol, f"RSI overbought ({rsi:.0f}) blocks LONG"))
                    continue
                
                # Passed Stage 1 - keep for Stage 2
                stage1_passed.append({
                    'pair': pair,
                    'long_score': long_score,
                    'short_score': short_score,
                    'context': context
                })
            
            pipeline_stats['stage1_output'] = len(stage1_passed)
            logger.info(f"📋 STAGE 1 RESULT: {len(stage1_passed)}/{len(pairs_data)} passed pre-filter")
            if stage1_rejected:
                reject_summary = {}
                for sym, reason in stage1_rejected[:10]:
                    if reason not in reject_summary:
                        reject_summary[reason] = 0
                    reject_summary[reason] += 1
                for reason, count in reject_summary.items():
                    logger.debug(f"   Rejected ({count}): {reason}")
            
            if not stage1_passed:
                return {
                    'has_opportunity': False,
                    'reason': 'STAGE 1: No pairs passed proactive pre-filter',
                    'ranked_pairs': [],
                    'pipeline_stats': pipeline_stats
                }
            
            # ═══════════════════════════════════════════════════════════════
            # STAGE 2: UNIFIED MATH ANALYSIS (Deep analysis)
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"📊 STAGE 2: Deep Math Analysis ({len(stage1_passed)} pairs)")
            
            math_ranked = []
            
            for item in stage1_passed:
                pair = item['pair']
                symbol = pair.get('symbol', '')
                df = pair.get('df')
                price = pair.get('price', 0)
                atr = pair.get('atr', 0)
                context = item['context']
                long_score = item['long_score']
                short_score = item['short_score']
                
                # Stage 2: Determine best direction based on MOMENTUM ALIGNMENT
                STAGE2_MIN_SCORE = 45  # Minimum score to proceed
                
                # Get momentum from context
                momentum_5 = context.get('roc_5', 0)
                momentum_10 = context.get('roc_10', 0)
                rsi = context.get('rsi', 50)
                
                # ═══════════════════════════════════════════════════════════════
                # SIMPLE RULE: GO WITH MOMENTUM!
                # If momentum is bullish → LONG, if bearish → SHORT
                # Only reject if the aligned direction score is too low
                # ═══════════════════════════════════════════════════════════════
                
                bullish_momentum = momentum_5 > 0.2 or (momentum_5 > 0 and momentum_10 > 0.1)
                bearish_momentum = momentum_5 < -0.2 or (momentum_5 < 0 and momentum_10 < -0.1)
                
                # Pick direction based on momentum, not complex math
                if bullish_momentum:
                    best_signal = 1  # LONG
                    best_math_score = long_score
                    if long_score < STAGE2_MIN_SCORE:
                        logger.info(f"   ⚠️ {symbol}: Bullish momentum but LONG score too low ({long_score:.0f}<{STAGE2_MIN_SCORE})")
                        continue
                elif bearish_momentum:
                    best_signal = -1  # SHORT
                    best_math_score = short_score
                    if short_score < STAGE2_MIN_SCORE:
                        logger.info(f"   ⚠️ {symbol}: Bearish momentum but SHORT score too low ({short_score:.0f}<{STAGE2_MIN_SCORE})")
                        continue
                else:
                    # Neutral momentum - use higher score if meets threshold
                    if long_score > short_score and long_score >= STAGE2_MIN_SCORE:
                        best_signal = 1
                        best_math_score = long_score
                    elif short_score > long_score and short_score >= STAGE2_MIN_SCORE:
                        best_signal = -1
                        best_math_score = short_score
                    else:
                        logger.debug(f"   {symbol}: Neutral momentum and neither direction >= {STAGE2_MIN_SCORE}")
                        continue
                
                # Do comprehensive check for the selected direction
                best_check = self._comprehensive_math_check(best_signal, df, price, atr, context)
                
                # Log momentum-aligned decision
                dir_str = "LONG" if best_signal == 1 else "SHORT"
                mom_str = f"ROC5={momentum_5:+.2f}%"
                logger.info(f"   ✅ {symbol}: {dir_str} follows momentum | Score={best_math_score:.0f} | {mom_str}")
                
                # ═══════════════════════════════════════════════════════════════
                # RESISTANCE/SUPPORT LEVEL CHECK
                # Don't go LONG at 24h resistance, don't go SHORT at 24h support
                # ═══════════════════════════════════════════════════════════════
                levels = self._detect_resistance_support_levels(df, price)
                
                # Debug log for resistance/support check
                r_touch = levels.get('resistance_touches', 0)
                s_touch = levels.get('support_touches', 0)
                r_dist = levels.get('distance_to_resistance_pct', 99)
                s_dist = levels.get('distance_to_support_pct', 99)
                pos_range = levels.get('position_in_range_pct', 50)
                zone_pct = levels.get('zone_width_pct', 2.0)
                in_r_zone = levels.get('in_resistance_zone', False)
                in_s_zone = levels.get('in_support_zone', False)
                
                # Log zone info with 4 price levels
                r_upper = levels.get('resistance_upper', 0)
                r_lower = levels.get('resistance_lower', 0)
                s_upper = levels.get('support_upper', 0)
                s_lower = levels.get('support_lower', 0)
                
                zone_status = ""
                if in_r_zone:
                    zone_status = " 🔴IN_R_ZONE"
                elif in_s_zone:
                    zone_status = " 🔵IN_S_ZONE"
                elif pos_range >= 88:
                    zone_status = " ⚠️NEAR_RESISTANCE"
                elif pos_range <= 12:
                    zone_status = " ⚠️NEAR_SUPPORT"
                
                logger.info(f"   📍 {symbol}: Range={pos_range:.0f}% | Zone={zone_pct:.1f}% | R[${r_lower:.4f}-${r_upper:.4f}] S[${s_lower:.4f}-${s_upper:.4f}]{zone_status}")
                
                # ═══════════════════════════════════════════════════════════════
                # ZONE BLOCKING: Only block if BOTH in zone AND has 3+ bounces
                # This prevents blocking valid breakout trades
                # ═══════════════════════════════════════════════════════════════
                
                # Block LONG only if in resistance zone AND 3+ bounces confirmed rejection
                if best_signal == 1 and levels.get('in_resistance_zone') and r_touch >= 3:
                    logger.warning(f"   🚫🚫 {symbol}: LONG BLOCKED - IN RESISTANCE ZONE with {r_touch} bounces [${r_lower:.4f} - ${r_upper:.4f}]")
                    continue
                
                # Block SHORT only if in support zone AND 3+ bounces confirmed support
                if best_signal == -1 and levels.get('in_support_zone') and s_touch >= 3:
                    logger.warning(f"   🚫🚫 {symbol}: SHORT BLOCKED - IN SUPPORT ZONE with {s_touch} bounces [${s_lower:.4f} - ${s_upper:.4f}]")
                    continue
                
                # NEW: Block LONG if price is in top 30% of range (near resistance even if not in zone)
                # Changed from 80% to 70% for symmetry with SHORT blocking at 30%
                if best_signal == 1 and pos_range >= 70:
                    logger.warning(f"   🚫🚫 {symbol}: LONG BLOCKED - TOO CLOSE TO RESISTANCE (Range={pos_range:.0f}% >= 70%)")
                    continue
                
                # NEW: Block SHORT if price is in bottom 30% of range (near support even if not in zone)
                # Changed from 20% to 30% because ZORA short at 24% bounced and lost money
                if best_signal == -1 and pos_range <= 30:
                    logger.warning(f"   🚫🚫 {symbol}: SHORT BLOCKED - TOO CLOSE TO SUPPORT (Range={pos_range:.0f}% <= 30%)")
                    continue
                
                # ═══════════════════════════════════════════════════════════════
                # REMOVED: Legacy blocking based on just being near a level
                # The zone + bounce requirement above is sufficient
                # ═══════════════════════════════════════════════════════════════
                
                # REMOVED: Old "at_resistance" / "at_support" blocks - too strict
                # The new zone + 3 bounces requirement above is the ONLY zone check now
                
                # ═══════════════════════════════════════════════════════════════
                # EXTREME SENTIMENT FILTER: When market is in extreme fear/greed,
                # require stronger confirmation before entry
                # ═══════════════════════════════════════════════════════════════
                news_ctx = self._get_news_context()
                fear_greed = news_ctx.get('fear_greed_index', 50)
                
                # In EXTREME FEAR (<20), be cautious about LONGs - market might keep falling
                # Require price to be in lower part of range (near support) for better entries
                if fear_greed < 20 and best_signal == 1 and pos_range >= 50:
                    logger.warning(f"   🚫 {symbol}: LONG BLOCKED - EXTREME FEAR ({fear_greed}) + Price in upper range ({pos_range:.0f}%). Wait for price near support.")
                    continue
                
                # In EXTREME GREED (>80), be cautious about SHORTs - market might keep rising
                # Require price to be in upper part of range (near resistance) for better entries  
                if fear_greed > 80 and best_signal == -1 and pos_range <= 50:
                    logger.warning(f"   🚫 {symbol}: SHORT BLOCKED - EXTREME GREED ({fear_greed}) + Price in lower range ({pos_range:.0f}%). Wait for price near resistance.")
                    continue
                
                # Add level warning to context for AI to consider (info only, not blocking)
                level_warning = levels.get('warning')
                
                # Get reasons from math check (for AI context, NOT filtering)
                reasons_for = best_check.get('reasons_for', [])
                reasons_against = best_check.get('reasons_against', [])
                
                # ═══════════════════════════════════════════════════════════════
                # MOMENTUM DIRECTION MUST MATCH TRADE DIRECTION!
                # This is the MOST IMPORTANT filter - we don't trade against momentum
                # ═══════════════════════════════════════════════════════════════
                
                # Only one remaining filter: Is momentum STRONG enough?
                MIN_MOMENTUM_STRENGTH = 0.75  # Need at least 0.75% ROC5 to proceed (increased from 0.3%)
                momentum_strength = abs(momentum_5)
                if momentum_strength < MIN_MOMENTUM_STRENGTH:
                    logger.info(f"   ❌ {symbol}: WEAK momentum ({momentum_strength:.2f}% < {MIN_MOMENTUM_STRENGTH}%)")
                    continue
                
                # 🚨 CRITICAL: Momentum direction MUST match trade direction!
                # LONG requires positive momentum (price rising)
                # SHORT requires negative momentum (price falling)
                if best_signal == 1 and momentum_5 < 0:
                    logger.warning(f"   🚫 {symbol}: LONG BLOCKED - Momentum is NEGATIVE ({momentum_5:+.2f}%). Price is falling, not rising!")
                    continue
                    
                if best_signal == -1 and momentum_5 > 0:
                    logger.warning(f"   🚫 {symbol}: SHORT BLOCKED - Momentum is POSITIVE ({momentum_5:+.2f}%). Price is rising, not falling!")
                    continue
                
                logger.info(f"📊 STAGE 2 PASS: {symbol} {'LONG' if best_signal == 1 else 'SHORT'} | Score={best_math_score:.0f} | ROC5={momentum_5:+.2f}%")
                
                # Get PhD math score from comprehensive check for better sorting
                phd_math_score = best_check.get('score', best_math_score)
                
                math_ranked.append({
                    'symbol': symbol,
                    'signal': best_signal,
                    'direction': 'LONG' if best_signal == 1 else 'SHORT',
                    'math_score': best_math_score,  # Stage 2 score (momentum weighted)
                    'phd_score': phd_math_score,    # PhD comprehensive score (for sorting)
                    'math_check': best_check,
                    'price': price,
                    'atr': atr,
                    'df': df,
                    'reasons_for': reasons_for,
                    'reasons_against': reasons_against,
                    'level_info': levels,  # Include resistance/support info for AI
                    'level_warning': level_warning  # Include any warning
                })
            
            pipeline_stats['stage2_output'] = len(math_ranked)
            logger.info(f"📊 STAGE 2 RESULT: {len(math_ranked)}/{len(stage1_passed)} passed deep analysis")
            
            if not math_ranked:
                return {
                    'has_opportunity': False,
                    'reason': 'STAGE 2: No pairs passed deep math analysis (need score >= 45)',
                    'ranked_pairs': [],
                    'pipeline_stats': pipeline_stats
                }
            
            # Sort by PhD math score (comprehensive analysis) for Stage 3
            # This prioritizes trades with better mathematical backing over pure momentum
            math_ranked.sort(key=lambda x: x.get('phd_score', x['math_score']), reverse=True)
            pipeline_stats['stage2_passed'] = len(math_ranked)
            
            # ═══════════════════════════════════════════════════════════════
            # STAGE 3: AI FINAL DECISION (Strategic validation)
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"🤖 STAGE 3: AI Final Decision (top {min(3, len(math_ranked))} candidates)")
            logger.info(f"   Candidates: {[(c['symbol'], c['direction'], c['math_score']) for c in math_ranked[:3]]}")
            
            # Only send top 3 to AI to save API calls
            top_candidates = math_ranked[:3]
            ai_approved_candidates = []
            best_rejected = None
            
            if self.use_ai:
                for candidate in top_candidates:
                    try:
                        # === AI FINAL ENTRY DECISION ===
                        # AI receives BOTH direction math scores and makes the FINAL call
                        # AI can: APPROVE the math direction, REVERSE it, or REJECT completely
                        
                        context = self._build_market_context(candidate['df'], candidate['price'], candidate['atr'])
                        
                        # Get BOTH direction scores for AI to see
                        long_check = self._comprehensive_math_check(1, candidate['df'], candidate['price'], candidate['atr'], context)
                        short_check = self._comprehensive_math_check(-1, candidate['df'], candidate['price'], candidate['atr'], context)
                        
                        long_score = long_check.get('score', 0)
                        short_score = short_check.get('score', 0)
                        math_direction = candidate['direction']
                        math_score = candidate['math_score']
                        
                        # Get level info for AI to consider
                        level_info = candidate.get('level_info', {})
                        
                        # Build advanced math section for the winning direction
                        # IMPORTANT: Use Stage 2 momentum-blended score for consistency
                        # PhD metrics are shown as details, but the overall score must match
                        # what was used to rank candidates (prevents AI seeing conflicting scores)
                        winning_check = long_check if math_direction == 'LONG' else short_check
                        winning_check_for_display = winning_check.copy()
                        winning_check_for_display['score'] = math_score  # Use Stage 2 score, not PhD score
                        winning_check_for_display['approved'] = math_score >= 45  # Match Stage 2 threshold
                        advanced_math_section = self._build_math_analysis_section(winning_check_for_display)
                        
                        # Log advanced math integration for Stage 3
                        detailed = winning_check.get('detailed_analysis', {})
                        if detailed:
                            kalman_val = detailed.get('kalman_momentum')
                            poc_val = detailed.get('poc_price')
                            hurst_val = detailed.get('hurst_exponent')
                            kalman_str = f"{kalman_val:.2f}" if isinstance(kalman_val, (int, float)) else "N/A"
                            poc_str = f"${poc_val:,.0f}" if isinstance(poc_val, (int, float)) else "N/A"
                            hurst_str = f"{hurst_val:.2f}" if isinstance(hurst_val, (int, float)) else "N/A"
                            logger.info(f"🧮 STAGE3 MATH: {candidate['symbol']} | Kalman={kalman_str} | POC={poc_str} | Hurst={hurst_str}")
                        
                        # Call enhanced AI decision that can choose direction
                        ai_result = self._ai_final_entry_decision(
                            symbol=candidate['symbol'],
                            math_direction=math_direction,
                            math_score=math_score,
                            long_score=long_score,
                            short_score=short_score,
                            long_reasons_for=long_check.get('reasons_for', []),
                            long_reasons_against=long_check.get('reasons_against', []),
                            short_reasons_for=short_check.get('reasons_for', []),
                            short_reasons_against=short_check.get('reasons_against', []),
                            context=context,
                            level_info=level_info,  # Pass resistance/support info to AI
                            advanced_math=advanced_math_section  # NEW: Pass advanced math to AI
                        )
                        
                        # AI has FULL POWER to decide
                        ai_decision = ai_result.get('decision', 'REJECT')  # LONG, SHORT, or REJECT
                        ai_confidence = ai_result.get('confidence', 0.5)
                        ai_reasoning = ai_result.get('reasoning', '')
                        
                        if ai_decision == 'REJECT':
                            # AI says NO TRADE
                            # TESTING OVERRIDE DISABLED - was causing AI flip-flop (approve then immediate exit)
                            candidate['ai_approved'] = False
                            candidate['ai_confidence'] = ai_confidence
                            candidate['ai_score'] = 20 + (1 - ai_confidence) * 30
                            candidate['ai_reasoning'] = ai_reasoning
                            logger.info(f"📊 {candidate['symbol']}: AI ❌ REJECT (conf={ai_confidence:.0%}) - {ai_reasoning[:60]}")
                        elif ai_confidence < 0.50:
                            # AI says yes but not confident enough - TREAT AS REJECT
                            # Restored from 45% to 50% (normal is 55%)
                            candidate['ai_approved'] = False
                            candidate['ai_confidence'] = ai_confidence
                            candidate['ai_score'] = ai_confidence * 50  # Reduced score for low confidence
                            candidate['ai_reasoning'] = f"Low confidence ({ai_confidence:.0%}) - need 50%+: {ai_reasoning}"
                            logger.warning(f"📊 {candidate['symbol']}: AI {ai_decision} but LOW CONF ({ai_confidence:.0%} < 50%) - treating as REJECT")
                        else:
                            # AI approved with 50%+ confidence (normal is 55%)
                            candidate['ai_approved'] = True
                            candidate['ai_confidence'] = ai_confidence
                            candidate['ai_score'] = ai_confidence * 100
                            candidate['ai_reasoning'] = ai_reasoning
                            
                            # Check if AI changed direction from math
                            if ai_decision != math_direction:
                                # AI REVERSED the direction!
                                logger.warning(f"🔄 {candidate['symbol']}: AI REVERSED direction! Math said {math_direction}, AI says {ai_decision}")
                                candidate['direction'] = ai_decision
                                candidate['signal'] = 1 if ai_decision == 'LONG' else -1
                                candidate['ai_reversed_direction'] = True  # Flag for penalty
                                # Update math score for new direction
                                if ai_decision == 'LONG':
                                    candidate['math_score'] = long_score
                                    candidate['math_check'] = long_check
                                else:
                                    candidate['math_score'] = short_score
                                    candidate['math_check'] = short_check
                            else:
                                candidate['ai_reversed_direction'] = False
                            
                            logger.info(f"📊 {candidate['symbol']}: AI ✅ {ai_decision} (conf={ai_confidence:.0%}) - {ai_reasoning[:60]}")
                        
                        # Combined score: 60% Math (PhD analysis) + 40% AI (Gemini judgment)
                        candidate['combined_score'] = (candidate['math_score'] * 0.6) + (candidate['ai_score'] * 0.4)
                        
                    except Exception as ai_err:
                        logger.warning(f"AI final entry decision failed for {candidate['symbol']}: {ai_err}")
                        # AI REQUIRED - if AI fails, mark as not approved
                        candidate['ai_approved'] = False
                        candidate['ai_confidence'] = 0.0
                        candidate['ai_score'] = 0
                        candidate['ai_reasoning'] = f"AI error: {ai_err}"
                        candidate['combined_score'] = 0  # Cannot open without AI
            else:
                # No AI available - cannot open positions without AI
                logger.warning("⚠️ AI not available - positions require AI approval")
                for candidate in top_candidates:
                    candidate['ai_approved'] = False
                    candidate['ai_confidence'] = 0.0
                    candidate['ai_score'] = 50
                    candidate['combined_score'] = candidate['math_score']
            
            # Re-sort by combined score
            top_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # === STAGE 3 FINAL DECISION ===
            # AI APPROVAL IS MANDATORY - no position opens without AI saying YES
            
            # Filter to only AI-approved candidates
            ai_approved_candidates = [c for c in top_candidates if c.get('ai_approved', False)]
            pipeline_stats['stage3_approved'] = len(ai_approved_candidates)
            
            if not ai_approved_candidates:
                # NO AI APPROVAL - NO TRADE
                best_rejected = top_candidates[0] if top_candidates else None
                if best_rejected:
                    logger.warning(f"🚫 STAGE 3 FAILED: AI did not approve any candidates")
                    logger.warning(f"   Best rejected: {best_rejected['symbol']} {best_rejected['direction']}")
                    logger.warning(f"   Math: {best_rejected['math_score']:.0f} | AI rejected: {best_rejected.get('ai_reasoning', 'unknown')[:50]}")
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                return {
                    'has_opportunity': False,
                    'reason': f"STAGE 3: AI rejected all candidates. Best was {best_rejected['symbol']} {best_rejected['direction']}" if best_rejected else 'No candidates',
                    'ranked_pairs': top_candidates,
                    'best_score': best_rejected['combined_score'] if best_rejected else 0,
                    'pipeline_stats': pipeline_stats
                }
            
            # Select best AI-approved candidate
            best = ai_approved_candidates[0]
            
            # === THRESHOLDS FOR AI-APPROVED TRADES ===
            # Balanced mode: Allow good trades while maintaining quality
            MIN_COMBINED_SCORE = 55  # Reasonable combined score
            MIN_MATH_SCORE = 45      # Standard math threshold
            MIN_MATH_SCORE_AI_HIGH_CONF = 38  # Lower for high confidence AI (75%+)
            MIN_MATH_SCORE_LONG_PREFERRED = 35  # Lower for historically winning LONG
            MIN_MATH_SCORE_REVERSAL = 50  # Higher when AI reverses math direction
            
            ai_confidence = best.get('ai_confidence', 0.5)
            ai_direction = best.get('direction', '')
            ai_reversed = best.get('ai_reversed_direction', False)
            
            # NOTE: Removed market regime filter (Fear & Greed based blocking)
            # Reason: API unreliable, data stale (daily), and the math analysis
            # already accounts for momentum. The real fix is the reversal penalty below.
            
            # Get historical direction performance  
            dir_perf = self._get_direction_performance()
            long_wr = dir_perf.get('long_wr', 50)
            
            # === DIRECTION REVERSAL PENALTY ===
            # When AI reverses math's direction, require HIGHER math score
            # This prevents AI from fighting strong trends
            if ai_reversed:
                effective_min_math = MIN_MATH_SCORE_REVERSAL
                logger.warning(f"   ⚠️ AI REVERSED direction - requiring higher math: {effective_min_math}")
            # SPECIAL RULE: If AI says LONG and our LONG WR is excellent, trust the AI!
            # The math scores low for LONG in bearish momentum, but LONG historically wins
            elif ai_direction == 'LONG' and long_wr >= 65:
                effective_min_math = MIN_MATH_SCORE_LONG_PREFERRED
                logger.info(f"   📈 LONG preferred (WR={long_wr:.0f}%) - using lower math threshold: {effective_min_math}")
            elif ai_confidence >= 0.75:
                effective_min_math = MIN_MATH_SCORE_AI_HIGH_CONF
            else:
                effective_min_math = MIN_MATH_SCORE
            
            # Check math score - but give AI some flexibility with high confidence
            if best['math_score'] < effective_min_math:
                logger.warning(f"🚫 {best['symbol']}: Math score {best['math_score']:.0f} too low (min {effective_min_math}) - AI conf={ai_confidence:.0%}")
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                return {
                    'has_opportunity': False,
                    'reason': f'STAGE 3: Math score too low ({best["math_score"]:.0f} < {effective_min_math}) - AI cannot override',
                    'ranked_pairs': top_candidates,
                    'best_score': best['combined_score'],
                    'pipeline_stats': pipeline_stats
                }
            
            # When LONG is preferred, lower combined score requirement
            # (since math score will be low due to counter-trend design)
            MIN_COMBINED_SCORE_LONG_PREFERRED = 50  # Lower threshold when trusting AI for LONG
            effective_min_combined = MIN_COMBINED_SCORE_LONG_PREFERRED if (ai_direction == 'LONG' and long_wr >= 65) else MIN_COMBINED_SCORE
            
            # Require minimum combined score even with AI approval
            if best['combined_score'] >= effective_min_combined:
                # We have a winner! AI approved AND score is good
                logger.info(f"✅ ALL 3 STAGES PASSED! Approving {best['symbol']} {best['direction']}")
                logger.info(f"   Math: {best['math_score']:.0f} | AI: {best['ai_score']:.0f} | Combined: {best['combined_score']:.0f}")
                logger.info(f"   AI Reasoning: {best.get('ai_reasoning', 'N/A')[:80]}")
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                
                return {
                    'has_opportunity': True,
                    'best_pair': best,
                    'ranked_pairs': top_candidates,
                    'ai_bias': best['direction'].lower(),
                    'market_regime': best['math_check'].get('detailed_analysis', {}).get('regime', 'UNKNOWN'),
                    'recommended_action': f"Open {best['direction']} on {best['symbol']}",
                    'pipeline_stats': pipeline_stats
                }
            else:
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                return {
                    'has_opportunity': False,
                    'reason': f'STAGE 3: AI approved {best["symbol"]} but combined score too low ({best["combined_score"]:.0f} < {effective_min_combined})',
                    'ranked_pairs': top_candidates,
                    'best_score': best['combined_score'],
                    'pipeline_stats': pipeline_stats
                }
                
        except Exception as e:
            logger.error(f"Unified Math+AI scan error: {e}")
            return {'has_opportunity': False, 'reason': str(e), 'ranked_pairs': []}
    
    def unified_position_decision(
        self,
        position: Dict[str, Any],
        df: pd.DataFrame,
        current_price: float,
        atr: float
    ) -> Dict[str, Any]:
        """
        UNIFIED MATH + AI POSITION DECISION
        
        Makes hold/close/adjust decisions using BOTH Math and AI together.
        
        SMART EXIT RULES:
        1. IN PROFIT + High reversal risk → Exit to protect gains
        2. IN DEEP LOSS (>-1.5% or >-$15) → Force exit to prevent deeper loss
           (Server-side SL should have triggered - if we're here, exit manually)
        3. SMALL LOSS → Can exit if strong reversal confirmed
        4. AI + Math must BOTH agree for exit (except deep loss protection which is automatic)
        
        Args:
            position: {symbol, side, entry_price, size, tp1_hit, tp2_hit, ...}
            df: OHLCV DataFrame
            current_price: Current market price
            atr: Current ATR
            
        Returns:
            {
                'action': 'hold' | 'close' | 'tighten_sl',
                'confidence': 0.0-1.0,
                'math_score': float,
                'ai_validated': bool,
                'reasoning': str,
                'sl_adjustment': float or None
            }
        """
        try:
            symbol = position.get('symbol', 'UNKNOWN')
            side = position.get('side', 'LONG').upper()
            entry_price = position.get('entry_price', current_price)
            size = position.get('size', 0)
            tp1_hit = position.get('tp1_hit', False)
            tp2_hit = position.get('tp2_hit', False)
            
            # Calculate PnL for logging
            if side == 'LONG':
                quick_pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                quick_pnl = ((entry_price - current_price) / entry_price) * 100
            
            logger.info(f"📊 POSITION CHECK: {symbol} {side} | Entry=${entry_price:.4f} | Now=${current_price:.4f} | PnL={quick_pnl:+.2f}%")
            
            # === S/R DISTANCE CHECK ===
            # Show how close we are to support and resistance levels
            try:
                sr_levels = self._detect_resistance_support_levels(df, current_price)
                dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
                dist_to_support = sr_levels.get('distance_to_support_pct', 99)
                range_position = sr_levels.get('position_in_range_pct', 50)
                in_r_zone = sr_levels.get('in_resistance_zone', False)
                in_s_zone = sr_levels.get('in_support_zone', False)
                
                # Build S/R status string
                sr_status = f"📍 Range: {range_position:.0f}% | "
                if in_r_zone:
                    sr_status += "🚫 IN RESISTANCE ZONE"
                elif in_s_zone:
                    sr_status += "🚫 IN SUPPORT ZONE"
                elif dist_to_resistance < 1.0:
                    sr_status += f"⚠️ Near R ({dist_to_resistance:.1f}% away)"
                elif dist_to_support < 1.0:
                    sr_status += f"⚠️ Near S ({dist_to_support:.1f}% away)"
                else:
                    sr_status += f"R: {dist_to_resistance:.1f}% | S: {dist_to_support:.1f}%"
                
                logger.info(f"📊 {symbol}: {sr_status}")
            except Exception as sr_err:
                logger.debug(f"S/R check error: {sr_err}")
            
            # Calculate PnL
            if side == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Estimate position value
            position_value = size * entry_price if size > 0 else 500
            pnl_usd = pnl_pct * position_value / 100
            
            # === MINIMUM HOLD TIME CHECK ===
            # Don't run reversal detection on freshly opened positions
            # Give trades at least 2 minutes to develop before considering exit
            import time
            entry_time = position.get('entry_time', 0) or position.get('open_time', 0)
            if entry_time:
                try:
                    if isinstance(entry_time, str):
                        # String datetime - parse with fromisoformat
                        entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        entry_timestamp = entry_dt.timestamp()
                    elif isinstance(entry_time, datetime):
                        # Datetime object - get timestamp directly
                        entry_timestamp = entry_time.timestamp()
                    else:
                        # Assume it's a numeric timestamp
                        entry_timestamp = float(entry_time)
                    hold_seconds = time.time() - entry_timestamp
                    logger.debug(f"📊 Hold time calculated: {hold_seconds:.0f}s (entry_type={type(entry_time).__name__})")
                except Exception as e:
                    logger.warning(f"⚠️ Could not calculate hold time: {e} (entry_time={entry_time}, type={type(entry_time)})")
                    hold_seconds = 999  # Assume old position
            else:
                hold_seconds = 999  # Assume old position
            
            MIN_HOLD_SECONDS = 120  # 2 minutes minimum before reversal detection
            is_new_position = hold_seconds < MIN_HOLD_SECONDS
            
            # === PHASE 0: SMART TINY WIN MANAGEMENT ===
            # Problem: AI was closing positions with +$0.09 profit on weak signals
            # Solution: Check if momentum STILL FAVORS us - if yes, HOLD. If reversing, exit.
            if 0 < pnl_pct < 0.6:  # Small win (0% to 0.6%)
                logger.info(f"💰 {symbol}: SMALL WIN zone ({pnl_pct:.2f}%) - checking if momentum still favors us")
                try:
                    context = self._build_market_context(df, current_price, atr)
                    roc_5 = context.get('roc_5', 0)
                    roc_10 = context.get('roc_10', 0)
                    rsi = context.get('rsi', 50)
                    
                    sr_levels = self._detect_resistance_support_levels(df, current_price)
                    in_r_zone = sr_levels.get('in_resistance_zone', False)
                    in_s_zone = sr_levels.get('in_support_zone', False)
                    dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
                    dist_to_support = sr_levels.get('distance_to_support_pct', 99)
                    
                    # Check if momentum STILL FAVORS our direction
                    momentum_still_good = False
                    reversal_confirmed = False
                    
                    if side == 'LONG':
                        # LONG: Good if momentum still up, bad if turning down
                        if roc_5 > 0.1 and roc_10 > 0:
                            momentum_still_good = True
                            logger.info(f"   ✅ Momentum still UP (roc_5={roc_5:+.2f}%, roc_10={roc_10:+.2f}%) - HOLD")
                        elif roc_5 < -0.2 and (in_r_zone or dist_to_resistance < 0.5):
                            reversal_confirmed = True
                            logger.info(f"   🔴 REVERSAL: Momentum turning DOWN at resistance (roc_5={roc_5:+.2f}%) - EXIT")
                        elif roc_5 < -0.3:
                            reversal_confirmed = True
                            logger.info(f"   🔴 REVERSAL: Strong downward momentum (roc_5={roc_5:+.2f}%) - EXIT")
                    else:
                        # SHORT: Good if momentum still down, bad if turning up
                        if roc_5 < -0.1 and roc_10 < 0:
                            momentum_still_good = True
                            logger.info(f"   ✅ Momentum still DOWN (roc_5={roc_5:+.2f}%, roc_10={roc_10:+.2f}%) - HOLD")
                        elif roc_5 > 0.2 and (in_s_zone or dist_to_support < 0.5):
                            reversal_confirmed = True
                            logger.info(f"   🔴 REVERSAL: Momentum turning UP at support (roc_5={roc_5:+.2f}%) - EXIT")
                        elif roc_5 > 0.3:
                            reversal_confirmed = True
                            logger.info(f"   🔴 REVERSAL: Strong upward momentum (roc_5={roc_5:+.2f}%) - EXIT")
                    
                    # DECISION LOGIC
                    if reversal_confirmed:
                        # Clear reversal happening - OK to exit even with small profit
                        logger.info(f"✅ {symbol}: REVERSAL CONFIRMED - Exit small win before it turns to loss")
                        # Continue to normal exit logic (don't block)
                    elif momentum_still_good:
                        # Momentum still in our favor - HOLD for bigger profit
                        logger.info(f"🔒 {symbol}: MOMENTUM STILL GOOD - Hold for bigger profit (no reversal yet)")
                        return {
                            'action': 'hold',
                            'confidence': 0.75,
                            'math_score': 0,
                            'exit_score': 0,
                            'ai_validated': False,
                            'reasoning': f"💰 SMALL WIN ({pnl_pct:.2f}%) but momentum still favors us - holding for bigger profit. No reversal detected yet.",
                            'sl_adjustment': None
                        }
                    else:
                        # Neutral momentum - let AI decide
                        logger.info(f"⚖️ {symbol}: NEUTRAL momentum - let AI decide on exit")
                        # Continue to normal exit logic
                
                except Exception as tiny_err:
                    logger.warning(f"Small win check failed: {tiny_err}")
            
            if is_new_position and pnl_pct > -0.5:  # Only skip if not in loss
                logger.info(f"📊 {symbol}: New position ({hold_seconds:.0f}s old) - skipping reversal check, letting it develop")
                return {
                    'action': 'hold',
                    'confidence': 0.7,
                    'math_score': 50,
                    'exit_score': 30,
                    'ai_validated': True,
                    'reasoning': f"New position (held {hold_seconds:.0f}s) - letting it develop",
                    'sl_adjustment': None,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd
                }
            
            # === PHASE 1: LOSS MANAGEMENT - WITH GRACE PERIOD ===
            # Only cut losses aggressively AFTER the grace period has elapsed
            # During grace period, allow normal volatility swings
            
            # INCREASED THRESHOLDS - give trades room to breathe!
            # For a $76 balance, these are reasonable risk levels
            if is_new_position:
                # GRACE PERIOD: Use much wider thresholds for new positions
                max_loss_threshold_pct = -2.5   # Only exit on major moves during grace period
                max_loss_threshold_usd = -15    # ~20% of balance max during grace
                logger.debug(f"⏳ {symbol}: Grace period - wider loss thresholds: {max_loss_threshold_pct}% / ${max_loss_threshold_usd}")
            else:
                # AFTER GRACE PERIOD: Use tighter but still reasonable thresholds
                max_loss_threshold_pct = -1.5   # 1.5% loss after grace period (was -0.6%!)
                max_loss_threshold_usd = -10    # $10 max loss after grace
            
            in_deep_loss = (pnl_pct < max_loss_threshold_pct) or (pnl_usd < max_loss_threshold_usd)
            
            if in_deep_loss:
                # BEYOND ACCEPTABLE LOSS - EXIT NOW to prevent deeper loss
                # The server-side SL should have triggered - if we're here, exit manually
                grace_status = "DURING GRACE" if is_new_position else "AFTER GRACE"
                logger.warning(f"🚨 {symbol}: LOSS EXCEEDED THRESHOLD {grace_status} (PnL: {pnl_pct:.2f}% / ${pnl_usd:.2f}) - EXITING!")
                return {
                    'action': 'close',
                    'confidence': 0.95,
                    'math_score': 0,
                    'exit_score': 100,
                    'ai_validated': True,
                    'reasoning': f"🚨 LOSS PROTECTION: {pnl_pct:.2f}% loss exceeds {max_loss_threshold_pct}% threshold ({grace_status})",
                    'sl_adjustment': None,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd
                }
            
            # === PHASE 0.5: ADVANCED POSITION RECOVERY CHECK ===
            # For small/moderate losses (-0.15% to -0.5%), check if we should hold for recovery
            # Uses MATH (momentum + S/R) + AI validation for smart recovery decisions
            in_recovery_zone = -0.5 < pnl_pct < -0.15 and pnl_usd > -5
            
            if in_recovery_zone:
                # Build context for recovery decision
                context = self._build_market_context(df, current_price, atr)
                roc_5 = context.get('roc_5', 0)
                roc_10 = context.get('roc_10', 0)
                roc_20 = context.get('roc_20', 0)  # Longer-term momentum
                rsi = context.get('rsi', 50)
                volume_ratio = context.get('volume_ratio', 1.0)
                
                # Get S/R info
                try:
                    sr_levels = self._detect_resistance_support_levels(df, current_price)
                    dist_to_support = sr_levels.get('distance_to_support_pct', 99)
                    dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
                    in_support_zone = sr_levels.get('in_support_zone', False)
                    in_resistance_zone = sr_levels.get('in_resistance_zone', False)
                except:
                    dist_to_support = 99
                    dist_to_resistance = 99
                    in_support_zone = False
                    in_resistance_zone = False
                
                # === ADVANCED MOMENTUM ANALYSIS ===
                # Check if momentum is REVERSING (turning in our favor)
                momentum_reversing = False
                momentum_strength = 0
                
                if side == 'LONG':
                    # For LONG: Need momentum turning UP (positive acceleration)
                    if roc_5 > 0 and roc_5 > roc_10:
                        momentum_reversing = True
                        momentum_strength = min(abs(roc_5), 30)  # Cap at 30 points
                    elif roc_10 > 0 and roc_10 > roc_20:
                        momentum_reversing = True
                        momentum_strength = min(abs(roc_10) * 0.5, 15)  # Less weight for slower momentum
                else:
                    # For SHORT: Need momentum turning DOWN (negative acceleration)
                    if roc_5 < 0 and roc_5 < roc_10:
                        momentum_reversing = True
                        momentum_strength = min(abs(roc_5), 30)
                    elif roc_10 < 0 and roc_10 < roc_20:
                        momentum_reversing = True
                        momentum_strength = min(abs(roc_10) * 0.5, 15)
                
                # Calculate recovery score (0-100)
                recovery_score = 0
                recovery_reasons = []
                
                # === MOMENTUM FACTORS (0-45 points) ===
                if momentum_reversing:
                    recovery_score += momentum_strength
                    recovery_reasons.append(f"🔄 Momentum REVERSING in our favor (strength: {momentum_strength:.0f})")
                elif side == 'LONG' and roc_5 > 0:
                    recovery_score += 15
                    recovery_reasons.append("📈 Short-term momentum turning up")
                elif side == 'SHORT' and roc_5 < 0:
                    recovery_score += 15
                    recovery_reasons.append("📉 Short-term momentum turning down")
                
                # === RSI FACTORS (0-20 points) ===
                if side == 'LONG':
                    if rsi < 30:
                        recovery_score += 20
                        recovery_reasons.append(f"✅ RSI EXTREME oversold ({rsi:.0f}) - strong bounce expected")
                    elif rsi < 40:
                        recovery_score += 12
                        recovery_reasons.append(f"✅ RSI oversold ({rsi:.0f}) - bounce likely")
                else:
                    if rsi > 70:
                        recovery_score += 20
                        recovery_reasons.append(f"✅ RSI EXTREME overbought ({rsi:.0f}) - strong pullback expected")
                    elif rsi > 60:
                        recovery_score += 12
                        recovery_reasons.append(f"✅ RSI overbought ({rsi:.0f}) - pullback likely")
                
                # === S/R FACTORS (0-30 points) ===
                if side == 'LONG':
                    if in_support_zone:
                        recovery_score += 30
                        recovery_reasons.append(f"💪 IN SUPPORT ZONE - very strong bounce zone")
                    elif dist_to_support < 0.5:
                        recovery_score += 25
                        recovery_reasons.append(f"💪 Near support ({dist_to_support:.1f}% away) - strong bounce zone")
                    elif dist_to_support < 1.5:
                        recovery_score += 15
                        recovery_reasons.append(f"📍 Approaching support ({dist_to_support:.1f}% away)")
                    # Penalty
                    if in_resistance_zone:
                        recovery_score -= 25
                        recovery_reasons.append("⚠️ At resistance - recovery very difficult")
                    elif dist_to_resistance < 1.0:
                        recovery_score -= 15
                        recovery_reasons.append("⚠️ Near resistance - limited upside")
                else:
                    if in_resistance_zone:
                        recovery_score += 30
                        recovery_reasons.append(f"💪 IN RESISTANCE ZONE - very strong rejection zone")
                    elif dist_to_resistance < 0.5:
                        recovery_score += 25
                        recovery_reasons.append(f"💪 Near resistance ({dist_to_resistance:.1f}% away) - strong rejection zone")
                    elif dist_to_resistance < 1.5:
                        recovery_score += 15
                        recovery_reasons.append(f"📍 Approaching resistance ({dist_to_resistance:.1f}% away)")
                    # Penalty
                    if in_support_zone:
                        recovery_score -= 25
                        recovery_reasons.append("⚠️ At support - recovery very difficult")
                    elif dist_to_support < 1.0:
                        recovery_score -= 15
                        recovery_reasons.append("⚠️ Near support - limited downside")
                
                # === VOLUME CONFIRMATION (0-10 points) ===
                if volume_ratio > 1.5:
                    recovery_score += 10
                    recovery_reasons.append(f"📊 High volume ({volume_ratio:.1f}x) - strong conviction")
                elif volume_ratio < 0.7:
                    recovery_score -= 5
                    recovery_reasons.append(f"⚠️ Low volume ({volume_ratio:.1f}x) - weak move")
                
                # === DECISION LOGIC ===
                # Math score >= 50: Strong recovery signals, ask AI for validation
                # Math score 40-49: Borderline, ask AI to decide
                # Math score < 40: Too weak, exit normally
                
                logger.info(f"🔄 {symbol}: RECOVERY CHECK (PnL: {pnl_pct:.2f}%) | Math Score: {recovery_score}/100")
                for reason in recovery_reasons[:4]:  # Show top 4 reasons
                    logger.info(f"   {reason}")
                
                if recovery_score >= 40:
                    # === AI VALIDATION FOR RECOVERY ===
                    # Ask AI to validate the recovery decision
                    ai_recovery_decision = None
                    
                    if self.use_ai and recovery_score >= 40:
                        logger.info(f"🤖 {symbol}: Consulting AI for recovery validation (math score: {recovery_score})")
                        
                        # Extract real-time tracking data from position
                        deepest_loss_pct = position.get('deepest_loss_pct', pnl_pct)
                        deepest_loss_usd = position.get('deepest_loss_usd', pnl_usd)
                        recovery_attempts = position.get('recovery_attempts', 0)
                        in_recovery_mode = position.get('in_recovery_mode', False)
                        urgency = position.get('urgency', 'normal')
                        
                        # Calculate if recovering
                        is_recovering = deepest_loss_pct < pnl_pct < 0  # Loss getting smaller
                        recovery_progress = abs(pnl_pct - deepest_loss_pct) if is_recovering else 0
                        
                        # Build recovery-specific AI prompt with real-time context
                        recovery_context = f"""You are analyzing a position that's in a SMALL LOSS and deciding if it should be HELD for RECOVERY.

Position: {side} {symbol}
Entry: ${entry_price:.4f} → Current: ${current_price:.4f}
Loss: {pnl_pct:.2f}% (${pnl_usd:.2f})

REAL-TIME TRACKING:
• Deepest Loss: {deepest_loss_pct:.2f}% (worst point)
• Recovery Status: {'✅ RECOVERING (+' + f'{recovery_progress:.2f}%)' if is_recovering else '❌ STILL DEEPENING'}
• Recovery Attempts: #{recovery_attempts} (times AI has evaluated this)
• Already in Recovery Mode: {'YES' if in_recovery_mode else 'NO'}
• Urgency Level: {urgency.upper()} {'🚨' if urgency == 'urgent' else '⚠️' if urgency == 'high' else ''}

MATH RECOVERY SCORE: {recovery_score}/100 (40+ = consider recovery)

Recovery Factors Detected:
{chr(10).join(recovery_reasons[:5])}

Market Context:
• Momentum: roc_5={roc_5:+.2f}%, roc_10={roc_10:+.2f}%, roc_20={roc_20:+.2f}%
• RSI: {rsi:.0f} ({('OVERSOLD' if rsi < 40 else 'OVERBOUGHT' if rsi > 60 else 'NEUTRAL')})
• S/R: {dist_to_resistance:.1f}% to R | {dist_to_support:.1f}% to S
• Volume: {volume_ratio:.1f}x average

QUESTION: Should we HOLD this position for recovery, or EXIT to prevent further loss?

Consider:
1. Is momentum TRULY reversing in our favor?
2. Are we at a KEY support/resistance level likely to bounce/reject?
3. Is the loss small enough that a reversal could quickly turn profitable?
4. Are there any hidden risks (trend, volume, market conditions)?

Your decision (respond EXACTLY in this format):
<DECISION>
Action: HOLD_RECOVERY or EXIT
Confidence: [0-100]
Reasoning: [Your 1-sentence analysis]
</DECISION>"""
                        
                        try:
                            ai_response = self._call_ai_api(recovery_context)
                            
                            if ai_response and '<DECISION>' in ai_response:
                                decision_text = ai_response.split('<DECISION>')[1].split('</DECISION>')[0].strip()
                                
                                # Parse AI decision
                                ai_action = None
                                ai_confidence = 0
                                ai_reasoning = ""
                                
                                for line in decision_text.split('\n'):
                                    if 'Action:' in line:
                                        ai_action = 'hold_recovery' if 'HOLD' in line.upper() else 'close'
                                    elif 'Confidence:' in line:
                                        try:
                                            ai_confidence = int(''.join(filter(str.isdigit, line)))
                                        except:
                                            ai_confidence = 50
                                    elif 'Reasoning:' in line:
                                        ai_reasoning = line.split('Reasoning:')[1].strip()
                                
                                logger.info(f"🤖 AI Recovery Decision: {ai_action.upper()} ({ai_confidence}%) - {ai_reasoning[:80]}")
                                
                                # AI validates recovery
                                if ai_action == 'hold_recovery' and ai_confidence >= 55:
                                    return {
                                        'action': 'hold_recovery',
                                        'confidence': min(0.85, ai_confidence / 100),
                                        'math_score': recovery_score,
                                        'exit_score': 0,
                                        'ai_validated': True,
                                        'reasoning': f"🔄 AI+MATH RECOVERY: {ai_reasoning[:100]}. Math: {recovery_score}/100, AI: {ai_confidence}%",
                                        'sl_adjustment': None,
                                        'pnl_pct': pnl_pct,
                                        'pnl_usd': pnl_usd,
                                        'recovery_score': recovery_score,
                                        'ai_confidence': ai_confidence
                                    }
                                else:
                                    logger.info(f"❌ AI rejected recovery: {ai_reasoning}")
                        
                        except Exception as ai_err:
                            logger.warning(f"⚠️ AI recovery validation failed: {ai_err}")
                    
                    # Fallback: If AI unavailable or didn't validate, use math score
                    if recovery_score >= 60:
                        # Very high math score - hold even without AI
                        return {
                            'action': 'hold_recovery',
                            'confidence': 0.70,
                            'math_score': recovery_score,
                            'exit_score': 0,
                            'ai_validated': False,
                            'reasoning': f"🔄 MATH RECOVERY: Strong signals (score: {recovery_score}/100). " + "; ".join(recovery_reasons[:2]),
                            'sl_adjustment': None,
                            'pnl_pct': pnl_pct,
                            'pnl_usd': pnl_usd,
                            'recovery_score': recovery_score
                        }
                else:
                    logger.info(f"❌ {symbol}: NO RECOVERY - Math score {recovery_score}/100 too low (need 40+)")
            
            # === PHASE 1B: SMART PROFIT PROTECTION ===
            # CRITICAL: This is NOT about hitting a profit number!
            # It's about detecting WHEN profit is REVERSING and capturing it!
            # 
            # LOGIC: 
            # - Track momentum using roc_5 and roc_10 (rate of change)
            # - If we're in good profit AND momentum is reversing against us = EXIT
            # - Don't exit just because profit hit $8 - exit when $8 profit starts DROPPING
            #
            # The AI should react FAST when it sees:
            # 1. Good profit ($5+ or 1%+)
            # 2. Momentum turning against the position
            # 3. Price dropping from recent highs (for LONG) or rising from lows (for SHORT)
            
            # === PHASE 1C: INTELLIGENT PROFIT REVERSAL DETECTION ===
            # DYNAMIC SYSTEM: Track peak profit and detect drawdown from peak
            # Using PERCENTAGE as primary metric (like Bybit display: $0.05 (+0.72%))
            # This is better because:
            # 1. Independent of position size
            # 2. Matches exchange display
            # 3. 0.72% peak → 0.10% current = 86% of gain lost (clear signal!)
            
            # Use existing momentum indicators (roc_5, roc_10) for faster reaction
            context = self._build_market_context(df, current_price, atr)
            roc_5 = context.get('roc_5', 0)   # Rate of change over 5 bars
            roc_10 = context.get('roc_10', 0)  # Rate of change over 10 bars
            
            # === TRACK PEAK PROFIT using PERCENTAGE (primary metric) ===
            # Initialize from position or use current PnL
            peak_profit_pct = position.get('peak_profit_pct', 0)
            peak_profit_usd = position.get('peak_profit_usd', 0)
            
            # Update peak if current profit is higher (use PCT as primary)
            if pnl_pct > peak_profit_pct and pnl_pct > 0:  # Only track positive peaks
                peak_profit_pct = pnl_pct
                peak_profit_usd = pnl_usd  # Update USD too for logging
                logger.info(f"📈 {symbol}: NEW PEAK! {pnl_pct:.2f}% (${pnl_usd:.2f})")
            
            # Calculate DRAWDOWN from peak using PERCENTAGE
            # Example: Peak was 0.72%, now 0.10% → lost 86% of the gain!
            if peak_profit_pct > 0.05:  # Only track if we had meaningful profit (>0.05%)
                drawdown_pct_of_peak = ((peak_profit_pct - pnl_pct) / peak_profit_pct * 100) if peak_profit_pct > 0 else 0
                drawdown_usd = peak_profit_usd - pnl_usd  # For logging
                # Log peak tracking status if we have a meaningful peak
                if peak_profit_pct > 0.1:
                    if pnl_pct < 0:
                        # Gone from profit to loss - special messaging
                        logger.info(f"📊 {symbol}: Peak: {peak_profit_pct:.2f}% → Now: {pnl_pct:.2f}% | ⚠️ IN LOSS (was profitable!)")
                    else:
                        logger.info(f"📊 {symbol}: Peak: {peak_profit_pct:.2f}% → Now: {pnl_pct:.2f}% | Lost {drawdown_pct_of_peak:.0f}% of gain")
            else:
                drawdown_pct_of_peak = 0
                drawdown_usd = 0
            
            # Track reversal signals for AI to consider
            profit_reversal_detected = False
            reversal_urgency = 'none'  # none, low, medium, high, critical
            reversal_reason = ''
            
            # Initialize fast reversal score (used for AI+Momentum combo)
            fast_reversal_score = 0
            fast_reversal_signals = []
            very_strong_reversal = False
            
            is_long = side == 'LONG'
            
            # === MINIMUM HOLD TIME CHECK ===
            # Don't run aggressive reversal detection on brand new positions
            # They need time to develop - normal market noise can trigger false reversals
            position_open_time = position.get('open_time')
            min_hold_seconds = 30  # 30 seconds minimum before fast reversal triggers
            position_age_seconds = 0
            
            if position_open_time:
                if isinstance(position_open_time, datetime):
                    position_age_seconds = (datetime.now() - position_open_time).total_seconds()
                else:
                    try:
                        open_dt = datetime.fromisoformat(str(position_open_time).replace('Z', '+00:00'))
                        position_age_seconds = (datetime.now(timezone.utc) - open_dt).total_seconds()
                    except:
                        position_age_seconds = 999  # Assume old position if can't parse
            
            is_new_position = position_age_seconds < min_hold_seconds
            
            # === REVERSAL DETECTION - RUN IF WE HAVE/HAD PROFIT ===
            # CRITICAL FIX: Also run if we HAD meaningful profit (peak_profit_pct > 0.1)
            # This catches the case where we go from profit into loss!
            # Example: Peak was +0.26%, now -0.68% - we MUST detect this reversal!
            has_any_profit = pnl_usd > 0 or pnl_pct > 0
            had_meaningful_profit = peak_profit_pct > 0.1  # We were profitable before
            
            # For NEW positions (< 30 sec), just skip - let them settle
            if is_new_position:
                logger.debug(f"⏳ {symbol}: New position ({position_age_seconds:.0f}s < 30s) - letting it settle")
                has_any_profit = False  # Skip reversal detection for brand new positions
                had_meaningful_profit = False  # Also skip if new
            
            # Initialize reversal_score here so it's always defined
            reversal_score = 0
            
            # CRITICAL: Run reversal detection if currently in profit OR if we HAD profit before
            if len(df) >= 5 and (has_any_profit or had_meaningful_profit):
                # Check recent price action
                recent_closes = df['close'].tail(10).values if len(df) >= 10 else df['close'].tail(5).values
                recent_high = max(recent_closes)
                recent_low = min(recent_closes)
                current_close = recent_closes[-1]
                
                # === FAST REVERSAL DETECTION (1-3 candles) ===
                # Get last 3 candles for immediate pattern detection
                last_3_candles = df.tail(3)
                if len(last_3_candles) >= 3:
                    c1_open, c1_high, c1_low, c1_close = last_3_candles.iloc[-3][['open', 'high', 'low', 'close']]
                    c2_open, c2_high, c2_low, c2_close = last_3_candles.iloc[-2][['open', 'high', 'low', 'close']]
                    c3_open, c3_high, c3_low, c3_close = last_3_candles.iloc[-1][['open', 'high', 'low', 'close']]  # Current
                    
                    # Calculate ATR for context
                    recent_atr = context.get('atr', 0) or df['close'].tail(14).std() * 1.5
                    atr_pct = (recent_atr / current_close * 100) if current_close > 0 else 1.0
                    
                    # FAST detection signals (immediate, 1-2 candle patterns)
                    fast_reversal_signals = []
                    fast_reversal_score = 0
                    
                    if is_long:
                        # === LONG POSITION FAST REVERSAL SIGNALS ===
                        # Signal 1: Big red candle (bearish engulfing or strong sell)
                        c3_body_pct = abs(c3_close - c3_open) / c3_open * 100 if c3_open > 0 else 0
                        if c3_close < c3_open and c3_body_pct > atr_pct * 0.5:
                            fast_reversal_score += 20
                            fast_reversal_signals.append(f"🔴 Big red candle ({c3_body_pct:.2f}%)")
                        
                        # Signal 2: Current close below previous low (breakdown)
                        if c3_close < c2_low:
                            fast_reversal_score += 25
                            fast_reversal_signals.append("⬇️ Breakdown below prev low")
                        
                        # Signal 3: Sharp 1-candle drop from high (rejected at top)
                        one_candle_drop = (c3_high - c3_close) / c3_high * 100 if c3_high > 0 else 0
                        if one_candle_drop > atr_pct * 0.7:
                            fast_reversal_score += 20
                            fast_reversal_signals.append(f"📉 Sharp 1-bar drop ({one_candle_drop:.2f}%)")
                        
                        # Signal 4: Two consecutive red candles
                        if c3_close < c3_open and c2_close < c2_open:
                            fast_reversal_score += 15
                            fast_reversal_signals.append("🔻 2 consecutive red bars")
                        
                        # Signal 5: Price dropped from 2-candle high (quick reversal)
                        two_candle_high = max(c2_high, c3_high)
                        drop_from_2c_high = (two_candle_high - c3_close) / two_candle_high * 100 if two_candle_high > 0 else 0
                        if drop_from_2c_high > atr_pct * 1.0:
                            fast_reversal_score += 25
                            fast_reversal_signals.append(f"🚨 Quick {drop_from_2c_high:.2f}% drop from peak")
                        
                        # Signal 6: Upper wick rejection (buying exhaustion)
                        upper_wick = c3_high - max(c3_open, c3_close)
                        body = abs(c3_close - c3_open)
                        if body > 0 and upper_wick > body * 1.5:
                            fast_reversal_score += 15
                            fast_reversal_signals.append("📍 Upper wick rejection")
                            
                    else:  # SHORT
                        # === SHORT POSITION FAST REVERSAL SIGNALS ===
                        # Signal 1: Big green candle (bullish)
                        c3_body_pct = abs(c3_close - c3_open) / c3_open * 100 if c3_open > 0 else 0
                        if c3_close > c3_open and c3_body_pct > atr_pct * 0.5:
                            fast_reversal_score += 20
                            fast_reversal_signals.append(f"🟢 Big green candle ({c3_body_pct:.2f}%)")
                        
                        # Signal 2: Current close above previous high (breakout)
                        if c3_close > c2_high:
                            fast_reversal_score += 25
                            fast_reversal_signals.append("⬆️ Breakout above prev high")
                        
                        # Signal 3: Sharp 1-candle pump from low
                        one_candle_pump = (c3_close - c3_low) / c3_low * 100 if c3_low > 0 else 0
                        if one_candle_pump > atr_pct * 0.7:
                            fast_reversal_score += 20
                            fast_reversal_signals.append(f"📈 Sharp 1-bar pump ({one_candle_pump:.2f}%)")
                        
                        # Signal 4: Two consecutive green candles
                        if c3_close > c3_open and c2_close > c2_open:
                            fast_reversal_score += 15
                            fast_reversal_signals.append("🔺 2 consecutive green bars")
                        
                        # Signal 5: Price jumped from 2-candle low
                        two_candle_low = min(c2_low, c3_low)
                        jump_from_2c_low = (c3_close - two_candle_low) / two_candle_low * 100 if two_candle_low > 0 else 0
                        if jump_from_2c_low > atr_pct * 1.0:
                            fast_reversal_score += 25
                            fast_reversal_signals.append(f"🚨 Quick {jump_from_2c_low:.2f}% pump from low")
                        
                        # Signal 6: Lower wick rejection (selling exhaustion)
                        lower_wick = min(c3_open, c3_close) - c3_low
                        body = abs(c3_close - c3_open)
                        if body > 0 and lower_wick > body * 1.5:
                            fast_reversal_score += 15
                            fast_reversal_signals.append("📍 Lower wick rejection")
                    
                    # Log fast reversal if detected
                    if fast_reversal_score >= 25:
                        logger.info(f"⚡ FAST REVERSAL {symbol}: Score {fast_reversal_score} | {' | '.join(fast_reversal_signals[:3])}")
                
                else:
                    fast_reversal_score = 0
                    fast_reversal_signals = []
                
                # === MOMENTUM ANALYSIS (slower, 5-10 bars) - MORE SENSITIVE ===
                if is_long:
                    momentum_against = roc_5 < -0.1  # Slightly negative = turning
                    strong_momentum_against = roc_5 < -0.3 or roc_10 < -0.2  # Was -0.5/-0.3
                    accelerating_against = roc_5 < roc_10
                    price_drop_pct = ((recent_high - current_close) / recent_high) * 100 if recent_high > 0 else 0
                    very_strong_reversal = roc_5 < -0.7  # Was -1.0
                else:  # SHORT
                    momentum_against = roc_5 > 0.1  # Slightly positive = turning
                    strong_momentum_against = roc_5 > 0.3 or roc_10 > 0.2  # Was 0.5/0.3
                    accelerating_against = roc_5 > roc_10
                    price_drop_pct = ((current_close - recent_low) / recent_low) * 100 if recent_low > 0 else 0
                    very_strong_reversal = roc_5 > 0.7  # Was 1.0
                
                # === INTELLIGENT REVERSAL DETECTION ===
                # Combines FAST signals + DRAWDOWN + MOMENTUM
                
                # Calculate reversal severity score (0-100)
                reversal_score = 0
                reversal_factors = []
                
                # Factor 0: FAST REVERSAL SIGNALS (immediate, 1-3 candles) - 0-40 points
                if fast_reversal_score >= 40:
                    reversal_score += min(40, fast_reversal_score * 0.8)  # Cap at 40
                    reversal_factors.extend(fast_reversal_signals[:2])
                elif fast_reversal_score >= 25:
                    reversal_score += min(25, fast_reversal_score * 0.7)
                    reversal_factors.extend(fast_reversal_signals[:1])
                
                # Factor 1: Drawdown from peak (0-70 points) - MOST IMPORTANT!
                # Using PERCENTAGE-based thresholds (like Bybit display)
                # Lost 20% = 15pts, 30% = 25pts, 40% = 35pts, 50%+ = 50pts
                if peak_profit_pct > 0.1 and drawdown_pct_of_peak > 0:  # Any meaningful profit (>0.1%)
                    # Aggressive scaling - protect profits early!
                    if drawdown_pct_of_peak >= 50:
                        drawdown_score = 50  # Lost half = BIG penalty
                    elif drawdown_pct_of_peak >= 40:
                        drawdown_score = 35
                    elif drawdown_pct_of_peak >= 30:
                        drawdown_score = 25
                    elif drawdown_pct_of_peak >= 20:
                        drawdown_score = 15
                    else:
                        drawdown_score = drawdown_pct_of_peak * 0.5
                    
                    reversal_score += drawdown_score
                    logger.debug(f"📊 REVERSAL: drawdown_score={drawdown_score:.1f}, total reversal_score={reversal_score:.1f}")
                    
                    # CRITICAL: If we went from PROFIT into LOSS, this is a MAJOR reversal!
                    # Peak was positive, now we're negative = 100%+ of peak lost
                    if pnl_pct < 0 and peak_profit_pct > 0.1:
                        reversal_score += 35  # SEVERE penalty - we lost ALL profit and went negative
                        reversal_factors.append(f"🚨🚨 PROFIT→LOSS: Was +{peak_profit_pct:.2f}% now {pnl_pct:.2f}%!")
                        logger.warning(f"🚨 {symbol}: PROFIT→LOSS REVERSAL! Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:.2f}%")
                    # ALERT: If we lost 30%+ of peak profit, this is concerning
                    elif drawdown_pct_of_peak >= 30 and peak_profit_pct >= 0.3:
                        reversal_score += 15  # Bonus penalty for significant profit loss
                        reversal_factors.append(f"🚨 PROFIT DECAY: Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak!")
                    elif drawdown_pct_of_peak >= 20:
                        reversal_factors.append(f"📉 Lost {drawdown_pct_of_peak:.0f}% of peak ({peak_profit_pct:.2f}%→{pnl_pct:.2f}%)")
                
                # Factor 2: Momentum against position (0-25 points)
                if momentum_against:
                    if very_strong_reversal:
                        reversal_score += 25
                        reversal_factors.append(f"🔥 STRONG momentum reversal (ROC5={roc_5:+.2f}%)")
                    elif strong_momentum_against:
                        reversal_score += 18
                        reversal_factors.append(f"⚡ Strong momentum against (ROC5={roc_5:+.2f}%)")
                    else:
                        reversal_score += 10
                        reversal_factors.append(f"📊 Momentum turning (ROC5={roc_5:+.2f}%)")
                
                # Factor 3: Accelerating against (0-15 points)
                if accelerating_against and momentum_against:
                    reversal_score += 15
                    reversal_factors.append("⬇️ Momentum accelerating against")
                
                # Factor 4: Price action reversal (0-10 points)
                if price_drop_pct > 0.5:
                    reversal_score += min(10, price_drop_pct * 2)
                    reversal_factors.append(f"📉 Price dropped {price_drop_pct:.1f}% from peak")
                
                # Factor 5: Position already profitable - protect it! (0-10 points)
                if pnl_pct > 0.5:
                    profit_bonus = min(10, pnl_pct * 3)
                    reversal_score += profit_bonus
                    reversal_factors.append(f"💰 Profit at risk: ${pnl_usd:.2f}/{pnl_pct:.1f}%")
                
                # Factor 6: RSI OVERBOUGHT/OVERSOLD - EXIT EXTREME CONDITIONS (0-25 points)
                # LONG in overbought = likely to reverse down
                # SHORT in oversold = likely to reverse up
                rsi = context.get('rsi', 50)
                if is_long and rsi >= 70:
                    # LONG position with RSI overbought - danger zone!
                    if rsi >= 80:
                        reversal_score += 25
                        reversal_factors.append(f"🔥 EXTREME OVERBOUGHT RSI={rsi:.0f}")
                        logger.warning(f"⚠️ {symbol}: LONG in EXTREME overbought RSI={rsi:.0f}")
                    elif rsi >= 75:
                        reversal_score += 18
                        reversal_factors.append(f"🔴 OVERBOUGHT RSI={rsi:.0f}")
                    else:
                        reversal_score += 12
                        reversal_factors.append(f"⚠️ RSI overbought ({rsi:.0f})")
                elif not is_long and rsi <= 30:
                    # SHORT position with RSI oversold - danger zone!
                    if rsi <= 20:
                        reversal_score += 25
                        reversal_factors.append(f"🔥 EXTREME OVERSOLD RSI={rsi:.0f}")
                        logger.warning(f"⚠️ {symbol}: SHORT in EXTREME oversold RSI={rsi:.0f}")
                    elif rsi <= 25:
                        reversal_score += 18
                        reversal_factors.append(f"🟢 OVERSOLD RSI={rsi:.0f}")
                    else:
                        reversal_score += 12
                        reversal_factors.append(f"⚠️ RSI oversold ({rsi:.0f})")
                
                # === MOMENTUM HOPE - CAN REDUCE REVERSAL SCORE ===
                # If momentum is STILL WITH US despite drawdown, give it a chance
                # This prevents exiting during temporary dips when trend is intact
                momentum_hope = 0
                momentum_hope_factors = []
                
                if is_long:
                    # LONG: Positive momentum = hope
                    if roc_5 > 0.2:  # Strong positive momentum
                        momentum_hope += 15
                        momentum_hope_factors.append(f"📈 ROC5 still positive ({roc_5:+.2f}%)")
                    elif roc_5 > 0:  # Slight positive
                        momentum_hope += 8
                        momentum_hope_factors.append(f"📊 ROC5 flat/positive ({roc_5:+.2f}%)")
                    
                    if roc_10 > 0.3:  # Longer term trend intact
                        momentum_hope += 10
                        momentum_hope_factors.append(f"📈 10-bar trend up ({roc_10:+.2f}%)")
                    
                    # RSI not overbought yet = room to run
                    if 40 <= rsi <= 65:
                        momentum_hope += 8
                        momentum_hope_factors.append(f"✅ RSI healthy ({rsi:.0f})")
                    
                    # Price still above key MAs
                    if current_close > context.get('ema_20', 0) and context.get('ema_20', 0) > 0:
                        momentum_hope += 10
                        momentum_hope_factors.append("📊 Above EMA20")
                        
                else:  # SHORT
                    # SHORT: Negative momentum = hope
                    if roc_5 < -0.2:  # Strong downward momentum
                        momentum_hope += 15
                        momentum_hope_factors.append(f"📉 ROC5 still negative ({roc_5:+.2f}%)")
                    elif roc_5 < 0:  # Slight negative
                        momentum_hope += 8
                        momentum_hope_factors.append(f"📊 ROC5 flat/negative ({roc_5:+.2f}%)")
                    
                    if roc_10 < -0.3:  # Longer term trend intact
                        momentum_hope += 10
                        momentum_hope_factors.append(f"📉 10-bar trend down ({roc_10:+.2f}%)")
                    
                    # RSI not oversold yet = room to fall
                    if 35 <= rsi <= 60:
                        momentum_hope += 8
                        momentum_hope_factors.append(f"✅ RSI healthy ({rsi:.0f})")
                    
                    # Price still below key MAs
                    if current_close < context.get('ema_20', float('inf')):
                        momentum_hope += 10
                        momentum_hope_factors.append("📊 Below EMA20")
                
                # Apply momentum hope reduction to reversal score
                # BUT only if drawdown is not critical (under 40%)
                if momentum_hope > 0 and drawdown_pct_of_peak < 40:
                    hope_reduction = min(momentum_hope, 25)  # Cap reduction at 25 points
                    original_score = reversal_score
                    reversal_score = max(0, reversal_score - hope_reduction)
                    if hope_reduction > 10:
                        logger.info(f"💪 {symbol}: MOMENTUM HOPE reduces reversal score {original_score:.0f}→{reversal_score:.0f} | {momentum_hope_factors[:2]}")
                        reversal_factors.append(f"💪 Hope -{hope_reduction}pts: {momentum_hope_factors[0]}")
                
                # Log momentum hope if significant
                if momentum_hope >= 20:
                    logger.info(f"💪 {symbol}: Strong momentum hope ({momentum_hope}pts): {' | '.join(momentum_hope_factors[:3])}")
                
                # === HARDCODED FALLBACK SAFETY NET ===
                # These trigger REGARDLESS of intelligent score - absolute protection
                # In case intelligent detection fails, these ensure we don't lose big profits
                # 
                # KEY INSIGHT: Exit when we've lost 30-40% of peak, NOT 50-70%!
                # Example: Peak 0.72%, lost 40% = exit at 0.43%, NOT at 0.10%
                # 
                # Using PERCENTAGE thresholds (matches Bybit display)
                # BUT: If momentum hope is VERY strong (30+), allow more drawdown before trigger
                hardcoded_triggered = False
                strong_hope = momentum_hope >= 30  # Very strong momentum with us
                
                # Adjust thresholds based on momentum hope
                # Strong hope = give trade more room to breathe
                drawdown_threshold_1 = 35 if strong_hope else 25  # Good profit peak (0.5%+)
                drawdown_threshold_2 = 45 if strong_hope else 35  # Medium profit peak (0.3%+)
                drawdown_threshold_3 = 55 if strong_hope else 45  # Small profit peak (0.2%+)
                
                # === PERCENTAGE-BASED PROFIT PROTECTION ===
                # Exit at 25-35% drawdown instead of waiting for 50%+
                
                # TIER 1: Good profit (0.5%+) - protect at 25-35% drawdown
                if peak_profit_pct >= 0.5 and drawdown_pct_of_peak >= drawdown_threshold_1:
                    reversal_score = max(reversal_score, 70)
                    hardcoded_triggered = True
                    remaining_pct = 100 - drawdown_pct_of_peak
                    hope_note = " (momentum hope gave extra room)" if strong_hope else ""
                    reversal_factors.append(f"🚨 PROTECT: Keep {remaining_pct:.0f}% of {peak_profit_pct:.2f}% peak!{hope_note}")
                    logger.warning(f"🚨 {symbol}: PROFIT PROTECTION - Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak, exiting at {pnl_pct:.2f}%!{hope_note}")
                
                # TIER 2: Medium profit (0.3%+) - protect at 35-45% drawdown
                elif peak_profit_pct >= 0.3 and drawdown_pct_of_peak >= drawdown_threshold_2:
                    reversal_score = max(reversal_score, 75)
                    hardcoded_triggered = True
                    hope_note = " (momentum gave extra room)" if strong_hope else ""
                    reversal_factors.append(f"🚨 CRITICAL: Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak!{hope_note}")
                    logger.warning(f"🚨 {symbol}: CRITICAL - Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak!{hope_note}")
                
                # TIER 3: Small profit (0.2%+) - protect at 45-55% drawdown
                elif peak_profit_pct >= 0.2 and drawdown_pct_of_peak >= drawdown_threshold_3:
                    reversal_score = max(reversal_score, 65)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚠️ Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak")
                    logger.warning(f"⚠️ {symbol}: Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak!")
                
                # ABSOLUTE FLOOR: Never let meaningful profit disappear completely!
                # If we had 0.4%+ peak and now under 0.1% = EXIT NOW
                elif peak_profit_pct >= 0.4 and pnl_pct <= 0.1 and pnl_pct > 0:
                    reversal_score = max(reversal_score, 80)
                    hardcoded_triggered = True
                    reversal_factors.append(f"🚨 FLOOR: {peak_profit_pct:.2f}% peak → {pnl_pct:.2f}% remaining!")
                    logger.warning(f"🚨🚨 {symbol}: PROFIT FLOOR - Had {peak_profit_pct:.2f}%, now only {pnl_pct:.2f}%! EXIT NOW!")
                
                # LARGE USD PROFIT: $10+ with strong reversal = CRITICAL
                elif pnl_usd >= 10.0 and momentum_against and very_strong_reversal:
                    if reversal_urgency not in ['critical']:
                        reversal_score = max(reversal_score, 75)
                        hardcoded_triggered = True
                        reversal_factors.append("🛡️ HARDCODED: $10+ with strong reversal")
                        logger.warning(f"🛡️ {symbol}: HARDCODED FALLBACK triggered - $10+ profit at risk!")
                
                # MEDIUM USD PROFIT: $5+ with strong momentum against = HIGH minimum
                elif pnl_usd >= 5.0 and momentum_against and strong_momentum_against:
                    if reversal_urgency not in ['critical', 'high']:
                        reversal_score = max(reversal_score, 55)
                        hardcoded_triggered = True
                        reversal_factors.append("🛡️ HARDCODED: $5+ with momentum against")
                        logger.warning(f"🛡️ {symbol}: HARDCODED FALLBACK triggered - $5+ profit at risk!")
                
                elif pnl_pct >= 2.0 and momentum_against:
                    # 2%+ profit with any momentum against = HIGH minimum
                    if reversal_urgency not in ['critical', 'high']:
                        reversal_score = max(reversal_score, 55)
                        hardcoded_triggered = True
                        reversal_factors.append("🛡️ HARDCODED: 2%+ profit reversal")
                        logger.warning(f"🛡️ {symbol}: HARDCODED FALLBACK triggered - {pnl_pct:.1f}% profit at risk!")
                
                elif pnl_usd >= 3.0 and very_strong_reversal:
                    # $3+ profit with very strong reversal = MEDIUM minimum
                    if reversal_urgency not in ['critical', 'high', 'medium']:
                        reversal_score = max(reversal_score, 40)
                        hardcoded_triggered = True
                        reversal_factors.append("🛡️ HARDCODED: $3+ very strong reversal")
                
                # === FAST REVERSAL HARDCODED TRIGGERS ===
                # These use the FAST 1-3 candle detection for immediate response
                if fast_reversal_score >= 50 and pnl_usd >= 1.5:
                    # Strong fast reversal with meaningful profit = ACT NOW
                    reversal_score = max(reversal_score, 65)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ FAST: Score {fast_reversal_score} + ${pnl_usd:.2f} profit")
                    logger.warning(f"⚡ {symbol}: FAST REVERSAL TRIGGERED - {fast_reversal_signals[:2]}")
                
                elif fast_reversal_score >= 40 and pnl_usd >= 3.0:
                    # Medium fast reversal with good profit = HIGH urgency
                    reversal_score = max(reversal_score, 55)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ FAST: ${pnl_usd:.2f} at risk")
                    logger.warning(f"⚡ {symbol}: FAST REVERSAL - ${pnl_usd:.2f} profit at risk!")
                
                elif fast_reversal_score >= 35 and pnl_pct >= 1.5:
                    # Fast reversal with % profit = HIGH urgency
                    reversal_score = max(reversal_score, 50)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ FAST: {pnl_pct:.1f}% at risk")
                
                # === SMALL PROFIT REVERSAL - BALANCED APPROACH ===
                # Don't cut tiny profits too early, but protect meaningful ones
                # Score 60+ with $0.15+ profit = worth protecting
                elif fast_reversal_score >= 60 and pnl_usd >= 0.15:
                    # Strong reversal on small but meaningful profit = consider exit
                    reversal_score = max(reversal_score, 50)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ Strong reversal on ${pnl_usd:.2f}")
                    logger.warning(f"⚡ {symbol}: STRONG REVERSAL on profit ${pnl_usd:.2f}")
                
                elif fast_reversal_score >= 45 and very_strong_reversal and pnl_usd >= 0.10:
                    # Fast reversal + strong ROC momentum on any profit = protect it
                    reversal_score = max(reversal_score, 45)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ Momentum+Fast on ${pnl_usd:.2f}")
                    logger.warning(f"⚡ {symbol}: MOMENTUM+FAST reversal on ${pnl_usd:.2f}")
                
                # === RSI EXTREME + PROFIT = EXIT NOW ===
                # If we're in profit and RSI is extreme against our position, exit!
                if pnl_usd >= 1.0:
                    if is_long and rsi >= 78:
                        # LONG with RSI very overbought + profit = EXIT
                        reversal_score = max(reversal_score, 60)
                        hardcoded_triggered = True
                        reversal_factors.append(f"🔴 RSI {rsi:.0f} + ${pnl_usd:.2f} profit")
                        logger.warning(f"🔴 {symbol}: LONG overbought RSI={rsi:.0f} + profit ${pnl_usd:.2f} - EXIT!")
                    elif not is_long and rsi <= 22:
                        # SHORT with RSI very oversold + profit = EXIT
                        reversal_score = max(reversal_score, 60)
                        hardcoded_triggered = True
                        reversal_factors.append(f"🟢 RSI {rsi:.0f} + ${pnl_usd:.2f} profit")
                        logger.warning(f"🟢 {symbol}: SHORT oversold RSI={rsi:.0f} + profit ${pnl_usd:.2f} - EXIT!")
                
                # === CLASSIFY URGENCY BASED ON REVERSAL SCORE ===
                # MORE AGGRESSIVE thresholds to exit earlier and protect profits!
                if reversal_score >= 55:  # Was 70
                    profit_reversal_detected = True
                    reversal_urgency = 'critical'
                    reversal_reason = f"CRITICAL: Score {reversal_score:.0f}/100 | Peak ${peak_profit_usd:.2f}→${pnl_usd:.2f} | " + " | ".join(reversal_factors[:2])
                    logger.warning(f"🚨🚨 {symbol}: {reversal_reason}")
                    
                elif reversal_score >= 40:  # Was 50
                    profit_reversal_detected = True
                    reversal_urgency = 'high'
                    reversal_reason = f"HIGH: Score {reversal_score:.0f}/100 | Peak ${peak_profit_usd:.2f}→${pnl_usd:.2f} | " + " | ".join(reversal_factors[:2])
                    logger.warning(f"🔴 {symbol}: {reversal_reason}")
                    
                elif reversal_score >= 28:  # Was 35
                    profit_reversal_detected = True
                    reversal_urgency = 'medium'
                    reversal_reason = f"MEDIUM: Score {reversal_score:.0f}/100 | Peak ${peak_profit_usd:.2f}→${pnl_usd:.2f}"
                    logger.warning(f"🟠 {symbol}: {reversal_reason}")
                    
                elif reversal_score >= 15 and pnl_pct > 0:  # Was 20
                    profit_reversal_detected = True
                    reversal_urgency = 'low'
                    reversal_reason = f"LOW: Score {reversal_score:.0f}/100 | Watching..."
                    logger.info(f"🟡 {symbol}: {reversal_reason}")
                
                # Log peak tracking for debugging (like Bybit: $0.05 (+0.72%))
                if peak_profit_pct > pnl_pct and peak_profit_pct > 0.3:
                    logger.info(f"📊 {symbol}: Peak ${peak_profit_usd:.2f} ({peak_profit_pct:.2f}%) → Now ${pnl_usd:.2f} ({pnl_pct:.2f}%) | Lost {drawdown_pct_of_peak:.0f}% of gain | RevScore: {reversal_score:.0f}")
                
                # Always log peak status for positions in meaningful profit 
                if pnl_pct > 0.3 or pnl_usd > 0.5:
                    hc_tag = " [HC]" if hardcoded_triggered else ""
                    logger.info(f"📈 {symbol}: PnL ${pnl_usd:.2f} ({pnl_pct:+.2f}%) | Peak {peak_profit_pct:.2f}% | RevScore: {reversal_score:.0f}{hc_tag}")
            
            # Store updated peak values for next cycle (return to caller to save)
            peak_tracking = {
                'peak_profit_usd': peak_profit_usd,
                'peak_profit_pct': peak_profit_pct,
                'drawdown_usd': drawdown_usd,
                'drawdown_pct_of_peak': drawdown_pct_of_peak,
                'reversal_score': reversal_score if 'reversal_score' in dir() else 0,
                'hardcoded_triggered': hardcoded_triggered if 'hardcoded_triggered' in dir() else False
            }
            
            # === PHASE 2: MATH ANALYSIS ===
            # context already built above for momentum checks
            
            # Get scores for holding vs exiting
            hold_signal = 1 if side == 'LONG' else -1
            exit_signal = -hold_signal
            
            hold_check = self._comprehensive_math_check(hold_signal, df, current_price, atr, context)
            exit_check = self._comprehensive_math_check(exit_signal, df, current_price, atr, context)
            
            hold_score = hold_check.get('score', 50)
            exit_score = exit_check.get('score', 50)
            
            # === PHASE 3: SMART PROFIT PROTECTION ===
            # Be MORE aggressive about protecting profits - our balance is under $500!
            # Key insight: Small profits ($5-10) matter when total balance is <$500
            profit_urgency = 0
            profit_context = []
            
            if pnl_pct > 0:  # In profit
                if pnl_usd >= 25 or pnl_pct >= 1.5:
                    profit_urgency = 35  # VERY strong - protect this profit!
                    profit_context.append(f"💰 GOOD PROFIT (${pnl_usd:.0f}/{pnl_pct:.1f}%) - PROTECT IT!")
                elif pnl_usd >= 10 or pnl_pct >= 0.75:
                    profit_urgency = 25  # Strong protection
                    profit_context.append(f"💰 Solid profit (${pnl_usd:.0f}/{pnl_pct:.1f}%) - protect it!")
                elif pnl_usd >= 5 or pnl_pct >= 0.4:
                    profit_urgency = 18  # Moderate protection - lowered threshold!
                    profit_context.append(f"📈 Profit (${pnl_usd:.0f}/{pnl_pct:.1f}%) - watching closely")
                elif pnl_usd >= 2 or pnl_pct >= 0.2:
                    profit_urgency = 10  # Light protection - capture small gains
                    profit_context.append(f"📊 Small profit (${pnl_usd:.0f}) - considering lock-in")
            elif pnl_pct > -1.0:  # Small loss (can exit if strong signal)
                profit_context.append(f"📉 Small loss ({pnl_pct:.2f}%) - can exit on strong signal")
            
            # Boost exit score based on profit urgency
            adjusted_exit_score = exit_score + profit_urgency
            
            # === PHASE 4: REVERSAL DETECTION ===
            # Look for signs the market is about to reverse against us
            reversal_signals = []
            reversal_strength = 0
            
            # Check momentum reversal
            if 'close' in df.columns and len(df) >= 10:
                recent_closes = df['close'].tail(10)
                momentum_5 = (recent_closes.iloc[-1] / recent_closes.iloc[-5] - 1) * 100
                momentum_10 = (recent_closes.iloc[-1] / recent_closes.iloc[-10] - 1) * 100
                
                if side == 'LONG':
                    if momentum_5 < -0.5:
                        reversal_signals.append(f"Negative 5-bar momentum ({momentum_5:.2f}%)")
                        reversal_strength += 15
                    if momentum_10 < -0.3 and momentum_5 < momentum_10:
                        reversal_signals.append("Accelerating downward momentum")
                        reversal_strength += 10
                else:  # SHORT
                    if momentum_5 > 0.5:
                        reversal_signals.append(f"Positive 5-bar momentum ({momentum_5:.2f}%)")
                        reversal_strength += 15
                    if momentum_10 > 0.3 and momentum_5 > momentum_10:
                        reversal_signals.append("Accelerating upward momentum")
                        reversal_strength += 10
            
            # Check RSI extremes
            if 'rsi' in df.columns and len(df) > 0:
                rsi = df['rsi'].iloc[-1]
                if side == 'LONG' and rsi > 75:
                    reversal_signals.append(f"Overbought RSI ({rsi:.0f})")
                    reversal_strength += 12
                elif side == 'SHORT' and rsi < 25:
                    reversal_signals.append(f"Oversold RSI ({rsi:.0f})")
                    reversal_strength += 12
            
            # Check volume spike (potential reversal)
            volume_ratio = context.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                reversal_signals.append(f"Volume spike ({volume_ratio:.1f}x)")
                reversal_strength += 8
            
            # Add reversal strength to exit score if in profit
            if pnl_pct > 0 and reversal_strength > 0:
                adjusted_exit_score += reversal_strength
                profit_context.append(f"⚠️ Reversal risk: {', '.join(reversal_signals)}")
            
            # === PHASE 5: MATH DECISION ===
            math_action = 'hold'
            math_reasoning = []
            
            # === FAST EXIT: Only at 0.0% to -0.1% loss IF price WANTS to go against us ===
            # This catches bad entries where direction was wrong AND there's NO recovery chance
            # Must be VERY strict - only exit if ALL indicators confirm no hope
            FAST_EXIT_WINDOW = 45  # First 45 seconds only
            FAST_EXIT_MAX_LOSS = -0.1  # Between 0.0% and -0.1% (breakeven to tiny loss)
            FAST_EXIT_MIN_LOSS = 0.0  # Don't exit if we're in profit!
            
            # Calculate "no recovery chance" score - must be VERY high to trigger fast exit
            no_recovery_score = 0
            no_recovery_reasons = []
            
            # Factor 1: Strong momentum against (ROC5 and ROC10 both against us)
            if is_long:
                if roc_5 < -0.4:  # Strong bearish
                    no_recovery_score += 30
                    no_recovery_reasons.append(f"ROC5={roc_5:+.2f}%")
                if roc_10 < -0.3:  # Sustained bearish
                    no_recovery_score += 20
                    no_recovery_reasons.append(f"ROC10={roc_10:+.2f}%")
                if accelerating_against:  # Getting worse
                    no_recovery_score += 15
                    no_recovery_reasons.append("Accelerating down")
            else:  # SHORT
                if roc_5 > 0.4:  # Strong bullish
                    no_recovery_score += 30
                    no_recovery_reasons.append(f"ROC5={roc_5:+.2f}%")
                if roc_10 > 0.3:  # Sustained bullish
                    no_recovery_score += 20
                    no_recovery_reasons.append(f"ROC10={roc_10:+.2f}%")
                if accelerating_against:  # Getting worse
                    no_recovery_score += 15
                    no_recovery_reasons.append("Accelerating up")
            
            # Factor 2: No momentum hope at all
            momentum_hope_score = momentum_hope if 'momentum_hope' in dir() else 0
            if momentum_hope_score == 0:
                no_recovery_score += 20
                no_recovery_reasons.append("No momentum hope")
            elif momentum_hope_score < 10:
                no_recovery_score += 10
                no_recovery_reasons.append(f"Low hope ({momentum_hope_score})")
            
            # Factor 3: Fast reversal already detected
            if 'fast_reversal_score' in dir() and fast_reversal_score >= 40:
                no_recovery_score += 15
                no_recovery_reasons.append(f"Fast rev={fast_reversal_score}")
            
            # ONLY trigger fast exit if:
            # 1. Within first 45 seconds
            # 2. At breakeven to -0.1% loss (not in profit, not too deep loss)
            # 3. No recovery score >= 70 (very high confidence no recovery)
            no_recovery_chance = no_recovery_score >= 70
            
            in_fast_exit_zone = pnl_pct <= FAST_EXIT_MIN_LOSS and pnl_pct >= FAST_EXIT_MAX_LOSS
            
            if hold_seconds <= FAST_EXIT_WINDOW and in_fast_exit_zone and no_recovery_chance:
                # Price went against us immediately AND multiple factors confirm no recovery
                logger.warning(f"🚨 {symbol}: FAST EXIT! PnL={pnl_pct:.2f}% in {hold_seconds:.0f}s | No Recovery Score={no_recovery_score} | {', '.join(no_recovery_reasons[:3])}")
                math_action = 'close'
                math_reasoning.append(f"🚨 FAST EXIT: No recovery ({no_recovery_score}pts) | {', '.join(no_recovery_reasons[:2])}")
                return {
                    'action': 'close',
                    'confidence': 0.95,
                    'math_score': 90,
                    'ai_validated': True,
                    'reasoning': '; '.join(math_reasoning),
                    'sl_adjustment': None,
                    'peak_profit_pct': peak_profit_pct,
                    'peak_profit_usd': peak_profit_usd
                }
            elif hold_seconds <= FAST_EXIT_WINDOW and in_fast_exit_zone:
                # Log why we're NOT fast exiting
                logger.info(f"⏳ {symbol}: In fast exit zone ({pnl_pct:.2f}%) but recovery possible (score={no_recovery_score}/70)")
            
            # === NO GRACE PERIOD - Let math and AI decide ===
            # Grace period was causing losses by holding bad positions too long
            # Now we rely on math exit score vs hold score + fast exit for bad entries
            
            # Decision thresholds - based on P&L state
            if pnl_pct > 0:
                exit_threshold = 12  # Easier to exit when in profit (after grace)
            else:
                exit_threshold = 25  # Harder to cut losses (was 15, increased to give more time)
            
            if adjusted_exit_score > hold_score + exit_threshold:
                math_action = 'close'
                math_reasoning.append(f"Exit score ({adjusted_exit_score:.0f}) > Hold ({hold_score:.0f}) by {exit_threshold}+")
            elif exit_score > hold_score and pnl_pct > 0.5:
                math_action = 'tighten_sl'
                math_reasoning.append(f"Exit strengthening - tighten SL to lock profit")
            else:
                math_reasoning.append(f"Hold score ({hold_score:.0f}) favored over exit ({adjusted_exit_score:.0f})")
            
            # Add context to reasoning
            math_reasoning.extend(profit_context)
            if reversal_signals and pnl_pct > 0:
                math_reasoning.append(f"Reversal indicators: {len(reversal_signals)}")
            
            # === PHASE 5.5: INTELLIGENT PROFIT REVERSAL MATH OVERRIDE ===
            # Uses the dynamic reversal_score calculated above (0-100)
            # No hardcoded dollar thresholds - decision based on:
            # 1. Reversal score (drawdown + momentum + acceleration)
            # 2. Math analysis (exit_score vs hold_score)
            # 3. Drawdown from peak profit
            math_override_exit = False
            
            # Get reversal score from peak_tracking (calculated above)
            reversal_score = peak_tracking.get('reversal_score', 0)
            drawdown_pct = peak_tracking.get('drawdown_pct_of_peak', 0)
            
            # IMPORTANT: Only trigger reversal protection if peak profit was meaningful
            # Don't trigger on tiny fluctuations (< $1.00 or < 0.5%)
            # Increased from $0.50/0.30% because small peaks were causing premature exits
            min_peak_for_reversal = 1.00  # At least $1.00 peak profit to trigger reversal protection
            min_peak_pct_for_reversal = 0.50  # At least 0.5% profit to trigger reversal protection
            
            peak_was_meaningful = peak_profit_usd >= min_peak_for_reversal or peak_profit_pct >= min_peak_pct_for_reversal
            
            if profit_reversal_detected and peak_was_meaningful:
                # CRITICAL: Reversal score >= 70 OR lost 50%+ of peak profit → EXIT NOW
                if reversal_urgency == 'critical' or (drawdown_pct >= 50 and peak_profit_usd > 1.0):
                    math_override_exit = True
                    math_action = 'close'
                    math_reasoning.append(f"🚨 MATH+AI OVERRIDE: CRITICAL (Score={reversal_score:.0f}) Peak ${peak_profit_usd:.2f}→${pnl_usd:.2f} ({drawdown_pct:.0f}% lost)")
                    logger.warning(f"🚨🚨 INTELLIGENT EXIT {symbol}: Score {reversal_score:.0f}/100, lost {drawdown_pct:.0f}% of peak profit")
                    
                # HIGH: Score >= 50 AND (math confirms OR lost 30%+ of peak)
                elif reversal_urgency == 'high' and (exit_score >= hold_score or drawdown_pct >= 30):
                    math_override_exit = True
                    math_action = 'close'
                    math_reasoning.append(f"⚡ MATH+AI OVERRIDE: HIGH (Score={reversal_score:.0f}) + Math confirms exit")
                    logger.warning(f"⚡⚡ INTELLIGENT EXIT {symbol}: Score {reversal_score:.0f}/100, math exit={exit_score:.0f} vs hold={hold_score:.0f}")
                    
                # MEDIUM: Score >= 35 AND math strongly favors exit AND losing profit
                elif reversal_urgency == 'medium' and adjusted_exit_score > hold_score + 10 and drawdown_pct >= 20:
                    math_override_exit = True
                    math_action = 'close'
                    math_reasoning.append(f"📊 MATH+AI OVERRIDE: MEDIUM (Score={reversal_score:.0f}) + Strong exit signal")
                    logger.warning(f"📊 INTELLIGENT EXIT {symbol}: Score {reversal_score:.0f}/100, drawdown {drawdown_pct:.0f}%")
                    
                # LOW: Just flag for AI consideration, don't override
                elif reversal_urgency == 'low':
                    math_reasoning.append(f"⚠️ Reversal detected (Score={reversal_score:.0f}) - AI will decide")
            elif profit_reversal_detected and not peak_was_meaningful:
                # Peak profit was too small to trigger reversal protection
                logger.info(f"💡 {symbol}: Reversal detected but peak was small (${peak_profit_usd:.2f}/{peak_profit_pct:.2f}%) - letting position develop")
                math_reasoning.append(f"💡 Small peak (${peak_profit_usd:.2f}) - holding for bigger move")
            
            # === PHASE 6: AI VALIDATION FOR ALL DECISIONS ===
            # AI now has POWERFUL control - it can override or refine decisions
            # EXCEPT when math_override_exit is True (profit protection)
            ai_validated = False
            ai_agrees_to_exit = False
            ai_confidence = 0.5
            ai_suggested_action = 'hold'
            
            # Log whether AI will be consulted
            if math_override_exit:
                logger.info(f"🛡️ {symbol}: Math Override active - skipping AI consultation for profit protection")
            
            # Always consult AI for position decisions (not just when math says close)
            # BUT skip AI consultation if math override is active (profit protection takes priority)
            if self.use_ai and not math_override_exit:
                logger.info(f"🤖 {symbol}: Consulting AI for exit decision (reversal_score={reversal_score:.0f}, hope={momentum_hope if 'momentum_hope' in dir() else 0})")
                # Get AI opinion with deep math analysis
                # Include momentum hope so AI knows if there's reason to hold
                ai_result = self._ai_smart_exit_decision(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    current_price=current_price,
                    pnl_pct=pnl_pct,
                    pnl_usd=pnl_usd,
                    hold_score=hold_score,
                    exit_score=adjusted_exit_score,
                    reversal_signals=reversal_signals,
                    context=context,
                    df=df,
                    profit_reversal_detected=profit_reversal_detected,
                    reversal_urgency=reversal_urgency,
                    reversal_reason=reversal_reason,
                    momentum_hope=momentum_hope if 'momentum_hope' in dir() else 0,
                    momentum_hope_factors=momentum_hope_factors if 'momentum_hope_factors' in dir() else [],
                    reversal_score=reversal_score if 'reversal_score' in dir() else 0
                )
                
                if ai_result:
                    ai_validated = True
                    ai_agrees_to_exit = ai_result.get('should_exit', False)
                    ai_confidence = ai_result.get('confidence', 0.5)
                    ai_reason = ai_result.get('reasoning', '')
                    ai_suggested_action = ai_result.get('suggested_action', 'hold')
                    
                    # === SMART GRACE PERIOD - ALLOWS INTELLIGENT REVERSALS ===
                    # During grace period, AI can still force exit IF:
                    # 1. Loss exceeds -1.0% (meaningful loss)
                    # 2. AI has 85%+ confidence (strong signal)
                    # 3. Reversal score >= 50 (strong reversal detected)
                    # 4. Fast reversal score >= 40 (quick momentum shift)
                    # 5. We're in profit AND momentum is strongly against us
                    # 6. PROFIT→LOSS: We had meaningful profit but now in loss
                    # NOTE: Grace period REMOVED - was causing losses by blocking exits
                    
                    # AI has POWERFUL control - respect its suggestion
                    # AI needs 85% confidence to override on LOSSES (was 75% but exited too quickly)
                    # On PROFITS, 75% is still enough to protect gains
                    ai_override_threshold = 0.75 if pnl_usd > 0 else 0.85
                    if ai_suggested_action == 'exit' and ai_confidence >= ai_override_threshold:
                        # Additional check for losses: only AI override if loss is meaningful (>0.5%)
                        if pnl_pct >= 0 or (pnl_pct < 0 and pnl_pct <= -0.5):
                            math_action = 'close'
                            ai_agrees_to_exit = True  # CRITICAL: Mark AI agrees to exit!
                            math_reasoning.append(f"🤖 AI OVERRIDE: EXIT ({ai_confidence:.0%}): {ai_reason}")
                            logger.warning(f"🚨 AI OVERRIDE CLOSE for {symbol}: {ai_reason} (conf={ai_confidence:.0%})")
                        else:
                            math_reasoning.append(f"🤖 AI wanted exit but loss too small ({pnl_pct:.2f}%) - HOLDING")
                            logger.info(f"⏳ {symbol}: AI EXIT blocked - loss {pnl_pct:.2f}% < 0.5%, waiting for recovery")
                    # AI + FAST REVERSAL COMBO: Even small profit, if AI says exit + fast signals strong
                    elif ai_suggested_action == 'exit' and ai_confidence >= 0.60 and fast_reversal_score >= 35 and pnl_usd > 0:
                        math_action = 'close'
                        ai_agrees_to_exit = True
                        math_reasoning.append(f"🤖⚡ AI+FAST COMBO: Exit ${pnl_usd:.2f} (AI={ai_confidence:.0%}, Fast={fast_reversal_score})")
                        logger.warning(f"🤖⚡ {symbol}: AI+FAST REVERSAL CLOSE - AI {ai_confidence:.0%} + Fast score {fast_reversal_score}")
                    elif ai_suggested_action == 'wait_for_bounce' and pnl_pct < 0 and ai_confidence >= 0.65:
                        # DISABLED: Don't wait for bounce - cut losses fast!
                        # Old behavior held losers hoping for mean reversion
                        # Now we exit if math says exit, regardless of AI bounce hope
                        math_reasoning.append(f"🤖 AI wanted bounce but we CUT LOSSES FAST now")
                        # Don't override math_action - let it proceed
                    elif ai_suggested_action == 'tighten_sl' and pnl_pct > 0:
                        math_action = 'tighten_sl'
                        ai_agrees_to_exit = False
                        math_reasoning.append(f"🤖 AI: TIGHTEN SL ({ai_confidence:.0%}): {ai_reason}")
                    elif ai_suggested_action == 'hold':
                        # AI says hold - if math said close, AI vetoes it
                        # BUT NOT if math_override_exit is active (profit protection)
                        ai_agrees_to_exit = False
                        if math_action == 'close' and ai_confidence >= 0.60 and not math_override_exit:
                            if pnl_pct > 0.5:
                                math_action = 'tighten_sl'
                                math_reasoning.append(f"🤖 AI VETO exit → tighten SL: {ai_reason}")
                            else:
                                math_action = 'hold'
                                math_reasoning.append(f"🤖 AI VETO exit → hold: {ai_reason}")
                        elif math_override_exit:
                            math_reasoning.append(f"🤖 AI said HOLD but MATH OVERRIDE active - exiting anyway!")
                            logger.warning(f"⚡ {symbol}: AI wanted to hold but MATH OVERRIDE takes priority for profit protection!")
                            ai_agrees_to_exit = True  # Force this for confidence calculation
                        else:
                            math_reasoning.append(f"🤖 AI confirms HOLD ({ai_confidence:.0%}): {ai_reason}")
            
            # For tighten_sl, AI validation is optional
            if math_action == 'tighten_sl' and not ai_validated:
                ai_validated = True  # Auto-approve SL tightening
            
            # === PHASE 7: FINAL DECISION ===
            # Calculate confidence based on agreement
            score_diff = abs(hold_score - adjusted_exit_score)
            base_confidence = min(0.5 + (score_diff / 100), 0.95)
            
            if math_action == 'close':
                # MATH OVERRIDE gets high confidence automatically
                if math_override_exit:
                    final_confidence = 0.90  # High confidence for profit protection
                    ai_validated = True  # Mark as validated
                    ai_agrees_to_exit = True  # Mark agreement
                    logger.warning(f"💰 {symbol}: MATH OVERRIDE EXIT - Protecting ${pnl_usd:.2f} profit with 90% confidence")
                # For normal exits, require both math and AI agreement
                elif ai_validated and ai_agrees_to_exit:
                    final_confidence = min(base_confidence, ai_confidence)
                else:
                    final_confidence = base_confidence * 0.7  # Reduce confidence without AI
            else:
                final_confidence = base_confidence
            
            # Calculate SL adjustment if needed
            sl_adjustment = None
            if math_action == 'tighten_sl':
                current_sl = position.get('stop_loss', entry_price)
                if side == 'LONG':
                    # Move SL up to lock profit (at least breakeven)
                    breakeven_sl = entry_price * 1.001  # Tiny buffer above entry
                    profit_lock_sl = current_price - atr * 1.2  # Tighter than usual
                    new_sl = max(current_sl, breakeven_sl, profit_lock_sl)
                    if new_sl > current_sl:
                        sl_adjustment = new_sl
                        math_reasoning.append(f"SL tightened: ${current_sl:.4f} → ${new_sl:.4f}")
                else:
                    breakeven_sl = entry_price * 0.999
                    profit_lock_sl = current_price + atr * 1.2
                    new_sl = min(current_sl, breakeven_sl, profit_lock_sl)
                    if new_sl < current_sl:
                        sl_adjustment = new_sl
                        math_reasoning.append(f"SL tightened: ${current_sl:.4f} → ${new_sl:.4f}")
            
            # === FINAL DECISION READY ===
            # Grace period safety removed - was causing losses by blocking exits
            
            result = {
                'action': math_action,
                'confidence': final_confidence,
                'math_score': hold_score,
                'exit_score': adjusted_exit_score,
                'ai_validated': ai_validated,
                'math_override': math_override_exit,  # Track if this was a math override
                'reversal_urgency': reversal_urgency if profit_reversal_detected else None,
                'reversal_score': peak_tracking.get('reversal_score', 0),  # NEW: Dynamic score
                'reasoning': '; '.join(math_reasoning),
                'sl_adjustment': sl_adjustment,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'reversal_strength': reversal_strength,
                # NEW: Peak profit tracking for intelligent reversal detection
                'peak_tracking': peak_tracking
            }
            
            override_tag = " ⚡MATH_OVERRIDE" if math_override_exit else ""
            action_emoji = {'hold': '⏸️', 'close': '🚪', 'tighten_sl': '🔒'}.get(math_action, '❓')
            logger.info(f"📊 UNIFIED POSITION: {symbol} {side} | {action_emoji} {math_action.upper()}{override_tag} | Conf: {final_confidence:.0%}")
            logger.info(f"   Hold: {hold_score:.0f} | Exit: {adjusted_exit_score:.0f} | PnL: {pnl_pct:+.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Unified position decision error: {e}")
            return {
                'action': 'hold',
                'confidence': 0.5,
                'math_score': 50,
                'ai_validated': False,
                'reasoning': f'Error: {e}',
                'sl_adjustment': None
            }
    
    def _detect_resistance_support_levels(
        self,
        df: pd.DataFrame,
        current_price: float,
        lookback_bars: int = 96  # ~24 hours on 15m timeframe
    ) -> Dict[str, Any]:
        """
        Detect RESISTANCE and SUPPORT ZONES using 4 key price levels:
        
        RESISTANCE ZONE (Top):
        - resistance_upper = 24h HIGH
        - resistance_lower = 24h HIGH - (ATR * zone_multiplier)
        - If price is BETWEEN these 2 points → DON'T GO LONG
        
        SUPPORT ZONE (Bottom):
        - support_upper = 24h LOW + (ATR * zone_multiplier)
        - support_lower = 24h LOW
        - If price is BETWEEN these 2 points → DON'T GO SHORT
        
        Zone size is calculated dynamically based on market volatility (ATR).
        More volatile markets = wider zones for safety.
        
        Returns:
            {
                'in_resistance_zone': bool,    # Price in top danger zone (no LONG)
                'in_support_zone': bool,       # Price in bottom danger zone (no SHORT)
                'resistance_upper': float,     # Top of resistance zone (24h high)
                'resistance_lower': float,     # Bottom of resistance zone
                'support_upper': float,        # Top of support zone
                'support_lower': float,        # Bottom of support zone (24h low)
                'zone_width_pct': float,       # Zone width as % of price
                ...
            }
        """
        try:
            if df is None or len(df) < 20:
                return {
                    'in_resistance_zone': False,
                    'in_support_zone': False,
                    'at_resistance': False,
                    'at_support': False,
                    'resistance_level': None,
                    'support_level': None,
                    'warning': None
                }
            
            # Get recent highs and lows (last 24 hours ~ 96 bars on 15m)
            recent_df = df.tail(min(lookback_bars, len(df)))
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            closes = recent_df['close'].values
            
            # Find the 24h high and low
            high_24h = highs.max()
            low_24h = lows.min()
            
            # ═══════════════════════════════════════════════════════════════
            # SMART SUPPORT/RESISTANCE DETECTION
            # Find price levels where price has bounced multiple times
            # This is more accurate than simple 24h high/low
            # ═══════════════════════════════════════════════════════════════
            
            # Calculate ATR for clustering tolerance
            if 'atr' in df.columns and len(df) > 0:
                atr = df['atr'].iloc[-1]
            else:
                tr_values = []
                for i in range(1, min(14, len(recent_df))):
                    high = recent_df['high'].iloc[i]
                    low = recent_df['low'].iloc[i]
                    prev_close = recent_df['close'].iloc[i-1]
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    tr_values.append(tr)
                atr = sum(tr_values) / len(tr_values) if tr_values else current_price * 0.02
            
            # Clustering tolerance: prices within this distance are considered "same level"
            cluster_tolerance = atr * 0.5
            
            # ═══════════════════════════════════════════════════════════════
            # FIND RESISTANCE LEVELS (Price highs that got rejected/bounced down)
            # A valid resistance is where price touched HIGH and then DROPPED
            # ═══════════════════════════════════════════════════════════════
            resistance_clusters = []
            for i in range(2, len(recent_df) - 2):
                high = recent_df['high'].iloc[i]
                # Check if this is a local high (swing high)
                is_swing_high = (
                    high > recent_df['high'].iloc[i-1] and
                    high > recent_df['high'].iloc[i-2] and
                    high > recent_df['high'].iloc[i+1] and
                    high > recent_df['high'].iloc[i+2]
                )
                if is_swing_high:
                    # Check if price bounced down after (rejection)
                    next_close = recent_df['close'].iloc[i+1]
                    if next_close < high * 0.995:  # Price dropped at least 0.5%
                        resistance_clusters.append(high)
            
            # ═══════════════════════════════════════════════════════════════
            # FIND SUPPORT LEVELS (Price lows that got bounced up)
            # A valid support is where price touched LOW and then ROSE
            # ═══════════════════════════════════════════════════════════════
            support_clusters = []
            for i in range(2, len(recent_df) - 2):
                low = recent_df['low'].iloc[i]
                # Check if this is a local low (swing low)
                is_swing_low = (
                    low < recent_df['low'].iloc[i-1] and
                    low < recent_df['low'].iloc[i-2] and
                    low < recent_df['low'].iloc[i+1] and
                    low < recent_df['low'].iloc[i+2]
                )
                if is_swing_low:
                    # Check if price bounced up after
                    next_close = recent_df['close'].iloc[i+1]
                    if next_close > low * 1.005:  # Price rose at least 0.5%
                        support_clusters.append(low)
            
            # ═══════════════════════════════════════════════════════════════
            # CLUSTER THE LEVELS (Group nearby bounces into zones)
            # ═══════════════════════════════════════════════════════════════
            def cluster_levels(levels, tolerance):
                """Group nearby price levels and count touches"""
                if not levels:
                    return []
                levels = sorted(levels)
                clusters = []
                current_cluster = [levels[0]]
                
                for level in levels[1:]:
                    if level - current_cluster[-1] <= tolerance:
                        current_cluster.append(level)
                    else:
                        # Save cluster: (avg_price, touch_count)
                        avg = sum(current_cluster) / len(current_cluster)
                        clusters.append({'level': avg, 'touches': len(current_cluster)})
                        current_cluster = [level]
                
                # Don't forget last cluster
                if current_cluster:
                    avg = sum(current_cluster) / len(current_cluster)
                    clusters.append({'level': avg, 'touches': len(current_cluster)})
                
                return clusters
            
            resistance_zones = cluster_levels(resistance_clusters, cluster_tolerance)
            support_zones = cluster_levels(support_clusters, cluster_tolerance)
            
            # ═══════════════════════════════════════════════════════════════
            # SELECT STRONGEST ZONES (Most touches near current price)
            # ═══════════════════════════════════════════════════════════════
            # Filter to zones above current price for resistance
            valid_resistances = [z for z in resistance_zones if z['level'] > current_price]
            # Sort by touches (most touched first), then by distance (closest first)
            valid_resistances.sort(key=lambda z: (-z['touches'], z['level'] - current_price))
            
            # Filter to zones below current price for support
            valid_supports = [z for z in support_zones if z['level'] < current_price]
            # Sort by touches (most touched first), then by distance (closest first)
            valid_supports.sort(key=lambda z: (-z['touches'], current_price - z['level']))
            
            # ═══════════════════════════════════════════════════════════════
            # DETERMINE FINAL ZONE LEVELS
            # Use bounce-based levels if we have enough data, else fall back to 24h high/low
            # ═══════════════════════════════════════════════════════════════
            zone_width = atr * 0.5  # REDUCED: Zone width based on ATR (was 1.5x, now 0.5x)
            zone_width_pct = (zone_width / current_price) * 100 if current_price > 0 else 1.0
            
            # Clamp zone width - REDUCED for tighter zones
            min_zone_pct, max_zone_pct = 0.3, 1.5  # Was 1.0-5.0%, now 0.3-1.5%
            if zone_width_pct < min_zone_pct:
                zone_width = current_price * (min_zone_pct / 100)
                zone_width_pct = min_zone_pct
            elif zone_width_pct > max_zone_pct:
                zone_width = current_price * (max_zone_pct / 100)
                zone_width_pct = max_zone_pct
            
            # RESISTANCE: Use strongest bounce level if we have 3+ touches, else 24h high
            resistance_touches = 0
            if valid_resistances and valid_resistances[0]['touches'] >= 3:  # CHANGED: Require 3+ bounces (was 2)
                resistance_level = valid_resistances[0]['level']
                resistance_touches = valid_resistances[0]['touches']
                logger.debug(f"Using bounce-based resistance: ${resistance_level:.4f} ({resistance_touches} touches)")
            else:
                resistance_level = high_24h
                # Count how many times price touched 24h high
                for h in highs:
                    if abs(h - high_24h) <= cluster_tolerance:
                        resistance_touches += 1
            
            # SUPPORT: Use strongest bounce level if we have 3+ touches, else 24h low
            support_touches = 0
            if valid_supports and valid_supports[0]['touches'] >= 3:  # CHANGED: Require 3+ bounces (was 2)
                support_level = valid_supports[0]['level']
                support_touches = valid_supports[0]['touches']
                logger.debug(f"Using bounce-based support: ${support_level:.4f} ({support_touches} touches)")
            else:
                support_level = low_24h
                # Count how many times price touched 24h low
                for l in lows:
                    if abs(l - low_24h) <= cluster_tolerance:
                        support_touches += 1
            
            # Define the 4 zone boundaries
            resistance_upper = resistance_level
            resistance_lower = resistance_level - zone_width
            support_upper = support_level + zone_width
            support_lower = support_level
            
            # Shadow/Caution zones (1.5x width of danger zone)
            shadow_width = zone_width * 1.5
            resistance_caution_lower = resistance_lower - shadow_width
            support_caution_upper = support_upper + shadow_width
            
            # Check if current price is in the danger zones
            in_resistance_zone = resistance_lower <= current_price <= resistance_upper
            in_support_zone = support_lower <= current_price <= support_upper
            
            # Check if in caution/shadow zones (allowed but reduce size)
            in_resistance_caution = resistance_caution_lower <= current_price < resistance_lower
            in_support_caution = support_upper < current_price <= support_caution_upper
            
            # ═══════════════════════════════════════════════════════════════
            # NOTE: Touch counts already calculated above using smart detection
            # Legacy counting removed to prevent overwriting smart counts
            # resistance_touches and support_touches are preserved from lines 3602-3624
            # ═══════════════════════════════════════════════════════════════
            
            # Calculate distances
            distance_to_resistance = high_24h - current_price
            distance_to_support = current_price - low_24h
            distance_to_resistance_pct = (distance_to_resistance / current_price) * 100 if current_price > 0 else 0
            distance_to_support_pct = (distance_to_support / current_price) * 100 if current_price > 0 else 0
            
            # Position in 24h range
            range_24h = high_24h - low_24h
            if range_24h > 0:
                position_in_range_pct = ((current_price - low_24h) / range_24h) * 100
            else:
                position_in_range_pct = 50
            
            # Legacy flags (for backward compatibility)
            at_24h_high = distance_to_resistance_pct < 0.15
            at_24h_low = distance_to_support_pct < 0.15
            near_top_of_range = position_in_range_pct >= 95
            near_bottom_of_range = position_in_range_pct <= 5
            at_resistance = distance_to_resistance_pct < 0.5 and resistance_touches >= 2
            at_support = distance_to_support_pct < 0.5 and support_touches >= 2
            
            # ═══════════════════════════════════════════════════════════════
            # GENERATE WARNING MESSAGE
            # ═══════════════════════════════════════════════════════════════
            warning = None
            if in_resistance_zone:
                warning = f"🚫 IN RESISTANCE ZONE [${resistance_lower:.4f} - ${resistance_upper:.4f}] | Zone={zone_width_pct:.1f}% | NO LONG - will likely drop!"
            elif in_support_zone:
                warning = f"🚫 IN SUPPORT ZONE [${support_lower:.4f} - ${support_upper:.4f}] | Zone={zone_width_pct:.1f}% | NO SHORT - will likely bounce!"
            elif in_resistance_caution:
                warning = f"⚠️ CAUTION ZONE (near resistance) - reduce position size"
            elif in_support_caution:
                warning = f"⚠️ CAUTION ZONE (near support) - reduce position size"
            elif distance_to_resistance_pct < zone_width_pct * 1.5:
                warning = f"⚠️ APPROACHING RESISTANCE ({distance_to_resistance_pct:.1f}% away from zone)"
            elif distance_to_support_pct < zone_width_pct * 1.5:
                warning = f"⚠️ APPROACHING SUPPORT ({distance_to_support_pct:.1f}% away from zone)"
            
            return {
                # NEW: Zone-based detection
                'in_resistance_zone': in_resistance_zone,
                'in_support_zone': in_support_zone,
                'in_resistance_caution': in_resistance_caution,  # NEW: Caution zone
                'in_support_caution': in_support_caution,        # NEW: Caution zone
                'resistance_upper': resistance_upper,
                'resistance_lower': resistance_lower,
                'support_upper': support_upper,
                'support_lower': support_lower,
                'zone_width': zone_width,
                'zone_width_pct': zone_width_pct,
                # Legacy fields
                'at_resistance': at_resistance or in_resistance_zone,
                'at_support': at_support or in_support_zone,
                'at_24h_high': at_24h_high,
                'at_24h_low': at_24h_low,
                'near_top_of_range': near_top_of_range or in_resistance_zone,
                'near_bottom_of_range': near_bottom_of_range or in_support_zone,
                'position_in_range_pct': position_in_range_pct,
                'resistance_level': high_24h,
                'support_level': low_24h,
                'resistance_touches': resistance_touches,
                'support_touches': support_touches,
                'distance_to_resistance_pct': distance_to_resistance_pct,
                'distance_to_support_pct': distance_to_support_pct,
                'warning': warning
            }
            
        except Exception as e:
            logger.debug(f"Resistance/support detection error: {e}")
            return {
                'in_resistance_zone': False,
                'in_support_zone': False,
                'at_resistance': False,
                'at_support': False,
                'resistance_level': None,
                'support_level': None,
                'warning': None
            }
    
    def _comprehensive_math_check(
        self,
        signal: int,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        PhD-Level Comprehensive Mathematical Analysis of Trade Setup.
        
        Implements advanced quantitative methods:
        - Statistical hypothesis testing
        - Bayesian probability updates  
        - Time series analysis (Hurst, autocorrelation)
        - Risk metrics (VaR, Sharpe, Sortino, Calmar)
        - Information theory (entropy-based uncertainty)
        - Regression analysis (price momentum slopes)
        - Order flow imbalance estimation
        - Fractal market analysis
        
        Returns objective scores that can override AI decisions.
        """
        scores = {}
        reasons_for = []
        reasons_against = []
        detailed_analysis = {}
        
        # Get price and return series
        closes = df['close'] if 'close' in df.columns else pd.Series([current_price])
        returns = closes.pct_change().dropna()
        
        # ═══════════════════════════════════════════════════════════════════
        # 0. SOFT PENALTY: OVERBOUGHT LONGS / OVERSOLD SHORTS
        # ═══════════════════════════════════════════════════════════════════
        # These conditions apply heavy penalties but don't hard-block.
        # Math evaluates, AI sees the score, combined decision is made.
        # This allows AI recommendation + math scoring to work together.
        if 'rsi' in df.columns:
            current_rsi = df['rsi'].iloc[-1] if len(df) > 0 else 50
        else:
            # Calculate RSI if not present
            from indicator import calculate_rsi
            rsi_series = calculate_rsi(closes)
            current_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50
        
        rsi_penalty = 0
        rsi_warning = None
        
        # SOFT PENALTY for overbought LONG (was hard block at 75)
        if signal == 1 and current_rsi >= 70:
            if current_rsi >= 80:
                rsi_penalty = -50  # Extreme overbought: massive penalty
                rsi_warning = f"⚠️ EXTREME OVERBOUGHT: RSI={current_rsi:.0f} (≥80) - Very high reversal risk"
            elif current_rsi >= 75:
                rsi_penalty = -35  # Very overbought: heavy penalty
                rsi_warning = f"⚠️ OVERBOUGHT: RSI={current_rsi:.0f} (≥75) - High reversal risk"
            else:  # 70-74
                rsi_penalty = -20  # Moderately overbought: moderate penalty
                rsi_warning = f"⚠️ RSI elevated: {current_rsi:.0f} (≥70) - Caution for LONG"
            logger.debug(f"📊 [SOFT PENALTY] LONG gets RSI penalty: {rsi_penalty} (RSI={current_rsi:.1f})")
            reasons_against.append(rsi_warning)
            detailed_analysis['rsi_penalty'] = rsi_penalty
        
        # SOFT PENALTY for oversold SHORT (was hard block at 25)
        # Moderate penalties - between harsh original and weak test
        elif signal == -1 and current_rsi <= 30:
            if current_rsi <= 20:
                rsi_penalty = -35  # Moderate: Was -50 original, -20 test
                rsi_warning = f"⚠️ EXTREME OVERSOLD: RSI={current_rsi:.0f} (≤20) - Very high bounce risk"
            elif current_rsi <= 25:
                rsi_penalty = -25  # Moderate: Was -35 original, -15 test
                rsi_warning = f"⚠️ OVERSOLD: RSI={current_rsi:.0f} (≤25) - High bounce risk"
            else:  # 26-30
                rsi_penalty = -15  # Moderate: Was -20 original, -10 test
                rsi_warning = f"⚠️ RSI depressed: {current_rsi:.0f} (≤30) - Caution for SHORT"
            logger.debug(f"📊 [SOFT PENALTY] SHORT gets RSI penalty: {rsi_penalty} (RSI={current_rsi:.1f})")
            reasons_against.append(rsi_warning)
            detailed_analysis['rsi_penalty'] = rsi_penalty
        
        detailed_analysis['rsi'] = current_rsi
        
        # ═══════════════════════════════════════════════════════════════════
        # 0b. SOFT PENALTY: COUNTER-TREND TRADES IN STRONG TRENDS
        # ═══════════════════════════════════════════════════════════════════
        # Apply penalties but allow math to evaluate the full picture.
        # AI + Math combined can make better decisions than hard blocks.
        sma_15 = closes.rolling(15).mean()
        sma_40 = closes.rolling(40).mean()
        
        trend_penalty = 0
        trend_warning = None
        
        if len(sma_15) >= 40 and len(sma_40) >= 40:
            trend_bullish = sma_15.iloc[-1] > sma_40.iloc[-1]
            trend_bearish = sma_15.iloc[-1] < sma_40.iloc[-1]
            
            # Calculate trend strength (how far apart are the SMAs)
            trend_strength = abs((sma_15.iloc[-1] - sma_40.iloc[-1]) / sma_40.iloc[-1] * 100)
            detailed_analysis['trend_strength'] = trend_strength
            
            # SOFT PENALTY: Don't SHORT in a BULLISH trend (reduced penalties)
            if signal == -1 and trend_bullish and trend_strength > 0.5:
                if trend_strength > 2.0:
                    trend_penalty = -25  # Very strong bullish trend
                    trend_warning = f"⚠️ STRONG UPTREND: SMA15 > SMA40 by {trend_strength:.2f}% - SHORT risky"
                elif trend_strength > 1.0:
                    trend_penalty = -15  # Moderate bullish trend
                    trend_warning = f"⚠️ UPTREND: SMA15 > SMA40 by {trend_strength:.2f}% - SHORT caution"
                else:  # 0.5-1.0
                    trend_penalty = -8  # Weak bullish trend
                    trend_warning = f"⚠️ Mild uptrend: SMA15 > SMA40 by {trend_strength:.2f}%"
                logger.debug(f"📊 [SOFT PENALTY] SHORT gets trend penalty: {trend_penalty} (trend={trend_strength:.2f}%)")
                reasons_against.append(trend_warning)
                detailed_analysis['trend_penalty'] = trend_penalty
                detailed_analysis['trend'] = 'bullish'
            
            # SOFT PENALTY: Don't LONG in a BEARISH trend (reduced penalties)
            elif signal == 1 and trend_bearish and trend_strength > 0.5:
                if trend_strength > 2.0:
                    trend_penalty = -25  # Very strong bearish trend
                    trend_warning = f"⚠️ STRONG DOWNTREND: SMA15 < SMA40 by {trend_strength:.2f}% - LONG risky"
                elif trend_strength > 1.0:
                    trend_penalty = -15  # Moderate bearish trend
                    trend_warning = f"⚠️ DOWNTREND: SMA15 < SMA40 by {trend_strength:.2f}% - LONG caution"
                else:  # 0.5-1.0
                    trend_penalty = -8  # Weak bearish trend
                    trend_warning = f"⚠️ Mild downtrend: SMA15 < SMA40 by {trend_strength:.2f}%"
                logger.debug(f"📊 [SOFT PENALTY] LONG gets trend penalty: {trend_penalty} (trend={trend_strength:.2f}%)")
                reasons_against.append(trend_warning)
                detailed_analysis['trend_penalty'] = trend_penalty
                detailed_analysis['trend'] = 'bearish'
        
        # Store penalties to apply at the end
        detailed_analysis['total_soft_penalty'] = rsi_penalty + trend_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. RISK-REWARD OPTIMIZATION & HYPOTHESIS TESTING (PhD Enhancement)
        # ═══════════════════════════════════════════════════════════════════
        stop_distance = atr * 1.5
        tp1_distance = atr * 1.5
        tp2_distance = atr * 2.5
        tp3_distance = atr * 4.0
        
        # Expected R:R using probability-weighted payoff
        # P(TP1) = 0.50, P(TP2|TP1) = 0.60, P(TP3|TP2) = 0.40
        # This is more realistic than simple averaging
        p_tp1 = 0.50
        p_tp2_given_tp1 = 0.60
        p_tp3_given_tp2 = 0.40
        
        expected_profit = (
            p_tp1 * tp1_distance +
            p_tp1 * p_tp2_given_tp1 * (tp2_distance - tp1_distance) +
            p_tp1 * p_tp2_given_tp1 * p_tp3_given_tp2 * (tp3_distance - tp2_distance)
        )
        expected_loss = (1 - p_tp1) * stop_distance
        expected_rr = expected_profit / stop_distance if stop_distance > 0 else 1
        
        # Kelly optimal fraction: f* = (p*b - q) / b where b = win/loss ratio
        win_prob = p_tp1
        win_loss_ratio = expected_profit / expected_loss if expected_loss > 0 else 1
        kelly_f = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_f = max(0, min(0.25, kelly_f))  # Cap at 25%
        
        scores['risk_reward'] = min(100, expected_rr * 40 + kelly_f * 200)
        detailed_analysis['kelly_fraction'] = kelly_f
        detailed_analysis['expected_rr'] = expected_rr
        
        # PhD Enhancement: Test if returns have statistically significant edge
        try:
            from indicator import calculate_hypothesis_test
            
            hypo_result = calculate_hypothesis_test(returns, null_mean=0)
            detailed_analysis['hypothesis_test'] = hypo_result
            
            p_value = hypo_result.get('p_value', 1.0)
            significant = hypo_result.get('significant', False)
            mean_return = hypo_result.get('mean_return', 0)
            
            if significant and mean_return > 0:
                reasons_for.append(f"Statistically significant edge (p={p_value:.4f}, mean={mean_return:.4f}%)")
                scores['risk_reward'] = min(100, scores['risk_reward'] * 1.2)
            elif significant and mean_return < 0:
                reasons_against.append(f"Significant negative returns (p={p_value:.4f})")
                scores['risk_reward'] = max(0, scores['risk_reward'] * 0.7)
            else:
                # No statistical edge - this goes AGAINST the trade, not for it
                reasons_against.append(f"No significant edge yet (p={p_value:.3f})")
                scores['risk_reward'] = max(0, scores['risk_reward'] * 0.85)  # Penalize score
                detailed_analysis['no_statistical_edge'] = True
        except Exception as e:
            logger.debug(f"Hypothesis test failed: {e}")
        
        if expected_rr >= 1.8:
            reasons_for.append(f"Excellent R:R ({expected_rr:.2f}:1), Kelly f*={kelly_f:.1%}")
        elif expected_rr >= 1.2:
            reasons_for.append(f"Good R:R ({expected_rr:.2f}:1)")
        else:
            reasons_against.append(f"Poor R:R ({expected_rr:.2f}:1)")
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. TREND STRENGTH VIA LINEAR REGRESSION SLOPE & R²
        # ═══════════════════════════════════════════════════════════════════
        if len(closes) >= 20:
            # Fit OLS regression to log prices (better for exponential trends)
            log_prices = np.log(closes.tail(20).values)
            x = np.arange(len(log_prices))
            
            # Calculate regression coefficients
            x_mean = x.mean()
            y_mean = log_prices.mean()
            
            numerator = np.sum((x - x_mean) * (log_prices - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                
                # Calculate R² (coefficient of determination)
                y_pred = slope * x + intercept
                ss_res = np.sum((log_prices - y_pred) ** 2)
                ss_tot = np.sum((log_prices - y_mean) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Annualized slope (assuming 15-min bars)
                bars_per_year = 365 * 24 * 4  # 4 bars per hour
                # Clip to prevent overflow (max reasonable annual return is ~10000%)
                exp_arg = np.clip(slope * bars_per_year, -10, 10)
                annualized_return = (np.exp(exp_arg) - 1) * 100
                
                detailed_analysis['regression_slope'] = slope
                detailed_analysis['r_squared'] = r_squared
                detailed_analysis['annualized_trend'] = annualized_return
                
                # Score based on slope direction matching signal AND R²
                trend_matches = (signal == 1 and slope > 0) or (signal == -1 and slope < 0)
                slope_strength = abs(slope) * 10000  # Scale for scoring
                
                if trend_matches:
                    scores['trend'] = min(100, 50 + slope_strength * 50 + r_squared * 30)
                    if r_squared > 0.7:
                        reasons_for.append(f"Strong trend (R²={r_squared:.2f}, ann. {annualized_return:+.0f}%)")
                    elif r_squared > 0.4:
                        reasons_for.append(f"Moderate trend (R²={r_squared:.2f})")
                else:
                    scores['trend'] = max(0, 50 - slope_strength * 50)
                    reasons_against.append(f"Counter-trend trade (R²={r_squared:.2f})")
            else:
                scores['trend'] = 50
        else:
            scores['trend'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. MARKET REGIME DETECTION - HIDDEN MARKOV MODEL (PhD Enhancement)
        # ═══════════════════════════════════════════════════════════════════
        try:
            from indicator import calculate_regime_hmm
            
            hmm_result = calculate_regime_hmm(returns)
            if hmm_result.get('status') == 'success':
                regime = hmm_result.get('regime', 'NORMAL')
                regime_confidence = hmm_result.get('confidence', 0.5)
                persistence = hmm_result.get('persistence', 0.5)
                
                detailed_analysis['market_regime'] = regime
                detailed_analysis['regime_confidence'] = regime_confidence
                detailed_analysis['regime_persistence'] = persistence
                
                # Use HMM regime instead of Hurst for scoring
                if regime == 'CALM' and signal == 1:
                    scores['hurst'] = 85
                    reasons_for.append(f"HMM: CALM regime (conf={regime_confidence:.0%}) - good for breakout LONG")
                elif regime == 'CALM' and signal == -1:
                    scores['hurst'] = 50
                    reasons_against.append(f"HMM: CALM regime - SHORT less favorable")
                elif regime == 'VOLATILE' and signal == 1:
                    scores['hurst'] = 40
                    reasons_against.append(f"HMM: VOLATILE regime - risky LONG")
                elif regime == 'VOLATILE' and signal == -1:
                    scores['hurst'] = 80
                    reasons_for.append(f"HMM: VOLATILE regime (conf={regime_confidence:.0%}) - good for SHORT")
                else:
                    scores['hurst'] = 70
                    reasons_for.append(f"HMM: {regime} regime (persistence={persistence:.0%})")
            else:
                # Fallback to Hurst calculation
                hurst = self._calculate_hurst(closes)
                detailed_analysis['hurst_exponent'] = hurst
                
                if hurst > 0.6:
                    scores['hurst'] = 90
                    reasons_for.append(f"Trending market (H={hurst:.2f}>0.5)")
                elif hurst < 0.4:
                    scores['hurst'] = 70
                    reasons_for.append(f"Mean-reverting (H={hurst:.2f})")
                else:
                    scores['hurst'] = 60
        except Exception as e:
            logger.warning(f"HMM regime detection failed: {e}")
            # Fallback to Hurst
            hurst = self._calculate_hurst(closes)
            detailed_analysis['hurst_exponent'] = hurst
            scores['hurst'] = 80 if hurst > 0.55 else 60 if hurst > 0.45 else 70
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. VOLATILITY CLUSTERING & GARCH-STYLE ANALYSIS (PhD Enhancement)
        # ═══════════════════════════════════════════════════════════════════
        try:
            # Import GARCH function from indicator.py
            from indicator import calculate_garch_volatility
            
            garch_result = calculate_garch_volatility(returns)
            if garch_result.get('status') == 'success':
                current_vol = garch_result.get('current_vol', 0)
                forecast_vol = garch_result.get('forecast_vol', 0)
                vol_trend = garch_result.get('vol_trend', 'unknown')
                vol_change = garch_result.get('vol_change_pct', 0)
                
                detailed_analysis['garch_vol'] = current_vol
                detailed_analysis['garch_forecast'] = forecast_vol
                detailed_analysis['vol_trend'] = vol_trend
                detailed_analysis['vol_change_pct'] = vol_change
                
                # Score based on volatility regime and forecast
                if vol_trend == 'decreasing' and 0.7 <= current_vol / forecast_vol <= 1.3:
                    scores['volatility'] = 90
                    reasons_for.append(f"GARCH: Stable vol (curr={current_vol:.2%}, forecast {vol_trend})")
                elif vol_trend == 'increasing':
                    scores['volatility'] = 40 + vol_change / 10
                    reasons_against.append(f"GARCH: Vol expanding (+{vol_change:.1f}%) - rising risk")
                else:
                    scores['volatility'] = 75
                    reasons_for.append(f"GARCH: Vol contracting ({vol_change:.1f}%)")
            else:
                # Fallback to simple volatility if GARCH fails
                short_vol = returns.tail(10).std() * np.sqrt(252 * 24 * 4)
                long_vol = returns.tail(30).std() * np.sqrt(252 * 24 * 4)
                vol_ratio = short_vol / long_vol if long_vol > 0 else 1
                
                detailed_analysis['short_vol'] = short_vol
                detailed_analysis['long_vol'] = long_vol
                detailed_analysis['vol_ratio'] = vol_ratio
                
                if 0.7 <= vol_ratio <= 1.3:
                    scores['volatility'] = 85
                else:
                    scores['volatility'] = 70
        except Exception as e:
            logger.warning(f"GARCH analysis failed: {e}")
            # Simple fallback
            short_vol = returns.tail(10).std() * np.sqrt(252 * 24 * 4)
            long_vol = returns.tail(30).std() * np.sqrt(252 * 24 * 4)
            vol_ratio = short_vol / long_vol if long_vol > 0 else 1
            scores['volatility'] = 75 if 0.7 <= vol_ratio <= 1.3 else 60
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. Z-SCORE MEAN REVERSION ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        if len(closes) >= 50:
            rolling_mean = closes.rolling(20).mean()
            rolling_std = closes.rolling(20).std()
            
            z_score = (current_price - rolling_mean.iloc[-1]) / rolling_std.iloc[-1] if rolling_std.iloc[-1] > 0 else 0
            detailed_analysis['z_score'] = z_score
            
            # Extreme z-scores suggest mean reversion
            if signal == 1:  # LONG
                if z_score < -2:
                    scores['zscore'] = 95
                    reasons_for.append(f"Extreme oversold (z={z_score:.2f}σ) - high reversion probability")
                elif z_score < -1:
                    scores['zscore'] = 80
                    reasons_for.append(f"Oversold (z={z_score:.2f}σ)")
                elif z_score > 2:
                    scores['zscore'] = 30
                    reasons_against.append(f"Overbought (z={z_score:.2f}σ) - risky LONG")
                else:
                    scores['zscore'] = 60
            else:  # SHORT
                if z_score > 2:
                    scores['zscore'] = 95
                    reasons_for.append(f"Extreme overbought (z={z_score:.2f}σ) - high reversion probability")
                elif z_score > 1:
                    scores['zscore'] = 80
                    reasons_for.append(f"Overbought (z={z_score:.2f}σ)")
                elif z_score < -2:
                    scores['zscore'] = 30
                    reasons_against.append(f"Oversold (z={z_score:.2f}σ) - risky SHORT")
                else:
                    scores['zscore'] = 60
        else:
            scores['zscore'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 6. ENHANCED RSI DIVERGENCE DETECTION (Multi-Timeframe)
        # ═══════════════════════════════════════════════════════════════════
        if 'rsi' in df.columns and len(df) >= 20:
            rsi = df['rsi'].iloc[-1]
            rsi_prev = df['rsi'].iloc[-10] if len(df) >= 10 else rsi
            price_change = (current_price / closes.iloc[-10] - 1) * 100 if len(closes) >= 10 else 0
            rsi_change = rsi - rsi_prev
            
            # Basic divergence detection (fallback)
            bullish_divergence = price_change < 0 and rsi_change > 5
            bearish_divergence = price_change > 0 and rsi_change < -5
            
            detailed_analysis['rsi'] = rsi
            detailed_analysis['bullish_divergence'] = bullish_divergence
            detailed_analysis['bearish_divergence'] = bearish_divergence
            
            # === ENHANCED: Multi-Timeframe RSI Divergence ===
            try:
                from indicator import calculate_rsi_divergence_mtf
                
                # Get higher timeframe data from context if available
                df_higher = context.get('df_higher_tf', None)
                div_result = calculate_rsi_divergence_mtf(df, df_higher)
                
                if div_result.get('status') == 'success':
                    reg_bullish = div_result['regular_bullish_divergence']
                    reg_bearish = div_result['regular_bearish_divergence']
                    hid_bullish = div_result['hidden_bullish_divergence']
                    hid_bearish = div_result['hidden_bearish_divergence']
                    mtf_confirm = div_result['mtf_confirmation']
                    div_strength = div_result['signal_confidence']
                    
                    detailed_analysis['rsi_divergence_mtf'] = div_result
                    
                    # Override basic divergence with enhanced detection
                    bullish_divergence = reg_bullish or hid_bullish
                    bearish_divergence = reg_bearish or hid_bearish
                    
                    if signal == 1:  # LONG
                        if reg_bullish:
                            mtf_bonus = 10 if mtf_confirm else 0
                            scores['momentum'] = min(100, 90 + mtf_bonus)
                            mtf_str = " (MTF confirmed!)" if mtf_confirm else ""
                            reasons_for.append(f"🔄 Regular bullish RSI divergence{mtf_str} (RSI={rsi:.0f}, strength={div_strength:.0f})")
                        elif hid_bullish:
                            scores['momentum'] = 85
                            reasons_for.append(f"🔄 Hidden bullish divergence - continuation signal (RSI={rsi:.0f})")
                        elif 30 <= rsi <= 50:
                            scores['momentum'] = 80
                            reasons_for.append(f"RSI {rsi:.0f} - optimal LONG zone")
                        elif rsi > 70:
                            scores['momentum'] = 35
                            reasons_against.append(f"RSI {rsi:.0f} - overbought")
                        elif bearish_divergence:
                            scores['momentum'] = 40
                            reasons_against.append(f"⚠️ Bearish RSI divergence conflicts with LONG (RSI={rsi:.0f})")
                        else:
                            scores['momentum'] = 65
                    else:  # SHORT
                        if reg_bearish:
                            mtf_bonus = 10 if mtf_confirm else 0
                            scores['momentum'] = min(100, 90 + mtf_bonus)
                            mtf_str = " (MTF confirmed!)" if mtf_confirm else ""
                            reasons_for.append(f"🔄 Regular bearish RSI divergence{mtf_str} (RSI={rsi:.0f}, strength={div_strength:.0f})")
                        elif hid_bearish:
                            scores['momentum'] = 85
                            reasons_for.append(f"🔄 Hidden bearish divergence - continuation signal (RSI={rsi:.0f})")
                        elif 50 <= rsi <= 70:
                            scores['momentum'] = 80
                            reasons_for.append(f"RSI {rsi:.0f} - optimal SHORT zone")
                        elif rsi < 30:
                            scores['momentum'] = 35
                            reasons_against.append(f"RSI {rsi:.0f} - oversold")
                        elif bullish_divergence:
                            scores['momentum'] = 40
                            reasons_against.append(f"⚠️ Bullish RSI divergence conflicts with SHORT (RSI={rsi:.0f})")
                        else:
                            scores['momentum'] = 65
                else:
                    # Fallback to basic divergence
                    if signal == 1:
                        if bullish_divergence:
                            scores['momentum'] = 95
                            reasons_for.append(f"Bullish RSI divergence detected (RSI={rsi:.0f})")
                        elif 30 <= rsi <= 50:
                            scores['momentum'] = 85
                            reasons_for.append(f"RSI {rsi:.0f} - optimal LONG zone")
                        elif rsi > 70:
                            scores['momentum'] = 35
                            reasons_against.append(f"RSI {rsi:.0f} - overbought")
                        else:
                            scores['momentum'] = 65
                    else:
                        if bearish_divergence:
                            scores['momentum'] = 95
                            reasons_for.append(f"Bearish RSI divergence detected (RSI={rsi:.0f})")
                        elif 50 <= rsi <= 70:
                            scores['momentum'] = 85
                            reasons_for.append(f"RSI {rsi:.0f} - optimal SHORT zone")
                        elif rsi < 30:
                            scores['momentum'] = 35
                            reasons_against.append(f"RSI {rsi:.0f} - oversold")
                        else:
                            scores['momentum'] = 65
                            
            except Exception as div_error:
                logger.debug(f"MTF RSI divergence failed: {div_error}")
                # Fallback to basic logic
                if signal == 1:
                    if bullish_divergence:
                        scores['momentum'] = 95
                        reasons_for.append(f"Bullish RSI divergence detected (RSI={rsi:.0f})")
                    elif 30 <= rsi <= 50:
                        scores['momentum'] = 85
                        reasons_for.append(f"RSI {rsi:.0f} - optimal LONG zone")
                    elif rsi > 70:
                        scores['momentum'] = 35
                        reasons_against.append(f"RSI {rsi:.0f} - overbought")
                    else:
                        scores['momentum'] = 65
                else:
                    if bearish_divergence:
                        scores['momentum'] = 95
                        reasons_for.append(f"Bearish RSI divergence detected (RSI={rsi:.0f})")
                    elif 50 <= rsi <= 70:
                        scores['momentum'] = 85
                        reasons_for.append(f"RSI {rsi:.0f} - optimal SHORT zone")
                    elif rsi < 30:
                        scores['momentum'] = 35
                        reasons_against.append(f"RSI {rsi:.0f} - oversold")
                    else:
                        scores['momentum'] = 65
        else:
            scores['momentum'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 7. ADVANCED VOLUME PROFILE ANALYSIS (TPO-style with POC)
        # ═══════════════════════════════════════════════════════════════════
        volume_ratio = context.get('volume_ratio', 1.0)
        
        if 'volume' in df.columns and len(df) >= 20:
            volumes = df['volume'].tail(20)
            vol_mean = volumes.mean()
            vol_std = volumes.std()
            current_vol = volumes.iloc[-1]
            
            # Volume z-score
            vol_zscore = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0
            detailed_analysis['volume_zscore'] = vol_zscore
            
            # Calculate approximate VWAP
            if 'close' in df.columns:
                typical_price = (df['high'].tail(20) + df['low'].tail(20) + df['close'].tail(20)) / 3
                vwap = (typical_price * volumes).sum() / volumes.sum()
                vwap_distance = (current_price - vwap) / vwap * 100
                detailed_analysis['vwap_distance'] = vwap_distance
                
                # === ENHANCED: Full Volume Profile with POC ===
                try:
                    from indicator import calculate_volume_profile
                    vp_result = calculate_volume_profile(df)
                    
                    if vp_result.get('status') == 'success':
                        poc_price = vp_result['poc_price']
                        va_high = vp_result['value_area_high']
                        va_low = vp_result['value_area_low']
                        in_value_area = vp_result['in_value_area']
                        price_vs_poc = vp_result['current_vs_poc_pct']
                        support_distance = vp_result['support_distance_pct']
                        resistance_distance = vp_result['resistance_distance_pct']
                        
                        detailed_analysis['poc_price'] = poc_price
                        detailed_analysis['value_area_high'] = va_high
                        detailed_analysis['value_area_low'] = va_low
                        detailed_analysis['in_value_area'] = in_value_area
                        detailed_analysis['price_vs_poc'] = price_vs_poc
                        detailed_analysis['nearest_support'] = vp_result['nearest_support']
                        detailed_analysis['nearest_resistance'] = vp_result['nearest_resistance']
                        
                        # Volume Profile Scoring:
                        # - LONG: Better if near support (POC below price) with room to resistance
                        # - SHORT: Better if near resistance (POC above price) with room to support
                        if signal == 1:  # LONG
                            if price_vs_poc < -0.5 and support_distance < 1.0:
                                # Price below POC, near support - excellent for LONG
                                scores['volume'] = min(100, 85 + vol_zscore * 5)
                                reasons_for.append(f"VP: Price below POC ({price_vs_poc:.1f}%), near support, VA:{va_low:.0f}-{va_high:.0f}")
                            elif in_value_area and current_price > vwap:
                                # In value area above VWAP - good for LONG
                                scores['volume'] = min(100, 75 + vol_zscore * 5)
                                reasons_for.append(f"VP: In value area, above VWAP ({vwap_distance:+.2f}%)")
                            elif price_vs_poc > 1.0:
                                # Far above POC - risky LONG, may revert
                                scores['volume'] = max(30, 50 - abs(price_vs_poc) * 5)
                                reasons_against.append(f"VP: Extended above POC ({price_vs_poc:.1f}%) - mean reversion risk")
                            else:
                                scores['volume'] = 60 + vol_zscore * 5
                        else:  # SHORT
                            if price_vs_poc > 0.5 and resistance_distance < 1.0:
                                # Price above POC, near resistance - excellent for SHORT
                                scores['volume'] = min(100, 85 + vol_zscore * 5)
                                reasons_for.append(f"VP: Price above POC ({price_vs_poc:.1f}%), near resistance, VA:{va_low:.0f}-{va_high:.0f}")
                            elif in_value_area and current_price < vwap:
                                # In value area below VWAP - good for SHORT
                                scores['volume'] = min(100, 75 + vol_zscore * 5)
                                reasons_for.append(f"VP: In value area, below VWAP ({vwap_distance:.2f}%)")
                            elif price_vs_poc < -1.0:
                                # Far below POC - risky SHORT, may bounce
                                scores['volume'] = max(30, 50 - abs(price_vs_poc) * 5)
                                reasons_against.append(f"VP: Extended below POC ({price_vs_poc:.1f}%) - bounce risk")
                            else:
                                scores['volume'] = 60 + vol_zscore * 5
                    else:
                        # Fallback to basic VWAP analysis
                        if signal == 1 and current_price > vwap:
                            scores['volume'] = min(100, 70 + vol_zscore * 10)
                            reasons_for.append(f"Price above VWAP (+{vwap_distance:.2f}%), volume z={vol_zscore:.1f}")
                        elif signal == -1 and current_price < vwap:
                            scores['volume'] = min(100, 70 + vol_zscore * 10)
                            reasons_for.append(f"Price below VWAP ({vwap_distance:.2f}%), volume z={vol_zscore:.1f}")
                        else:
                            scores['volume'] = 50
                except Exception as vp_error:
                    logger.debug(f"Volume Profile calculation failed: {vp_error}")
                    # Fallback to basic VWAP
                    if signal == 1 and current_price > vwap:
                        scores['volume'] = min(100, 70 + vol_zscore * 10)
                        reasons_for.append(f"Price above VWAP (+{vwap_distance:.2f}%), volume z={vol_zscore:.1f}")
                    elif signal == -1 and current_price < vwap:
                        scores['volume'] = min(100, 70 + vol_zscore * 10)
                        reasons_for.append(f"Price below VWAP ({vwap_distance:.2f}%), volume z={vol_zscore:.1f}")
                    else:
                        scores['volume'] = 50
            else:
                scores['volume'] = 50 + volume_ratio * 25
        else:
            if volume_ratio >= 1.5:
                scores['volume'] = 90
                reasons_for.append(f"Strong volume confirmation ({volume_ratio:.1f}x)")
            elif volume_ratio >= 1.0:
                scores['volume'] = 70
            else:
                scores['volume'] = 40
                reasons_against.append(f"Weak volume ({volume_ratio:.1f}x)")
        
        # ═══════════════════════════════════════════════════════════════════
        # 8. KALMAN FILTER SMOOTHED MOMENTUM ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        try:
            from indicator import calculate_kalman_momentum
            
            kalman_result = calculate_kalman_momentum(closes)
            
            if kalman_result.get('status') == 'success':
                kalman_momentum = kalman_result['kalman_momentum']
                kalman_accel = kalman_result['kalman_acceleration']
                momentum_zscore = kalman_result['momentum_zscore']
                momentum_conf = kalman_result['momentum_confidence']
                momentum_trend = kalman_result['momentum_trend']
                signal_strength = kalman_result['signal_strength']
                
                detailed_analysis['kalman_momentum'] = kalman_momentum
                detailed_analysis['kalman_acceleration'] = kalman_accel
                detailed_analysis['kalman_momentum_zscore'] = momentum_zscore
                detailed_analysis['kalman_momentum_trend'] = momentum_trend
                detailed_analysis['kalman_signal_strength'] = signal_strength
                
                # Kalman Momentum Scoring:
                # - Positive momentum + strengthening = good for LONG
                # - Negative momentum + strengthening = good for SHORT
                # - High signal strength = high confidence
                
                if signal == 1:  # LONG
                    if kalman_momentum > 0 and momentum_trend == 'strengthening':
                        kalman_score = min(100, 75 + signal_strength * 15)
                        reasons_for.append(f"⚡ Kalman: Bullish momentum strengthening (z={momentum_zscore:.1f}, conf={momentum_conf:.0%})")
                    elif kalman_momentum > 0:
                        kalman_score = 70 + signal_strength * 10
                        reasons_for.append(f"⚡ Kalman: Positive momentum (z={momentum_zscore:.1f})")
                    elif kalman_momentum < 0 and momentum_trend == 'weakening':
                        kalman_score = 60  # Bearish momentum weakening - potential reversal
                        reasons_for.append(f"⚡ Kalman: Bearish momentum weakening - potential reversal")
                    elif kalman_momentum < 0 and momentum_trend == 'strengthening':
                        kalman_score = 30  # Bearish momentum strengthening - bad for LONG
                        reasons_against.append(f"⚡ Kalman: Bearish momentum strengthening (z={momentum_zscore:.1f})")
                    else:
                        kalman_score = 50
                else:  # SHORT
                    if kalman_momentum < 0 and momentum_trend == 'strengthening':
                        kalman_score = min(100, 75 + signal_strength * 15)
                        reasons_for.append(f"⚡ Kalman: Bearish momentum strengthening (z={momentum_zscore:.1f}, conf={momentum_conf:.0%})")
                    elif kalman_momentum < 0:
                        kalman_score = 70 + signal_strength * 10
                        reasons_for.append(f"⚡ Kalman: Negative momentum (z={momentum_zscore:.1f})")
                    elif kalman_momentum > 0 and momentum_trend == 'weakening':
                        kalman_score = 60  # Bullish momentum weakening - potential reversal
                        reasons_for.append(f"⚡ Kalman: Bullish momentum weakening - potential reversal")
                    elif kalman_momentum > 0 and momentum_trend == 'strengthening':
                        kalman_score = 30  # Bullish momentum strengthening - bad for SHORT
                        reasons_against.append(f"⚡ Kalman: Bullish momentum strengthening (z={momentum_zscore:.1f})")
                    else:
                        kalman_score = 50
                
                scores['kalman'] = kalman_score
            else:
                scores['kalman'] = 50
        except Exception as kalman_error:
            logger.debug(f"Kalman momentum failed: {kalman_error}")
            scores['kalman'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 9. AUTOCORRELATION ANALYSIS (Predictability)
        # ═══════════════════════════════════════════════════════════════════
        if len(returns) >= 30:
            # Lag-1 autocorrelation
            autocorr_1 = returns.autocorr(lag=1) if hasattr(returns, 'autocorr') else 0
            autocorr_1 = autocorr_1 if not np.isnan(autocorr_1) else 0
            detailed_analysis['autocorrelation'] = autocorr_1
            
            # Positive autocorr = momentum, Negative = mean reversion
            if abs(autocorr_1) > 0.1:
                if autocorr_1 > 0:
                    scores['autocorr'] = 80
                    reasons_for.append(f"Momentum persistence (ρ={autocorr_1:.2f})")
                else:
                    scores['autocorr'] = 75
                    reasons_for.append(f"Mean-reversion pattern (ρ={autocorr_1:.2f})")
            else:
                scores['autocorr'] = 50
                reasons_against.append(f"Low predictability (ρ={autocorr_1:.2f})")
        else:
            scores['autocorr'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 10. SKEWNESS & KURTOSIS (Tail Risk)
        # ═══════════════════════════════════════════════════════════════════
        if len(returns) >= 30:
            from scipy import stats as scipy_stats
            skewness = scipy_stats.skew(returns.dropna())
            kurtosis = scipy_stats.kurtosis(returns.dropna())
            
            detailed_analysis['skewness'] = skewness
            detailed_analysis['kurtosis'] = kurtosis
            
            # Positive skew = more upside potential (good for longs)
            # Negative skew = more downside risk (good for shorts)
            # High kurtosis = fat tails (more extreme moves)
            if signal == 1:
                if skewness > 0.3:
                    scores['tail_risk'] = 85
                    reasons_for.append(f"Positive skew ({skewness:.2f}) - upside potential")
                elif skewness < -0.5:
                    scores['tail_risk'] = 40
                    reasons_against.append(f"Negative skew ({skewness:.2f}) - downside risk")
                else:
                    scores['tail_risk'] = 65
            else:
                if skewness < -0.3:
                    scores['tail_risk'] = 85
                    reasons_for.append(f"Negative skew ({skewness:.2f}) - downside potential")
                elif skewness > 0.5:
                    scores['tail_risk'] = 40
                    reasons_against.append(f"Positive skew ({skewness:.2f}) - upside risk")
                else:
                    scores['tail_risk'] = 65
            
            # Fat tails = higher risk of extreme moves
            if kurtosis > 3:
                reasons_against.append(f"Fat tails (kurt={kurtosis:.1f}) - extreme move risk")
                scores['tail_risk'] = scores.get('tail_risk', 50) * 0.9
        else:
            scores['tail_risk'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 11. SHARPE & SORTINO RATIO PROJECTION
        # ═══════════════════════════════════════════════════════════════════
        if len(returns) >= 30:
            # Historical Sharpe
            excess_returns = returns - 0.05 / (252 * 24 * 4)  # 5% annual risk-free rate
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252 * 24 * 4) if excess_returns.std() > 0 else 0
            
            # Sortino (downside deviation only)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 5 else returns.std()
            sortino = excess_returns.mean() / downside_std * np.sqrt(252 * 24 * 4) if downside_std > 0 else 0
            
            detailed_analysis['sharpe_ratio'] = sharpe
            detailed_analysis['sortino_ratio'] = sortino
            
            # Trade direction alignment with historical performance
            if signal == 1 and sharpe > 0.5:
                scores['risk_metrics'] = min(100, 60 + sharpe * 20)
                reasons_for.append(f"Favorable risk metrics (Sharpe={sharpe:.2f}, Sortino={sortino:.2f})")
            elif signal == -1 and sharpe < -0.5:
                scores['risk_metrics'] = min(100, 60 + abs(sharpe) * 20)
                reasons_for.append(f"Favorable risk metrics for SHORT (Sharpe={sharpe:.2f})")
            else:
                scores['risk_metrics'] = 50
        else:
            scores['risk_metrics'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 11. SYSTEM SCORE & ML ALIGNMENT
        # ═══════════════════════════════════════════════════════════════════
        system_combined = context.get('system_score', {}).get('combined', 50)
        scores['system'] = system_combined
        if system_combined >= 75:
            reasons_for.append(f"System score {system_combined}/100 (excellent)")
        elif system_combined >= 60:
            reasons_for.append(f"System score {system_combined}/100 (good)")
        elif system_combined < 40:
            reasons_against.append(f"System score {system_combined}/100 (weak)")
        
        ml_prob = context.get('ml_insight', {}).get('probability', 0.5)
        ml_available = context.get('ml_insight', {}).get('ml_available', False)
        scores['ml'] = ml_prob * 100
        if ml_available:
            if ml_prob >= 0.65:
                reasons_for.append(f"ML: {ml_prob*100:.0f}% win probability")
            elif ml_prob < 0.4:
                reasons_against.append(f"ML: only {ml_prob*100:.0f}% win probability")
        
        # ═══════════════════════════════════════════════════════════════════
        # FINAL BAYESIAN-WEIGHTED SCORE CALCULATION WITH CONFIDENCE INTERVALS
        # ═══════════════════════════════════════════════════════════════════
        # Weights based on predictive importance (can be updated via Bayesian learning)
        # FIXED: Added Kalman momentum (was calculated but not weighted)
        weights = {
            'risk_reward': 0.11,   # Kelly/R:R
            'trend': 0.13,         # Regression-based trend
            'hurst': 0.09,         # Fractal analysis / HMM regime
            'volatility': 0.07,    # GARCH-style
            'zscore': 0.09,        # Mean reversion
            'momentum': 0.11,      # RSI with divergence
            'volume': 0.09,        # VWAP analysis
            'kalman': 0.08,        # Kalman filter momentum (NEW!)
            'autocorr': 0.05,      # Predictability
            'tail_risk': 0.05,     # Skew/Kurtosis
            'risk_metrics': 0.05,  # Sharpe/Sortino
            'system': 0.04,        # System score
            'ml': 0.04            # ML prediction
        }
        
        # Calculate weighted score
        final_score = sum(scores.get(k, 50) * w for k, w in weights.items())
        
        # ═══════════════════════════════════════════════════════════════════
        # APPLY SOFT PENALTIES FROM RSI AND TREND CHECKS
        # These replace the old hard blocks - penalties reduce score but don't 
        # block. AI + Math can now work together on final decision.
        # ═══════════════════════════════════════════════════════════════════
        soft_penalty = detailed_analysis.get('total_soft_penalty', 0)
        if soft_penalty != 0:
            pre_penalty_score = final_score
            final_score = max(0, final_score + soft_penalty)  # soft_penalty is negative
            logger.debug(f"📊 Score after soft penalties: {pre_penalty_score:.0f} → {final_score:.0f} (penalty: {soft_penalty})")
        
        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM DIRECTION PENALTY - CRITICAL!
        # If momentum is going AGAINST the signal direction, heavily penalize!
        # This is the #1 cause of losses - entering against momentum
        # ═══════════════════════════════════════════════════════════════════
        roc_5 = context.get('roc_5', 0) if context else 0
        roc_10 = context.get('roc_10', 0) if context else 0
        kalman_momentum = context.get('kalman_momentum', 0) if context else 0
        
        momentum_direction_penalty = 0
        if signal == 1:  # LONG - need POSITIVE momentum
            if roc_5 < -0.3:  # Strong negative momentum
                momentum_direction_penalty = -30
                reasons_against.append(f"🚫 WRONG DIRECTION: LONG with FALLING price (ROC5={roc_5:+.2f}%)")
            elif roc_5 < 0:  # Negative momentum
                momentum_direction_penalty = -15
                reasons_against.append(f"⚠️ LONG against momentum (ROC5={roc_5:+.2f}%)")
        elif signal == -1:  # SHORT - need NEGATIVE momentum
            if roc_5 > 0.3:  # Strong positive momentum
                momentum_direction_penalty = -30
                reasons_against.append(f"🚫 WRONG DIRECTION: SHORT with RISING price (ROC5={roc_5:+.2f}%)")
            elif roc_5 > 0:  # Positive momentum
                momentum_direction_penalty = -15
                reasons_against.append(f"⚠️ SHORT against momentum (ROC5={roc_5:+.2f}%)")
        
        if momentum_direction_penalty != 0:
            final_score = max(0, final_score + momentum_direction_penalty)
            logger.info(f"📊 Momentum direction penalty: {momentum_direction_penalty} (ROC5={roc_5:+.2f}%)")
        
        # ═══════════════════════════════════════════════════════════════════
        # HIGH VOLATILITY PENALTY (Added after SUI/XRP loss analysis)
        # When absolute volatility is high, direction is harder to predict
        # ═══════════════════════════════════════════════════════════════════
        garch_vol = detailed_analysis.get('garch_vol', 0)
        if garch_vol > 0.20:  # >20% annualized vol = very high
            vol_penalty = 15
            reasons_against.append(f"Very high volatility ({garch_vol:.1%}) - direction uncertain")
            final_score = max(0, final_score - vol_penalty)
            logger.debug(f"Applied -15 penalty for very high vol ({garch_vol:.1%})")
        elif garch_vol > 0.15:  # >15% = high
            vol_penalty = 8
            reasons_against.append(f"High volatility ({garch_vol:.1%}) - increased uncertainty")
            final_score = max(0, final_score - vol_penalty)
            logger.debug(f"Applied -8 penalty for high vol ({garch_vol:.1%})")
        
        # Also require STRONGER trend confirmation when volatility is high
        r_squared = detailed_analysis.get('r_squared', 0)
        if garch_vol > 0.15 and r_squared < 0.5:
            # High vol + weak trend = very uncertain direction
            weak_trend_penalty = 10
            reasons_against.append(f"Weak trend (R²={r_squared:.2f}) in high vol environment")
            final_score = max(0, final_score - weak_trend_penalty)
            logger.debug(f"Applied -10 penalty for weak trend in high vol")
        
        # PhD Enhancement: Calculate confidence intervals on final score
        try:
            from indicator import calculate_confidence_interval
            
            score_values = np.array(list(scores.values()))
            ci_result = calculate_confidence_interval(score_values, confidence=0.95)
            
            detailed_analysis['score_ci'] = ci_result
            detailed_analysis['score_ci_lower'] = ci_result.get('lower', final_score)
            detailed_analysis['score_ci_upper'] = ci_result.get('upper', final_score)
            
            # Confidence level based on CI width
            ci_width = ci_result.get('upper', final_score) - ci_result.get('lower', final_score)
            if ci_width < 20:
                confidence_level = 'very_high'
            elif ci_width < 40:
                confidence_level = 'high'
            elif ci_width < 60:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
        except Exception as e:
            logger.debug(f"Confidence interval calculation failed: {e}")
            score_values = np.array(list(scores.values()))
            score_variance = np.var(score_values) if score_values else 0
            confidence_penalty = min(10, score_variance / 50)
            
            final_score = max(0, final_score - confidence_penalty)
            confidence_level = 'high' if score_variance < 200 else 'medium' if score_variance < 400 else 'low'
            detailed_analysis['score_variance'] = score_variance
        
        detailed_analysis['component_scores'] = scores
        detailed_analysis['weights'] = weights
        detailed_analysis['confidence_level'] = confidence_level
        
        # Determine approval thresholds
        math_approved = final_score >= 55
        math_strong = final_score >= 70
        can_override = final_score >= 65
        
        # PhD Enhancement: Add walk-forward validation diagnostics
        try:
            from indicator import calculate_walk_forward_performance
            
            # Create a minimal dataframe for walk-forward analysis
            if 'returns' not in df.columns:
                df_analysis = df.copy()
                df_analysis['returns'] = df_analysis['close'].pct_change()
            else:
                df_analysis = df.copy()
            
            wf_result = calculate_walk_forward_performance(df_analysis, window=50, step=10)
            if wf_result.get('status') == 'success':
                detailed_analysis['walk_forward'] = wf_result
                mean_oos = wf_result.get('mean_oos_return', 0)
                mean_win_rate = wf_result.get('mean_win_rate', 0)
                
                if mean_oos > 0 and mean_win_rate > 0.5:
                    reasons_for.append(f"Walk-forward validated: {mean_win_rate:.0%} win rate, OOS return {mean_oos:.4f}%")
                    # Boost score if walk-forward is positive
                    final_score = min(100, final_score * 1.05)
        except Exception as e:
            logger.debug(f"Walk-forward analysis skipped: {e}")
        
        return {
            'score': final_score,
            'approved': math_approved,
            'strong': math_strong,
            'scores': scores,
            'reasons_for': reasons_for,
            'reasons_against': reasons_against,
            'can_override_ai': can_override,
            'detailed_analysis': detailed_analysis,
            'confidence_level': confidence_level if 'confidence_level' in locals() else 'high'
        }
    
    def _calculate_hurst(self, prices: pd.Series, max_lag: int = 20) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        if len(prices) < max_lag * 2:
            return 0.5
        
        try:
            lags = range(2, max_lag)
            rs_values = []
            
            for lag in lags:
                returns = prices.pct_change(lag).dropna()
                if len(returns) < 10:
                    continue
                
                mean_ret = returns.mean()
                std_ret = returns.std()
                
                if std_ret == 0:
                    continue
                
                cumdev = (returns - mean_ret).cumsum()
                r = cumdev.max() - cumdev.min()
                s = std_ret
                
                if s > 0:
                    rs_values.append((lag, r/s))
            
            if len(rs_values) < 3:
                return 0.5
            
            log_lags = np.log([x[0] for x in rs_values])
            log_rs = np.log([x[1] for x in rs_values])
            
            slope, _ = np.polyfit(log_lags, log_rs, 1)
            return float(np.clip(slope, 0, 1))
        except Exception:
            return 0.5
    
    def calculate_cointegration_analysis(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        symbol1: str = "Asset1",
        symbol2: str = "Asset2"
    ) -> Dict[str, Any]:
        """
        Perform cointegration analysis for pairs trading.
        
        Uses Engle-Granger or Johansen tests to determine if two assets
        share a common stochastic trend (are cointegrated).
        
        Cointegrated pairs allow for statistical arbitrage:
        - When spread deviates from mean, expect mean-reversion
        - Long spread when oversold, short when overbought
        
        Args:
            prices1: Price series for first asset
            prices2: Price series for second asset
            symbol1: Name of first asset (for logging)
            symbol2: Name of second asset (for logging)
            
        Returns:
            Dict containing:
            - cointegrated: bool - Whether pair is cointegrated
            - hedge_ratio: float - Optimal hedge ratio
            - spread_zscore: float - Current spread z-score
            - trade_signal: str - 'long_spread', 'short_spread', 'close_spread', 'hold'
            - half_life: float - Mean reversion half-life in periods
            - entry/exit levels
        """
        try:
            from indicator import calculate_cointegration_test
            
            # Run Engle-Granger cointegration test
            result = calculate_cointegration_test(prices1, prices2, test_type='engle_granger')
            
            if result.get('status') != 'success':
                logger.warning(f"Cointegration test failed for {symbol1}/{symbol2}: {result.get('error', 'unknown')}")
                return {
                    'cointegrated': False,
                    'reason': result.get('error', 'test_failed'),
                    'symbol1': symbol1,
                    'symbol2': symbol2
                }
            
            cointegrated = result.get('cointegrated', False)
            hedge_ratio = result.get('hedge_ratio', 1.0)
            spread_zscore = result.get('spread_zscore', 0.0)
            half_life = result.get('half_life', np.inf)
            trade_signal = result.get('trade_signal', 'hold')
            
            logger.info(f"🔗 Cointegration {symbol1}/{symbol2}: "
                       f"{'✓ COINTEGRATED' if cointegrated else '✗ NOT cointegrated'} | "
                       f"Hedge={hedge_ratio:.4f} | Z={spread_zscore:.2f} | "
                       f"Half-life={half_life:.1f} | Signal={trade_signal}")
            
            return {
                'cointegrated': cointegrated,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'hedge_ratio': hedge_ratio,
                'spread_zscore': spread_zscore,
                'half_life': half_life,
                'trade_signal': trade_signal,
                'trade_reason': result.get('trade_reason', ''),
                'adf_pvalue': result.get('adf_pvalue', 1.0),
                'entry_long': result.get('entry_long', 0),
                'entry_short': result.get('entry_short', 0),
                'exit_level': result.get('exit_level', 0),
                'spread_mean': result.get('spread_mean', 0),
                'spread_std': result.get('spread_std', 0),
                # Additional Johansen test if available
                'johansen_available': False  # Placeholder for future enhancement
            }
            
        except ImportError as ie:
            logger.warning(f"Cointegration analysis unavailable: {ie}")
            return {
                'cointegrated': False,
                'reason': 'missing_dependencies',
                'symbol1': symbol1,
                'symbol2': symbol2
            }
        except Exception as e:
            logger.error(f"Cointegration analysis error for {symbol1}/{symbol2}: {e}")
            return {
                'cointegrated': False,
                'reason': str(e),
                'symbol1': symbol1,
                'symbol2': symbol2
            }
    
    def analyze_pairs_cointegration(
        self,
        pairs_data: List[Dict[str, Any]],
        reference_symbol: str = 'BTCUSDT'
    ) -> Dict[str, Any]:
        """
        Analyze cointegration of multiple pairs with a reference asset.
        
        Useful for identifying pairs that move together and can be used
        for hedging or statistical arbitrage strategies.
        
        Args:
            pairs_data: List of {symbol, df} dictionaries
            reference_symbol: Reference asset to test against (default: BTC)
            
        Returns:
            Dict with cointegrated pairs and trading signals
        """
        cointegrated_pairs = []
        signals = []
        
        # Find reference asset
        ref_data = None
        for pair in pairs_data:
            if reference_symbol in pair.get('symbol', ''):
                ref_data = pair
                break
        
        if ref_data is None:
            return {
                'status': 'no_reference',
                'message': f'Reference {reference_symbol} not found in pairs_data',
                'cointegrated_pairs': [],
                'signals': []
            }
        
        ref_prices = ref_data['df']['close'] if 'df' in ref_data else None
        if ref_prices is None:
            return {
                'status': 'no_reference_prices',
                'cointegrated_pairs': [],
                'signals': []
            }
        
        for pair in pairs_data:
            symbol = pair.get('symbol', '')
            if symbol == reference_symbol:
                continue
            
            df = pair.get('df')
            if df is None or 'close' not in df.columns:
                continue
            
            prices = df['close']
            
            # Perform cointegration test
            coint_result = self.calculate_cointegration_analysis(
                ref_prices, prices, reference_symbol, symbol
            )
            
            if coint_result.get('cointegrated', False):
                cointegrated_pairs.append({
                    'symbol': symbol,
                    'hedge_ratio': coint_result['hedge_ratio'],
                    'half_life': coint_result['half_life'],
                    'adf_pvalue': coint_result['adf_pvalue']
                })
                
                if coint_result['trade_signal'] in ['long_spread', 'short_spread']:
                    signals.append({
                        'symbol': symbol,
                        'signal': coint_result['trade_signal'],
                        'spread_zscore': coint_result['spread_zscore'],
                        'reason': coint_result['trade_reason']
                    })
        
        return {
            'status': 'success',
            'reference': reference_symbol,
            'total_tested': len(pairs_data) - 1,
            'cointegrated_count': len(cointegrated_pairs),
            'cointegrated_pairs': cointegrated_pairs,
            'active_signals': signals
        }
    
    def _validate_ai_decision_with_math(
        self,
        ai_result: Dict[str, Any],
        math_check: Dict[str, Any],
        signal: int,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Validate AI decision against mathematical analysis.
        Override AI if math strongly disagrees (SYMMETRICALLY in both directions).
        
        - If AI blocks but math score >= 65: Math can OVERRIDE to APPROVE
        - If AI approves but math score < 30: Math can OVERRIDE to BLOCK
        """
        ai_approved = ai_result.get('approved', False)
        ai_confidence = ai_result.get('confidence', 0.5)
        math_score = math_check.get('score', 50)
        math_approved = math_check.get('approved', False)
        math_can_override = math_check.get('can_override_ai', False)
        
        # Math can BLOCK if AI approves but math score is very low
        math_can_block = math_score < 30  # Symmetric threshold: strong rejection
        
        signal_type = "LONG" if signal == 1 else "SHORT"
        
        # === CASE 1: AI approves, math confirms ===
        if ai_approved and math_approved:
            logger.info(f"✅ AI + Math AGREE: Approve {signal_type} (AI: {ai_confidence:.0%}, Math: {math_score:.0f}/100)")
            return ai_result
        
        # === CASE 2: AI approves, but math STRONGLY disagrees (score < 30) ===
        if ai_approved and math_can_block:
            # SYMMETRIC OVERRIDE: Math blocks the trade
            logger.warning(f"🔄 MATH OVERRIDE (BLOCK): AI approved but Math score VERY LOW ({math_score:.0f}/100) - BLOCKING trade")
            return {
                'approved': False,
                'confidence': (100 - math_score) / 100,  # High confidence in blocking
                'reasoning': f"AI approved but blocked by very weak math analysis (score: {math_score:.0f}/100). " +
                            f"Math concerns: {', '.join(math_check.get('reasons_against', [])[:3])}",
                'risk_assessment': 'high',
                'override_reason': 'math_override_block',
                'original_ai_decision': ai_result,
                'math_score': math_score
            }
        
        # === CASE 2B: AI approves, math says NO but not strongly ===
        if ai_approved and not math_approved:
            # Let AI decision stand but log warning
            logger.warning(f"⚠️ AI approves but Math score low ({math_score:.0f}/100) - proceeding with caution")
            ai_result['math_warning'] = f"Math score {math_score:.0f}/100 below threshold"
            return ai_result
        
        # === CASE 3: AI blocks, but math says it's a GOOD trade ===
        if not ai_approved and math_can_override:
            logger.info(f"🔄 MATH OVERRIDE (APPROVE): AI blocked but Math score {math_score:.0f}/100 is strong - APPROVING trade")
            
            # Override AI decision
            return {
                'approved': True,
                'confidence': math_score / 100,
                'reasoning': f"AI blocked but overridden by strong math analysis (score: {math_score:.0f}/100). " +
                            f"Math reasons: {', '.join(math_check.get('reasons_for', [])[:3])}",
                'risk_assessment': 'medium',
                'override_reason': 'math_override_approve',
                'original_ai_decision': ai_result,
                'math_score': math_score
            }
        
        # === CASE 4: AI blocks, math agrees ===
        if not ai_approved and not math_approved:
            logger.info(f"❌ AI + Math AGREE: Block {signal_type} (AI: {ai_confidence:.0%}, Math: {math_score:.0f}/100)")
            ai_result['math_confirms'] = True
            ai_result['math_score'] = math_score
            return ai_result
        
        # Default: return AI result
        return ai_result
    
    def _build_market_context(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float
    ) -> Dict[str, Any]:
        """Build market context from DataFrame with defensive handling."""
        
        recent = df.tail(20) if len(df) >= 20 else df
        
        # Ensure we have data to work with
        if len(recent) < 1:
            return {
                "current_price": current_price,
                "price_change_1h": 0,
                "price_change_5m": 0,
                "volume_ratio": 1,
                "trend": "unknown",
                "volatility_pct": 0,
                "atr": atr,
                "sma_10": current_price,
                "sma_20": current_price
            }
        
        # Calculate key metrics with defensive checks
        try:
            price_change_1h = (current_price - recent.iloc[-12]["close"]) / recent.iloc[-12]["close"] * 100 if len(recent) >= 12 else 0
        except (IndexError, KeyError):
            price_change_1h = 0
            
        try:
            price_change_5m = (current_price - recent.iloc[-1]["close"]) / recent.iloc[-1]["close"] * 100
        except (IndexError, KeyError, ZeroDivisionError):
            price_change_5m = 0
        
        # Volume analysis
        # Use previous completed candle (not current incomplete candle) for fair comparison
        try:
            # Average of completed candles (excluding current)
            avg_volume = recent["volume"].iloc[:-1].mean() if len(recent) > 1 else recent["volume"].mean()
            # Use second-to-last candle (most recent completed) for comparison
            current_volume = recent.iloc[-2]["volume"] if len(recent) > 1 else recent.iloc[-1]["volume"]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        except (KeyError, IndexError):
            volume_ratio = 1
        
        # Trend analysis
        try:
            sma_10 = recent["close"].tail(10).mean()
            sma_20 = recent["close"].tail(20).mean()
            trend = "bullish" if sma_10 > sma_20 else "bearish"
        except (KeyError, ValueError):
            sma_10 = sma_20 = current_price
            trend = "unknown"
        
        # Volatility
        volatility = atr / current_price * 100 if current_price > 0 else 0
        
        # === ENHANCED TECHNICAL MATH DATA ===
        # These are the actual numbers the signal generation uses
        
        # SMA15/SMA40 crossover (the core signal)
        try:
            sma_15 = df['close'].rolling(15).mean().iloc[-1] if len(df) >= 15 else current_price
            sma_40 = df['close'].rolling(40).mean().iloc[-1] if len(df) >= 40 else current_price
            sma_15_prev = df['close'].rolling(15).mean().iloc[-2] if len(df) >= 16 else sma_15
            sma_40_prev = df['close'].rolling(40).mean().iloc[-2] if len(df) >= 41 else sma_40
            sma_crossover = "BULLISH" if sma_15 > sma_40 and sma_15_prev <= sma_40_prev else \
                           "BEARISH" if sma_15 < sma_40 and sma_15_prev >= sma_40_prev else \
                           "NONE"
            sma_spread_pct = ((sma_15 - sma_40) / sma_40 * 100) if sma_40 > 0 else 0
        except Exception:
            sma_15 = sma_40 = current_price
            sma_crossover = "UNKNOWN"
            sma_spread_pct = 0
        
        # RSI calculation
        try:
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
            else:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, 0.001)
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
        except Exception:
            rsi = 50  # Default neutral RSI
        
        # ADX calculation (trend strength)
        try:
            if 'adx' in df.columns:
                adx = df['adx'].iloc[-1]
            else:
                from indicator import calculate_adx
                adx = calculate_adx(df)
        except Exception:
            adx = 25  # Default moderate trend
        
        # Momentum (rate of change)
        try:
            roc_5 = ((df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100) if len(df) > 6 else 0
            roc_10 = ((df['close'].iloc[-1] - df['close'].iloc[-11]) / df['close'].iloc[-11] * 100) if len(df) > 11 else 0
        except Exception:
            roc_5 = roc_10 = 0  # Default no momentum
        
        return {
            "current_price": current_price,
            "price_change_1h": round(price_change_1h, 2),
            "price_change_5m": round(price_change_5m, 3),
            "volume_ratio": round(volume_ratio, 2),
            "trend": trend,
            "volatility_pct": round(volatility, 2),
            "atr": round(atr, 4),
            "sma_10": round(sma_10, 4),
            "sma_20": round(sma_20, 4),
            # Enhanced math data
            "sma_15": round(sma_15, 4),
            "sma_40": round(sma_40, 4),
            "sma_crossover": sma_crossover,
            "sma_spread_pct": round(sma_spread_pct, 3),
            "rsi": round(rsi, 1),
            "adx": round(adx, 1),
            "roc_5": round(roc_5, 2),
            "roc_10": round(roc_10, 2)
        }
    
    def _ai_final_entry_decision(
        self,
        symbol: str,
        math_direction: str,
        math_score: float,
        long_score: float,
        short_score: float,
        long_reasons_for: List[str],
        long_reasons_against: List[str],
        short_reasons_for: List[str],
        short_reasons_against: List[str],
        context: Dict[str, Any],
        level_info: Dict[str, Any] = None,  # Resistance/support info
        advanced_math: str = ""  # NEW: Advanced math analysis section (Kalman, POC, RSI divergence, etc.)
    ) -> Dict[str, Any]:
        """
        AI FINAL ENTRY DECISION - AI has full power to choose direction.
        
        AI receives:
        - Both LONG and SHORT math scores
        - All reasons for/against each direction
        - Market context
        
        AI outputs:
        - decision: LONG, SHORT, or REJECT
        - confidence: 0.0-1.0
        - reasoning: explanation
        """
        try:
            # Get performance and market context
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            # Get direction-based performance (CRITICAL for decision making)
            dir_perf = self._get_direction_performance()
            
            # Build extra caution warnings
            extra_caution = ""
            if perf["last_trade_was_loss"]:
                extra_caution = f"\n⚠️ CAUTION: Last {perf['consecutive_losses']} trade(s) were losses. Be extra skeptical!"
            if perf["consecutive_losses"] >= 2:
                extra_caution += "\n🛑 LOSING STREAK: Require very high confidence to approve any trade."
            if market["is_weekend"]:
                extra_caution += "\n📅 WEEKEND: Lower liquidity, higher risk."
            if market["activity_level"] == "low":
                extra_caution += "\n🌙 LOW ACTIVITY: Increased slippage risk."
            
            # Historical direction info (secondary - for context only)
            # NOTE: Don't add strong warnings - math is primary!
            
            # Get symbol-specific performance WITH per-direction breakdown
            symbol_perf = self._get_symbol_performance(symbol)
            symbol_warning = ""
            if symbol_perf["has_history"]:
                # Just provide context, not strong warnings
                # Math analysis is PRIMARY, history is SECONDARY
                if symbol_perf.get("long_trades", 0) > 0 or symbol_perf.get("short_trades", 0) > 0:
                    long_str = f"LONG: {symbol_perf['long_wins']}W/{symbol_perf['long_losses']}L" if symbol_perf.get("long_trades", 0) > 0 else "LONG: new"
                    short_str = f"SHORT: {symbol_perf['short_wins']}W/{symbol_perf['short_losses']}L" if symbol_perf.get("short_trades", 0) > 0 else "SHORT: new"
                    symbol_warning = f"\n📊 {symbol} past trades (context only): {long_str} | {short_str}"
                
                # Only warn on extreme cases (5+ more losses than wins)
                if symbol_perf["losses"] > symbol_perf["wins"] + 4:
                    symbol_warning += f"\n⚠️ Note: {symbol} has struggled - but math may have found new pattern!"
            
            # Check for momentum contradiction
            momentum_5 = context.get('roc_5', 0)
            momentum_10 = context.get('roc_10', 0)
            momentum_warning = ""
            
            if math_direction == "SHORT" and (momentum_5 > 0.5 or momentum_10 > 0.3):
                momentum_warning = f"\n📈 MOMENTUM CONFLICT: Math says SHORT but price is RISING (5bar={momentum_5:+.2f}%, 10bar={momentum_10:+.2f}%). Consider: 1) LONG if LONG score >= 45, or 2) REJECT if neither direction is good."
            elif math_direction == "LONG" and (momentum_5 < -0.5 or momentum_10 < -0.3):
                momentum_warning = f"\n📉 MOMENTUM CONFLICT: Math says LONG but price is FALLING (5bar={momentum_5:+.2f}%, 10bar={momentum_10:+.2f}%). Consider: 1) SHORT if SHORT score >= 45, or 2) REJECT if neither direction is good."
            
            # Build resistance/support level warning
            level_warning = ""
            if level_info:
                if level_info.get('warning'):
                    level_warning = f"\n{level_info['warning']}"
                    
                # Add specific context
                dist_to_res = level_info.get('distance_to_resistance_pct', 999)
                dist_to_sup = level_info.get('distance_to_support_pct', 999)
                res_touches = level_info.get('resistance_touches', 0)
                sup_touches = level_info.get('support_touches', 0)
                
                if math_direction == "LONG" and dist_to_res < 2.0 and res_touches >= 2:
                    level_warning += f"\n🚨 DANGER FOR LONG: Price is only {dist_to_res:.1f}% below 24h resistance (tested {res_touches}x and rejected). High probability of reversal!"
                elif math_direction == "SHORT" and dist_to_sup < 2.0 and sup_touches >= 2:
                    level_warning += f"\n🚨 DANGER FOR SHORT: Price is only {dist_to_sup:.1f}% above 24h support (tested {sup_touches}x and held). High probability of bounce!"
            
            # Get market news and sentiment context (RECOMMENDATION ONLY!)
            news_ctx = self._get_news_context()
            news_section = ""
            if news_ctx.get('fear_greed_index', 50) != 50 or news_ctx.get('warning'):
                news_section = f"""
=== 📰 MARKET SENTIMENT (RECOMMENDATION ONLY - NOT A HARD RULE) ===
Fear & Greed Index: {news_ctx.get('fear_greed_index', 50)} ({news_ctx.get('fear_greed_label', 'Unknown')})
Market Cap 24h Change: {news_ctx.get('market_cap_change_24h', 0):+.2f}%
News Sentiment: {news_ctx.get('news_sentiment', 0):+.2f} (Bullish: {news_ctx.get('bullish_news_count', 0)}, Bearish: {news_ctx.get('bearish_news_count', 0)})
{f"Critical News: {news_ctx.get('critical_news_count', 0)} alerts" if news_ctx.get('critical_news_count', 0) > 0 else ''}
{chr(10).join('• ' + h for h in news_ctx.get('critical_headlines', [])[:2]) if news_ctx.get('critical_headlines') else ''}
⚠️ NOTE: Use this as CONTEXT only. Extreme fear can be a BUY opportunity (contrarian). 
   Math analysis (momentum, Kalman) is MORE RELIABLE than daily sentiment.
"""
            
            # Add advanced math section if provided
            advanced_math_section = ""
            if advanced_math:
                advanced_math_section = "\n" + advanced_math
            
            prompt = f"""You are the FINAL DECISION MAKER for a crypto trading bot. Your role is to PROTECT CAPITAL first, then find profitable opportunities.

=== TRADE CANDIDATE ===
Symbol: {symbol}
Math's Preferred Direction: {math_direction} (score: {math_score:.0f}/100)
{momentum_warning}
{level_warning}
{news_section}
{advanced_math_section}
=== {symbol} SPECIFIC HISTORY (PER-PAIR DIRECTION STATS) ===
{self._build_symbol_history_section(symbol_perf, symbol)}
{symbol_warning}

=== BOTH DIRECTIONS COMPARED ===

📈 LONG ANALYSIS (Score: {long_score:.0f}/100):
Strengths: {', '.join(long_reasons_for[:4]) if long_reasons_for else 'None'}
Weaknesses: {', '.join(long_reasons_against[:4]) if long_reasons_against else 'None'}

📉 SHORT ANALYSIS (Score: {short_score:.0f}/100):
Strengths: {', '.join(short_reasons_for[:4]) if short_reasons_for else 'None'}
Weaknesses: {', '.join(short_reasons_against[:4]) if short_reasons_against else 'None'}

=== CURRENT MARKET STATE ===
Price: ${context.get('current_price', 0)}
1h Change: {context.get('price_change_1h', 0)}%
RSI: {context.get('rsi', 50):.1f} (<30=oversold bounce likely, >70=overbought drop likely)
ADX: {context.get('adx', 25):.1f} (>25=strong trend, >50=extreme)
Momentum 5-bar: {context.get('roc_5', 0):+.2f}% {'⬆️ RISING' if context.get('roc_5', 0) > 0 else '⬇️ FALLING'}
Momentum 10-bar: {context.get('roc_10', 0):+.2f}% {'⬆️ RISING' if context.get('roc_10', 0) > 0 else '⬇️ FALLING'}
Trend: {context.get('trend', 'UNKNOWN')}
Volume: {context.get('volume_ratio', 1):.1f}x average

=== HISTORICAL DIRECTION PERFORMANCE (SECONDARY - for context only) ===
📈 LONG trades: {dir_perf['long_trades']} total, {dir_perf['long_wins']}W/{dir_perf['long_losses']}L ({dir_perf['long_wr']:.0f}% WR), P&L: ${dir_perf['long_pnl']:+.2f}
📉 SHORT trades: {dir_perf['short_trades']} total, {dir_perf['short_wins']}W/{dir_perf['short_losses']}L ({dir_perf['short_wr']:.0f}% WR), P&L: ${dir_perf['short_pnl']:+.2f}
Note: Historical data is from OLD system - use as minor context only, NOT as decision maker!

=== PERFORMANCE CONTEXT ===
Overall Win Rate: {perf['win_rate']}%
Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L
{extra_caution}

=== INTELLIGENT DECISION FRAMEWORK ===

**🧮 MATH ANALYSIS IS PRIMARY (70% weight):**
- The math score incorporates: Kalman momentum, POC distance, Hurst exponent, RSI divergence, GARCH volatility
- Math score 55+ = STRONG signal, trust it!
- Math score 50-54 = Good signal, approve if momentum aligns
- Math score 45-49 = Marginal signal, need strong momentum confirmation

**📊 HISTORICAL DATA IS SECONDARY (30% weight):**
- Historical data is advisory only - it should NOT override good math
- Use it as a TIE-BREAKER when math scores are similar for both directions
- A good math signal should be approved even if historical data is limited

**MINIMUM CONFIDENCE: 55%**
- You MUST provide at least 55% confidence to approve

**APPROVAL GUIDELINES (FOLLOW THESE!):**
✅ Math score 70+ AND momentum aligns → APPROVE with 65-75% confidence
✅ Math score 60-69 AND momentum aligns → APPROVE with 60-65% confidence
✅ Math score 55-59 AND momentum aligns → APPROVE with 55-60% confidence
⚠️ Math score 45-54 → APPROVE only with STRONG momentum confirmation

**🚨 CRITICAL: CHECK MOMENTUM DIRECTION!**
- For LONG: ROC5 MUST be positive (price rising) - otherwise REJECT!
- For SHORT: ROC5 MUST be negative (price falling) - otherwise REJECT!
- Math score alone is NOT enough - momentum direction MUST align!

**REJECTION IS REQUIRED FOR THESE CASES:**
❌ Math score < 45 (weak signal - insufficient edge)
❌ MOMENTUM CONFLICT: Math says LONG but ROC5 is negative (price falling!)
❌ MOMENTUM CONFLICT: Math says SHORT but ROC5 is positive (price rising!)
❌ RSI extreme AND against direction (RSI<20 for SHORT, RSI>80 for LONG)

**DO NOT REJECT FOR:**
✓ "Low volume" - already factored into math score
✓ "Limited history" - fresh start is intentional
✓ "Uncertain direction" - math score tells you direction
✓ "Multiple factors" - vague reasons are not valid

**PICK THE BEST DIRECTION:**
- If LONG score > SHORT score by 5+ AND ROC5 positive → approve LONG
- If SHORT score > LONG score by 5+ AND ROC5 negative → approve SHORT
- If momentum conflicts with math direction → REJECT (don't trade against momentum!)

**CONFIDENCE GUIDELINES:**
- Math 45-54 → confidence 0.55-0.60 (requires momentum alignment)
- Math 55-64 → confidence 0.60-0.65
- Math 65+ → confidence 0.65-0.75

**YOUR ROLE:**
1. Validate that MOMENTUM (ROC5) aligns with the trade direction
2. ADD your own analysis of momentum, RSI, volume patterns
3. CATCH edge cases math might miss (news events, unusual patterns)
4. Choose the BEST direction if both are viable

Your opinion MATTERS! If you see something math missed, say it.
If momentum conflicts with math, you MUST reject - don't fight the trend!

**DECISION WEIGHT: Math 50% + Momentum 30% + Your AI Analysis 20%**

=== YOUR RESPONSE ===
Respond ONLY with this JSON (nothing else):
{{"decision": "LONG", "confidence": 0.65, "reasoning": "Brief explanation"}}

Valid decisions: "LONG", "SHORT", "REJECT"
Confidence: 0.50-1.00 (0.55+ to approve)
"""
            
            # Make API call
            for attempt in range(2):
                result_text = self._generate_content(prompt)
                if not result_text:
                    continue
                    
                result_text = result_text.strip()
                
                # Parse JSON
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                
                result_text = result_text.strip()
                result = json.loads(result_text)
                
                # Validate and normalize
                decision = result.get('decision', 'REJECT').upper()
                if decision not in ['LONG', 'SHORT', 'REJECT']:
                    decision = 'REJECT'
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
                
                reasoning = result.get('reasoning', 'AI decision')
                
                # Apply loss cooldown - require slightly higher confidence after losses
                # Moderate settings - not too strict, not too loose
                if perf["consecutive_losses"] >= 3:
                    required_conf = 0.60  # After 3+ losses, need solid confidence
                elif perf["consecutive_losses"] >= 2:
                    required_conf = 0.55  # After 2 losses, reasonable confidence
                elif perf["last_trade_was_loss"]:
                    required_conf = 0.52  # After 1 loss, slightly cautious
                else:
                    required_conf = 0.50  # Normal operation: trade when reasonable
                
                # If confidence too low, force REJECT
                if decision != 'REJECT' and confidence < required_conf:
                    logger.info(f"AI chose {decision} but confidence {confidence:.0%} < required {required_conf:.0%}, forcing REJECT")
                    decision = 'REJECT'
                    reasoning = f"Low confidence ({confidence:.0%}) below threshold ({required_conf:.0%})"
                
                logger.info(f"🤖 AI FINAL DECISION for {symbol}: {decision} (conf={confidence:.0%})")
                
                return {
                    'decision': decision,
                    'confidence': confidence,
                    'reasoning': reasoning
                }
            
            # API failed - default to REJECT (safety)
            logger.warning(f"AI API failed for {symbol} entry decision - defaulting to REJECT")
            return {
                'decision': 'REJECT',
                'confidence': 0.0,
                'reasoning': 'AI API failed - safety reject'
            }
            
        except Exception as e:
            import traceback
            logger.error(f"AI final entry decision error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'decision': 'REJECT',
                'confidence': 0.0,
                'reasoning': f'AI error: {e}'
            }
    
    def _ai_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Use Google Gemini API for signal analysis with SKEPTIC MODE."""
        try:
            signal_type = "LONG" if signal == 1 else "SHORT"
            
            # Get additional context
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            # Extract system score and ML insight from context
            system_score = context.get('system_score', {})
            ml_insight = context.get('ml_insight', {})
            tech_score_data = context.get('tech_score', {})
            # NEW: Extract PhD Math Check
            math_check = context.get('math_check', {})
            
            # Build PhD Math section
            math_section = ""
            if math_check:
                math_score = math_check.get('final_score', 50)
                kelly = math_check.get('detailed', {}).get('kelly_fraction', 0)
                hurst = math_check.get('detailed', {}).get('hurst_exponent', 0.5)
                r_squared = math_check.get('detailed', {}).get('r_squared', 0)
                reasons_for = math_check.get('reasons_for', [])
                reasons_against = math_check.get('reasons_against', [])
                
                math_section = f"""
=== PHD MATHEMATICAL VERIFICATION ===
Overall Math Score: {math_score}/100
• Kelly Criterion (Optimal Size): {kelly:.1%} (Risk Management)
• Hurst Exponent (Trend State): {hurst:.2f} (0.5=Random, >0.6=Trend, <0.4=Revert)
• Trend Fit (R²): {r_squared:.2f}
• Quant Strengths: {', '.join(reasons_for[:3]) if reasons_for else 'None'}
• Quant Weaknesses: {', '.join(reasons_against[:3]) if reasons_against else 'None'}"""

            # Build technical score section
            tech_section = ""
            if tech_score_data:
                t_score = tech_score_data.get('score', 50)
                t_quality = tech_score_data.get('quality', 'UNKNOWN')
                t_factors = tech_score_data.get('factors', [])
                t_breakdown = tech_score_data.get('breakdown', {})
                factors_str = ", ".join(t_factors) if t_factors else "None notable"
                
                # Build breakdown string if available
                breakdown_parts = []
                if t_breakdown:
                    for k, v in t_breakdown.items():
                        if v > 0:
                            breakdown_parts.append(f"{k}={v}")
                breakdown_str = " + ".join(breakdown_parts) if breakdown_parts else "N/A"
                
                tech_section = f"""
=== TECHNICAL SCORE (Math-based quality) ===
Score: {t_score}/100 ({t_quality})
Breakdown: {breakdown_str}
Notable Factors: {factors_str}
NOTE: Score based on ADX, Hurst exponent, Volume, RSI, Regime clarity"""
            
            # Build system assessment section
            system_section = ""
            if system_score:
                score = system_score.get('combined', 50)
                rec = system_score.get('recommendation', 'NEUTRAL')
                breakdown = system_score.get('breakdown', 'N/A')
                short_bonus = system_score.get('short_bonus', 0)
                system_section = f"""
=== SYSTEM SCORE (Pre-computed) ===
Combined Score: {score:.0f}/100
Recommendation: {rec}
Breakdown: {breakdown}
{"Short Bias Applied: +5 points" if short_bonus > 0 else ""}
NOTE: Score ≥60 = favorable setup, ≥75 = strong setup"""
            
            # Build ML section
            ml_section = ""
            if ml_insight.get('ml_available'):
                win_prob = ml_insight.get('ml_win_probability', 0.5)
                ml_conf = ml_insight.get('ml_confidence', 'low')
                ml_acc = ml_insight.get('ml_accuracy', 0.5)
                ml_samples = ml_insight.get('ml_samples', 0)
                ml_section = f"""
=== ML MODEL PREDICTION ===
Win Probability: {win_prob:.1%}
Confidence: {ml_conf}
Model Accuracy: {ml_acc:.1%} (trained on {ml_samples} samples)
NOTE: ML needs 500+ samples for reliability, currently {'LEARNING' if ml_samples < 500 else 'RELIABLE'}"""
            else:
                ml_section = """
=== ML MODEL ===
Status: Not available (still learning)"""
            
            # Build market scanner context section
            scanner = context.get('market_scanner', {})
            scanner_section = ""
            if scanner.get('best_pair'):
                best = scanner.get('best_pair', '')
                current_rank = scanner.get('current_pair_rank', 0)
                is_recommended = scanner.get('is_recommended_pair', False)
                scanner_section = f"""
=== MARKET SCANNER CONTEXT ===
Scanner's Best Pair: {best}
Current Pair Rank: #{current_rank} in market scan
Is This The Recommended Pair: {'YES ✓' if is_recommended else 'NO - scanner prefers ' + best}
NOTE: If scanner recommends a different pair, consider if this trade is worth taking"""
            
            # Determine if we need extra caution
            extra_caution = ""
            if perf["last_trade_was_loss"]:
                extra_caution = f"\n⚠️ CAUTION: Last {perf['consecutive_losses']} trade(s) were losses. Be extra skeptical!"
            if perf["consecutive_losses"] >= 2:
                extra_caution += "\n🛑 LOSING STREAK: Require very high confidence to approve."
            if market["is_weekend"]:
                extra_caution += "\n📅 WEEKEND: Lower liquidity, higher risk of false moves."
            if market["activity_level"] == "low":
                extra_caution += "\n🌙 LOW ACTIVITY HOURS: Increased slippage risk."
            
            prompt = f"""You are a SKEPTICAL crypto trading supervisor. Your job is to PROTECT capital by rejecting bad trades.

=== SIGNAL ===
Proposed Trade: {signal_type} on {symbol}

=== PHD MATHEMATICAL VERIFICATION ===
{math_section}

=== TECHNICAL MATH (Signal Generation) ===
SMA Crossover Signal: {context.get('sma_crossover', 'UNKNOWN')}
SMA15: ${context.get('sma_15', 0):.4f}
SMA40: ${context.get('sma_40', 0):.4f}
SMA Spread: {context.get('sma_spread_pct', 0):+.3f}% (positive = bullish)
RSI: {context.get('rsi', 50):.1f} (30-70 normal, <30 oversold, >70 overbought)
ADX: {context.get('adx', 0):.1f} (>25 trending, >35 strong trend, 35-40 DANGER ZONE)
Momentum 5-bar: {context.get('roc_5', 0):+.2f}%
Momentum 10-bar: {context.get('roc_10', 0):+.2f}%

=== MARKET DATA ===
Current Price: ${context['current_price']}
1-Hour Price Change: {context['price_change_1h']}%
Volume Ratio (vs avg): {context['volume_ratio']}x
Trend (SMA10 vs SMA20): {context['trend']}
Volatility (ATR%): {context['volatility_pct']}%
{tech_section}
{system_section}
{ml_section}
{scanner_section}

=== TRADING PERFORMANCE ===
Total Trades: {perf['total_trades']}
Win Rate: {perf['win_rate']}%
Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L
Recent P&L (last 5): ${perf['recent_pnl']}

=== MARKET SESSION ===
Session: {market['session']} ({market['hour_utc']}:00 UTC)
Activity Level: {market['activity_level']}
Weekend: {market['is_weekend']}
{extra_caution}

=== MATH-BASED DECISION RULES ===
APPROVE signals when:
- PhD Math Score is high (>70)
- Kelly Criterion suggests positive sizing (>0%)
- SMA crossover matches signal direction
- Hurst Exponent confirms regime

REJECT signals when:
- PhD Math detects "Quant Weaknesses" that are critical
- Kelly ~0% (Negative expectancy)
- ADX 35-40 (DANGER ZONE)
- Momentum diverges from signal

=== YOUR TASK ===
1. Analyze the "PHD MATHEMATICAL VERIFICATION" section first.
2. List 3 quantified reasons why this trade could FAIL (e.g., low R², poor Hurst).
3. Decide if the mathematical edge is sufficient to risk capital.

Only approve if the MATH is compelling.

Respond ONLY with this JSON format, no other text:
{{"reasons_against": ["reason1", "reason2", "reason3"], "approved": false, "confidence": 0.65, "reasoning": "why approved or rejected", "risk_assessment": "low/medium/high"}}"""

            # Retry loop for robustness
            for attempt in range(2):
                result_text = self._generate_content(prompt)
                if not result_text:
                    continue
                result_text = result_text.strip()
            
                # Parse JSON from response
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
            
                # Clean up common issues
                result_text = result_text.strip()
            
                result = json.loads(result_text)
            
                # Ensure required fields exist
                result.setdefault("approved", False)
                result.setdefault("confidence", 0.5)
                result.setdefault("reasoning", "AI analysis")
                result.setdefault("risk_assessment", "medium")
                result.setdefault("reasons_against", [])
            
                # Apply confidence threshold with LOSS COOLDOWN
                perf = self._get_performance_context()
            
                if perf["consecutive_losses"] >= 2:
                    # After 2+ consecutive losses, require 90% confidence
                    required_threshold = self.loss_cooldown_threshold
                    logger.info(f"Loss cooldown active: requiring {required_threshold:.0%} confidence")
                elif perf["last_trade_was_loss"]:
                    # After 1 loss, require 85% confidence
                    required_threshold = 0.85
                else:
                    required_threshold = self.confidence_threshold
            
                result["approved"] = result["approved"] and result["confidence"] >= required_threshold
                result["threshold_used"] = required_threshold
            
                # Log the skeptic analysis
                if result.get("reasons_against"):
                    logger.info(f"AI reasons against trade: {result['reasons_against']}")
            
                return result
            
            # If we get here, both attempts failed
            return self._rule_based_analysis(signal, context, symbol)
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}, falling back to rules")
            return self._rule_based_analysis(signal, context, symbol)
    
    def _rule_based_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Rule-based signal validation when AI is unavailable.
        Uses comprehensive math check as primary decision maker.
        """
        signal_type = "LONG" if signal == 1 else "SHORT"
        
        # Use math check if available
        math_check = context.get('math_check')
        if math_check:
            math_score = math_check.get('score', 50)
            math_approved = math_check.get('approved', False)
            reasons_for = math_check.get('reasons_for', [])
            reasons_against = math_check.get('reasons_against', [])
            
            logger.info(f"📊 Rule-based using Math Check: {math_score:.0f}/100, Approved: {math_approved}")
            
            if math_approved:
                risk = "low" if math_score >= 70 else "medium"
            else:
                risk = "high" if math_score < 40 else "medium"
            
            return {
                "approved": math_approved,
                "confidence": math_score / 100,
                "reasoning": f"Math-based decision (AI unavailable). Score: {math_score:.0f}/100. " +
                           f"Pros: {', '.join(reasons_for[:3]) or 'None'}. " +
                           f"Cons: {', '.join(reasons_against[:3]) or 'None'}.",
                "risk_assessment": risk,
                "source": "math_rules",
                "math_score": math_score
            }
        
        # Fallback to basic rules if no math check
        confidence = 0.5  # Start neutral
        reasons = []
        risk = "medium"
        
        # Rule 1: Trend alignment
        if (signal == 1 and context["trend"] == "bullish") or \
           (signal == -1 and context["trend"] == "bearish"):
            confidence += 0.15
            reasons.append(f"Signal aligns with {context['trend']} trend")
        else:
            confidence -= 0.1
            reasons.append(f"Counter-trend trade ({context['trend']} market)")
        
        # Rule 2: Volume confirmation
        if context["volume_ratio"] > 1.2:
            confidence += 0.1
            reasons.append("Strong volume confirmation")
        elif context["volume_ratio"] < 0.5:
            confidence -= 0.1
            reasons.append("Low volume - weak confirmation")
        
        # Rule 3: Volatility check
        if context["volatility_pct"] > 3:
            confidence -= 0.1
            risk = "high"
            reasons.append("High volatility environment")
        elif context["volatility_pct"] < 1:
            confidence += 0.05
            risk = "low"
            reasons.append("Low volatility - stable conditions")
        
        # Rule 4: Recent momentum
        if (signal == 1 and context["price_change_1h"] > 0) or \
           (signal == -1 and context["price_change_1h"] < 0):
            confidence += 0.1
            reasons.append("Momentum supports direction")
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "approved": confidence >= self.confidence_threshold,
            "confidence": round(confidence, 2),
            "reasoning": "; ".join(reasons),
            "risk_assessment": risk
        }
    
    def _log_analysis(
        self,
        signal: int,
        symbol: str,
        result: Dict[str, Any]
    ):
        """Log analysis for tracking."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "signal": "LONG" if signal == 1 else "SHORT",
            **result
        }
        self.trade_history.append(entry)
        
        status = "✅ APPROVED" if result["approved"] else "❌ REJECTED"
        logger.info(
            f"AI Filter {status}: {symbol} {entry['signal']} | "
            f"Confidence: {result['confidence']:.0%} | "
            f"Risk: {result['risk_assessment']} | "
            f"Reason: {result['reasoning']}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics from memory and/or file."""
        # First try in-memory history
        history = self.trade_history.copy() if self.trade_history else []
        
        # If empty, load from ai_decisions.json
        if not history:
            try:
                decisions_file = Path(__file__).parent / "ai_decisions.json"
                if decisions_file.exists():
                    import json
                    with open(decisions_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'decisions' in data:
                            history = data['decisions']
                        elif isinstance(data, list):
                            history = data
            except Exception as e:
                logger.debug(f"Could not load ai_decisions.json: {e}")
        
        if not history:
            return {"total_signals": 0, "approved": 0, "rejected": 0}
        
        approved = sum(1 for t in history if t.get("approved", False))
        return {
            "total_signals": len(history),
            "approved": approved,
            "rejected": len(history) - approved,
            "approval_rate": f"{approved / len(history):.1%}"
        }
    
    def _math_proactive_scan(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Math-based proactive scan when AI is unavailable.
        Uses comprehensive math check to identify high-quality opportunities.
        Requires higher thresholds than signal-based trades.
        """
        try:
            context = self._build_market_context(df, current_price, atr)
            
            # Check both directions and pick the better one
            long_check = self._comprehensive_math_check(1, df, current_price, atr, context)
            short_check = self._comprehensive_math_check(-1, df, current_price, atr, context)
            
            # Log BOTH scores for debugging direction choices
            long_score = long_check.get('score', 0)
            short_score = short_check.get('score', 0)
            logger.info(f"📊 Math direction comparison {symbol}: LONG={long_score:.0f}, SHORT={short_score:.0f} (diff: {abs(long_score-short_score):.0f})")
            
            # Proactive trades threshold - configurable, default 65
            # Lower threshold = more aggressive, higher = more conservative
            PROACTIVE_THRESHOLD = getattr(self, 'proactive_threshold', 65)
            
            best_direction = None
            best_score = 0
            best_check = None
            
            # === PRE-FILTERS: Block contradictory signals before flagging opportunity ===
            volume_ratio = context.get('volume_ratio', 1.0)
            
            # Filter 1: Volume too low (illiquid market)
            if volume_ratio < 0.1:
                logger.info(f"📊 Math scan: BLOCKED - Volume too low ({volume_ratio:.2f}x) - illiquid market")
                return None
            
            # Choose best direction
            if long_check['score'] >= PROACTIVE_THRESHOLD and long_check['score'] > short_check['score']:
                best_direction = "LONG"
                best_score = long_check['score']
                best_check = long_check
            elif short_check['score'] >= PROACTIVE_THRESHOLD and short_check['score'] > long_check['score']:
                best_direction = "SHORT"
                best_score = short_check['score']
                best_check = short_check
            
            if not best_direction:
                # Log at INFO level so scores are visible in logs
                logger.info(f"📊 Math scan: No opportunity (LONG: {long_check['score']:.0f}, SHORT: {short_check['score']:.0f}, need {PROACTIVE_THRESHOLD}+)")
                return None
            
            # Filter 2: Direction margin check - if LONG and SHORT too close, direction is unclear
            score_margin = abs(long_check['score'] - short_check['score'])
            MIN_DIRECTION_MARGIN = 8  # Require at least 8-point difference to be confident in direction
            if score_margin < MIN_DIRECTION_MARGIN:
                logger.info(f"📊 Math scan: BLOCKED - Direction unclear for {symbol} (LONG={long_check['score']:.0f}, SHORT={short_check['score']:.0f}, margin={score_margin:.0f} < {MIN_DIRECTION_MARGIN})")
                return None
            
            # Filter 3: No statistical edge detected (DISABLED - too strict for live trading)
            # The score already gets penalized by 15% when p-value isn't significant
            # Blocking entirely prevents too many valid trades
            has_no_edge = best_check.get('detailed_analysis', {}).get('no_statistical_edge', False)
            if has_no_edge:
                logger.debug(f"📊 Math scan: Note - p-value not significant for {best_direction} {symbol}, but proceeding (score {best_score:.0f})")
                # Don't block - the score penalty is sufficient
            
            # Filter 4: Too many reasons against vs for
            reasons_for_count = len(best_check.get('reasons_for', []))
            reasons_against_count = len(best_check.get('reasons_against', []))
            if reasons_against_count > reasons_for_count:
                logger.info(f"📊 Math scan: BLOCKED - More negatives ({reasons_against_count}) than positives ({reasons_for_count}) for {best_direction} {symbol}")
                return None
            
            # ═══════════════════════════════════════════════════════════════
            # Filter 5: RESISTANCE/SUPPORT ZONE CHECK (NEW!)
            # Don't go LONG in resistance zone, don't go SHORT in support zone
            # Zones are calculated dynamically based on ATR (market volatility)
            # ═══════════════════════════════════════════════════════════════
            levels = self._detect_resistance_support_levels(df, current_price)
            
            in_r_zone = levels.get('in_resistance_zone', False)
            in_s_zone = levels.get('in_support_zone', False)
            zone_pct = levels.get('zone_width_pct', 2.0)
            r_upper = levels.get('resistance_upper', 0)
            r_lower = levels.get('resistance_lower', 0)
            s_upper = levels.get('support_upper', 0)
            s_lower = levels.get('support_lower', 0)
            pos_range = levels.get('position_in_range_pct', 50)
            
            # Log zone info
            zone_status = ""
            if in_r_zone:
                zone_status = " 🔴IN_RESISTANCE_ZONE"
            elif in_s_zone:
                zone_status = " 🔵IN_SUPPORT_ZONE"
            elif pos_range >= 88:
                zone_status = " ⚠️NEAR_RESISTANCE"
            elif pos_range <= 12:
                zone_status = " ⚠️NEAR_SUPPORT"
            logger.info(f"📊 {symbol}: Range={pos_range:.0f}% | Zone={zone_pct:.1f}% | R[${r_lower:.4f}-${r_upper:.4f}] S[${s_lower:.4f}-${s_upper:.4f}]{zone_status}")
            
            # Block LONG if in resistance zone
            if best_direction == "LONG" and in_r_zone:
                logger.warning(f"🚫 Math scan: BLOCKED LONG {symbol} - IN RESISTANCE ZONE [${r_lower:.4f}-${r_upper:.4f}] | Price at {pos_range:.0f}% of 24h range")
                return None
            
            # Block SHORT if in support zone
            if best_direction == "SHORT" and in_s_zone:
                logger.warning(f"🚫 Math scan: BLOCKED SHORT {symbol} - IN SUPPORT ZONE [${s_lower:.4f}-${s_upper:.4f}] | Price at {pos_range:.0f}% of 24h range")
                return None
            
            # NEW: Block LONG if price is in top 12% of range (near resistance even if not in zone)
            # This catches cases like BTC where price is at 91% but just below the resistance zone
            if best_direction == "LONG" and pos_range >= 88:
                logger.warning(f"🚫 Math scan: BLOCKED LONG {symbol} - TOO CLOSE TO RESISTANCE (Range={pos_range:.0f}% >= 88%) | Risk of rejection")
                return None
            
            # NEW: Block SHORT if price is in bottom 12% of range (near support even if not in zone)
            if best_direction == "SHORT" and pos_range <= 12:
                logger.warning(f"🚫 Math scan: BLOCKED SHORT {symbol} - TOO CLOSE TO SUPPORT (Range={pos_range:.0f}% <= 12%) | Risk of bounce")
                return None
            
            # Determine risk based on score
            if best_score >= 80:
                risk_level = "low"
                risk_pct = 0.025  # 2.5%
            elif best_score >= 75:
                risk_level = "medium"
                risk_pct = 0.02  # 2%
            else:
                risk_level = "medium"
                risk_pct = 0.015  # 1.5%
            
            reasons = best_check.get('reasons_for', [])[:3]
            reasoning = f"Math scan score {best_score:.0f}/100. " + ", ".join(reasons)
            
            logger.info(
                f"📊 MATH OPPORTUNITY: {best_direction} {symbol} | "
                f"Score: {best_score:.0f}/100 | {reasoning}"
            )
            
            return {
                "action": best_direction,
                "signal": 1 if best_direction == "LONG" else -1,
                "confidence": best_score / 100,
                "reasoning": reasoning,
                "risk_assessment": risk_level,
                "suggested_risk_pct": risk_pct,
                "source": "math_proactive",
                "math_score": best_score
            }
            
        except Exception as e:
            logger.error(f"Math proactive scan error: {type(e).__name__}: {e}")
            return None
    
    def _ai_validate_position_decision(
        self,
        position_side: str,
        entry_price: float,
        current_price: float,
        unrealized_pnl_pct: float,
        math_action: str,
        math_score: float,
        hold_score: float,
        exit_score: float,
        math_reasoning: str,
        symbol: str
    ) -> Dict[str, Any]:
        """
        AI validates the math-based position monitoring decision.
        Returns adjusted recommendation or None if AI unavailable.
        Math decision is ALWAYS the fallback.
        """
        if not self.use_ai:
            return None
        
        try:
            pnl_emoji = "🟢" if unrealized_pnl_pct >= 0 else "🔴"
            prompt = f"""You are a position management AI validating a math-based decision.

═══════════════════════════════════════════════════════════════════════
CURRENT POSITION STATUS
═══════════════════════════════════════════════════════════════════════
Symbol: {symbol}
Side: {position_side}
Entry: ${entry_price:.4f}
Current: ${current_price:.4f}
{pnl_emoji} PnL: {unrealized_pnl_pct:+.2f}%

═══════════════════════════════════════════════════════════════════════
MATH ANALYSIS RESULT
═══════════════════════════════════════════════════════════════════════
Recommended Action: {math_action.upper()}
Hold Score: {hold_score:.0f}/100
Exit Score: {exit_score:.0f}/100
Adjusted Score: {math_score:.0f}/100
Reasoning: {math_reasoning}

═══════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════
Validate the math decision. You can:
1. CONFIRM: Agree with math recommendation
2. REFINE: Adjust confidence slightly based on market insight
3. OVERRIDE: Only if you see a CRITICAL issue (use sparingly)

Respond ONLY with JSON:
{{"validate": "confirm" or "refine" or "override", "action": "hold" or "close", "confidence_adjustment": -0.1 to +0.1, "note": "brief reason (max 50 chars)"}}"""

            result_text = self._generate_content(prompt)
            if not result_text:
                return None
            
            result_text = result_text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            ai_result = json.loads(result_text.strip())
            
            validation = ai_result.get("validate", "confirm")
            ai_action = ai_result.get("action", math_action)
            confidence_adj = ai_result.get("confidence_adjustment", 0)
            note = ai_result.get("note", "")
            
            logger.info(f"🤖 AI Position Validation: {validation} | Action: {ai_action} | Note: {note}")
            
            return {
                "validation": validation,
                "action": ai_action,
                "confidence_adjustment": confidence_adj,
                "note": note
            }
            
        except Exception as e:
            logger.warning(f"AI position validation failed: {e} - using math decision")
            return None

    def _ai_smart_exit_decision(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        pnl_pct: float,
        pnl_usd: float,
        hold_score: float,
        exit_score: float,
        reversal_signals: List[str],
        context: Dict[str, Any],
        df: pd.DataFrame = None,
        profit_reversal_detected: bool = False,
        reversal_urgency: str = 'none',
        reversal_reason: str = '',
        momentum_hope: int = 0,
        momentum_hope_factors: List[str] = None,
        reversal_score: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        AI makes a POWERFUL exit decision with deep math understanding.
        Uses market microstructure, momentum analysis, and risk assessment.
        
        KEY PRINCIPLE: Balance between cutting losses quickly and giving
        positions room to breathe through normal market noise.
        
        Returns:
            {
                'should_exit': bool,
                'confidence': 0.0-1.0,
                'reasoning': str,
                'suggested_action': 'hold' | 'exit' | 'tighten_sl' | 'wait_for_bounce'
            }
        """
        if not self.use_ai:
            return None
        
        try:
            # === DEEP MATH ANALYSIS ===
            math_analysis = self._calculate_deep_position_metrics(
                df, side, entry_price, current_price, context
            )
            
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            reversal_text = '\n'.join([f"  • {s}" for s in reversal_signals]) if reversal_signals else "  • None detected"
            
            # Build comprehensive context for AI
            trend_status = context.get('trend', 'unknown')
            rsi = context.get('rsi', 50)
            volume_ratio = context.get('volume_ratio', 1.0)
            volatility = context.get('volatility_pct', 0)
            
            # === S/R LEVEL ANALYSIS ===
            sr_levels = self._detect_resistance_support_levels(df, current_price)
            dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
            dist_to_support = sr_levels.get('distance_to_support_pct', 99)
            range_position = sr_levels.get('position_in_range_pct', 50)
            in_r_zone = sr_levels.get('in_resistance_zone', False)
            in_s_zone = sr_levels.get('in_support_zone', False)
            
            # Build S/R context for AI with side-specific warnings
            sr_context = f"Position in 24h range: {range_position:.0f}%\n"
            sr_context += f"  • Distance to Resistance: {dist_to_resistance:.2f}%"
            if in_r_zone:
                sr_context += " 🚫 IN DANGER ZONE"
            elif dist_to_resistance < 1.0:
                sr_context += " ⚠️ VERY CLOSE"
            sr_context += f"\n  • Distance to Support: {dist_to_support:.2f}%"
            if in_s_zone:
                sr_context += " 🚫 IN DANGER ZONE"
            elif dist_to_support < 1.0:
                sr_context += " ⚠️ VERY CLOSE"
            
            # Add side-specific S/R warnings
            if side == 'LONG':
                if in_r_zone:
                    sr_context += "\n\n🚨 LONG IN RESISTANCE ZONE - HIGH EXIT URGENCY!"
                    sr_context += f"\n   Price likely to reject here. Only {dist_to_resistance:.2f}% upside before resistance."
                elif dist_to_resistance < 0.5:
                    sr_context += f"\n\n⚠️ LONG very close to resistance ({dist_to_resistance:.2f}% away) - limited upside!"
                if dist_to_support < 1.0:
                    sr_context += f"\n   ✅ Support nearby ({dist_to_support:.2f}% below) - good protection"
            else:  # SHORT
                if in_s_zone:
                    sr_context += "\n\n🚨 SHORT IN SUPPORT ZONE - HIGH EXIT URGENCY!"
                    sr_context += f"\n   Price likely to bounce here. Only {dist_to_support:.2f}% downside before support."
                elif dist_to_support < 0.5:
                    sr_context += f"\n\n⚠️ SHORT very close to support ({dist_to_support:.2f}% away) - limited downside!"
                if dist_to_resistance < 1.0:
                    sr_context += f"\n   ✅ Resistance nearby ({dist_to_resistance:.2f}% above) - good protection"
            
            # Determine market regime
            if rsi < 30:
                rsi_status = "OVERSOLD (bounce likely)"
            elif rsi > 70:
                rsi_status = "OVERBOUGHT (pullback likely)"
            else:
                rsi_status = f"Neutral ({rsi:.0f})"
            
            prompt = f"""You are an EXPERT AI position manager with deep mathematical understanding. Analyze this position and decide the OPTIMAL action.

═══════════════════════════════════════════════════════════════════════════════
📊 POSITION STATUS
═══════════════════════════════════════════════════════════════════════════════
Symbol: {symbol}
Direction: {side}
Entry Price: ${entry_price:.4f}
Current Price: ${current_price:.4f}
{pnl_emoji} Unrealized PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})
Distance to Entry: {abs(pnl_pct):.2f}%

═══════════════════════════════════════════════════════════════════════════════
📈 MATHEMATICAL ANALYSIS (PhD-Level)
═══════════════════════════════════════════════════════════════════════════════
Math Hold Score: {hold_score:.0f}/100 (higher = position looks good)
Math Exit Score: {exit_score:.0f}/100 (higher = should consider exit)
Score Difference: {hold_score - exit_score:+.0f} (positive = favor HOLD)

Momentum Analysis:
  • 3-bar momentum: {math_analysis.get('momentum_3', 0):+.3f}%
  • 5-bar momentum: {math_analysis.get('momentum_5', 0):+.3f}%
  • 10-bar momentum: {math_analysis.get('momentum_10', 0):+.3f}%
  • Momentum trend: {math_analysis.get('momentum_trend', 'unknown')}

Mean Reversion Analysis:
  • Price vs 20-SMA: {math_analysis.get('price_vs_sma20', 0):+.2f}%
  • Bollinger position: {math_analysis.get('bb_position', 'middle')}
  • Mean reversion probability: {math_analysis.get('mean_reversion_prob', 50):.0f}%

Volatility & Risk:
  • Current volatility: {volatility:.2f}%
  • ATR distance to SL: {math_analysis.get('atr_to_sl', 0):.1f}x ATR
  • Risk/reward at current price: {math_analysis.get('current_rr', 0):.2f}

═══════════════════════════════════════════════════════════════════════════════
📍 SUPPORT/RESISTANCE LEVELS
═══════════════════════════════════════════════════════════════════════════════
{sr_context}

CRITICAL FOR EXITS:
  • LONG near resistance → Higher chance of rejection, consider exit
  • LONG near support → Support may hold, can hold
  • SHORT near support → Higher chance of bounce, consider exit
  • SHORT near resistance → Resistance may reject, can hold

═══════════════════════════════════════════════════════════════════════════════
🔮 MARKET CONDITIONS
═══════════════════════════════════════════════════════════════════════════════
Trend: {trend_status.upper()}
RSI: {rsi_status}
Volume: {volume_ratio:.1f}x average {'(HIGH ACTIVITY)' if volume_ratio > 1.5 else '(normal)'}

Reversal Warning Signals:
{reversal_text}

{'🚨🚨🚨 PROFIT REVERSAL ALERT 🚨🚨🚨' if profit_reversal_detected else ''}
{f'URGENCY: {reversal_urgency.upper()}' if profit_reversal_detected else ''}
{f'REASON: {reversal_reason}' if profit_reversal_detected else ''}
{f'REVERSAL SCORE: {reversal_score}/100 (55+=CRITICAL, 40+=HIGH)' if reversal_score > 0 else ''}
{f'The code detected momentum reversing against our profitable position!' if profit_reversal_detected else ''}
{f'YOU MUST DECIDE: Capture this profit NOW or let it potentially evaporate?' if profit_reversal_detected else ''}

═══════════════════════════════════════════════════════════════════════════════
💪 MOMENTUM HOPE ANALYSIS (Reasons to HOLD)
═══════════════════════════════════════════════════════════════════════════════
Hope Score: {momentum_hope}/50 (higher = more reason to hold through dip)
{chr(10).join(['  • ' + f for f in (momentum_hope_factors or [])]) if momentum_hope_factors else '  • No favorable momentum factors detected'}

{'⚡ STRONG HOPE: Momentum is still WITH us despite the dip!' if momentum_hope >= 30 else ''}
{'📊 MODERATE HOPE: Some favorable factors present' if 15 <= momentum_hope < 30 else ''}
{'⚠️ LOW HOPE: Few reasons to expect recovery' if 0 < momentum_hope < 15 else ''}

IMPORTANT: If REVERSAL SCORE is high but HOPE SCORE is also high, this is a critical decision point!
- High reversal + Low hope = EXIT to protect profit
- High reversal + High hope = Consider holding if trend is intact
- Low reversal + High hope = HOLD, let it run

═══════════════════════════════════════════════════════════════════════════════
🧠 DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════
IMPORTANT CONSIDERATIONS:

FOR POSITIONS IN LOSS:
  • Small loss (< -0.5%): Normal noise - HOLD if momentum supports our direction
  • Medium loss (-0.5% to -1.0%): Check momentum and mean reversion potential
  • Large loss (> -1.0%): Analyze carefully - exit if no recovery signals

FOR POSITIONS IN PROFIT:
  • Small profit (< +0.5%): LET IT RUN - only exit on VERY strong reversal (score>70)
  • Medium profit (+0.5% to +1.0%): Can tighten SL but don't exit prematurely
  • Good profit (> +1.0%): Protect actively, exit on clear reversal signals
  
{'⚡ PROFIT REVERSAL DETECTED - Weight this heavily in your decision!' if profit_reversal_detected else ''}

KEY QUESTION: Based on momentum, mean reversion, and volatility analysis:
Is the current price move likely to CONTINUE against us, or is this a temporary fluctuation?

═══════════════════════════════════════════════════════════════════════════════
📋 YOUR TASK
═══════════════════════════════════════════════════════════════════════════════
Analyze all the math and decide:

1. should_exit: true/false
2. confidence: 0.0 to 1.0 (be honest - only high confidence for clear signals)
3. reasoning: Brief explanation (max 80 chars)
4. suggested_action: One of:
   - "hold": Keep position, math favors our direction
   - "exit": Close now, clear risk of further loss
   - "tighten_sl": Keep but move SL to protect (for profits)
   - "wait_for_bounce": In loss but mean reversion likely, give it a few bars

Respond ONLY with JSON:
{{"should_exit": bool, "confidence": 0.0-1.0, "reasoning": "...", "suggested_action": "hold|exit|tighten_sl|wait_for_bounce"}}"""

            result_text = self._generate_content(prompt)
            if not result_text:
                return None
            
            result_text = result_text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            ai_result = json.loads(result_text.strip())
            
            should_exit = ai_result.get("should_exit", False)
            confidence = ai_result.get("confidence", 0.5)
            reasoning = ai_result.get("reasoning", "")
            suggested_action = ai_result.get("suggested_action", "hold")
            
            logger.info(f"🤖 AI Exit Decision: {suggested_action.upper()} ({confidence:.0%}) - {reasoning}")
            
            return {
                "should_exit": should_exit,
                "confidence": confidence,
                "reasoning": reasoning,
                "suggested_action": suggested_action
            }
            
        except Exception as e:
            logger.warning(f"AI smart exit decision failed: {e}")
            return None
    
    def _calculate_deep_position_metrics(
        self, 
        df: pd.DataFrame, 
        side: str, 
        entry_price: float, 
        current_price: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate deep mathematical metrics for position analysis."""
        metrics = {
            'momentum_3': 0,
            'momentum_5': 0,
            'momentum_10': 0,
            'momentum_trend': 'unknown',
            'price_vs_sma20': 0,
            'bb_position': 'middle',
            'mean_reversion_prob': 50,
            'atr_to_sl': 0,
            'current_rr': 0
        }
        
        try:
            if df is None or len(df) < 20:
                return metrics
            
            closes = df['close'].values if 'close' in df.columns else None
            if closes is None or len(closes) < 20:
                return metrics
            
            current = closes[-1]
            
            # Momentum calculations
            if len(closes) >= 3:
                metrics['momentum_3'] = (current / closes[-3] - 1) * 100
            if len(closes) >= 5:
                metrics['momentum_5'] = (current / closes[-5] - 1) * 100
            if len(closes) >= 10:
                metrics['momentum_10'] = (current / closes[-10] - 1) * 100
            
            # Determine momentum trend
            m3, m5, m10 = metrics['momentum_3'], metrics['momentum_5'], metrics['momentum_10']
            if side == 'LONG':
                if m3 > 0 and m5 > 0:
                    metrics['momentum_trend'] = 'favorable (bullish)'
                elif m3 < 0 and m5 < 0:
                    metrics['momentum_trend'] = 'against us (bearish)'
                else:
                    metrics['momentum_trend'] = 'mixed'
            else:  # SHORT
                if m3 < 0 and m5 < 0:
                    metrics['momentum_trend'] = 'favorable (bearish)'
                elif m3 > 0 and m5 > 0:
                    metrics['momentum_trend'] = 'against us (bullish)'
                else:
                    metrics['momentum_trend'] = 'mixed'
            
            # SMA and mean reversion
            sma20 = np.mean(closes[-20:])
            metrics['price_vs_sma20'] = (current / sma20 - 1) * 100
            
            # Bollinger Band position
            std20 = np.std(closes[-20:])
            upper_bb = sma20 + 2 * std20
            lower_bb = sma20 - 2 * std20
            
            if current > upper_bb:
                metrics['bb_position'] = 'above upper band (overbought)'
                metrics['mean_reversion_prob'] = 70 if side == 'SHORT' else 30
            elif current < lower_bb:
                metrics['bb_position'] = 'below lower band (oversold)'
                metrics['mean_reversion_prob'] = 70 if side == 'LONG' else 30
            elif current > sma20:
                metrics['bb_position'] = 'above SMA (bullish zone)'
                metrics['mean_reversion_prob'] = 55 if side == 'SHORT' else 45
            else:
                metrics['bb_position'] = 'below SMA (bearish zone)'
                metrics['mean_reversion_prob'] = 55 if side == 'LONG' else 45
            
            # ATR-based metrics
            atr = context.get('atr', current * 0.01)
            if atr > 0:
                # How many ATRs from entry
                distance_from_entry = abs(current - entry_price)
                metrics['atr_to_sl'] = distance_from_entry / atr
            
            # Current risk/reward estimate
            pnl_pct = ((current - entry_price) / entry_price * 100) if side == 'LONG' else ((entry_price - current) / entry_price * 100)
            if pnl_pct < 0:
                # In loss - R/R is potential upside vs current drawdown
                potential_upside = 1.5  # Assume TP at 1.5%
                metrics['current_rr'] = potential_upside / abs(pnl_pct) if abs(pnl_pct) > 0 else 10
            else:
                # In profit - R/R is current profit vs SL distance
                sl_distance = 1.5  # Assume 1.5% SL
                metrics['current_rr'] = pnl_pct / sl_distance if sl_distance > 0 else 1
            
        except Exception as e:
            logger.debug(f"Error calculating deep metrics: {e}")
        
        return metrics

    def monitor_position(
        self,
        df: pd.DataFrame,
        position_side: str,  # "LONG" or "SHORT"
        entry_price: float,
        current_price: float,
        atr: float,
        symbol: str,
        unrealized_pnl_pct: float,
        tp1_hit: bool = False,
        tp2_hit: bool = False,
        position_size: float = 0.0  # Actual position size in base currency
    ) -> Dict[str, Any]:
        """
        Monitor an open position and decide whether to hold or close.
        Uses PhD-level math analysis to make the decision.
        SPEED: Target < 50ms for math, < 1.5s with AI validation.
        
        Returns:
            {
                "action": "hold" | "close" | "partial_close",
                "confidence": 0.0-1.0,
                "reasoning": str,
                "math_score": float
            }
        """
        import time
        start_time = time.perf_counter()
        
        try:
            if df is None or len(df) < 20:
                return {"action": "hold", "confidence": 0.5, "reasoning": "Insufficient data", "math_score": 50}
            
            context = self._build_market_context(df, current_price, atr)
            context_time = (time.perf_counter() - start_time) * 1000
            
            # Get math scores for both directions
            position_signal = 1 if position_side == "LONG" else -1
            opposite_signal = -position_signal
            
            hold_check = self._comprehensive_math_check(position_signal, df, current_price, atr, context)
            exit_check = self._comprehensive_math_check(opposite_signal, df, current_price, atr, context)
            
            hold_score = hold_check.get('score', 50)
            exit_score = exit_check.get('score', 50)
            
            # === GRADUATED PROFIT PROTECTION ===
            # Balance between letting profits run to TP1 vs protecting meaningful gains
            # 
            # Philosophy:
            # - Small profit (<$50): Be patient, let it develop toward TP1
            # - Medium profit ($50-100): Light protection, still favor holding
            # - Good profit ($100-200): Real protection, AI/math can decide
            # - Large profit ($200+): Strong protection, don't lose it!
            
            profit_protection_boost = 0
            patience_penalty = 0
            profit_protection_reason = ""
            
            # Calculate unrealized PnL in dollars using actual position value
            position_notional = position_size * entry_price if position_size > 0 else 500  # Default to $500 if unknown
            estimated_pnl_usd = abs(unrealized_pnl_pct * position_notional / 100)
            
            # Track peak profit for this position
            position_key = f"{symbol}_{position_side}"
            if not hasattr(self, '_peak_profits'):
                self._peak_profits = {}
            
            current_peak = self._peak_profits.get(position_key, 0)
            if estimated_pnl_usd > current_peak:
                self._peak_profits[position_key] = estimated_pnl_usd
                current_peak = estimated_pnl_usd
            
            # === GRADUATED APPROACH ===
            if tp1_hit:
                # TP1 already hit - position has proven itself, protect remaining profit
                if unrealized_pnl_pct >= 1.5:
                    profit_protection_boost = 25
                    profit_protection_reason = f"TP1 HIT + HIGH PROFIT (+{unrealized_pnl_pct:.2f}%)"
                elif unrealized_pnl_pct >= 1.0:
                    profit_protection_boost = 20
                    profit_protection_reason = f"TP1 HIT + PROFIT (+{unrealized_pnl_pct:.2f}%)"
                elif unrealized_pnl_pct >= 0.5:
                    profit_protection_boost = 15
                    profit_protection_reason = f"TP1 hit - protecting (+{unrealized_pnl_pct:.2f}%)"
            else:
                # TP1 NOT hit yet - use graduated protection based on dollar profit
                if estimated_pnl_usd >= 200:
                    # $200+ profit - STRONG protection, don't let it evaporate!
                    profit_protection_boost = 25
                    patience_penalty = 0  # No patience penalty for large profits
                    profit_protection_reason = f"LARGE PROFIT (${estimated_pnl_usd:.0f}) - protect it!"
                elif estimated_pnl_usd >= 100:
                    # $100-200 profit - MODERATE protection, let AI/math decide
                    profit_protection_boost = 15
                    patience_penalty = 5  # Small patience penalty
                    profit_protection_reason = f"GOOD PROFIT (${estimated_pnl_usd:.0f}) - consider protecting"
                elif estimated_pnl_usd >= 50:
                    # $50-100 profit - LIGHT protection, still favor holding for TP1
                    profit_protection_boost = 5
                    patience_penalty = 10
                    profit_protection_reason = f"Building profit (${estimated_pnl_usd:.0f})"
                elif unrealized_pnl_pct > 0:
                    # <$50 profit - BE PATIENT, wait for TP1
                    patience_penalty = 15
                    logger.debug(f"⏳ Small profit ${estimated_pnl_usd:.0f} - patience penalty -{patience_penalty}")
                else:
                    # In the red - standard patience
                    patience_penalty = 10
            
            # Profit erosion detection: protect if we've lost significant gains
            if current_peak > 100 and estimated_pnl_usd < current_peak * 0.5:
                # Lost 50%+ of peak profit that was >$100
                erosion_boost = 20
                if erosion_boost > profit_protection_boost:
                    profit_protection_boost = erosion_boost
                    profit_protection_reason = f"PROFIT EROSION (was ${current_peak:.0f}, now ${estimated_pnl_usd:.0f})"
            
            # === S/R ZONE PENALTY: Exit when near danger zones ===
            sr_penalty_to_hold = 0
            sr_boost_to_exit = 0
            sr_reason = ""
            
            sr_levels = self._detect_resistance_support_levels(df, current_price)
            dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
            dist_to_support = sr_levels.get('distance_to_support_pct', 99)
            in_r_zone = sr_levels.get('in_resistance_zone', False)
            in_s_zone = sr_levels.get('in_support_zone', False)
            
            if position_side == 'LONG':
                if in_r_zone or dist_to_resistance < 0.5:
                    # LONG hitting resistance - high exit urgency
                    sr_penalty_to_hold = -25
                    sr_boost_to_exit = 30
                    sr_reason = f"🚫 LONG in/near resistance ({dist_to_resistance:.1f}% away)"
                elif dist_to_resistance < 1.0:
                    # LONG very close to resistance
                    sr_penalty_to_hold = -15
                    sr_boost_to_exit = 20
                    sr_reason = f"⚠️ LONG close to resistance ({dist_to_resistance:.1f}% away)"
                elif dist_to_resistance < 2.0:
                    # LONG approaching resistance
                    sr_penalty_to_hold = -8
                    sr_boost_to_exit = 10
                    sr_reason = f"📍 LONG approaching resistance ({dist_to_resistance:.1f}% away)"
            else:  # SHORT
                if in_s_zone or dist_to_support < 0.5:
                    # SHORT hitting support - high exit urgency
                    sr_penalty_to_hold = -25
                    sr_boost_to_exit = 30
                    sr_reason = f"🚫 SHORT in/near support ({dist_to_support:.1f}% away)"
                elif dist_to_support < 1.0:
                    # SHORT very close to support
                    sr_penalty_to_hold = -15
                    sr_boost_to_exit = 20
                    sr_reason = f"⚠️ SHORT close to support ({dist_to_support:.1f}% away)"
                elif dist_to_support < 2.0:
                    # SHORT approaching support
                    sr_penalty_to_hold = -8
                    sr_boost_to_exit = 10
                    sr_reason = f"📍 SHORT approaching support ({dist_to_support:.1f}% away)"
            
            # Apply S/R adjustments
            if sr_penalty_to_hold != 0:
                hold_score += sr_penalty_to_hold
                exit_score += sr_boost_to_exit
                logger.info(f"{sr_reason} | Hold penalty: {sr_penalty_to_hold}, Exit boost: +{sr_boost_to_exit}")
            
            # Apply profit protection adjustments
            exit_score += profit_protection_boost - patience_penalty
            
            if profit_protection_boost > 0:
                logger.info(f"💰 {profit_protection_reason} | Exit boost: +{profit_protection_boost} → Exit:{exit_score:.0f}")
            
            # Calculate position health metrics
            pnl_factor = 0
            if unrealized_pnl_pct > 2.0:
                pnl_factor = 15  # Good profit, slight bias to hold
            elif unrealized_pnl_pct > 1.0:
                pnl_factor = 10
            elif unrealized_pnl_pct > 0.5:
                pnl_factor = 5  # Getting close to TP1, lean toward hold
            elif unrealized_pnl_pct < -1.5:
                pnl_factor = -20  # Losing, bias toward exit
            elif unrealized_pnl_pct < -0.5:
                pnl_factor = -10
            
            # Bonus for hitting TPs
            tp_bonus = 0
            if tp2_hit:
                tp_bonus = 10  # Already took profit, can be more aggressive
            elif tp1_hit:
                tp_bonus = 5
            
            # TP proximity bonus - removed in graduated approach (handled by profit tiers)
            tp_proximity_bonus = 0
            
            # Final adjusted hold score includes: base hold + pnl factor + tp bonus
            adjusted_hold_score = hold_score + pnl_factor + tp_bonus + tp_proximity_bonus
            
            # Decision thresholds - RAISED to prevent premature exits
            STRONG_EXIT_THRESHOLD = 80  # Was 70 - need stronger signal to exit
            WEAK_HOLD_THRESHOLD = 35    # Was 40 - more tolerant of weak holds
            
            reasons_for_hold = hold_check.get('reasons_for', [])[:2]
            reasons_for_exit = exit_check.get('reasons_for', [])[:2]
            
            action = "hold"
            reasoning = ""
            confidence = 0.5
            
            # Decision logic - MODIFIED to be more patient
            if exit_score >= STRONG_EXIT_THRESHOLD and exit_score > adjusted_hold_score + 20:  # Was +15
                # Strong reversal signal - but require higher margin
                action = "close"
                confidence = exit_score / 100
                reasoning = f"Strong reversal signal (Exit:{exit_score:.0f} vs Hold:{adjusted_hold_score:.0f}). {', '.join(reasons_for_exit)}"
            
            elif adjusted_hold_score < WEAK_HOLD_THRESHOLD and unrealized_pnl_pct < -1.0:  # Was < 0
                # Weak hold + SIGNIFICANT loss = exit (not just any loss)
                action = "close"
                confidence = (100 - adjusted_hold_score) / 100
                reasoning = f"Weak hold score ({adjusted_hold_score:.0f}) + losing position ({unrealized_pnl_pct:+.2f}%)"
            
            elif adjusted_hold_score < 45 and unrealized_pnl_pct > 2.0 and tp1_hit:  # Was > 1.0, no TP1 check
                # Weak hold but good profit AND TP1 already hit = take remaining profit
                action = "close"
                confidence = 0.7
                reasoning = f"Weakening after TP1 (score:{adjusted_hold_score:.0f}) - securing {unrealized_pnl_pct:+.2f}% remaining"
            
            elif tp2_hit and adjusted_hold_score < 55:  # Was < 60
                # TP2 hit and momentum fading = close remaining
                action = "close"
                confidence = 0.65
                reasoning = f"TP2 hit + fading momentum (score:{adjusted_hold_score:.0f})"
            
            else:
                # Hold position
                action = "hold"
                confidence = adjusted_hold_score / 100
                reasoning = f"Position healthy (Hold:{adjusted_hold_score:.0f}, Exit:{exit_score:.0f}). {', '.join(reasons_for_hold)}"
            
            # === AI VALIDATION (optional, math is primary) ===
            ai_validation = None
            final_action = action
            final_confidence = confidence
            final_reasoning = reasoning
            
            # Only call AI for borderline cases to save API calls
            is_borderline = (
                (action == "hold" and adjusted_hold_score < 55) or  # Weak hold
                (action == "close" and confidence < 0.7) or  # Uncertain close
                (abs(hold_score - exit_score) < 15)  # Close scores
            )
            
            if self.use_ai and is_borderline:
                ai_validation = self._ai_validate_position_decision(
                    position_side=position_side,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    math_action=action,
                    math_score=adjusted_hold_score,
                    hold_score=hold_score,
                    exit_score=exit_score,
                    math_reasoning=reasoning,
                    symbol=symbol
                )
                
                if ai_validation:
                    validation_type = ai_validation.get("validation", "confirm")
                    
                    if validation_type == "override":
                        # AI wants to override - use graduated logic based on profit
                        ai_action = ai_validation.get("action", action)
                        
                        # Calculate estimated profit in dollars
                        est_profit_usd = abs(unrealized_pnl_pct * 23000 / 100)
                        
                        # GRADUATED AI OVERRIDE RULES:
                        # - $200+ profit: ALWAYS allow AI to protect (close or hold)
                        # - $100-200: Allow AI override if it wants to CLOSE (protect profit)
                        # - $50-100: Allow only if math is truly borderline (45-55)
                        # - <$50: Block AI override to close, let position develop
                        
                        should_accept = False
                        block_reason = ""
                        
                        if est_profit_usd >= 200:
                            # Large profit - trust AI judgment fully
                            should_accept = True
                            logger.info(f"💰 Large profit ${est_profit_usd:.0f} - AI override allowed")
                        elif est_profit_usd >= 100 and ai_action == "close":
                            # Good profit and AI wants to protect it
                            should_accept = True
                            logger.info(f"💰 Good profit ${est_profit_usd:.0f} - AI protection allowed")
                        elif est_profit_usd >= 50 and 45 <= adjusted_hold_score <= 55:
                            # Medium profit, truly borderline math
                            should_accept = True
                        elif est_profit_usd < 50 and ai_action == "close":
                            # Small profit - don't let AI close prematurely
                            block_reason = f"Profit too small (${est_profit_usd:.0f}). Wait for development."
                        elif 40 <= adjusted_hold_score <= 60:
                            # Standard borderline case
                            should_accept = True
                        else:
                            block_reason = f"Math score {adjusted_hold_score:.0f} outside borderline range"
                        
                        if should_accept:
                            final_action = ai_action
                            final_confidence = min(1.0, max(0.4, confidence + ai_validation.get("confidence_adjustment", 0)))
                            final_reasoning = f"{reasoning} [AI override: {ai_validation.get('note', '')}]"
                            logger.info(f"⚠️ AI override accepted: {action} → {final_action}")
                        else:
                            logger.info(f"🛡️ AI override BLOCKED - {block_reason}")
                    
                    elif validation_type == "refine":
                        # AI refines confidence
                        final_confidence = min(1.0, max(0.4, confidence + ai_validation.get("confidence_adjustment", 0)))
                        final_reasoning = f"{reasoning} [AI: {ai_validation.get('note', '')}]"
                    
                    # "confirm" - no changes needed
            
            # Calculate total time
            total_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                f"🔍 Position Monitor [{position_side}]: {final_action.upper()} | "
                f"Hold:{adjusted_hold_score:.0f} Exit:{exit_score:.0f} | "
                f"PnL:{unrealized_pnl_pct:+.2f}% | AI:{ai_validation is not None} | ⚡{total_time:.0f}ms | {final_reasoning[:50]}..."
            )
            
            return {
                "action": final_action,
                "confidence": final_confidence,
                "reasoning": final_reasoning,
                "math_score": adjusted_hold_score,
                "hold_score": hold_score,
                "exit_score": exit_score,
                "pnl_factor": pnl_factor,
                "ai_validated": ai_validation is not None,
                "ai_validation": ai_validation.get("validation") if ai_validation else None
            }
            
        except Exception as e:
            logger.error(f"Position monitor error: {type(e).__name__}: {e}")
            return {"action": "hold", "confidence": 0.5, "reasoning": f"Error: {e}", "math_score": 50}
    
    def proactive_scan(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Proactively scans market for opportunities using PhD-level mathematical analysis.
        Math is the PRIMARY decision maker. AI only confirms/refines the math decision.
        Returns a trade suggestion or None.
        """
        # Require minimum data
        if df is None or len(df) < 20:
            logger.debug(f"Proactive scan skipped: insufficient data ({len(df) if df is not None else 0}/20 bars)")
            return None
        
        # === PHASE 1: PhD-LEVEL MATHEMATICAL ANALYSIS ===
        # Always run math scan first - it's the foundation
        math_result = self._math_proactive_scan(df, current_price, atr, symbol)
        
        # If math says no opportunity with high confidence, trust it
        if not math_result:
            # Already logged at INFO level in _math_proactive_scan
            return None
        
        # Math found an opportunity - if AI is not available, use math result directly
        if not self.use_ai:
            return math_result
        
        # === PHASE 2: AI VALIDATION OF MATH DECISION ===
        # AI can only CONFIRM or REFINE the math decision, not override it
        try:
            context = self._build_market_context(df, current_price, atr)
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            # Get math details for AI context
            math_score = math_result.get('math_score', 0)
            math_direction = math_result.get('action', 'UNKNOWN')
            math_reasoning = math_result.get('reasoning', '')
            math_confidence = math_result.get('confidence', 0)
            
            prompt = f"""You are validating a MATHEMATICAL TRADING DECISION. The math has already identified an opportunity.

═══════════════════════════════════════════════════════════════════════
MATHEMATICAL ANALYSIS RESULT (PhD-Level - This is the PRIMARY decision)
═══════════════════════════════════════════════════════════════════════
Direction: {math_direction}
Math Score: {math_score:.0f}/100
Confidence: {math_confidence:.0%}
Analysis: {math_reasoning}

═══════════════════════════════════════════════════════════════════════
MARKET DATA for {symbol}
═══════════════════════════════════════════════════════════════════════
Current Price: ${context['current_price']}
1-Hour Change: {context['price_change_1h']}%
5-Min Change: {context['price_change_5m']}%
Volume Ratio: {context['volume_ratio']}x
Trend: {context['trend']}
Volatility (ATR%): {context['volatility_pct']}%
SMA10: ${context['sma_10']} | SMA20: ${context['sma_20']}

═══════════════════════════════════════════════════════════════════════
PERFORMANCE CONTEXT
═══════════════════════════════════════════════════════════════════════
Total Trades: {perf.get('total_trades', 0)}
Win Rate: {perf.get('win_rate', 0)}%
Current Streak: {perf.get('consecutive_wins', 0)}W / {perf.get('consecutive_losses', 0)}L
Session: {market['session']} | Activity: {market['activity_level']}

═══════════════════════════════════════════════════════════════════════
YOUR TASK (VALIDATION ONLY)
═══════════════════════════════════════════════════════════════════════
The MATH has identified a {math_direction} opportunity with {math_score:.0f}/100 score.

You can:
1. CONFIRM: Agree with math (respond with same direction)
2. REFINE: Adjust risk assessment if you see specific concerns
3. VETO: Only if you see a CRITICAL flaw the math missed (rare)

IMPORTANT: Math is the primary decision maker. Only veto if there's a clear mathematical error or critical market condition the algorithm couldn't detect.

Respond ONLY with JSON:
{{"validate": "confirm" or "refine" or "veto", "direction": "{math_direction}", "confidence_adjustment": 0.0 to 0.1 (add or subtract), "risk_assessment": "low/medium/high", "note": "brief reason"}}"""

            result_text = self._generate_content(prompt)
            if not result_text:
                # AI failed, use math result
                return math_result
            
            result_text = result_text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            ai_validation = json.loads(result_text.strip())
            
            validation = ai_validation.get("validate", "confirm")
            
            if validation == "veto":
                # AI vetoed - log it but still require strong justification
                veto_reason = ai_validation.get("note", "No reason given")
                logger.warning(f"⚠️ AI vetoed math decision: {veto_reason}")
                # Only accept veto if math score wasn't very high
                if math_score >= 80:
                    logger.info(f"🛡️ Math score {math_score:.0f} is high - overriding AI veto")
                    return math_result
                return None
            
            # Confirm or refine - use math result with possible adjustments
            confidence_adj = ai_validation.get("confidence_adjustment", 0)
            final_confidence = min(1.0, max(0.5, math_confidence + confidence_adj))
            
            risk_assessment = ai_validation.get("risk_assessment", math_result.get("risk_assessment", "medium"))
            
            logger.info(
                f"🤖📊 MATH+AI OPPORTUNITY: {math_direction} {symbol} | "
                f"Math: {math_score:.0f}/100 | AI: {validation} | "
                f"Final Conf: {final_confidence:.0%}"
            )
            
            return {
                "action": math_direction,
                "signal": 1 if math_direction == "LONG" else -1,
                "confidence": final_confidence,
                "reasoning": f"Math({math_score:.0f}/100): {math_reasoning}. AI: {validation}",
                "risk_assessment": risk_assessment,
                "suggested_risk_pct": math_result.get("suggested_risk_pct", 0.02),
                "source": "math_ai_combined",
                "math_score": math_score
            }
            
        except Exception as e:
            logger.error(f"AI validation error: {e} - using math result")
            return math_result

    async def chat(self, user_message: str, trading_context: str = "") -> str:
        """
        Chat with the AI about trading, market analysis, or general questions.
        Maintains conversation history for context across messages.
        
        Args:
            user_message: The user's message
            trading_context: Current trading context (positions, balance, etc.)
            
        Returns:
            AI response string
        """
        logger.info(f"💬 CHAT: Received message: '{user_message[:50]}...'")
        logger.info(f"💬 CHAT: AI Provider={self.ai_provider}, use_ai={self.use_ai}, client={self.client is not None}, model={self.model is not None}, anthropic={self.anthropic_client is not None}")
        
        # Check for AI availability - support Claude (anthropic_client), new Gemini SDK (client), and legacy Gemini SDK (model)
        if not self.use_ai or (not self.anthropic_client and not self.client and not self.model):
            logger.warning("AI chat unavailable - using simple response")
            return self._simple_chat_response(user_message)
        
        try:
            # Build conversation history context
            history_text = ""
            if self.chat_history:
                history_text = "\n\nRecent Conversation History:\n"
                for msg in self.chat_history[-10:]:  # Last 10 exchanges
                    role = "Trader" if msg["role"] == "user" else "Julaba"
                    history_text += f"{role}: {msg['content']}\n"
            
            # Build trade performance summary
            trade_summary = ""
            if self.total_wins + self.total_losses > 0:
                win_rate = self.total_wins / (self.total_wins + self.total_losses) * 100
                trade_summary = f"\n\nMy Trade Performance: {self.total_wins}W/{self.total_losses}L ({win_rate:.1f}% win rate)"
                if self.consecutive_wins > 0:
                    trade_summary += f" | Current streak: {self.consecutive_wins} wins 🔥"
                elif self.consecutive_losses > 0:
                    trade_summary += f" | Current streak: {self.consecutive_losses} losses 📉"
            
            prompt = f"""You are Julaba, a concise AI trading assistant with PhD-level trading mathematics.

═══════════════════════════════════════════════════════════════════════
SYSTEM ARCHITECTURE (when user asks "how does X work")
═══════════════════════════════════════════════════════════════════════
**Trading Engine** (bot.py):
- Connects to Bybit futures via CCXT library
- Manages positions with dynamic stop-loss (ATR-based) and 3-tier take profits
- Multi-position support: up to 2 positions on DIFFERENT symbols simultaneously
- Smart exit: Deep loss protection (-3%), profit protection, reversal detection

**AI Filter** (ai_filter.py - that's me!):
- Uses Claude claude-opus-4-20250514 for analysis and chat
- Position monitoring: Real-time hold/exit scoring (0-100 scale)
- Math-based decisions: MTF alignment, volatility, trend strength, RSI, ADX

**Market Scanner** (unified scanner):
- Scans top 10 opportunities from 50+ pairs
- Combined score = 60% PhD Math + 40% AI confidence
- Threshold: 40 minimum to consider for trading
- Updates every 3 minutes (180s cache)

**Dashboard** (dashboard.py):
- Real-time web UI at port 5001
- Shows all positions, opportunities, PhD math scores
- Chart generation with technical indicators

**Risk Management**:
- Position sizing based on risk_pct (default 2% per trade)
- Max drawdown protection, consecutive loss limits
- ATR multiplier for stop-loss distance

**ML Classifier** (ml_predictor.py):
- Learns from trade outcomes (wins/losses)
- Provides probability scoring for new signals
- Can be retrained with /ml_train command

**Indicators** (indicator.py):
- RSI, MACD, Bollinger Bands, ATR, ADX
- Volume analysis, price action patterns
- Multi-timeframe analysis (15m, 1h, 4h)

**AI Modes**:
- AUTONOMOUS: Bot trades on its own when math criteria met
- ADVISORY: AI analyzes but user must confirm
- FILTER: Only validates signals, doesn't initiate
- HYBRID: Suggests trades but doesn't execute

═══════════════════════════════════════════════════════════════════════
MODE LOGIC
═══════════════════════════════════════════════════════════════════════
  - If mode is AUTONOMOUS: Only open/close trades if strict math criteria are met. If criteria fail, say "NO TRADE" and explain why.
  - If user requests a trade (manual override): Always do the math, give your recommendation (with logic), and provide the command for user to execute. Do NOT auto-execute.
  - If mode is ADVISORY: Always provide math verdict and command, never auto-execute.
  - If mode is FILTER: Only validate technical signals, never execute trades.
  - If mode is HYBRID: Suggest trades when no technical signal, but never auto-execute.
If action needed, include the command for the user to execute manually.
Always clarify which mode you are in at the start of your response.

═══════════════════════════════════════════════════════════════════════
CRITICAL RULES - FOLLOW THESE EXACTLY
═══════════════════════════════════════════════════════════════════════
1. BE CONCISE - Max 3-4 sentences unless user asks for details
2. BE ACTION-ORIENTED - If user wants to do something, DO IT (include command)
3. NEVER say "here is the command" without actually providing it
4. NEVER trail off or leave responses incomplete
5. If asked "can you X?" - answer briefly, then offer to do it
6. When asked for STATUS/SYSTEM STATUS - SHOW THE ACTUAL DATA from CURRENT STATUS section below

═══════════════════════════════════════════════════════════════════════
HOW TO SHOW SYSTEM STATUS
═══════════════════════════════════════════════════════════════════════
When user asks "status", "system status", "how is it going", "what's happening":
SHOW THE ACTUAL NUMBERS from the CURRENT STATUS section. Example format:

📊 **System Status**
• Position: [LONG/SHORT/NONE] on [SYMBOL]
• Entry: $X.XXXX | Current: $X.XXXX | P&L: $X.XX
• Balance: $X,XXX.XX
• AI Mode: [MODE] | ML: [trained/learning]
• Market: [REGIME] | Score: XX | ADX: XX | RSI: XX

DO NOT just say "we have a position with small profit" - SHOW THE NUMBERS!

═══════════════════════════════════════════════════════════════════════
HOW TO CLOSE A POSITION
═══════════════════════════════════════════════════════════════════════
If user wants to close (says "close", "exit", "close position"):
Say briefly "Closing position." and include:
```command
{{"action": "close_trade"}}
```

If user asks IF you CAN close: Say "Yes! Just say 'close' and I'll do it."

═══════════════════════════════════════════════════════════════════════
HOW TO OPEN A TRADE
═══════════════════════════════════════════════════════════════════════
MULTI-POSITION SUPPORT: We can hold up to 2 positions simultaneously on DIFFERENT symbols!
If user specifies a symbol (e.g., "open long ETH"), use that symbol in the command.

If user says "open long", "go long", "buy", "long" (without specifying symbol):
```command
{{"action": "open_trade", "side": "long"}}
```

If user says "open short", "go short", "sell", "short" (without specifying symbol):
```command
{{"action": "open_trade", "side": "short"}}
```

If user specifies a symbol (e.g., "open long on ETH", "buy LINK", "short SOL"):
```command
{{"action": "open_trade", "side": "long", "symbol": "ETHUSDT"}}
```
```command
{{"action": "open_trade", "side": "short", "symbol": "SOLUSDT"}}
```

IMPORTANT: We can have 2 positions at once on DIFFERENT symbols. Don't refuse if we already have 1 position - just open on a different symbol!

═══════════════════════════════════════════════════════════════════════
PERSONALITY
═══════════════════════════════════════════════════════════════════════
- Friendly and helpful but CONCISE
- For general chat: be conversational
- For trading: be brief and action-oriented
- You are the trader's companion

═══════════════════════════════════════════════════════════════════════
TRADING ANALYSIS (only when asked "should I trade?")
═══════════════════════════════════════════════════════════════════════
Quick criteria check:
- Regime: TRENDING=✅ CHOPPY=❌
- ADX>25=✅ ADX<20=❌
- Score>=60=✅ Score<50=❌
- RSI 30-70=safe

Give brief verdict: "TRADE [LONG/SHORT]" or "NO TRADE" + 1 reason.

═══════════════════════════════════════════════════════════════════════
PAIR SWITCHING (when asked about other pairs)
═══════════════════════════════════════════════════════════════════════
Switch only if: Score diff > 15, new pair Score >= 60, ADX >= 25
NEVER switch mid-trade. Give brief: "SWITCH to X" or "STAY with X" + 1 reason.

═══════════════════════════════════════════════════════════════════════
COMMANDS (when user asks to DO something)
═══════════════════════════════════════════════════════════════════════
**Settings Commands:**
```command {{"action": "set_param", "param": "PARAM_NAME", "value": VALUE}}```

Available Parameters:
- risk_pct: Position size as % of balance (0.01-0.05, default 0.02 = 2%)
- ai_confidence: Minimum AI confidence for trades (50-85, default 65)
- atr_mult: Stop-loss ATR multiplier (1.0-3.0, default 1.5)
- tp1_r: Take profit 1 risk multiple (1.0-3.0, default 1.5)
- tp2_r: Take profit 2 risk multiple (2.0-5.0, default 2.5)
- tp3_r: Take profit 3 risk multiple (3.0-8.0, default 4.0)
- ai_mode: Trading mode (autonomous, advisory, filter, hybrid)
- paused: Pause/resume bot (true/false)
- ml_learn_all_trades: Learn from ALL trades vs only autonomous (true/false)
- proactive_threshold: Scanner aggressiveness (50-85, lower=more trades)

**Symbol Commands:**
```command {{"action": "switch_symbol", "symbol": "SOL"}}```

**Trade Commands:**
```command {{"action": "close_trade"}}```
```command {{"action": "close_trade", "symbol": "ETHUSDT"}}```
```command {{"action": "open_trade", "side": "long"}}```
```command {{"action": "open_trade", "side": "short"}}```
```command {{"action": "open_trade", "side": "long", "symbol": "ETHUSDT"}}```
```command {{"action": "open_trade", "side": "short", "symbol": "LINKUSDT"}}```

**Telegram Slash Commands (tell user to type these):**
- /status - Current positions and balance
- /positions - Detailed position info
- /market - Market scanner opportunities
- /analyze - Full technical analysis
- /chart - Generate price chart
- /ml_stats - ML model performance
- /risk - Risk manager status
- /help - All available commands

═══════════════════════════════════════════════════════════════════════
CURRENT STATUS
═══════════════════════════════════════════════════════════════════════
{trading_context if trading_context else "No trading context available"}
{trade_summary}

═══════════════════════════════════════════════════════════════════════
HISTORY: {history_text if history_text else "None"}
═══════════════════════════════════════════════════════════════════════
Trader: "{user_message}"

REMEMBER: Be concise (3-4 sentences max). If action needed, include the command."""

            # Run blocking API call in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            ai_response = await loop.run_in_executor(None, self._generate_content, prompt)
            if not ai_response:
                return self._simple_chat_response(user_message)
            ai_response = ai_response.strip()
            
            # === STRICT MATHEMATICAL COMMAND VALIDATION ===
            # Commands require explicit action verbs - no guessing, no interpretation
            import re
            
            user_msg_lower = user_message.lower().strip()
            
            # Action verbs that indicate user wants to change something
            action_verbs = [
                'change', 'set', 'switch', 'reduce', 'increase', 'adjust', 
                'make', 'put', 'raise', 'lower', 'modify', 'update', 'enable',
                'disable', 'turn on', 'turn off', 'activate', 'deactivate',
                'open', 'close', 'exit', 'buy', 'sell', 'go long', 'go short', 'pause', 'resume',
                'execute', 'confirm', 'proceed', 'do it', 'yes'
            ]
            
            has_action_verb = any(verb in user_msg_lower for verb in action_verbs)
            
            # Acknowledgements - NEVER trigger commands
            acknowledgement_words = [
                'okay', 'ok', 'yes', 'got it', 'thanks', 'thank you', 'alright', 
                'cool', 'noted', 'understood', 'sure', 'fine', 'great', 'good',
                'nice', 'perfect', 'awesome', 'kk', 'k', 'yep', 'yup', 'right',
                'i see', 'makes sense', 'roger', 'copy', 'affirmative', 'hmm',
                'ah', 'oh', 'hm', 'interesting', 'wow', 'lol', 'haha'
            ]
            
            is_acknowledgement = (
                user_msg_lower in acknowledgement_words or 
                (len(user_msg_lower) < 12 and any(ack == user_msg_lower for ack in acknowledgement_words))
            )
            
            # Questions without action verbs don't trigger commands
            is_pure_question = (
                any(q in user_msg_lower for q in ['what ', 'how ', 'why ', 'when ', 'where ', '?']) 
                and not has_action_verb
            )
            
            # User explicitly asking for/about commands - don't strip!
            asking_for_command = any(phrase in user_msg_lower for phrase in [
                'command', 'show me', 'give me', 'what is the', 'how do i',
                'reset', 'daily loss', 'daily_loss', 'override', 'halt'
            ])
            
            # MATHEMATICAL RULE: Strip commands unless explicit action is requested
            if '```command' in ai_response:
                should_strip = False
                reason = ""
                
                if is_acknowledgement:
                    should_strip = True
                    reason = f"acknowledgement '{user_message}'"
                elif asking_for_command:
                    should_strip = False  # User wants to see the command!
                elif not has_action_verb:
                    should_strip = True
                    reason = f"no action verb in '{user_message[:30]}'"
                elif is_pure_question:
                    should_strip = True
                    reason = f"question without action"
                elif len(user_msg_lower) < 5:
                    should_strip = True
                    reason = f"message too short to be a command"
                
                if should_strip:
                    ai_response = re.sub(r'```command\s*\n?\{[^}]+\}\s*\n?```', '', ai_response)
                    ai_response = ai_response.strip()
                    logger.info(f"🛡️ MATH GUARD: Stripped unauthorized command - {reason}")
            
            # === COMPREHENSIVE ANTI-HALLUCINATION CHECK ===
            # Detect if AI is making claims that contradict the trading_context
            response_lower = ai_response.lower()
            hallucinations_detected = []
            
            # 1. POSITION HALLUCINATION
            position_claim_phrases = [
                "i have opened", "i opened", "i've opened", "position is open",
                "we have a", "we are in a", "holding a long", "holding a short",
                "current position", "your position", "our position"
            ]
            claims_position = any(phrase in response_lower for phrase in position_claim_phrases)
            no_actual_position = "POSITION: **NONE**" in trading_context or "POSITION: None" in trading_context
            if claims_position and no_actual_position:
                hallucinations_detected.append("position")
                logger.warning("🚨 AI hallucinated: claimed position but none exists!")
            
            # 2. WRONG SYMBOL CLAIMS
            # Extract actual symbol from context
            import re
            symbol_match = re.search(r'Symbol:\s*(\w+)', trading_context)
            actual_symbol = symbol_match.group(1) if symbol_match else None
            if actual_symbol:
                actual_base = actual_symbol.replace('USDT', '').lower()
                wrong_symbol_phrases = [
                    ("trading btc", "btc"), ("trading eth", "eth"), ("trading sol", "sol"),
                    ("on btc", "btc"), ("on eth", "eth"), ("on sol", "sol"),
                    ("trading link", "link"), ("trading tia", "tia"), ("trading inj", "inj")
                ]
                for phrase, sym in wrong_symbol_phrases:
                    if phrase in response_lower and sym != actual_base:
                        hallucinations_detected.append(f"symbol (said {sym.upper()}, actual {actual_symbol})")
                        logger.warning(f"🚨 AI hallucinated: said trading {sym.upper()} but actual is {actual_symbol}")
                        break
            
            # 3. BALANCE HALLUCINATION (check for wildly wrong numbers)
            balance_match = re.search(r'Balance:\s*\$([0-9,]+(?:\.[0-9]+)?)', trading_context)
            if balance_match:
                actual_balance = float(balance_match.group(1).replace(',', ''))
                # Find balance claims in response
                claimed_balances = re.findall(r'\$([0-9,]+(?:\.[0-9]{2})?)', ai_response)
                for claim in claimed_balances:
                    try:
                        claimed = float(claim.replace(',', ''))
                        if claimed > 1000 and abs(claimed - actual_balance) / actual_balance > 0.25:
                            hallucinations_detected.append(f"balance (claimed ${claimed:,.0f}, actual ${actual_balance:,.0f})")
                            logger.warning(f"🚨 AI hallucinated: claimed ${claimed} but actual is ${actual_balance:.2f}")
                            break
                    except (ValueError, TypeError):
                        pass  # Non-critical: couldn't parse balance claim
            
            # 4. WIN RATE HALLUCINATION
            winrate_match = re.search(r'Win Rate:\s*([0-9.]+)%', trading_context)
            if winrate_match:
                actual_wr = float(winrate_match.group(1))
                claimed_wr_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*win', response_lower)
                for claimed in claimed_wr_matches:
                    try:
                        claimed_wr = float(claimed)
                        if abs(claimed_wr - actual_wr) > 20:  # More than 20% off
                            hallucinations_detected.append(f"win rate (claimed {claimed_wr}%, actual {actual_wr:.1f}%)")
                            logger.warning(f"🚨 AI hallucinated: claimed {claimed_wr}% win rate but actual is {actual_wr:.1f}%")
                            break
                    except (ValueError, TypeError):
                        pass  # Non-critical: couldn't parse win rate claim
            
            # 5. MODE/STATUS HALLUCINATION
            if "AI Mode: autonomous" in trading_context:
                if "mode is filter" in response_lower or "in filter mode" in response_lower:
                    hallucinations_detected.append("mode (said filter, actual autonomous)")
            elif "AI Mode: filter" in trading_context:
                if "mode is autonomous" in response_lower or "in autonomous mode" in response_lower:
                    hallucinations_detected.append("mode (said autonomous, actual filter)")
            
            if "Paused: False" in trading_context and ("bot is paused" in response_lower or "i am paused" in response_lower):
                hallucinations_detected.append("status (said paused, but bot is running)")
            elif "Paused: True" in trading_context and ("bot is running" in response_lower or "bot is active" in response_lower):
                hallucinations_detected.append("status (said running, but bot is paused)")
            
            # If hallucinations detected, don't save to history and add warning
            if hallucinations_detected:
                logger.warning(f"🚨 AI hallucinations detected: {hallucinations_detected} - NOT saving to history")
                warning = "\n\n⚠️ *Reality Check - Please verify on dashboard:*\n"
                for h in hallucinations_detected[:3]:  # Max 3 corrections
                    warning += f"• Incorrect claim about {h}\n"
                return ai_response + warning
            
            # Save to conversation history - only if not hallucinating
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Keep only recent history (limit to prevent long context)
            if len(self.chat_history) > self.max_chat_history:
                self.chat_history = self.chat_history[-self.max_chat_history:]
            
            # Persist to disk
            self._save_chat_history()
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            return self._simple_chat_response(user_message)
    
    def clear_chat_history(self):
        """Clear chat history to prevent false memories."""
        self.chat_history = []
        self._save_chat_history()
        logger.info("Chat history cleared")
    
    def _save_chat_history(self):
        """Save chat history to disk for persistence."""
        try:
            with open(CHAT_HISTORY_FILE, 'w') as f:
                json.dump(self.chat_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
    
    def _load_chat_history(self):
        """Load chat history from disk."""
        try:
            if CHAT_HISTORY_FILE.exists():
                with open(CHAT_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                # Handle both old format {"messages": [...]} and new format [...]
                if isinstance(data, dict) and "messages" in data:
                    self.chat_history = data["messages"]
                elif isinstance(data, list):
                    self.chat_history = data
                else:
                    self.chat_history = []
                logger.debug(f"Loaded {len(self.chat_history)} chat history entries")
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            self.chat_history = []
    
    def _simple_chat_response(self, message: str) -> str:
        """Simple rule-based chat responses when AI is not available."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey", "sup"]):
            return "👋 Hey there! I'm Julaba, your trading assistant. How can I help you today?"
        
        if any(word in message_lower for word in ["how are you", "how's it going"]):
            return "🤖 I'm running smoothly and watching the markets! How can I help you?"
        
        if any(word in message_lower for word in ["help", "what can you do"]):
            return "🤖 I can help you with:\n\n• Check /status for bot status\n• Use /market for price info\n• See /positions for open trades\n• Try /pnl for profit/loss\n\nOr just chat with me about trading! 📊"
        
        if any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! 😊 Let me know if you need anything else."
        
        if any(word in message_lower for word in ["price", "market", "link"]):
            return "📊 Use /market to see current price, volume, and market data!"
        
        if any(word in message_lower for word in ["trade", "position", "buy", "sell"]):
            return "📈 Check /positions for open trades or /signals for recent trading signals!"
        
        if any(word in message_lower for word in ["profit", "loss", "pnl", "money"]):
            return "💰 Use /pnl to see your profit/loss summary or /balance for your current balance!"
        
        return "🤖 I'm here to help! Try asking about trading, or use /help to see all commands."
