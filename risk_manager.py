"""
Risk Manager Module for Julaba
Centralized risk management with dynamic position sizing, drawdown limits, and time-based cooldowns.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("Julaba.RiskManager")


@dataclass
class TradeOutcome:
    """Record of a completed trade."""
    timestamp: datetime
    symbol: str
    side: str
    pnl: float
    pnl_pct: float
    duration_minutes: int
    ai_approved: bool = True
    ai_confidence: float = 0.0


class RiskManager:
    """
    Centralized Risk Manager for Julaba.
    
    Features:
    - Dynamic position sizing (Kelly Criterion + volatility)
    - Daily/Weekly loss limits with circuit breakers
    - Time-based cooldown after losses
    - Streak-based risk adjustment
    - Correlation-aware position limits
    - Portfolio-level multi-pair risk management (ML Acceleration Plan)
    """
    
    def __init__(
        self,
        base_risk_pct: float = 0.02,
        max_risk_pct: float = 0.04,
        min_risk_pct: float = 0.005,
        daily_loss_limit: float = 0.05,
        weekly_loss_limit: float = 0.10,
        cooldown_minutes: int = 30,
        max_consecutive_losses: int = 3,
        # === PORTFOLIO-LEVEL LIMITS (ML Acceleration Plan) ===
        max_total_positions: int = 2,  # Max simultaneous positions across all pairs
        max_correlated_positions: int = 2,  # All crypto is correlated
        portfolio_max_risk_pct: float = 0.04  # Max total portfolio risk at any time
    ):
        # Base parameters
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        
        # Loss limits
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        
        # Cooldown settings
        self.cooldown_minutes = cooldown_minutes
        self.max_consecutive_losses = max_consecutive_losses
        
        # === PORTFOLIO-LEVEL LIMITS (ML Acceleration Plan) ===
        self.max_total_positions = max_total_positions
        self.max_correlated_positions = max_correlated_positions
        self.portfolio_max_risk_pct = portfolio_max_risk_pct
        self.current_positions: Dict[str, float] = {}  # symbol -> risk_amount
        
        # State tracking
        self.trade_outcomes: List[TradeOutcome] = []
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.last_trade_time: Optional[datetime] = None
        self.last_loss_time: Optional[datetime] = None
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        
        # Circuit breaker state
        self.daily_limit_hit: bool = False
        self.weekly_limit_hit: bool = False
        self.cooldown_active: bool = False
        self.cooldown_until: Optional[datetime] = None
        
        # Date tracking for resets
        self.current_date: datetime = datetime.now(timezone.utc).date()
        self.week_start: datetime = self._get_week_start()
        
        # Performance metrics for Kelly
        self.win_rate: float = 0.5
        self.avg_win: float = 0.01
        self.avg_loss: float = 0.01
        
        logger.info(f"RiskManager initialized | Base risk: {base_risk_pct:.1%} | "
                   f"Daily limit: {daily_loss_limit:.1%} | Weekly limit: {weekly_loss_limit:.1%} | "
                   f"Max positions: {max_total_positions}")
    
    def _get_week_start(self) -> datetime:
        """Get the start of the current week (Monday)."""
        today = datetime.now(timezone.utc)
        days_since_monday = today.weekday()
        return (today - timedelta(days=days_since_monday)).date()
    
    def _check_date_resets(self):
        """Reset daily/weekly counters if needed."""
        now = datetime.now(timezone.utc)
        today = now.date()
        week_start = self._get_week_start()
        
        # Daily reset
        if today != self.current_date:
            logger.info(f"New day detected - resetting daily P&L from ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.daily_limit_hit = False
            self.current_date = today
        
        # Weekly reset
        if week_start != self.week_start:
            logger.info(f"New week detected - resetting weekly P&L from ${self.weekly_pnl:.2f}")
            self.weekly_pnl = 0.0
            self.weekly_limit_hit = False
            self.week_start = week_start
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float,
        duration_minutes: int = 0,
        ai_approved: bool = True,
        ai_confidence: float = 0.0
    ):
        """Record a completed trade outcome."""
        now = datetime.now(timezone.utc)
        
        outcome = TradeOutcome(
            timestamp=now,
            symbol=symbol,
            side=side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_minutes=duration_minutes,
            ai_approved=ai_approved,
            ai_confidence=ai_confidence
        )
        self.trade_outcomes.append(outcome)
        
        # Keep only last 100 trades
        if len(self.trade_outcomes) > 100:
            self.trade_outcomes = self.trade_outcomes[-100:]
        
        # Update P&L tracking
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        
        # Update streaks
        is_win = pnl > 0
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.last_loss_time = now
            
            # Activate cooldown after consecutive losses
            if self.consecutive_losses >= 2:
                self._activate_cooldown(now)
        
        # Update performance metrics
        self._update_metrics()
        
        self.last_trade_time = now
        
        logger.info(f"Trade recorded: {side} {symbol} P&L=${pnl:+.2f} ({pnl_pct:+.2%}) | "
                   f"Streak: {self.consecutive_wins}W/{self.consecutive_losses}L | "
                   f"Daily: ${self.daily_pnl:+.2f} Weekly: ${self.weekly_pnl:+.2f}")
    
    def _activate_cooldown(self, now: datetime):
        """Activate trading cooldown."""
        self.cooldown_active = True
        self.cooldown_until = now + timedelta(minutes=self.cooldown_minutes)
        logger.warning(f"🛑 Cooldown activated for {self.cooldown_minutes}min after {self.consecutive_losses} losses")
    
    def _update_metrics(self):
        """Update win rate and average win/loss for Kelly calculation."""
        if len(self.trade_outcomes) < 5:
            return
        
        recent = self.trade_outcomes[-50:]  # Use last 50 trades
        wins = [t for t in recent if t.pnl > 0]
        losses = [t for t in recent if t.pnl <= 0]
        
        if len(recent) > 0:
            self.win_rate = len(wins) / len(recent)
        
        if wins:
            self.avg_win = np.mean([t.pnl_pct for t in wins])
        if losses:
            self.avg_loss = abs(np.mean([t.pnl_pct for t in losses]))
    
    def calculate_kelly_fraction(self) -> float:
        """Calculate optimal bet size using Kelly Criterion."""
        if self.avg_loss == 0:
            return self.base_risk_pct
        
        # Kelly: f* = (p * W - q * L) / (W * L)
        # Simplified: f* = W/L * p - (1-p) / (W/L)
        p = self.win_rate
        q = 1 - p
        w = self.avg_win
        l = self.avg_loss
        
        if l == 0:
            l = 0.01
        
        win_loss_ratio = w / l
        
        kelly = (p * win_loss_ratio - q) / win_loss_ratio
        
        # Use half-Kelly for safety
        half_kelly = kelly / 2
        
        # Clip to reasonable bounds
        return float(np.clip(half_kelly, self.min_risk_pct, self.max_risk_pct))
    
    def can_trade(self, balance: float, initial_balance: float) -> Dict[str, Any]:
        """
        Check if trading is allowed based on risk limits.
        
        Returns:
            Dict with 'allowed' bool, 'reason' str, and additional context
        """
        self._check_date_resets()
        now = datetime.now(timezone.utc)
        
        result = {
            'allowed': True,
            'reason': 'OK',
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'daily_pnl_pct': self.daily_pnl / initial_balance if initial_balance > 0 else 0,
            'weekly_pnl_pct': self.weekly_pnl / initial_balance if initial_balance > 0 else 0,
            'cooldown_active': False,
            'cooldown_remaining': 0
        }
        
        # Check cooldown
        if self.cooldown_active and self.cooldown_until:
            if now < self.cooldown_until:
                remaining = (self.cooldown_until - now).total_seconds() / 60
                result['allowed'] = False
                result['reason'] = f"Cooldown active ({remaining:.0f}min remaining after {self.consecutive_losses} losses)"
                result['cooldown_active'] = True
                result['cooldown_remaining'] = int(remaining)
                return result
            else:
                self.cooldown_active = False
                self.cooldown_until = None
                logger.info("Cooldown ended - trading resumed")
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl / initial_balance) if initial_balance > 0 else 0
        if self.daily_pnl < 0 and daily_loss_pct >= self.daily_loss_limit:
            self.daily_limit_hit = True
            result['allowed'] = False
            result['reason'] = f"Daily loss limit hit ({daily_loss_pct:.1%} >= {self.daily_loss_limit:.1%})"
            return result
        
        # Check weekly loss limit
        weekly_loss_pct = abs(self.weekly_pnl / initial_balance) if initial_balance > 0 else 0
        if self.weekly_pnl < 0 and weekly_loss_pct >= self.weekly_loss_limit:
            self.weekly_limit_hit = True
            result['allowed'] = False
            result['reason'] = f"Weekly loss limit hit ({weekly_loss_pct:.1%} >= {self.weekly_loss_limit:.1%})"
            return result
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            if not self.cooldown_active:
                self._activate_cooldown(now)
            result['allowed'] = False
            result['reason'] = f"Max consecutive losses reached ({self.consecutive_losses})"
            return result
        
        return result
    
    def get_adjusted_risk(
        self,
        balance: float,
        peak_balance: float,
        volatility_pct: float = 1.0,
        ai_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate dynamically adjusted risk percentage.
        
        Factors:
        - Kelly Criterion based on historical performance
        - Current drawdown level
        - Win/loss streak
        - Market volatility
        - AI confidence level
        """
        self._check_date_resets()
        
        # Start with Kelly-optimal or base risk
        if len(self.trade_outcomes) >= 10:
            kelly_risk = self.calculate_kelly_fraction()
        else:
            kelly_risk = self.base_risk_pct
        
        # === Factor 1: Drawdown Adjustment ===
        drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        drawdown_pct = drawdown * 100
        
        if drawdown_pct >= 20:
            dd_multiplier = 0.25
            dd_mode = 'EMERGENCY'
        elif drawdown_pct >= 15:
            dd_multiplier = 0.4
            dd_mode = 'SEVERE'
        elif drawdown_pct >= 10:
            dd_multiplier = 0.6
            dd_mode = 'CAUTIOUS'
        elif drawdown_pct >= 5:
            dd_multiplier = 0.8
            dd_mode = 'REDUCED'
        else:
            dd_multiplier = 1.0
            dd_mode = 'NORMAL'
        
        # === Factor 2: Streak Adjustment ===
        if self.consecutive_losses >= 3:
            streak_multiplier = 0.5
        elif self.consecutive_losses >= 2:
            streak_multiplier = 0.75
        elif self.consecutive_losses >= 1:
            streak_multiplier = 0.9
        elif self.consecutive_wins >= 5:
            streak_multiplier = 1.15  # Slight boost on hot streak
        elif self.consecutive_wins >= 3:
            streak_multiplier = 1.1
        else:
            streak_multiplier = 1.0
        
        # === Factor 3: Volatility Adjustment ===
        # High volatility = lower risk
        if volatility_pct > 2.0:
            vol_multiplier = 0.7
        elif volatility_pct > 1.5:
            vol_multiplier = 0.85
        elif volatility_pct < 0.5:
            vol_multiplier = 0.8  # Too quiet = suspicious
        else:
            vol_multiplier = 1.0
        
        # === Factor 4: AI Confidence Adjustment ===
        # Higher AI confidence = slightly higher risk allowed
        if ai_confidence >= 0.9:
            ai_multiplier = 1.1
        elif ai_confidence >= 0.8:
            ai_multiplier = 1.05
        elif ai_confidence < 0.6:
            ai_multiplier = 0.8
        else:
            ai_multiplier = 1.0
        
        # === Combine Factors ===
        # RISK SCALING LOGIC:
        # - Multiple reducing factors: Use the MOST restrictive (minimum multiplier)
        #   This prevents compounding effects from making position sizes too small
        # - Multiple increasing factors: Use the MOST conservative increase (minimum multiplier)
        # - Example: If DD=0.6 and streak=0.75, use 0.6 (most restrictive)
        # - This is intentional to prevent over-reduction but ensures at least one
        #   major risk factor is fully accounted for
        
        reducing_factors = [f for f in [dd_multiplier, streak_multiplier, vol_multiplier] if f < 1.0]
        increasing_factors = [f for f in [streak_multiplier, ai_multiplier] if f > 1.0]
        
        if reducing_factors:
            combined = min(reducing_factors)  # Most restrictive wins
        elif increasing_factors:
            combined = min(increasing_factors)  # Conservative increase
        else:
            combined = 1.0
        
        adjusted_risk = kelly_risk * combined
        adjusted_risk = float(np.clip(adjusted_risk, self.min_risk_pct, self.max_risk_pct))
        
        return {
            'base_risk': self.base_risk_pct,
            'kelly_risk': kelly_risk,
            'adjusted_risk': adjusted_risk,
            'dd_mode': dd_mode,
            'drawdown_pct': round(drawdown_pct, 2),
            'dd_multiplier': dd_multiplier,
            'streak_multiplier': streak_multiplier,
            'vol_multiplier': vol_multiplier,
            'ai_multiplier': ai_multiplier,
            'combined_multiplier': combined,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'win_rate': self.win_rate,
            'message': f"{dd_mode}: {adjusted_risk:.2%} risk (DD:{drawdown_pct:.1f}%, W{self.consecutive_wins}/L{self.consecutive_losses})"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get risk manager statistics."""
        recent_trades = self.trade_outcomes[-20:] if self.trade_outcomes else []
        
        return {
            'total_trades': len(self.trade_outcomes),
            'win_rate': round(self.win_rate * 100, 1),
            'avg_win_pct': round(self.avg_win * 100, 2),
            'avg_loss_pct': round(self.avg_loss * 100, 2),
            'kelly_fraction': round(self.calculate_kelly_fraction() * 100, 2),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': round(self.daily_pnl, 2),
            'weekly_pnl': round(self.weekly_pnl, 2),
            'daily_limit_hit': self.daily_limit_hit,
            'weekly_limit_hit': self.weekly_limit_hit,
            'cooldown_active': self.cooldown_active,
            'recent_trades': len(recent_trades)
        }
    
    def reset_daily(self):
        """Manually reset daily limits."""
        self.daily_pnl = 0.0
        self.daily_limit_hit = False
        logger.info("Daily risk limits manually reset")
    
    def reset_weekly(self):
        """Manually reset weekly limits."""
        self.weekly_pnl = 0.0
        self.weekly_limit_hit = False
        logger.info("Weekly risk limits manually reset")
    
    def reset_cooldown(self):
        """Manually reset cooldown."""
        self.cooldown_active = False
        self.cooldown_until = None
        self.consecutive_losses = 0
        logger.info("Cooldown manually reset")
    
    # === PORTFOLIO-LEVEL POSITION MANAGEMENT (ML Acceleration Plan) ===
    
    def register_position(self, symbol: str, risk_pct: float):
        """Register an open position for portfolio tracking."""
        self.current_positions[symbol] = risk_pct
        logger.debug(f"Position registered: {symbol} ({risk_pct:.2%} risk)")
    
    def unregister_position(self, symbol: str):
        """Remove a closed position from portfolio tracking."""
        if symbol in self.current_positions:
            del self.current_positions[symbol]
            logger.debug(f"Position unregistered: {symbol}")
    
    def can_open_position(self, symbol: str, proposed_risk_pct: float) -> Dict[str, Any]:
        """
        Check if a new position can be opened based on portfolio limits.
        
        Checks:
        1. Max total positions not exceeded
        2. Total portfolio risk not exceeded
        3. Symbol not already in position
        
        Returns:
            Dict with 'allowed' bool and 'reason' str
        """
        result = {
            'allowed': True,
            'reason': 'OK',
            'current_positions': len(self.current_positions),
            'max_positions': self.max_total_positions,
            'current_risk_pct': sum(self.current_positions.values()),
            'proposed_risk_pct': proposed_risk_pct
        }
        
        # Check if already in position for this symbol
        if symbol in self.current_positions:
            result['allowed'] = False
            result['reason'] = f"Already in position for {symbol}"
            return result
        
        # Check max total positions
        if len(self.current_positions) >= self.max_total_positions:
            result['allowed'] = False
            result['reason'] = f"Max positions reached ({len(self.current_positions)}/{self.max_total_positions})"
            return result
        
        # Check total portfolio risk
        total_risk = sum(self.current_positions.values()) + proposed_risk_pct
        if total_risk > self.portfolio_max_risk_pct:
            result['allowed'] = False
            result['reason'] = f"Portfolio risk exceeded ({total_risk:.2%} > {self.portfolio_max_risk_pct:.2%})"
            return result
        
        return result
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio risk status."""
        return {
            'open_positions': len(self.current_positions),
            'max_positions': self.max_total_positions,
            'positions': dict(self.current_positions),
            'total_risk_pct': sum(self.current_positions.values()),
            'max_risk_pct': self.portfolio_max_risk_pct,
            'available_slots': self.max_total_positions - len(self.current_positions),
            'can_add_position': len(self.current_positions) < self.max_total_positions
        }


# Singleton instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get the global risk manager instance."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager


def reset_risk_manager():
    """Reset the global risk manager."""
    global _risk_manager
    _risk_manager = None
