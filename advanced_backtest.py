"""
Advanced Backtesting Module for Julaba Trading Bot
===================================================
Walk-forward optimization, parameter sensitivity, and robustness testing.

Features:
- Walk-forward optimization (prevent overfitting)
- Parameter sensitivity analysis
- Robustness testing (out-of-sample validation)
- Integration with Monte Carlo
- Multi-period analysis
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("Julaba.AdvBacktest")

@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    params: Dict[str, Any]
    period_start: str
    period_end: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_trade_duration: float
    trades: List[dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

@dataclass  
class WalkForwardResult:
    """Results from walk-forward optimization"""
    optimization_periods: int
    in_sample_results: List[BacktestResult]
    out_sample_results: List[BacktestResult]
    best_params_per_period: List[Dict[str, Any]]
    combined_out_sample_pnl: float
    combined_out_sample_trades: int
    combined_win_rate: float
    combined_profit_factor: float
    combined_max_drawdown: float
    robustness_score: float  # How consistent are params across periods

@dataclass
class SensitivityResult:
    """Results from parameter sensitivity analysis"""
    base_params: Dict[str, Any]
    param_name: str
    param_values: List[Any]
    results: List[BacktestResult]
    optimal_value: Any
    sensitivity_score: float  # How much does performance vary

class AdvancedBacktester:
    """
    Advanced backtesting with walk-forward optimization and robustness testing
    """
    
    def __init__(self, data_path: str = "historical_data",
                 results_path: str = "backtest_results"):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Strategy function (to be set)
        self._strategy_fn: Optional[Callable] = None
        
        # Cache for loaded data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info("📊 Advanced Backtester initialized")
    
    def set_strategy(self, strategy_fn: Callable):
        """
        Set the strategy function to backtest.
        
        Strategy function should accept:
            - df: DataFrame with OHLCV data
            - params: Dict of parameters
        
        And return:
            - List of trade signals: [{'timestamp', 'direction', 'entry_price', ...}]
        """
        self._strategy_fn = strategy_fn
    
    def load_data(self, symbol: str, timeframe: str = '15m') -> Optional[pd.DataFrame]:
        """Load historical data for symbol"""
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # Try to find data file
        possible_files = [
            self.data_path / f"{symbol}_{timeframe}_90d.csv",
            self.data_path / f"{symbol}_{timeframe}.csv",
            self.data_path / f"{symbol.replace('/', '')}_{timeframe}.csv",
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    
                    # Ensure timestamp column
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    elif 'date' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'])
                    
                    # Sort by time
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    self._data_cache[cache_key] = df
                    logger.info(f"Loaded {len(df)} bars from {file_path}")
                    return df
                    
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.warning(f"No data found for {symbol}")
        return None
    
    # ==================== SINGLE BACKTEST ====================
    
    def run_backtest(self, df: pd.DataFrame, params: Dict[str, Any],
                     initial_balance: float = 1000,
                     risk_per_trade: float = 0.02) -> BacktestResult:
        """
        Run a single backtest with given parameters
        """
        if self._strategy_fn is None:
            raise ValueError("Strategy function not set. Call set_strategy() first.")
        
        # Get trade signals from strategy
        signals = self._strategy_fn(df, params)
        
        # Simulate trades
        balance = initial_balance
        peak_balance = initial_balance
        equity_curve = [initial_balance]
        trades = []
        
        for signal in signals:
            direction = signal.get('direction', 'LONG')
            entry_price = signal.get('entry_price', 0)
            exit_price = signal.get('exit_price', 0)
            
            if entry_price <= 0 or exit_price <= 0:
                continue
            
            # Calculate P&L
            risk_amount = balance * risk_per_trade
            
            if direction == 'LONG':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            # Assume 1:2 risk/reward for position sizing
            pnl = risk_amount * pnl_pct * 50  # Leverage effect
            
            # Apply P&L
            balance += pnl
            equity_curve.append(balance)
            
            # Track peak for drawdown
            if balance > peak_balance:
                peak_balance = balance
            
            trades.append({
                'timestamp': signal.get('timestamp'),
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'balance': balance
            })
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, params, df, initial_balance)
    
    def _calculate_metrics(self, trades: List[dict], equity_curve: List[float],
                          params: Dict[str, Any], df: pd.DataFrame,
                          initial_balance: float) -> BacktestResult:
        """Calculate backtest performance metrics"""
        if not trades:
            return BacktestResult(
                params=params,
                period_start=str(df['timestamp'].iloc[0]) if len(df) > 0 else "",
                period_end=str(df['timestamp'].iloc[-1]) if len(df) > 0 else "",
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                profit_factor=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                avg_trade_duration=0,
                trades=[],
                equity_curve=equity_curve
            )
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        total_wins = sum(t['pnl'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
        
        # Max drawdown
        peak = initial_balance
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        
        max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0
        
        # Returns for Sharpe/Sortino
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)
        
        avg_ret = np.mean(returns) if returns else 0
        std_ret = np.std(returns) if returns else 0.001
        
        neg_returns = [r for r in returns if r < 0]
        downside_std = np.std(neg_returns) if neg_returns else 0.001
        
        sharpe = (avg_ret / std_ret * np.sqrt(365 * 24 * 4)) if std_ret > 0 else 0  # 15m bars
        sortino = (avg_ret / downside_std * np.sqrt(365 * 24 * 4)) if downside_std > 0 else 0
        
        return BacktestResult(
            params=params,
            period_start=str(df['timestamp'].iloc[0]) if len(df) > 0 else "",
            period_end=str(df['timestamp'].iloc[-1]) if len(df) > 0 else "",
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=(len(wins) / len(trades) * 100) if trades else 0,
            total_pnl=sum(t['pnl'] for t in trades),
            profit_factor=(total_wins / total_losses) if total_losses > 0 else 0,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_trade_duration=0,  # Would need entry/exit times
            trades=trades,
            equity_curve=equity_curve
        )
    
    # ==================== WALK-FORWARD OPTIMIZATION ====================
    
    def walk_forward_optimize(self, df: pd.DataFrame,
                               param_grid: Dict[str, List[Any]],
                               n_periods: int = 5,
                               in_sample_pct: float = 0.7,
                               optimization_metric: str = 'profit_factor',
                               initial_balance: float = 1000) -> WalkForwardResult:
        """
        Walk-forward optimization to prevent overfitting.
        
        Splits data into n_periods, optimizes on in-sample portion,
        then tests on out-of-sample portion of each period.
        
        Args:
            df: Historical data
            param_grid: Dict of parameter names to lists of values to test
            n_periods: Number of walk-forward periods
            in_sample_pct: Percentage of each period for optimization
            optimization_metric: Metric to optimize ('profit_factor', 'sharpe_ratio', 'total_pnl')
        
        Returns:
            WalkForwardResult with combined performance
        """
        logger.info(f"🔄 Walk-Forward Optimization: {n_periods} periods, {in_sample_pct:.0%} in-sample")
        
        # Calculate period lengths
        total_bars = len(df)
        period_length = total_bars // n_periods
        
        in_sample_results = []
        out_sample_results = []
        best_params_list = []
        
        for period in range(n_periods):
            # Define period boundaries
            start_idx = period * period_length
            end_idx = start_idx + period_length if period < n_periods - 1 else total_bars
            
            period_df = df.iloc[start_idx:end_idx].copy()
            
            # Split into in-sample and out-of-sample
            split_idx = int(len(period_df) * in_sample_pct)
            in_sample_df = period_df.iloc[:split_idx]
            out_sample_df = period_df.iloc[split_idx:]
            
            logger.info(f"  Period {period+1}/{n_periods}: "
                       f"IS={len(in_sample_df)} bars, OOS={len(out_sample_df)} bars")
            
            # Optimize on in-sample
            best_params, best_result = self._optimize_params(
                in_sample_df, param_grid, optimization_metric, initial_balance
            )
            
            in_sample_results.append(best_result)
            best_params_list.append(best_params)
            
            # Test on out-of-sample with best params
            oos_result = self.run_backtest(out_sample_df, best_params, initial_balance)
            out_sample_results.append(oos_result)
            
            logger.info(f"    IS: PF={best_result.profit_factor:.2f}, WR={best_result.win_rate:.1f}%")
            logger.info(f"    OOS: PF={oos_result.profit_factor:.2f}, WR={oos_result.win_rate:.1f}%")
        
        # Combine out-of-sample results
        combined_trades = []
        for result in out_sample_results:
            combined_trades.extend(result.trades)
        
        combined_pnl = sum(r.total_pnl for r in out_sample_results)
        combined_trade_count = sum(r.total_trades for r in out_sample_results)
        combined_wins = sum(r.winning_trades for r in out_sample_results)
        
        total_profits = sum(sum(t['pnl'] for t in r.trades if t['pnl'] > 0) 
                          for r in out_sample_results)
        total_losses = abs(sum(sum(t['pnl'] for t in r.trades if t['pnl'] < 0) 
                              for r in out_sample_results)) or 0.001
        
        # Calculate robustness score (how stable are optimal params)
        robustness = self._calculate_robustness(best_params_list)
        
        return WalkForwardResult(
            optimization_periods=n_periods,
            in_sample_results=in_sample_results,
            out_sample_results=out_sample_results,
            best_params_per_period=best_params_list,
            combined_out_sample_pnl=combined_pnl,
            combined_out_sample_trades=combined_trade_count,
            combined_win_rate=(combined_wins / combined_trade_count * 100) if combined_trade_count > 0 else 0,
            combined_profit_factor=(total_profits / total_losses) if total_losses > 0 else 0,
            combined_max_drawdown=max(r.max_drawdown_pct for r in out_sample_results),
            robustness_score=robustness
        )
    
    def _optimize_params(self, df: pd.DataFrame, param_grid: Dict[str, List[Any]],
                         metric: str, initial_balance: float) -> Tuple[Dict[str, Any], BacktestResult]:
        """Find best parameters on given data"""
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        best_params = None
        best_result = None
        best_metric_value = float('-inf')
        
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            try:
                result = self.run_backtest(df, params, initial_balance)
                
                # Get metric value
                metric_value = getattr(result, metric, 0)
                
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
                    best_result = result
                    
            except Exception as e:
                logger.debug(f"Backtest failed for {params}: {e}")
        
        return best_params or {}, best_result or BacktestResult(
            params={}, period_start="", period_end="",
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_pnl=0, profit_factor=0,
            max_drawdown=0, max_drawdown_pct=0,
            sharpe_ratio=0, sortino_ratio=0, avg_trade_duration=0
        )
    
    def _calculate_robustness(self, params_list: List[Dict[str, Any]]) -> float:
        """Calculate how consistent optimal parameters are across periods"""
        if not params_list or len(params_list) < 2:
            return 1.0
        
        # For each parameter, calculate coefficient of variation
        param_names = params_list[0].keys()
        variations = []
        
        for name in param_names:
            values = [p.get(name) for p in params_list if p.get(name) is not None]
            
            if not values or not all(isinstance(v, (int, float)) for v in values):
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                variations.append(cv)
        
        if not variations:
            return 1.0
        
        # Robustness = 1 - avg(CV), capped at 0-1
        robustness = 1 - min(np.mean(variations), 1.0)
        return max(robustness, 0)
    
    # ==================== PARAMETER SENSITIVITY ====================
    
    def analyze_sensitivity(self, df: pd.DataFrame, base_params: Dict[str, Any],
                           param_name: str, param_values: List[Any],
                           initial_balance: float = 1000) -> SensitivityResult:
        """
        Analyze how sensitive performance is to changes in a single parameter.
        """
        logger.info(f"📊 Sensitivity Analysis: {param_name}")
        
        results = []
        
        for value in param_values:
            test_params = base_params.copy()
            test_params[param_name] = value
            
            result = self.run_backtest(df, test_params, initial_balance)
            results.append(result)
            
            logger.debug(f"  {param_name}={value}: PF={result.profit_factor:.2f}, "
                        f"WR={result.win_rate:.1f}%")
        
        # Find optimal value
        best_idx = np.argmax([r.profit_factor for r in results])
        optimal_value = param_values[best_idx]
        
        # Calculate sensitivity score (how much does performance vary)
        pf_values = [r.profit_factor for r in results]
        sensitivity = np.std(pf_values) / (np.mean(pf_values) + 0.001)
        
        return SensitivityResult(
            base_params=base_params,
            param_name=param_name,
            param_values=param_values,
            results=results,
            optimal_value=optimal_value,
            sensitivity_score=sensitivity
        )
    
    def full_sensitivity_analysis(self, df: pd.DataFrame, base_params: Dict[str, Any],
                                  param_ranges: Dict[str, List[Any]],
                                  initial_balance: float = 1000) -> Dict[str, SensitivityResult]:
        """Run sensitivity analysis for all parameters"""
        results = {}
        
        for param_name, values in param_ranges.items():
            results[param_name] = self.analyze_sensitivity(
                df, base_params, param_name, values, initial_balance
            )
        
        return results
    
    # ==================== VISUALIZATION ====================
    
    def plot_walk_forward(self, wf_result: WalkForwardResult, 
                          output_path: Path = None):
        """Visualize walk-forward optimization results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Walk-Forward Optimization Results', fontsize=14, fontweight='bold')
            
            # 1. In-Sample vs Out-of-Sample Profit Factors
            ax1 = axes[0, 0]
            periods = range(1, wf_result.optimization_periods + 1)
            is_pf = [r.profit_factor for r in wf_result.in_sample_results]
            oos_pf = [r.profit_factor for r in wf_result.out_sample_results]
            
            x = np.arange(len(periods))
            width = 0.35
            ax1.bar(x - width/2, is_pf, width, label='In-Sample', alpha=0.8)
            ax1.bar(x + width/2, oos_pf, width, label='Out-of-Sample', alpha=0.8)
            ax1.set_xlabel('Period')
            ax1.set_ylabel('Profit Factor')
            ax1.set_title('IS vs OOS Profit Factor')
            ax1.set_xticks(x)
            ax1.set_xticklabels(periods)
            ax1.legend()
            ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5)
            ax1.grid(alpha=0.3)
            
            # 2. Combined equity curve
            ax2 = axes[0, 1]
            combined_equity = []
            for result in wf_result.out_sample_results:
                if combined_equity:
                    # Scale to connect curves
                    scale = combined_equity[-1] / result.equity_curve[0]
                    combined_equity.extend([e * scale for e in result.equity_curve[1:]])
                else:
                    combined_equity.extend(result.equity_curve)
            
            ax2.plot(combined_equity, color='steelblue', linewidth=1)
            ax2.set_xlabel('Trade #')
            ax2.set_ylabel('Equity')
            ax2.set_title('Combined OOS Equity Curve')
            ax2.grid(alpha=0.3)
            
            # 3. Win rates comparison
            ax3 = axes[1, 0]
            is_wr = [r.win_rate for r in wf_result.in_sample_results]
            oos_wr = [r.win_rate for r in wf_result.out_sample_results]
            
            ax3.bar(x - width/2, is_wr, width, label='In-Sample', alpha=0.8)
            ax3.bar(x + width/2, oos_wr, width, label='Out-of-Sample', alpha=0.8)
            ax3.set_xlabel('Period')
            ax3.set_ylabel('Win Rate (%)')
            ax3.set_title('IS vs OOS Win Rate')
            ax3.set_xticks(x)
            ax3.set_xticklabels(periods)
            ax3.legend()
            ax3.axhline(50, color='red', linestyle='--', alpha=0.5)
            ax3.grid(alpha=0.3)
            
            # 4. Summary stats
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            summary_text = f"""
Walk-Forward Optimization Summary
{'='*35}

Periods:            {wf_result.optimization_periods}
Robustness Score:   {wf_result.robustness_score:.2f}

Combined OOS Performance:
  Total Trades:     {wf_result.combined_out_sample_trades}
  Win Rate:         {wf_result.combined_win_rate:.1f}%
  Profit Factor:    {wf_result.combined_profit_factor:.2f}
  Total P&L:        ${wf_result.combined_out_sample_pnl:.2f}
  Max Drawdown:     {wf_result.combined_max_drawdown:.1f}%

In-Sample Avg PF:   {np.mean(is_pf):.2f}
Out-Sample Avg PF:  {np.mean(oos_pf):.2f}
Degradation:        {(1 - np.mean(oos_pf)/np.mean(is_pf))*100:.1f}%
            """
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                output_path = self.results_path / "walk_forward_results.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            logger.info(f"Walk-forward plot saved: {output_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot walk-forward results: {e}")
    
    def plot_sensitivity(self, sensitivity_results: Dict[str, SensitivityResult],
                         output_path: Path = None):
        """Visualize parameter sensitivity analysis"""
        try:
            n_params = len(sensitivity_results)
            cols = min(3, n_params)
            rows = (n_params + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            fig.suptitle('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
            
            if n_params == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_params > 1 else [axes]
            
            for idx, (param_name, result) in enumerate(sensitivity_results.items()):
                ax = axes[idx]
                
                values = result.param_values
                pf_values = [r.profit_factor for r in result.results]
                wr_values = [r.win_rate for r in result.results]
                
                ax.plot(values, pf_values, 'b-o', label='Profit Factor', linewidth=2)
                ax.set_xlabel(param_name)
                ax.set_ylabel('Profit Factor', color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                
                ax2 = ax.twinx()
                ax2.plot(values, wr_values, 'r--s', label='Win Rate', linewidth=2, alpha=0.7)
                ax2.set_ylabel('Win Rate (%)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                ax.axvline(result.optimal_value, color='green', linestyle=':', 
                          label=f'Optimal: {result.optimal_value}')
                ax.set_title(f'{param_name}\n(Sensitivity: {result.sensitivity_score:.2f})')
                ax.grid(alpha=0.3)
                ax.legend(loc='upper left')
            
            # Hide unused subplots
            for idx in range(n_params, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                output_path = self.results_path / "sensitivity_analysis.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            logger.info(f"Sensitivity plot saved: {output_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot sensitivity: {e}")
    
    # ==================== REPORTING ====================
    
    def print_walk_forward_report(self, wf_result: WalkForwardResult):
        """Print formatted walk-forward report"""
        print("\n" + "="*70)
        print("🔄 WALK-FORWARD OPTIMIZATION REPORT")
        print("="*70)
        
        print(f"\n📊 Configuration:")
        print(f"  Optimization Periods: {wf_result.optimization_periods}")
        print(f"  Robustness Score:     {wf_result.robustness_score:.2f}")
        
        print(f"\n📈 Per-Period Results:")
        print(f"  {'Period':<8} {'IS PF':<10} {'OOS PF':<10} {'IS WR':<10} {'OOS WR':<10}")
        print(f"  {'-'*48}")
        
        for i in range(wf_result.optimization_periods):
            is_r = wf_result.in_sample_results[i]
            oos_r = wf_result.out_sample_results[i]
            print(f"  {i+1:<8} {is_r.profit_factor:<10.2f} {oos_r.profit_factor:<10.2f} "
                  f"{is_r.win_rate:<10.1f} {oos_r.win_rate:<10.1f}")
        
        print(f"\n💰 Combined Out-of-Sample Performance:")
        print(f"  Total Trades:     {wf_result.combined_out_sample_trades}")
        print(f"  Win Rate:         {wf_result.combined_win_rate:.1f}%")
        print(f"  Profit Factor:    {wf_result.combined_profit_factor:.2f}")
        print(f"  Total P&L:        ${wf_result.combined_out_sample_pnl:.2f}")
        print(f"  Max Drawdown:     {wf_result.combined_max_drawdown:.1f}%")
        
        # Assessment
        print(f"\n🎯 Assessment:")
        is_pf_avg = np.mean([r.profit_factor for r in wf_result.in_sample_results])
        oos_pf_avg = np.mean([r.profit_factor for r in wf_result.out_sample_results])
        degradation = (1 - oos_pf_avg / is_pf_avg) * 100 if is_pf_avg > 0 else 100
        
        print(f"  In-Sample Avg PF:     {is_pf_avg:.2f}")
        print(f"  Out-of-Sample Avg PF: {oos_pf_avg:.2f}")
        print(f"  Performance Drop:     {degradation:.1f}%")
        
        if degradation < 20 and oos_pf_avg > 1.2:
            print(f"  ✅ ROBUST - Low degradation, profitable OOS")
        elif degradation < 40 and oos_pf_avg > 1.0:
            print(f"  ⚠️  MODERATE - Some overfitting but still profitable")
        else:
            print(f"  🚨 OVERFITTED - High degradation, review strategy")
        
        print("\n" + "="*70 + "\n")
    
    def save_results(self, result: Any, filename: str):
        """Save results to JSON"""
        try:
            output_path = self.results_path / filename
            
            # Convert dataclass to dict
            if hasattr(result, '__dataclass_fields__'):
                data = asdict(result)
            else:
                data = result
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# Example strategy function (for testing)
def example_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> List[dict]:
    """
    Example strategy using moving average crossover.
    Replace with your actual strategy logic.
    """
    signals = []
    
    fast_period = params.get('fast_ma', 10)
    slow_period = params.get('slow_ma', 30)
    
    if len(df) < slow_period:
        return signals
    
    df = df.copy()
    df['fast_ma'] = df['close'].rolling(fast_period).mean()
    df['slow_ma'] = df['close'].rolling(slow_period).mean()
    
    in_position = False
    entry_price = 0
    entry_time = None
    direction = None
    
    for i in range(slow_period, len(df)):
        fast = df['fast_ma'].iloc[i]
        slow = df['slow_ma'].iloc[i]
        price = df['close'].iloc[i]
        timestamp = df['timestamp'].iloc[i]
        
        if not in_position:
            # Entry conditions
            if fast > slow:
                direction = 'LONG'
                entry_price = price
                entry_time = timestamp
                in_position = True
            elif fast < slow:
                direction = 'SHORT'
                entry_price = price
                entry_time = timestamp
                in_position = True
        else:
            # Exit conditions
            should_exit = False
            if direction == 'LONG' and fast < slow:
                should_exit = True
            elif direction == 'SHORT' and fast > slow:
                should_exit = True
            
            if should_exit:
                signals.append({
                    'timestamp': entry_time,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'exit_time': timestamp
                })
                in_position = False
    
    return signals


# CLI for testing
if __name__ == "__main__":
    backtester = AdvancedBacktester()
    backtester.set_strategy(example_strategy)
    
    # Try to load data
    df = backtester.load_data("BTCUSDT")
    
    if df is not None:
        # Run walk-forward optimization
        param_grid = {
            'fast_ma': [5, 10, 15, 20],
            'slow_ma': [20, 30, 40, 50]
        }
        
        wf_result = backtester.walk_forward_optimize(
            df, param_grid, n_periods=4, in_sample_pct=0.7
        )
        
        backtester.print_walk_forward_report(wf_result)
        backtester.plot_walk_forward(wf_result)
        
        # Sensitivity analysis
        base_params = {'fast_ma': 10, 'slow_ma': 30}
        sensitivity = backtester.full_sensitivity_analysis(
            df, base_params,
            {
                'fast_ma': [5, 8, 10, 12, 15, 20],
                'slow_ma': [20, 25, 30, 35, 40, 50]
            }
        )
        backtester.plot_sensitivity(sensitivity)
    else:
        print("No data available for backtesting")
