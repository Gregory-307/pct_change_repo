# Merge from:
# - core_analyses.py
# - data_4_analysis_plots.py
# - plot_all.py
# Core functionality:
# - Prediction vs actual plots
# - Portfolio performance visualization
# - Trade entry/exit analysis
# - Drawdown tracking 

"""Results analysis module for model predictions and portfolio performance.

Consolidates functionality from core_analyses.py and plot_all.py for analyzing and
visualizing model predictions and portfolio performance.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from config import OUTPUT_DIRS

logger = logging.getLogger(__name__)

def set_plotting_style():
    """Set consistent plotting style across all analysis modules."""
    # Use default style and apply our customizations
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': [12, 6],
        'figure.dpi': 100,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.autolayout': True  # This prevents cropping
    })

class ResultsAnalyzer:
    def __init__(self):
        """Initialize analyzer with output directories."""
        # Set up directories using config
        self.plots_dir = {
            'model': Path(OUTPUT_DIRS['plots']['model']),
            'portfolio': Path(OUTPUT_DIRS['plots']['portfolio'])
        }
        self.json_dir = {
            'model': Path(OUTPUT_DIRS['jsons']['model']),
            'portfolio': Path(OUTPUT_DIRS['jsons']['portfolio'])
        }
        
        # Create directories
        for dir_path in self.plots_dir.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        for dir_path in self.json_dir.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set consistent plotting style
        set_plotting_style()
    
    def analyze_predictions(self, 
                          train_true: np.ndarray, 
                          train_pred: np.ndarray,
                          val_true: Optional[np.ndarray] = None,
                          val_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze model prediction performance for both training and validation sets."""
        analysis = {
            'training': self._compute_prediction_metrics(train_true, train_pred, 'training'),
            'validation': self._compute_prediction_metrics(val_true, val_pred, 'validation') 
                         if val_true is not None and val_pred is not None else None
        }
        
        # Save analysis to new JSON directory
        output_file = self.json_dir['model'] / "prediction_performance.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
    
    def _compute_prediction_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  dataset_name: str) -> Dict[str, Any]:
        """Compute detailed prediction metrics for a dataset."""
        return {
            'dataset': dataset_name,
            'metrics': {
                'mse': float(np.mean((y_true - y_pred) ** 2)),
                'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                'mae': float(np.mean(np.abs(y_true - y_pred))),
                'r2': float(1 - np.var(y_true - y_pred) / np.var(y_true))
            },
            'distribution': {
                'true_stats': {
                    'mean': float(y_true.mean()),
                    'std': float(y_true.std()),
                    'percentiles': [float(np.percentile(y_true, p)) for p in [25, 50, 75]]
                },
                'pred_stats': {
                    'mean': float(y_pred.mean()),
                    'std': float(y_pred.std()),
                    'percentiles': [float(np.percentile(y_pred, p)) for p in [25, 50, 75]]
                }
            },
            'error_analysis': {
                'mean_error': float(np.mean(y_true - y_pred)),
                'std_error': float(np.std(y_true - y_pred)),
                'error_percentiles': {
                    str(p): float(np.percentile(np.abs(y_true - y_pred), p))
                    for p in [50, 75, 90, 95, 99]
                }
            }
        }
    
    def plot_prediction_scatter(self,
                              train_true: np.ndarray,
                              train_pred: np.ndarray,
                              val_true: Optional[np.ndarray] = None,
                              val_pred: Optional[np.ndarray] = None):
        """Create scatter plots of predicted vs actual values for both training and validation."""
        # Create figure with two subplots if validation data is provided
        n_plots = 2 if val_true is not None and val_pred is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(12*n_plots, 6))
        
        if n_plots == 1:
            axes = [axes]  # Make axes iterable for consistent handling
        
        # Plot training data
        self._plot_scatter_subplot(axes[0], train_true, train_pred, "Training Set")
        
        # Plot validation data if provided
        if n_plots == 2:
            self._plot_scatter_subplot(axes[1], val_true, val_pred, "Validation Set")
        
        plt.tight_layout()
        plt.savefig(self.plots_dir['model'] / 'prediction_scatter.png', bbox_inches='tight')
        plt.close()
        
        # Also create residual plots
        self.plot_prediction_residuals(train_true, train_pred, val_true, val_pred)
    
    def _plot_scatter_subplot(self, ax: plt.Axes, y_true: np.ndarray, 
                            y_pred: np.ndarray, title: str):
        """Helper function to create a scatter subplot."""
        ax.scatter(y_true, y_pred, alpha=0.5, color='#3498db', s=20)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Add R² value to plot
        r2 = 1 - np.var(y_true - y_pred) / np.var(y_true)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top')
        
        ax.set_xlabel('Actual Returns')
        ax.set_ylabel('Predicted Returns')
        ax.set_title(f'Prediction vs Actual ({title})')
        ax.grid(True, alpha=0.3)
    
    def plot_prediction_residuals(self,
                                train_true: np.ndarray,
                                train_pred: np.ndarray,
                                val_true: Optional[np.ndarray] = None,
                                val_pred: Optional[np.ndarray] = None):
        """Create residual plots for both training and validation sets."""
        n_plots = 2 if val_true is not None and val_pred is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(12*n_plots, 6))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot training residuals
        train_residuals = train_true - train_pred
        self._plot_residuals_subplot(axes[0], train_pred, train_residuals, "Training Set")
        
        # Plot validation residuals if provided
        if n_plots == 2:
            val_residuals = val_true - val_pred
            self._plot_residuals_subplot(axes[1], val_pred, val_residuals, "Validation Set")
        
        plt.tight_layout()
        plt.savefig(self.plots_dir['model'] / 'prediction_residuals.png', bbox_inches='tight')
        plt.close()
    
    def _plot_residuals_subplot(self, ax: plt.Axes, y_pred: np.ndarray, 
                              residuals: np.ndarray, title: str):
        """Helper function to create a residual subplot."""
        ax.scatter(y_pred, residuals, alpha=0.5, color='#3498db', s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Predicted Returns')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residual Plot ({title})')
        ax.grid(True, alpha=0.3)
        
        # Add residual stats
        stats_text = (f'Mean: {np.mean(residuals):.3f}\n'
                     f'Std: {np.std(residuals):.3f}')
        ax.text(0.05, 0.95, stats_text, 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top')
    
    def analyze_portfolio(self, simulation_results: Dict) -> Dict[str, Any]:
        """Analyze portfolio performance metrics."""
        analysis = {
            'performance_metrics': {
                'sharpe_ratio': simulation_results['sharpe_ratio'],
                'max_drawdown': simulation_results['max_drawdown'],
                'win_rate': self._calculate_win_rate(simulation_results['trades']),
                'profit_factor': self._calculate_profit_factor(simulation_results['trades'])
            },
            'lifecycle_data': {
                'dates': simulation_results['lifecycle']['dates'],
                'equity_curve': simulation_results['lifecycle']['portfolio_values'],
                'drawdown_curve': simulation_results['lifecycle'].get('drawdown_series', [])
            },
            'position_analysis': {
                'holding_periods': self._calculate_holding_periods(simulation_results['enhanced_trades'])
            }
        }
        
        # Save analysis to new JSON directory
        output_file = self.json_dir['portfolio'] / "performance.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        return analysis
    
    def plot_portfolio_performance(self, portfolio_data: Dict):
        """Plot portfolio equity curve with trades and drawdown."""
        try:
            # Convert dates and ensure matching lengths
            dates = pd.to_datetime(portfolio_data['lifecycle']['dates'])
            values = np.array(portfolio_data['lifecycle']['portfolio_values'])
            
            if len(dates) != len(values):
                logger.warning(f"Date/value length mismatch: dates={len(dates)}, values={len(values)}")
                min_len = min(len(dates), len(values))
                dates = dates[:min_len]
                values = values[:min_len]
            
            # 1. Equity curve with trades
            plt.figure(figsize=(15, 8))
            plt.plot(dates, values, label='Portfolio Value', color='#2ecc71', linewidth=2)
            
            # Add trade markers
            for trade in portfolio_data.get('trades', []):
                trade_date = pd.to_datetime(trade['date'])
                if trade['action'] == 'buy':
                    plt.scatter(trade_date, values[dates == trade_date], 
                              color='g', marker='^', s=100, label='Buy')
                else:
                    plt.scatter(trade_date, values[dates == trade_date], 
                              color='r', marker='v', s=100, label='Sell')
            
            plt.title('Portfolio Performance with Trade Points')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(self.plots_dir['portfolio'] / 'portfolio_performance.png')
            plt.close()
            
            # 2. Drawdown analysis
            self.plot_drawdown(dates, values)
            
            # 3. Returns distribution
            self.plot_returns_distribution(portfolio_data)
            
            # 4. Trade analysis
            self.plot_trade_analysis(portfolio_data['trades'])
            
            # 5. Rolling metrics
            self.plot_rolling_metrics(dates, values)
            
        except Exception as e:
            logger.error(f"Error plotting portfolio performance: {str(e)}")
            raise

    def plot_drawdown(self, dates: pd.Series, values: np.ndarray):
        """Plot drawdown analysis."""
        try:
            rolling_max = pd.Series(values).expanding().max()
            drawdown = (values - rolling_max) / rolling_max * 100
            
            plt.figure(figsize=(15, 6))
            plt.plot(dates, drawdown, color='#e74c3c', linewidth=2)
            plt.fill_between(dates, drawdown, 0, color='#e74c3c', alpha=0.3)
            
            plt.title('Portfolio Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir['portfolio'] / 'drawdown.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting drawdown: {str(e)}")
            raise

    def plot_returns_distribution(self, portfolio_data: Dict):
        """Plot returns distribution analysis."""
        try:
            returns = pd.Series(portfolio_data['lifecycle'].get('daily_returns', []))
            
            plt.figure(figsize=(12, 6))
            returns.hist(bins=50, density=True, alpha=0.75, color='#3498db')
            
            plt.title('Distribution of Daily Returns')
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir['portfolio'] / 'returns_distribution.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {str(e)}")
            raise

    def plot_trade_analysis(self, trades: List[Dict]):
        """Plot trade analysis visualizations."""
        try:
            # Convert trades to DataFrame
            df_trades = pd.DataFrame(trades)
            df_trades['date'] = pd.to_datetime(df_trades['date'])
            
            # 1. Trade PnL Distribution
            plt.figure(figsize=(12, 6))
            df_trades[df_trades['action'] == 'sell']['pnl'].hist(
                bins=30, alpha=0.75, color='#3498db')
            plt.title('Distribution of Trade PnL')
            plt.xlabel('PnL ($)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir['portfolio'] / 'trade_pnl_distribution.png')
            plt.close()
            
            # 2. Trade Volume Over Time
            plt.figure(figsize=(15, 6))
            df_trades.groupby('date').size().plot(kind='bar', alpha=0.75, color='#2ecc71')
            plt.title('Trading Volume Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Trades')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir['portfolio'] / 'trade_volume.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting trade analysis: {str(e)}")
            raise

    def plot_rolling_metrics(self, dates: pd.Series, values: np.ndarray):
        """Plot rolling performance metrics."""
        try:
            # Calculate rolling metrics
            returns = pd.Series(np.diff(values) / values[:-1])
            rolling_sharpe = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            
            # Plot rolling Sharpe ratio
            plt.figure(figsize=(15, 6))
            plt.plot(dates[1:], rolling_sharpe, color='#3498db', linewidth=2)
            plt.title('30-Day Rolling Sharpe Ratio')
            plt.xlabel('Date')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir['portfolio'] / 'rolling_sharpe.png')
            plt.close()
            
            # Plot rolling volatility
            plt.figure(figsize=(15, 6))
            plt.plot(dates[1:], rolling_vol * 100, color='#e74c3c', linewidth=2)
            plt.title('30-Day Rolling Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir['portfolio'] / 'rolling_volatility.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting rolling metrics: {str(e)}")
            raise

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        sell_trades = [t for t in trades if t['action'] == 'sell']
        if not sell_trades:
            return 0.0
        winning_trades = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
        return winning_trades / len(sell_trades)

    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor from trades."""
        if not trades:
            return 0.0
        sell_trades = [t for t in trades if t['action'] == 'sell']
        if not sell_trades:
            return 0.0
        gross_profit = sum(t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def _calculate_holding_periods(self, enhanced_trades: List[Dict]) -> Dict[str, float]:
        """Calculate trade holding period statistics."""
        holding_periods = []
        for trade in enhanced_trades:
            if trade.get('exit_date'):
                entry = pd.to_datetime(trade['entry_date'])
                exit = pd.to_datetime(trade['exit_date'])
                holding_periods.append((exit - entry).days)
        
        if not holding_periods:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0}
            
        return {
            'mean': float(np.mean(holding_periods)),
            'median': float(np.median(holding_periods)),
            'min': float(min(holding_periods)),
            'max': float(max(holding_periods))
        }