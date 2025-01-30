"""Portfolio simulator with flexible position sizing strategies."""

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from pathlib import Path
import json
import pickle
from config import PORTFOLIO_PARAMS, DATA_PARAMS, OUTPUT_DIRS
from config_features import FEATURE_CONFIG
from model_1_class import PctChangeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioSimulator:
    def __init__(self, model: PctChangeModel):
        self.model_wrapper = model  # Our custom model class
        self.booster = model.model   # The actual XGBoost booster
        
        # Use paths from config
        self.data_dir = Path(OUTPUT_DIRS['data'])
        self.models_dir = Path(OUTPUT_DIRS['models'])
        self.json_dir = Path(OUTPUT_DIRS['jsons']['portfolio'])
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # Load portfolio parameters
        self.params = PORTFOLIO_PARAMS
        logger.info(f"Portfolio parameters: {self.params}")
        
        # Use prediction horizon from config
        self.horizon = DATA_PARAMS['prediction_horizon'].days
        logger.info(f"Using {self.horizon}-day prediction horizon")
        
        # Load enabled features from config
        self.enabled_features = [
            feat
            for cat in FEATURE_CONFIG.values()
            if cat['enabled']
            for feat, enabled in cat['features'].items()
            if enabled
        ]
        
    def _load_data(self) -> pd.DataFrame:
        """Use model's known features and required price columns"""
        df = pd.read_csv(self.data_dir / "test.csv")
        required_cols = self.model_wrapper.feature_cols + ['COMPANYNAME', 'ASOFDATE', 'PRICECLOSE']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise KeyError(f"Missing required columns in test data: {missing_cols}")
        
        return df[required_cols].copy()
    
    def calculate_position_sizes(self, predictions, available_capital):
        """Calculate position sizes based on strategy"""
        try:
            if len(predictions) == 0 or available_capital <= 0:
                return {}
            
            company_col = 'COMPANYNAME'
            return_col = 'predicted_return'
            
            # Add validation for required columns
            if return_col not in predictions.columns:
                raise KeyError(f"Missing prediction column: {return_col}")
            
            # Filter by ranking threshold
            n_positions = min(len(predictions), self.params['max_positions'])
            threshold_idx = int(len(predictions) * self.params['ranking_threshold'])
            top_predictions = predictions.nlargest(threshold_idx, return_col)
            
            # Filter by entry threshold with debug logging
            valid_mask = top_predictions[return_col] >= self.params['entry_threshold']
            logger.info(f"Entries above threshold: {valid_mask.sum()}/{len(top_predictions)}")
            
            valid_predictions = top_predictions[valid_mask]
            
            if len(valid_predictions) == 0:
                logger.warning("No valid predictions above entry threshold")
                return {}
            
            # Calculate position sizes based on strategy
            if self.params['position_sizing'] == 'equal':
                position_size = available_capital / min(len(valid_predictions), n_positions)
                sizes = {row[company_col]: position_size 
                    for _, row in valid_predictions.head(n_positions).iterrows()}
            
            elif self.params['position_sizing'] == 'prediction_weighted':
                # Filter first before calculating weights
                top_predictions = valid_predictions.head(n_positions)
                weights = top_predictions[return_col]
                
                # Add safety checks
                if weights.sum() <= 0:
                    return {}
                
                weights = weights / weights.sum()  # Normalize
                sizes = {}
                for idx, (_, row) in enumerate(top_predictions.iterrows()):
                    ticker = row[company_col]
                    sizes[ticker] = available_capital * weights.iloc[idx]
            
            elif self.params['position_sizing'] == 'kelly':
                sizes = self._calculate_position_sizes_kelly(valid_predictions)
            
            # Apply maximum position size constraint
            max_size = available_capital * self.params['max_position_size']
            sizes = {ticker: min(size, max_size) for ticker, size in sizes.items()}
            
            return sizes
        
        except Exception as e:
            logger.error(f"Position sizing failed: {str(e)}")
            return {}
    
    def _calculate_position_sizes_kelly(self, valid_predictions):
        """Calculate position sizes using Kelly criterion"""
        if len(valid_predictions) == 0:
            return {}
        
        # Estimate win probability and win/loss ratio from predictions
        win_prob = (valid_predictions[self.enabled_features] > 0).mean()
        avg_win = valid_predictions[valid_predictions[self.enabled_features] > 0][self.enabled_features].mean()
        avg_loss = abs(valid_predictions[valid_predictions[self.enabled_features] < 0][self.enabled_features].mean())
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1

        # Calculate Kelly fraction for each prediction
        kelly_fractions = valid_predictions[self.enabled_features].apply(
            lambda x: max(0, min(1, (x * (1 + win_loss_ratio) - (1-x)) / win_loss_ratio))
        )
        
        total_fraction = kelly_fractions.sum()
        if total_fraction > 0:
            # Scale position sizes to respect max_leverage
            scale = min(1, self.params['max_leverage'] / total_fraction)
            sizes = {row['COMPANYNAME']: kelly_fractions[i] * scale 
                    for i, row in valid_predictions.iterrows()}
        else:
            sizes = {row['COMPANYNAME']: 0 for _, row in valid_predictions.iterrows()}
        
        return sizes
    
    def simulate_portfolio(self) -> dict:
        """Run portfolio simulation using the trained model's predictions."""
        # Load data with horizon cropping
        test_data = self._load_data()
        
        # Verify we have the trained booster
        if not self.booster:
            raise ValueError("No trained model available for simulation")

        # Initialize portfolio metrics
        portfolio_value = self.params['initial_capital']
        current_positions = {}  # {ticker: {'entry_price': float, 'shares': int}}
        daily_returns = []
        trades = []  # List to track all trades
        portfolio_values = []  # List to track portfolio values over time
        dates = sorted(test_data['ASOFDATE'].unique())
        daily_metrics = []

        for date in dates:
            # Store portfolio value at start of day
            portfolio_values.append(portfolio_value)
            
            date_data = test_data[test_data['ASOFDATE'] == date].copy()
            
            # Make predictions using the model wrapper
            date_data['predicted_return'] = self.model_wrapper.predict(date_data)
            
            # Calculate current position values and returns
            daily_pnl = 0
            closed_positions = []
            
            for company, position in current_positions.items():
                if company in date_data['COMPANYNAME'].values:
                    current_price = date_data[date_data['COMPANYNAME'] == company]['PRICECLOSE'].iloc[0]
                    prev_price = test_data[(test_data['COMPANYNAME'] == company) & 
                                      (test_data['ASOFDATE'] < date)]['PRICECLOSE'].iloc[-1]
                    
                    # Check stop loss and take profit
                    pct_change = (current_price - position['entry_price']) / position['entry_price']
                    position_value = position['shares'] * current_price
                    
                    if pct_change <= -self.params['stop_loss'] or pct_change >= self.params['take_profit']:
                        # Close position
                        daily_pnl += position['shares'] * (current_price - prev_price)
                        closed_positions.append(company)
                        trades.append({
                            'date': date,
                            'ticker': company,
                            'action': 'sell',
                            'price': current_price,
                            'shares': position['shares'],
                            'pnl': position['shares'] * (current_price - position['entry_price'])
                        })
                    else:
                        # Update position value
                        daily_pnl += position['shares'] * (current_price - prev_price)
                else:
                    closed_positions.append(company)
            
            # Remove closed positions
            for company in closed_positions:
                del current_positions[company]
            
            # Update portfolio value
            portfolio_value += daily_pnl
            daily_returns.append(daily_pnl / portfolio_value if portfolio_value > 0 else 0)
            
            # Calculate new positions based on predictions
            available_capital = portfolio_value - sum(pos['shares'] * price 
                                                    for company, pos in current_positions.items()
                                                    for price in [date_data[date_data['COMPANYNAME'] == company]['PRICECLOSE'].iloc[0]])
            
            if available_capital > 0:
                # Get top predictions above entry threshold
                valid_predictions = date_data[date_data['predicted_return'] > self.params['entry_threshold']]
                valid_predictions = valid_predictions.nlargest(self.params['max_positions'], 'predicted_return')
                
                # Calculate position sizes
                if len(valid_predictions) > 0:
                    position_sizes = self.calculate_position_sizes(valid_predictions, available_capital)
                
                for company, size in position_sizes.items():
                    if company not in current_positions and size > 0:
                        price = date_data[date_data['COMPANYNAME'] == company]['PRICECLOSE'].iloc[0]
                        shares = int(size / price)
                        if shares > 0:
                            current_positions[company] = {
                                'entry_price': price,
                                'shares': shares
                            }
                            trades.append({
                                'date': date,
                                'ticker': company,
                                'action': 'buy',
                                'price': price,
                                'shares': shares
                            })
            
            daily_metrics.append({
                'date': str(date),
                'value': portfolio_value,
                'cash': available_capital,
                'positions': len(current_positions)
            })
        
        # Calculate performance metrics
        total_return = (portfolio_value - self.params['initial_capital']) / self.params['initial_capital']
        daily_returns = pd.Series(daily_returns)
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_value,
            'portfolio_values': portfolio_values,
            'trades': trades,
            'lifecycle': {
                'dates': [str(d) for d in dates],
                'portfolio_values': portfolio_values,
                'daily_returns': daily_returns,
                'daily_metrics': daily_metrics
            },
            'enhanced_trades': [{
                'entry_date': t['date'],
                'ticker': t['ticker'],
                'exit_date': next((s['date'] for s in trades 
                                 if s['ticker'] == t['ticker'] and s['action'] == 'sell'), None),
                'entry_price': t['price'],
                'exit_price': next((s['price'] for s in trades 
                                  if s['ticker'] == t['ticker'] and s['action'] == 'sell'), None)
            } for t in trades if t['action'] == 'buy']
        }
        
        # Save results
        output_file = self.json_dir / "simulation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Simulation results saved to {output_file}")
        logger.info(f"Final Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"Performance Metrics: {results}")
        
        return results
    
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown from portfolio values"""
        if not portfolio_values:
            return 0
            
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown 
if __name__ == "__main__":
    simulator = PortfolioSimulator()
    logger.info("Running simulation")
    try:
        simulator.simulate_portfolio()
    except Exception as e:
        logger.error(f"Error simulating: {str(e)}") 
