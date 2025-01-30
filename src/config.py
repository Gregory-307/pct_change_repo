"""Configuration parameters for data processing, model training, and analysis.

This configuration file controls the entire pipeline:
1. Data Processing: Filtering, feature engineering, and dataset creation
2. Model Training: XGBoost parameters and training settings
3. Portfolio Simulation: Position sizing, risk management, and trading rules
"""

from datetime import timedelta
import pandas as pd

# =============================================================================
# ACTIVE SETTINGS - MODIFY THESE TO CONTROL THE RUN
# =============================================================================

ACTIVE_SETTINGS = {
    # Data Processing & Features
    'prediction_horizon': 10,  # Number of days to predict forward returns
    'feature_window': 21,  # Window size for technical indicators
    'volatility_window': 21,  # Window size for volatility calculations
    'price_features': True,  # Include basic price and volume features
    'technical_features': True,  # Include technical indicators
    'fundamental_features': True,  # Include fundamental factors
    'train_start_date': '2014-08-01',
    'train_end_date': '2017-12-29',
    'val_start_date': '2018-01-02',
    'val_end_date': '2018-12-28',
    'test_start_date': '2019-01-02',
    'test_end_date': '2019-12-11',
    
    # Data Quality Requirements
    'min_stocks_per_day': 8,  # Minimum number of stocks required per day
    'min_coverage_pct': 60,  # Increased minimum coverage requirement
    'min_history_days': 252,  # Increased to 1 year of history
    'min_factor_coverage': 0.4,  # Increased factor coverage requirement
    'data_start_date': '2014-08-01',  # Remove data before this date
    
    # Model Training
    'learning_rate': 0.01,  # Reduced learning rate for better generalization
    'max_depth': 8,  # Slightly increased depth
    'n_estimators': 1000,  # Increased number of trees
    'early_stopping_rounds': 500,  # More patience for early stopping
    
    # Portfolio Strategy
    'initial_capital': 1_000_000,
    'max_positions': 10,  # Reduced for better concentration
    'ranking_threshold': 0.9,  # More selective stock picking
    'position_sizing': 'equal',
    'max_position_size': 0.15,  # Reduced max position size
    'entry_threshold': 0.10,  # Increased entry threshold
    'stop_loss': 0.05,  # Wider stop loss
    'take_profit': 0.50,  # Increased take profit target
    
    # Risk Management
    'max_sector_exposure': 0.4,
    'max_leverage': 1.0,
    'transaction_cost': 0.001,
    
    # Analysis
    'rolling_window': 21
}

# =============================================================================
# DERIVED SETTINGS - DO NOT MODIFY DIRECTLY (modify ACTIVE_SETTINGS instead)
# =============================================================================

# Data Processing Parameters
DATA_PARAMS = {
    'prediction_horizon': timedelta(days=ACTIVE_SETTINGS['prediction_horizon']),
    'min_history_days': timedelta(days=ACTIVE_SETTINGS['min_history_days']),
    'min_factor_coverage': ACTIVE_SETTINGS['min_factor_coverage'],
    'feature_windows': [ACTIVE_SETTINGS['feature_window']],
    'vol_windows': [ACTIVE_SETTINGS['volatility_window']],
    'use_price_features': ACTIVE_SETTINGS['price_features'],
    'use_technical_features': ACTIVE_SETTINGS['technical_features'],
    'use_fundamental_features': ACTIVE_SETTINGS['fundamental_features'],
    'min_stocks_per_day': ACTIVE_SETTINGS['min_stocks_per_day'],
    'min_coverage_pct': ACTIVE_SETTINGS['min_coverage_pct'],
    'data_start_date': pd.Timestamp(ACTIVE_SETTINGS['data_start_date']),
    'train_start_date': pd.Timestamp(ACTIVE_SETTINGS['train_start_date']),
    'train_end_date': pd.Timestamp(ACTIVE_SETTINGS['train_end_date']),
    'val_start_date': pd.Timestamp(ACTIVE_SETTINGS['val_start_date']),
    'val_end_date': pd.Timestamp(ACTIVE_SETTINGS['val_end_date']),
    'test_start_date': pd.Timestamp(ACTIVE_SETTINGS['test_start_date']),
    'test_end_date': pd.Timestamp(ACTIVE_SETTINGS['test_end_date']),
    'raw_data_path': 'data/raw',
    'processed_data_folder_path': 'data/processed'
}

# Model Training Parameters
MODEL_PARAMS = {
    'learning_rate': ACTIVE_SETTINGS['learning_rate'],
    'max_depth': ACTIVE_SETTINGS['max_depth'],
    'n_estimators': ACTIVE_SETTINGS['n_estimators'],
    'early_stopping_rounds': ACTIVE_SETTINGS['early_stopping_rounds'],
    'min_child_weight': 3,  # Increased to reduce overfitting
    'subsample': 0.7,  # Reduced for better generalization
    'colsample_bytree': 0.7,  # Reduced for better generalization
    'objective': 'custom',  # Will use our DirectionalObjective
    'alpha': 0.3,  # Weight for directional component (0.3 = 30% directional, 70% MSE)
    'random_state': 42,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0   # L2 regularization
}

# Portfolio Parameters
PORTFOLIO_PARAMS = {
    'position_identifiers': 'COMPANYNAME',  # Instead of ticker-based params
    'initial_capital': ACTIVE_SETTINGS['initial_capital'],
    'max_positions': ACTIVE_SETTINGS['max_positions'],
    'position_size': 1.0 / ACTIVE_SETTINGS['max_positions'],
    'ranking_threshold': ACTIVE_SETTINGS['ranking_threshold'],
    'position_sizing': ACTIVE_SETTINGS['position_sizing'],
    'max_position_size': ACTIVE_SETTINGS['max_position_size'],
    'entry_threshold': ACTIVE_SETTINGS['entry_threshold'],
    'stop_loss': ACTIVE_SETTINGS['stop_loss'],
    'take_profit': ACTIVE_SETTINGS['take_profit'],
    'max_sector_exposure': ACTIVE_SETTINGS['max_sector_exposure'],
    'max_leverage': ACTIVE_SETTINGS['max_leverage']
}

# Analysis Parameters
ANALYSIS_PARAMS = {
    'rolling_window': ACTIVE_SETTINGS['rolling_window'],
    'benchmark': '^GSPC',  # S&P 500 as benchmark
    'risk_free_rate': 0.02,  # 2% risk-free rate
    'transaction_cost': ACTIVE_SETTINGS['transaction_cost']
}

# Data Split Parameters
SPLIT_DATES = {
    'train_start': '2011-01-03',
    'train_end': '2017-12-29',
    'val_start': '2018-01-02',
    'val_end': '2018-12-28',
    'test_start': '2019-01-02',
    'test_end': '2019-12-11'
}

# Output Directories
OUTPUT_DIRS = {
    'data': 'data/processed',
    'models': 'models',
    'plots': {
        'data': 'results/plots/data',
        'model': 'results/plots/model', 
        'portfolio': 'results/plots/portfolio'
    },
    'jsons': {
        'data': 'results/jsons/data',
        'model': 'results/jsons/model',
        'portfolio': 'results/jsons/portfolio'
    }
}

