"""Feature engineering and target calculation utilities.

This module handles:
1. Feature selection based on config
2. Target variable calculation
3. Train/val/test splitting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from config import DATA_PARAMS, SPLIT_DATES
from config_features import FEATURE_CONFIG, get_enabled_features

logger = logging.getLogger(__name__)

def validate_features(df: pd.DataFrame, enabled_features: List[str]) -> None:
    """Validate that required features are present."""
    missing_features = [f for f in enabled_features if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} enabled features:")
        logger.warning(f"First 5 missing: {missing_features[:5]}")
        logger.warning("These features will not be available for modeling")

def old_calculate_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Calculate forward returns for target variable."""
    logger.info(f"Calculating {horizon}-day forward returns...")
    
    df = df.sort_values(['ASOFDATE', 'COMPANYNAME'])
    df['price_in_{horzion}_days'] = df.groupby('COMPANYNAME')['PRICECLOSE'].shift(-horizon)
    df['target'] = (df['price_in_{horzion}_days'] / df['PRICECLOSE']) - 1
    
    missing = df['target'].isna().sum()
    logger.info(f"Missing target values: {missing:,} ({missing/len(df):.1%})")
    return df


def calculate_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Calculate forward returns using mean price over horizon window.
    
    Args:
        df: DataFrame with ASOFDATE, COMPANYNAME, and PRICECLOSE columns
        horizon: Number of days to look forward
    
    Returns:
        DataFrame with added target column representing forward returns
    """
    logger.info(f"Calculating {horizon}-day forward returns...")
    
    # Sort chronologically for each company
    df = df.sort_values(['COMPANYNAME', 'ASOFDATE'])
    
    # Calculate horizon window end dates
    df['horizon_end'] = df['ASOFDATE'] + pd.Timedelta(days=horizon)
    
    # For each company, calculate mean price in horizon window
    def get_horizon_mean(group):
        dates = group['ASOFDATE'].values
        prices = group['PRICECLOSE'].values
        horizon_ends = group['horizon_end'].values
        means = np.full_like(prices, np.nan, dtype=np.float64)
        
        for i in range(len(dates)):
            # Get prices in horizon window
            mask = (dates > dates[i]) & (dates <= horizon_ends[i])
            if mask.any():
                means[i] = prices[mask].mean()
        
        return pd.Series(means, index=group.index)
    
    # Calculate mean future price and return
    df['horizon_mean'] = df.groupby('COMPANYNAME', group_keys=False).apply(
        lambda x: pd.Series(get_horizon_mean(x), index=x.index)
    )
    df['target'] = ((df['horizon_mean'] / df['PRICECLOSE']) - 1) * 100

    # Move horizon_mean, horizon_end, and target to the beginning of the dataframe after index columns
    df = df[['ASOFDATE', 'COMPANYNAME', 'EXCHANGETICKER', 'PRICECLOSE', 'horizon_mean', 'horizon_end', 'target'] + [col for col in df.columns if col not in ['ASOFDATE', 'COMPANYNAME', 'EXCHANGETICKER', 'PRICECLOSE', 'SIMPLEINDUSTRYDESCRIPTION', 'horizon_mean', 'horizon_end', 'target']]]
    
    # Log statistics
    missing = df['target'].isna().sum()
    logger.info(f"Missing target values: {missing:,} ({missing/len(df):.1%})")
    
    return df

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets based on dates."""
    logger.info("Splitting data...")
    
    train = df[(df['ASOFDATE'] >= DATA_PARAMS['train_start_date']) & (df['ASOFDATE'] <= DATA_PARAMS['train_end_date'])]
    val = df[(df['ASOFDATE'] >= DATA_PARAMS['val_start_date']) & (df['ASOFDATE'] <= DATA_PARAMS['val_end_date'])]
    test = df[(df['ASOFDATE'] >= DATA_PARAMS['test_start_date']) & (df['ASOFDATE'] <= DATA_PARAMS['test_end_date'])]
    
    logger.info(f"Train: {len(train):,} rows ({train['ASOFDATE'].min()} to {train['ASOFDATE'].max()})")
    logger.info(f"Val: {len(val):,} rows ({val['ASOFDATE'].min()} to {val['ASOFDATE'].max()})")
    logger.info(f"Test: {len(test):,} rows ({test['ASOFDATE'].min()} to {test['ASOFDATE'].max()})")
    
    return train, val, test

def select_features(df: pd.DataFrame, enabled_features: List[str]) -> pd.DataFrame:
    """Select enabled features and metadata columns."""
    logger.info("Selecting features...")
    
    metadata_cols = ['ASOFDATE', 'COMPANYNAME', 'EXCHANGETICKER', 'PRICECLOSE']
    available_features = [f for f in enabled_features if f in df.columns]
    
    logger.info(f"Using {len(available_features)} out of {len(enabled_features)} enabled features")
    return df[metadata_cols + available_features]
