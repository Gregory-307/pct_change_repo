"""Data cleaning and preprocessing utilities.

This module handles:
1. Converting factor values to numeric format
2. Removing duplicates and invalid data
3. Pivoting data into wide format
4. Applying quality filters
5. Handling multiple exchange listings
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from pathlib import Path
from config import DATA_PARAMS
from config_features import FEATURE_CONFIG

logger = logging.getLogger(__name__)

def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate raw data structure and content."""
    # Check required columns
    required_cols = [
        'ASOFDATE', 'COMPANYNAME', 'EXCHANGETICKER', 'FACTORNAME',
        'FACTORVALUE', 'PRICECLOSE', 'SIMPLEINDUSTRYDESCRIPTION'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['ASOFDATE']):
        raise ValueError("ASOFDATE must be datetime")
    
    # Check for empty data
    if len(df) == 0:
        raise ValueError("Empty DataFrame")
    
    logger.info("Raw data validation passed")


    """Remove duplicate entries keeping first occurrence."""
    orig_len = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    exact_dupes = orig_len - len(df)
    
    # Remove duplicates by key columns
    orig_len = len(df)
    df = df.drop_duplicates(
        subset=['ASOFDATE', 'COMPANYNAME', 'FACTORNAME'],
        keep='first'
    )
    key_dupes = orig_len - len(df)
    
    logger.info(f"Removed {exact_dupes:,} exact duplicates")
    logger.info(f"Removed {key_dupes:,} key-based duplicates")
    return df

def pivot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot data into wide format with proper error handling."""
    logger.info("Pivoting data to wide format...")
    
    try:

        #print unique factor names and company names and shape  
        print("Unique factor names: ", df['FACTORNAME'].dropna().nunique())
        print("Unique company names: ", df['COMPANYNAME'].dropna().nunique())
        print("Shape before pivot: ", df.shape)
        
        # Check for duplicate entries that would break the pivot
        print("Checking for duplicate entries...")
        duplicate_mask = df.duplicated(subset=['ASOFDATE', 'COMPANYNAME', 'FACTORNAME', 'PRICECLOSE', 'EXCHANGETICKER'], keep=False)
        if duplicate_mask.any():
            logger.warning(f"Found {duplicate_mask.sum()} duplicate entries in pivot columns")
            logger.warning("Sample duplicates:\n" + 
                          str(df[duplicate_mask].sort_values(
                              ['ASOFDATE', 'COMPANYNAME', 'FACTORNAME', 'PRICECLOSE', 'EXCHANGETICKER']
                          ).head(10)))

            # Check if any FACTORVALUEs are different
            duplicate_factor_values = df[duplicate_mask]['FACTORVALUE'].nunique()
            if duplicate_factor_values > 1:
                logger.warning(f"{duplicate_factor_values} duplicate entries with different FACTORVALUE values found")

            # Keep last duplicate entry as a simple resolution
            df = df.drop_duplicates(
                subset=['ASOFDATE', 'COMPANYNAME', 'FACTORNAME', 'PRICECLOSE', 'EXCHANGETICKER'], 
                keep='last'
            )

        print("Shape after dropping duplicates: ", df.shape)

        # Pivot with aggregation as safety measure
        df_pivot = df.pivot(
            index=['ASOFDATE', 'COMPANYNAME', 'PRICECLOSE', 'EXCHANGETICKER'],
            columns='FACTORNAME',
            values='FACTORVALUE'
        ).reset_index()

        logger.info(f"Pivoted data shape: {df_pivot.shape}")
        return df_pivot
        
    except Exception as e:
        logger.error("Pivot operation failed")
        logger.error(f"Data shape before pivot: {df.shape}")
        logger.error(f"Duplicate count: {duplicate_mask.sum() if 'duplicate_mask' in locals() else 'Not calculated'}")
        logger.error(f"Unique FACTORNAME values: {df['FACTORNAME'].nunique()}")
        logger.error(f"Null values in FACTORNAME: {df['FACTORNAME'].isna().sum()}")
        raise ValueError(f"Pivot failed: {str(e)}") from e

def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data quality filters based on config parameters."""
    logger.info("Applying quality filters...")
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['ASOFDATE', 'COMPANYNAME', 'EXCHANGETICKER',
                                'PRICECLOSE', 'SIMPLEINDUSTRYDESCRIPTION']]
    
    # Apply minimum history requirement
    company_history = df.groupby('COMPANYNAME')['ASOFDATE'].agg(['min', 'max', 'count'])
    valid_companies = company_history[
        company_history['count'] >= DATA_PARAMS['min_history_days']
    ].index
    
    orig_len = len(df)
    df = df[df['COMPANYNAME'].isin(valid_companies)]
    history_filtered = orig_len - len(df)
    
    # Apply factor coverage requirement
    orig_len = len(df)
    factor_coverage = df[feature_cols].notna().mean(axis=1)
    df = df[factor_coverage >= DATA_PARAMS['min_factor_coverage']]
    coverage_filtered = orig_len - len(df)
    
    logger.info(f"Removed {history_filtered:,} rows due to insufficient history")
    logger.info(f"Removed {coverage_filtered:,} rows due to low factor coverage")
    logger.info(f"Final shape after filtering: {df.shape}")
    
    return df

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data by removing missing values and converting factor values to numeric."""
    logging.info(f"Initial data shape: {df.shape}")
    
    # Remove rows with missing company names
    df = df[~df['COMPANYNAME'].isna()]
    logging.info(f"Shape after removing missing company names: {df.shape}")

    # Remove rows with less than min_stocks_per_day stocks per day
    stocks_per_day = df.groupby('ASOFDATE')['COMPANYNAME'].nunique()
    valid_dates = stocks_per_day[stocks_per_day >= DATA_PARAMS['min_stocks_per_day']].index
    df = df[df['ASOFDATE'].isin(valid_dates)]
    logging.info(f"Shape after removing days with < {DATA_PARAMS['min_stocks_per_day']} stocks: {df.shape}")
    
    # # Clean and convert FACTORVALUE to numeric
    # try:
    #     # First try direct conversion
    #     df['FACTORVALUE'] = pd.to_numeric(df['FACTORVALUE'], errors='coerce')
    # except Exception as e:
    #     logging.warning(f"Direct numeric conversion failed: {e}")
    #     # If that fails, try cleaning the string values first
    #     df['FACTORVALUE'] = df['FACTORVALUE'].astype(str).str.strip()
    #     df['FACTORVALUE'] = df['FACTORVALUE'].replace('', np.nan)
    #     df['FACTORVALUE'] = df['FACTORVALUE'].replace('None', np.nan)
    #     df['FACTORVALUE'] = pd.to_numeric(df['FACTORVALUE'], errors='coerce')
    
    # Drop rows where conversion failed
    # df = df.dropna(subset=['FACTORVALUE'])
    # logging.info(f"Shape after cleaning and converting factor values: {df.shape}")
        
    # # Convert float64 to float32 to save memory
    # df['FACTORVALUE'] = df['FACTORVALUE'].astype('float32')
    
    return df

def handle_multiple_exchanges(df: pd.DataFrame) -> pd.DataFrame:
    """Handle multiple exchange listings after pivot.
    Selects the best exchange for each stock based on data coverage
    and checks for extreme price changes.
    """
    logging.info("Handling multiple exchange listings...")
    logging.info(f"Initial shape: {df.shape}")
    
    # Calculate coverage metrics for each company-exchange combination
    exchange_coverage = df.groupby(['COMPANYNAME', 'EXCHANGETICKER']).agg({
        'ASOFDATE': ['min', 'max', 'count']
        })
    
    # Flatten multi-level columns
    exchange_coverage.columns = ['date_min', 'date_max', 'date_count']
    exchange_coverage = exchange_coverage.reset_index()
    
    # Calculate date range in days
    exchange_coverage['date_range'] = (
        exchange_coverage['date_max'] - 
        exchange_coverage['date_min']
    ).dt.days
    
    # Select exchange with longest date range for each company
    best_exchanges = (exchange_coverage
                     .sort_values(['COMPANYNAME', 'date_range'], ascending=[True, False])
                     .groupby('COMPANYNAME', as_index=False)
                     .first()[['COMPANYNAME', 'EXCHANGETICKER']])
    
    # Filter data to keep only the best exchange per company
    df = df.merge(best_exchanges, on=['COMPANYNAME', 'EXCHANGETICKER'])
    logging.info(f"Shape after selecting best exchange per stock: {df.shape}")
    
    # Check for extreme price changes
    logging.info("Checking for extreme price changes...")
    df = df.sort_values(['COMPANYNAME', 'ASOFDATE'])
    price_changes = df.groupby('COMPANYNAME')['PRICECLOSE'].pct_change()
    
    # Flag extreme changes (>100% increase or >50% decrease in a day)
    extreme_changes = (price_changes > 1.0) | (price_changes < -0.5)
    if extreme_changes.any():
        n_extreme = extreme_changes.sum()
        logging.warning(f"Found {n_extreme} extreme price changes (>100% up or >50% down in a day)")
        
        # Log some examples
        extreme_examples = df[extreme_changes].sort_values('ASOFDATE').head()
        logging.warning("Example extreme changes:")
        for _, row in extreme_examples.iterrows():
            prev_price = df[(df['COMPANYNAME'] == row['COMPANYNAME']) & 
                          (df['ASOFDATE'] < row['ASOFDATE'])]['PRICECLOSE'].iloc[-1]
            change = (row['PRICECLOSE'] - prev_price) / prev_price * 100
            logging.warning(f"{row['COMPANYNAME']}: {row['ASOFDATE']:%Y-%m-%d} "
                          f"- Price change: {change:.1f}% "
                          f"({prev_price:.2f} -> {row['PRICECLOSE']:.2f})")
    
    return df

