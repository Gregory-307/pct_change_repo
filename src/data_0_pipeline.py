"""Main data processing pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"process_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import processing modules
from config import DATA_PARAMS
from config_features import get_enabled_features
from data_1_cleaning import clean_raw_data, pivot_data, handle_multiple_exchanges
from data_2_feature_engineering import calculate_target, split_data, select_features, validate_features
from _run_analysis import AnalysisPipeline

def process_data():
    """Process raw data into train/val/test sets."""
    try:
        # Load configuration
        start_date = pd.Timestamp(DATA_PARAMS['data_start_date'])
        horizon = DATA_PARAMS['prediction_horizon'].days
        train_end = DATA_PARAMS['train_end_date']
        val_end = DATA_PARAMS['val_end_date']
        enabled_features = get_enabled_features()
        
        logger.info(f"Processing data with:")
        logger.info(f"- Start date: {start_date}")
        logger.info(f"- Prediction horizon: {horizon} days")
        logger.info(f"- Train end: {train_end}")
        logger.info(f"- Val end: {val_end}")
        logger.info(f"- Enabled features: {len(enabled_features)}")
        
        # Create necessary directories
        for dir_path in ["data/processed"]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Load and filter raw data
        logger.info("\nLoading raw data...")
        raw_data = pd.read_csv(
            "data/raw/raw_data.csv",
            parse_dates=['ASOFDATE']
        )
        raw_data = raw_data[raw_data['ASOFDATE'] >= start_date]
        
        # Clean and pivot data
        logger.info("\nCleaning data...")
        cleaned_data = clean_raw_data(raw_data)

        logger.info("\nPivoting data...")
        pivoted_data = pivot_data(cleaned_data)
        pivoted_data.to_csv("data/processed/cleaned_pivoted_data.csv", index=False)

        # Select best exchange for each stock
        logger.info("\nSelecting best exchange for each stock...")
        pivoted_data = handle_multiple_exchanges(pivoted_data)
        # I've looked the extreme price changes up online and they are legit. 
        # This data is so shit. Do not buy S&P for data please.

        # Select features
        logger.info("\nSelecting features...")
        featured_data = select_features(pivoted_data, enabled_features)
        
        # Calculate target
        logger.info("\nCalculating target...")
        featured_data = calculate_target(featured_data, horizon)
        featured_data = featured_data.dropna(subset=['target'])
        featured_data.to_csv("data/processed/feature_target_data.csv", index=False)

        # Split data
        logger.info("\nSplitting data...")
        train_data, val_data, test_data = split_data(featured_data)
        
        for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            logger.info(f"Validating features for {name} dataset...")
            validate_features(data, enabled_features)

        # Save datasets
        logger.info("\nSaving datasets...")
        output_dir = Path("data/processed")
        
        for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            path = output_dir / f"{name}.csv"
            data.to_csv(path, index=False)
            logger.info(f"Saved {name} data with shape {data.shape} to {path}")
        
        # Run data analysis
        logger.info("\nRunning data analysis...")
        analysis_pipeline = AnalysisPipeline()
        analysis_pipeline.analyze_data({
            'full': featured_data,
            'train': train_data,
            'val': val_data,
            'test': test_data
        })
        
        logger.info("\nProcessing complete!")
        return train_data, val_data, test_data
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        process_data()
    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        sys.exit(1) 