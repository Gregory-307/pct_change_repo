"""Data analysis and validation utilities.

This module handles:
1. Dataset statistics and validation
2. Feature coverage analysis
3. Data quality checks
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from config import OUTPUT_DIRS

logger = logging.getLogger(__name__)

def analyze_split(df: pd.DataFrame, split_name: str) -> None:
    """Log basic statistics for a dataset split."""
    feature_cols = [col for col in df.columns 
                   if col not in ['ASOFDATE', 'COMPANYNAME', 'PRICECLOSE', 'target']]
    
    logger.info(f"\n{split_name} Dataset:")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Companies: {df['COMPANYNAME'].nunique():,}")
    logger.info(f"Date range: {df['ASOFDATE'].min():%Y-%m-%d} to {df['ASOFDATE'].max():%Y-%m-%d}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Mean feature coverage: {df[feature_cols].notna().mean().mean():.1%}")
    
    if 'target' in df.columns:
        logger.info(f"Target mean: {df['target'].mean()}")
        logger.info(f"Target std: {df['target'].std()}")

def plot_analysis(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 output_dir: Path) -> None:
    """Generate basic analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    for df, label in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
        plt.hist(df['target'].dropna(), bins=50, alpha=0.5, label=label, density=True)
    plt.title('Target Distribution')
    plt.xlabel('Forward Return')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_dir / 'target_dist.png', bbox_inches='tight')
    plt.close()
    
    # Plot stock coverage
    plt.figure(figsize=(10, 6))
    for df, label in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
        counts = df.groupby('ASOFDATE')['COMPANYNAME'].nunique()
        plt.plot(counts.index, counts.values, label=label, alpha=0.8)
    plt.title('Stocks per Day')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(output_dir / 'stock_coverage.png', bbox_inches='tight')
    plt.close()

def analyze_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Analyze all dataset splits and generate plots."""
    try:
        # Log statistics
        for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            analyze_split(df, split_name)
        
        # Generate plots using configured plot directory
        plot_analysis(train_df, val_df, test_df, Path(OUTPUT_DIRS['plots']['data']))
        logger.info(f"\nAnalysis complete. Plots saved to {OUTPUT_DIRS['plots']['data']}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise