"""Generate analysis plots for the processed data.

This script creates various visualizations to analyze:
1. Stock coverage over time
2. Feature availability over time
3. Average stock price trends
4. Feature coverage per stock
5. Train/Val/Test split analysis
6. Target distribution analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from config import DATA_PARAMS, OUTPUT_DIRS
from config_features import FEATURE_CONFIG
from _results_analysis import set_plotting_style
import json

logger = logging.getLogger(__name__)

def setup_directories() -> Tuple[Path, Path]:
    """Create standardized directory structure."""
    try:
        # Create plots directory using config
        plots_dir = Path(OUTPUT_DIRS['plots']['data'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create JSON directory using config (replaces old coverage_dir)
        json_dir = Path(OUTPUT_DIRS['jsons']['data'])
        json_dir.mkdir(parents=True, exist_ok=True)
        
        return plots_dir, json_dir
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        raise

def load_data() -> Dict[str, pd.DataFrame]:
    """Load all required datasets."""
    logger.info("Loading data...")
    data_dir = Path(OUTPUT_DIRS['data'])
    
    # Load main processed data
    df = pd.read_csv(data_dir / "feature_target_data.csv")
    df['ASOFDATE'] = pd.to_datetime(df['ASOFDATE'])
    
    # Load split data
    train = pd.read_csv(data_dir / "train.csv")
    train['ASOFDATE'] = pd.to_datetime(train['ASOFDATE'])
    
    val = pd.read_csv(data_dir / "val.csv")
    val['ASOFDATE'] = pd.to_datetime(val['ASOFDATE'])
    
    test = pd.read_csv(data_dir / "test.csv")
    test['ASOFDATE'] = pd.to_datetime(test['ASOFDATE'])
    
    return {
        'full': df,
        'train': train,
        'val': val,
        'test': test
    }

def save_plot(fig: plt.Figure, name: str, save_dir: Path):
    """Save plot with standardized settings."""
    try:
        output_path = save_dir / f"{name}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        logger.info(f"Saved plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {name}: {str(e)}")
        plt.close(fig)  # Ensure figure is closed even if save fails
        raise

def plot_stock_coverage(df: pd.DataFrame, save_dir: Path) -> Dict:
    """Plot number of stocks covered over time."""
    logger.info("Plotting stock coverage...")
    
    # Count unique stocks per day
    daily_counts = df.groupby('ASOFDATE')['COMPANYNAME'].nunique()
    
    fig, ax = plt.subplots()
    ax.plot(daily_counts.index, daily_counts.values, linewidth=2)
    ax.set_title('Number of Stocks Covered Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stocks')
    plt.xticks(rotation=45)
    
    save_plot(fig, 'stock_coverage', save_dir)
    
    metrics = {
        'mean_coverage': float(daily_counts.mean()),
        'min_coverage': int(daily_counts.min()),
        'max_coverage': int(daily_counts.max())
    }
    
    return metrics

def plot_feature_coverage(df: pd.DataFrame, save_dir: Path) -> Dict:
    """Plot average feature coverage per stock over time."""
    logger.info("Plotting feature coverage...")
    
    feature_cols = [col for col in df.columns 
                   if col not in ['ASOFDATE', 'COMPANYNAME', 'PRICECLOSE', 'SIMPLEINDUSTRYDESCRIPTION']]
    
    daily_coverage = df[feature_cols].notna().sum(axis=1).groupby(df['ASOFDATE']).mean()
    
    fig, ax = plt.subplots()
    ax.plot(daily_coverage.index, daily_coverage.values, linewidth=2)
    ax.set_title('Average Feature Coverage per Stock Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Number of Available Features')
    plt.xticks(rotation=45)
    
    total_features = len(feature_cols)
    ax.axhline(y=total_features, color='r', linestyle='--', 
               label='Total Possible Features')
    ax.legend()
    
    save_plot(fig, 'feature_coverage', save_dir)
    
    metrics = {
        'mean_coverage': float(daily_coverage.mean()),
        'coverage_pct': float(daily_coverage.mean() / total_features * 100),
        'total_features': total_features
    }
    
    return metrics

def plot_price_trends(df: pd.DataFrame, save_dir: Path) -> Dict:
    """Plot average stock price trends."""
    logger.info("Plotting price trends...")
    
    daily_avg_price = df.groupby('ASOFDATE')['PRICECLOSE'].mean()
    
    fig, ax = plt.subplots()
    ax.plot(daily_avg_price.index, daily_avg_price.values, linewidth=2)
    ax.set_title('Average Stock Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price ($)')
    plt.xticks(rotation=45)
    
    save_plot(fig, 'price_trends', save_dir)
    
    metrics = {
        'mean_price': float(daily_avg_price.mean()),
        'min_price': float(daily_avg_price.min()),
        'max_price': float(daily_avg_price.max())
    }
    
    return metrics

def plot_feature_categories(df: pd.DataFrame, save_dir: Path) -> Dict:
    """Plot feature coverage by category."""
    logger.info("Plotting feature categories...")
    
    category_features = {
        category: features for category, features in FEATURE_CONFIG.items()
        if isinstance(features, list)
    }
    
    category_coverage = {
        category: len([f for f in features if f in df.columns])
        for category, features in category_features.items()
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    categories = list(category_coverage.keys())
    coverage = list(category_coverage.values())
    
    ax.bar(categories, coverage)
    ax.set_title('Feature Count by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Features')
    plt.xticks(rotation=45, ha='right')
    
    save_plot(fig, 'feature_categories', save_dir)
    
    return {'category_counts': category_coverage}

def plot_target_distribution(train: pd.DataFrame, val: pd.DataFrame, 
                           test: pd.DataFrame, save_dir: Path) -> Dict:
    """Plot target distribution across splits."""
    logger.info("Plotting target distribution...")
    
    fig, ax = plt.subplots()
    metrics = {}
    
    for df, label in [(train, 'Train'), (val, 'Val'), (test, 'Test')]:
        target_data = df['target'].dropna()
        ax.hist(target_data, bins=50, alpha=0.5, label=label, density=True)
        
        metrics[label.lower()] = {
            'mean': float(target_data.mean()),
            'std': float(target_data.std()),
            'skew': float(target_data.skew())
        }
    
    ax.set_title('Target Distribution Across Splits')
    ax.set_xlabel('Forward Return (%)')
    ax.set_ylabel('Density')
    ax.legend()
    
    save_plot(fig, 'target_distribution', save_dir)
    
    return metrics

def plot_split_coverage(train: pd.DataFrame, val: pd.DataFrame, 
                       test: pd.DataFrame, save_dir: Path) -> Dict:
    """Plot stock coverage by split."""
    logger.info("Plotting split coverage...")
    
    fig, ax = plt.subplots()
    metrics = {}
    
    for df, label in [(train, 'Train'), (val, 'Val'), (test, 'Test')]:
        counts = df.groupby('ASOFDATE')['COMPANYNAME'].nunique()
        ax.plot(counts.index, counts.values, label=label, alpha=0.8)
        
        metrics[label.lower()] = {
            'mean_coverage': float(counts.mean()),
            'min_coverage': int(counts.min()),
            'max_coverage': int(counts.max())
        }
    
    ax.set_title('Stock Coverage by Split')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stocks')
    ax.legend()
    
    save_plot(fig, 'split_coverage', save_dir)
    
    return metrics

def main():
    """Generate all analysis plots."""
    logger.info("Starting analysis plots generation...")
    
    try:
        # Set plotting style
        set_plotting_style()
        
        # Create output directories
        plots_dir, json_dir = setup_directories()
        logger.info(f"Using plots directory: {plots_dir}")
        logger.info(f"Using JSON directory: {json_dir}")
        
        # Load all data
        data = load_data()
        
        # Generate plots and collect metrics
        metrics = {}
        
        # Generate each plot with proper error handling
        plot_functions = [
            ('stock_coverage', plot_stock_coverage),
            ('feature_coverage', plot_feature_coverage),
            ('price_trends', plot_price_trends),
            ('feature_categories', plot_feature_categories),
            ('target_distribution', lambda d, p: plot_target_distribution(data['train'], data['val'], data['test'], p)),
            ('split_coverage', lambda d, p: plot_split_coverage(data['train'], data['val'], data['test'], p))
        ]
        
        for name, func in plot_functions:
            try:
                if name in ['target_distribution', 'split_coverage']:
                    metrics[name] = func(None, plots_dir)
                else:
                    metrics[name] = func(data['full'], plots_dir)
                logger.info(f"Generated {name} plot successfully")
            except Exception as e:
                logger.error(f"Failed to generate {name} plot: {str(e)}")
                metrics[name] = {'error': str(e)}
        
        # Save metrics
        try:
            metrics_file = json_dir / "data_analysis_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved analysis metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
            raise
        
        logger.info(f"Analysis plots saved to {plots_dir}")
        logger.info(f"Analysis metrics saved to {metrics_file}")
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 