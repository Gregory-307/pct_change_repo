# Merge from:
# - analyze_raw.py
# - analyze_raw_data.py
# - analyze_data_coverage.py
# - check_coverage.py
# - check_stock_coverage.py
# Core functionality:
# - Raw data coverage analysis
# - Feature availability tracking
# - Stock coverage over time
# - Data quality checks 

"""Data analysis module for analyzing data coverage and quality.

Consolidates functionality from analyze_raw.py, analyze_data_coverage.py,
and check_coverage.py for tracking data availability and quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any
import logging
from config import OUTPUT_DIRS

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.json_dir = Path(OUTPUT_DIRS['jsons']['data'])
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_data_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data coverage and completeness."""
        # Handle date information safely
        date_info = {}
        if 'ASOFDATE' in df.columns:
            date_info = {
                'start': df['ASOFDATE'].min().strftime('%Y-%m-%d'),
                'end': df['ASOFDATE'].max().strftime('%Y-%m-%d'),
                'trading_days': len(df['ASOFDATE'].unique())
            }
        elif isinstance(df.index, pd.DatetimeIndex):
            date_info = {
                'start': df.index.min().strftime('%Y-%m-%d'),
                'end': df.index.max().strftime('%Y-%m-%d'),
                'trading_days': len(df.index.unique())
            }
        else:
            date_info = {
                'start': None,
                'end': None,
                'trading_days': len(df)
            }
            logger.warning("No date information found in data")

        analysis = {
            'overall_stats': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
                'date_range': date_info
            },
            'missing_data': {
                'columns': {
                    col: {
                        'missing_count': int(df[col].isna().sum()),
                        'missing_pct': float(df[col].isna().mean() * 100)
                    } for col in df.columns
                },
                'total_missing_cells': int(df.isna().sum().sum()),
                'total_missing_pct': float(df.isna().mean().mean() * 100)
            },
            'feature_stats': {
                col: {
                    'dtype': str(df[col].dtype),
                    'unique_values': int(df[col].nunique()),
                    'mean': float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    'std': float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    'min': float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    'max': float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None
                } for col in df.columns
            }
        }
        
        # Save to new JSON directory
        output_file = self.json_dir / "data_coverage.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        return analysis
    
    def analyze_stock_coverage(self, df: pd.DataFrame, 
                             ticker_col: str = 'ticker') -> Dict[str, Any]:
        """Analyze coverage across different stocks."""
        analysis = {
            'stock_counts': {
                'total_unique_stocks': int(df[ticker_col].nunique()),
                'stocks_per_day': {
                    'mean': float(df.groupby('ASOFDATE')[ticker_col].nunique().mean()),
                    'min': int(df.groupby('ASOFDATE')[ticker_col].nunique().min()),
                    'max': int(df.groupby('ASOFDATE')[ticker_col].nunique().max())
                }
            },
            'stock_history': {
                ticker: {
                    'days_present': int(df[df[ticker_col] == ticker].shape[0]),
                    'first_date': df[df[ticker_col] == ticker]['ASOFDATE'].min().strftime('%Y-%m-%d'),
                    'last_date': df[df[ticker_col] == ticker]['ASOFDATE'].max().strftime('%Y-%m-%d'),
                    'missing_days': int(len(df['ASOFDATE'].unique()) - df[df[ticker_col] == ticker].shape[0])
                } for ticker in df[ticker_col].unique()
            }
        }
        
        # Save analysis
        output_file = self.json_dir / "stock_coverage.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        return analysis
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality checks."""
        quality_checks = {
            'duplicates': {
                'total_duplicates': int(df.duplicated().sum()),
                'duplicate_pct': float(df.duplicated().mean() * 100)
            },
            'outliers': {
                col: self._analyze_outliers(df[col])
                for col in df.select_dtypes(include=[np.number]).columns
            },
            'consistency': {
                'index_gaps': self._check_index_gaps(df),
                'value_ranges': {
                    col: {
                        'out_of_range_count': int(np.sum((df[col] < -1000) | (df[col] > 1000)))
                        if pd.api.types.is_numeric_dtype(df[col]) else 0
                    } for col in df.columns
                }
            }
        }
        
        # Save analysis
        output_file = self.json_dir / "data_quality.json"
        with open(output_file, 'w') as f:
            json.dump(quality_checks, f, indent=2, default=str)
            
        return quality_checks
    
    def _analyze_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze outliers in a numeric series using IQR method."""
        if not pd.api.types.is_numeric_dtype(series):
            return {}
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'outlier_count': int(len(outliers)),
            'outlier_pct': float(len(outliers) / len(series) * 100),
            'bounds': {
                'lower': float(lower_bound),
                'upper': float(upper_bound)
            }
        }
    
    def _check_index_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for gaps in the datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return {}
            
        # Get unique dates and sort
        dates = pd.Series(df.index.unique()).sort_values()
        
        # Calculate gaps
        gaps = []
        for i in range(len(dates)-1):
            gap = (dates.iloc[i+1] - dates.iloc[i]).days
            if gap > 1:  # More than 1 day gap
                gaps.append({
                    'start_date': dates.iloc[i].strftime('%Y-%m-%d'),
                    'end_date': dates.iloc[i+1].strftime('%Y-%m-%d'),
                    'gap_days': int(gap)
                })
        
        return {
            'total_gaps': len(gaps),
            'gap_details': gaps[:10]  # Limit to first 10 gaps
        } 