"""Analysis pipeline orchestrator.

This script coordinates the execution of all analysis components:
1. Data analysis (coverage, quality)
2. Model analysis (feature importance, predictions)
3. Results analysis (performance, portfolio)
"""

import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from _data_analysis import DataAnalyzer
from _model_analysis import ModelAnalyzer
from _results_analysis import ResultsAnalyzer, set_plotting_style
from data_4_analysis_plots import (plot_stock_coverage, plot_feature_coverage,
                                 plot_price_trends, plot_feature_coverage,
                                 plot_split_coverage, plot_target_distribution,
                                 setup_directories)
from config import OUTPUT_DIRS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisPipeline:
    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = base_dir
        
        # Input directories
        self.data_dir = base_dir / "data"
        self.model_dir = base_dir / OUTPUT_DIRS['jsons']['model']
        self.portfolio_dir = base_dir / OUTPUT_DIRS['jsons']['portfolio']
        
        # Set up data plot directories
        self.plots_dir, self.analysis_dir = setup_directories()
        
        # Set plotting style
        set_plotting_style()
        
        # Initialize analyzers
        self.data_analyzer = DataAnalyzer()
        self.model_analyzer = ModelAnalyzer()
        self.results_analyzer = ResultsAnalyzer()
        
        # Create JSON directories from config
        for dir_path in OUTPUT_DIRS['jsons'].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_json(self, path: Path) -> Dict:
        """Load JSON file with error handling."""
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}
    
    def analyze_data_metrics(self) -> None:
        """Analyze data quality metrics from data pipeline."""
        logger.info("Analyzing data metrics...")
        
        try:
            # Load processed data
            data_dir = Path("data/processed")
            df = pd.read_csv(data_dir / "feature_target_data.csv")
            df['ASOFDATE'] = pd.to_datetime(df['ASOFDATE'])
            
            # Run data analysis
            coverage_analysis = self.data_analyzer.analyze_data_coverage(df)
            quality_analysis = self.data_analyzer.check_data_quality(df)
            stock_analysis = self.data_analyzer.analyze_stock_coverage(df, ticker_col='COMPANYNAME')
            
            # Log key findings
            logger.info("Data Analysis Summary:")
            logger.info(f"Total rows: {coverage_analysis['overall_stats']['total_rows']}")
            logger.info(f"Total features: {coverage_analysis['overall_stats']['total_columns']}")
            logger.info(f"Missing data: {coverage_analysis['missing_data']['total_missing_pct']:.1f}%")
            logger.info(f"Unique stocks: {stock_analysis['stock_counts']['total_unique_stocks']}")
            logger.info(f"Average stocks per day: {stock_analysis['stock_counts']['stocks_per_day']['mean']:.0f}")
            
            if quality_analysis['duplicates']['total_duplicates'] > 0:
                logger.warning(f"Found {quality_analysis['duplicates']['total_duplicates']} duplicate rows")
            
        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            raise
    
    def analyze_model_performance(self, model=None) -> None:
        """Analyze model predictions and feature importance."""
        logger.info("Analyzing model performance...")
        
        # Load prediction results
        train_pred = self.load_json(self.model_dir / "train_predictions.json")
        val_pred = self.load_json(self.model_dir / "validation_predictions.json")
        
        if train_pred and val_pred:
            # Analyze predictions
            self.results_analyzer.analyze_predictions(
                train_true=np.array(train_pred['actual']),
                train_pred=np.array(train_pred['predicted']),
                val_true=np.array(val_pred['actual']),
                val_pred=np.array(val_pred['predicted'])
            )
            
            # Create prediction plots
            self.results_analyzer.plot_prediction_scatter(
                train_true=np.array(train_pred['actual']),
                train_pred=np.array(train_pred['predicted']),
                val_true=np.array(val_pred['actual']),
                val_pred=np.array(val_pred['predicted'])
            )
            
            # Log metrics
            logger.info("Model Performance Metrics:")
            logger.info(f"Train MSE: {train_pred['metrics']['mse']:.6f}")
            logger.info(f"Train R²: {train_pred['metrics']['r2']:.3f}")
            logger.info(f"Validation MSE: {val_pred['metrics']['mse']:.6f}")
            logger.info(f"Validation R²: {val_pred['metrics']['r2']:.3f}")
    
    def analyze_portfolio_performance(self) -> None:
        """Analyze portfolio simulation results."""
        logger.info("Analyzing portfolio performance...")
        
        # Load simulation results
        sim_results = self.load_json(self.portfolio_dir / "simulation_results.json")
        
        if sim_results:
            # Run portfolio analysis
            analysis = self.results_analyzer.analyze_portfolio(sim_results)
            
            # Create portfolio plots
            self.results_analyzer.plot_portfolio_performance(sim_results)
            
            # Log key metrics
            logger.info("Portfolio Performance Metrics:")
            logger.info(f"Sharpe Ratio: {analysis['performance_metrics']['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {analysis['performance_metrics']['max_drawdown']:.1%}")
            logger.info(f"Win Rate: {analysis['performance_metrics']['win_rate']:.1%}")
    
    def analyze_data(self, data: Dict[str, pd.DataFrame]):
        """Run data analysis and generate plots."""
        logger.info("Running data analysis...")
        
        try:
            # Generate data analysis plots
            coverage_metrics = plot_stock_coverage(data['full'], self.plots_dir)
            feature_metrics = plot_feature_coverage(data['full'], self.plots_dir)
            price_metrics = plot_price_trends(data['full'], self.plots_dir)
            coverage_metrics.update(plot_feature_coverage(data['full'], self.plots_dir))
            split_metrics = plot_split_coverage(data['train'], data['val'], data['test'], self.plots_dir)
            target_metrics = plot_target_distribution(data['train'], data['val'], 
                                                   data['test'], self.plots_dir)
            
            # Save metrics
            metrics = {
                'coverage': coverage_metrics,
                'features': feature_metrics,
                'prices': price_metrics,
                'splits': split_metrics,
                'target': target_metrics
            }
            
            # Save to new JSON directory
            metrics_file = Path(OUTPUT_DIRS['jsons']['data']) / 'data_analysis_metrics.json'
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("Data analysis complete")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            raise
    
    def run_pipeline(self, model=None) -> None:
        """Run the complete analysis pipeline."""
        try:
            logger.info("Starting analysis pipeline...")
            
            # Run analyses
            self.analyze_data_metrics()
            self.analyze_model_performance(model)
            self.analyze_portfolio_performance()
            
            logger.info("Analysis pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}", exc_info=True)
            raise

def run_analysis(model=None):
    """Convenience function to run the analysis pipeline."""
    pipeline = AnalysisPipeline()
    pipeline.run_pipeline(model)

if __name__ == "__main__":
    run_analysis() 
