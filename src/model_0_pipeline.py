"""Model pipeline for training, validation and portfolio simulation.

This script handles:
1. Loading processed data
2. Training the model
3. Analyzing feature importance
4. Running portfolio simulation
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

from model_1_class import PctChangeModel
from model_2_portfolio_simulator import PortfolioSimulator
from _run_analysis import run_analysis
from config import ACTIVE_SETTINGS, MODEL_PARAMS, OUTPUT_DIRS
from config_features import FEATURE_CONFIG, get_enabled_features, validate_price_columns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create standardized directory structure."""
    dirs = [
        OUTPUT_DIRS['models'],
        OUTPUT_DIRS['plots']['model'],
        OUTPUT_DIRS['plots']['portfolio']
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def save_predictions(predictions: np.ndarray, actual: np.ndarray, split: str):
    """Save predictions and actual values to JSON."""
    output = {
        'predicted': predictions.tolist(),
        'actual': actual.tolist(),
        'metrics': {
            'mse': float(np.mean((actual - predictions) ** 2)),
            'r2': float(1 - np.var(actual - predictions) / np.var(actual))
        }
    }
    
    # Save to configured model JSON directory
    output_dir = Path(OUTPUT_DIRS['jsons']['model'])
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{split}_predictions.json"
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved {split} predictions to {filepath}")

def run_pipeline():
    """Run the complete model pipeline."""
    try:
        start_time = datetime.now()
        logger.info("Starting model pipeline...")

        # Create directory structure
        setup_directories()
        logger.info("Directory structure created")
        
        # Get data paths from config
        data_dir = Path(OUTPUT_DIRS['data'])
        train_path = data_dir / "train.csv"
        val_path = data_dir / "val.csv"
        
        # Get enabled features from config
        enabled_features = get_enabled_features()
        
        # Initialize model with data source and features
        model = PctChangeModel(
            train_path=train_path,
            val_path=val_path,
            enabled_features=enabled_features,
            params=MODEL_PARAMS
        )
        
        # Train model handles its own data loading
        model.train()
        
        # Predictions use same feature set
        test_data = pd.read_csv(data_dir / "test.csv")
        test_pred = model.predict(test_data)

        # Generate and save predictions
        logger.info("Generating predictions...")
        try:
            # Training set predictions
            train_data = pd.read_csv(train_path)
            train_pred = model.predict(train_data)
            save_predictions(train_pred, train_data['target'], 'train')
            logger.info("Training predictions saved")
            
            # Validation set predictions
            val_data = pd.read_csv(val_path)
            val_pred = model.predict(val_data)
            save_predictions(val_pred, val_data['target'], 'validation')
            logger.info("Validation predictions saved")

            # Validate price columns
            validate_price_columns(train_data)
            validate_price_columns(val_data)
            validate_price_columns(test_data)
        except Exception as e:
            logger.error(f"Prediction generation failed: {str(e)}")
            raise

        # Save model with features used
        logger.info("Saving model...")
        try:
            model_path = Path(OUTPUT_DIRS['models']) / "model.pkl"
            model.save(model_path)
            # Verify model was saved correctly
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            raise

        # Run portfolio simulation
        logger.info("Running portfolio simulation...")
        try:
            simulator = PortfolioSimulator(model)
            simulation_results = simulator.simulate_portfolio()
            logger.info("Portfolio simulation completed")
        except Exception as e:
            logger.error(f"Portfolio simulation failed: {str(e)}")
            raise

        # Run analysis pipeline
        logger.info("Running analysis pipeline...")
        try:
            run_analysis(model)
            logger.info("Analysis pipeline completed")
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            raise

        # Log completion
        duration = datetime.now() - start_time
        logger.info(f"Pipeline completed in {duration}")

        return {
            "simulation_results": simulation_results
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline()