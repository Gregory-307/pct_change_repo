"""
Model training and prediction utilities
"""

import logging
from pathlib import Path
import pickle
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from config import MODEL_PARAMS, OUTPUT_DIRS
import shap
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class DirectionalObjective:
    """Custom objective function that balances MSE with directional accuracy."""
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for directional component (0 to 1)
                  0 = pure MSE, 1 = pure directional
        """
        self.alpha = alpha
    
    def __call__(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gradient and hessian for the custom objective.
        
        Combines MSE with directional accuracy:
        Loss = (1-α)*(y-ŷ)² + α*max(0, -(y*ŷ))
        
        The second term penalizes wrong directional predictions.
        """
        y = dtrain.get_label()
        
        # MSE component
        mse_grad = predt - y
        mse_hess = np.ones_like(predt)
        
        # Directional component
        dir_grad = np.where(y * predt < 0, -y, 0)
        dir_hess = np.where(y * predt < 0, 1, 0)
        
        # Combine components
        grad = (1 - self.alpha) * mse_grad + self.alpha * dir_grad
        hess = (1 - self.alpha) * mse_hess + self.alpha * dir_hess
        
        return grad, hess
    
    def metric(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        """Calculate the custom metric for evaluation."""
        y = dtrain.get_label()
        
        # Calculate MSE
        mse = np.mean((y - predt) ** 2)
        
        # Calculate directional accuracy
        dir_acc = np.mean(np.sign(y) == np.sign(predt))
        
        # Combined metric
        score = (1 - self.alpha) * mse + self.alpha * (1 - dir_acc)
        
        return 'custom_metric', score

class PctChangeModel:
    def __init__(self, train_path: Path, val_path: Path, enabled_features: list, 
                 target_col: str = 'target', params: Dict = None, model_dir: str = "models"):
        self.train_path = train_path
        self.val_path = val_path
        self.enabled_features = enabled_features
        self.target_col = target_col
        self.params = params or MODEL_PARAMS.copy()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self._feature_cols = None
        
        # Initialize custom objective if specified
        if self.params.get('objective') == 'custom':
            self.objective = DirectionalObjective(alpha=self.params.get('alpha', 0.3))
            # Remove these from xgboost params
            self.params = {k: v for k, v in self.params.items() 
                         if k not in ['objective', 'alpha']}
    
    @property
    def feature_cols(self):
        """Public access to features used in training"""
        return self._feature_cols
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate training/validation data"""
        train_data = pd.read_csv(self.train_path)
        val_data = pd.read_csv(self.val_path) if self.val_path else None
        
        # Validate features
        for data, name in [(train_data, 'training'), (val_data, 'validation')]:
            if data is not None:
                missing = set(self.enabled_features) - set(data.columns)
                if missing:
                    raise ValueError(f"Missing features in {name} data: {missing}")
        
        return train_data, val_data
    
    def train(self):
        """Handle full training workflow from data loading to model fitting"""
        train_data, val_data = self.load_training_data()
        
        X_train = train_data[self.enabled_features]
        y_train = train_data[self.target_col]
        
        # Prepare validation set as DMatrix
        evals = []
        if val_data is not None:
            X_val = val_data[self.enabled_features]
            y_val = val_data[self.target_col]
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dval, 'validation')]  # XGBoost requires DMatrix for eval sets

        self._feature_cols = X_train.columns.tolist()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Training parameters
        train_params = self.params.copy()
        if hasattr(self, 'objective'):
            train_params['obj'] = self.objective
            train_params['custom_metric'] = self.objective.metric
        
        # Train model with proper eval set
        self.model = xgb.train(
            train_params,
            dtrain,
            evals=evals,  # Now contains DMatrix objects
            num_boost_round=train_params.get('n_estimators', 500),
            early_stopping_rounds=train_params.get('early_stopping_rounds'),
            verbose_eval=100
        )
        
        # Calculate feature importance
        importance_scores = self.model.get_score(importance_type='gain')
        self.feature_importance = pd.Series(
            {col: importance_scores.get(col, 0) for col in self.feature_cols}
        ).sort_values(ascending=False)
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained - call train() first")
        if not set(self.enabled_features).issubset(data.columns):
            missing = set(self.enabled_features) - set(data.columns)
            raise ValueError(f"Missing required features: {missing}")
            
        return self.model.predict(xgb.DMatrix(data[self.enabled_features]))
    
    def save(self, filename: str):
        """Save model to disk"""
        try:
            # Convert to Path if string
            path = Path(filename) if isinstance(filename, str) else filename
            
            # If relative path, make it relative to model_dir
            if not path.is_absolute() and self.model_dir:
                path = self.model_dir / path
                
            logging.info(f"Saving model to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                model_data = {
                    'model': self.model,
                    'feature_cols': self.feature_cols,
                    'feature_importance': self.feature_importance
                }
                pickle.dump(model_data, f)
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")
            raise
    
    def load(self, filename: str):
        """Load model from disk"""
        if self.model_dir:
            path = self.model_dir / filename
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self._feature_cols = data['feature_cols']
                self.feature_importance = data['feature_importance']

def train_model(data: pd.DataFrame, params: dict = None) -> PctChangeModel:
    """Train a model with the given data and parameters."""
    # Use default parameters if none provided
    params = params or MODEL_PARAMS
    
    # Initialize model
    model = PctChangeModel(train_path=Path(params['train_path']), val_path=Path(params['val_path']), enabled_features=params['enabled_features'], target_col=params['target_col'], params=params)
    
    # Train model
    model.train()
    
    return model

def analyze_features(model: PctChangeModel, data: pd.DataFrame) -> dict:
    """Analyze feature importance and relationships."""
    # Get feature importance
    importance = model.feature_importance.to_dict()
    
    # Get feature correlations
    feature_cols = model.feature_importance.index
    correlations = data[feature_cols].corr()
    
    # Save correlation matrix to configured path
    corr_path = Path(OUTPUT_DIRS['jsons']['model']) / "features/correlation_matrix.csv"
    corr_path.parent.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(corr_path)
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr = correlations.iloc[i, j]
            if abs(corr) > 0.8:
                high_corr_pairs.append({
                    'feature1': feature_cols[i],
                    'feature2': feature_cols[j],
                    'correlation': float(corr)
                })
    
    # Add SHAP values calculation
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(data[model.feature_cols])
    
    return {
        'feature_importance': importance,
        'shap_summary': {
            'mean_abs_shap': pd.Series(np.abs(shap_values).mean(0), 
                                     index=model.feature_cols).to_dict(),
            'interaction_effects': calculate_interactions(shap_values, data)
        },
        'correlation_matrix_path': str(corr_path),
        'high_correlations': high_corr_pairs
    }

def calculate_interactions(shap_values: np.ndarray, data: pd.DataFrame) -> Dict:
    """Calculate top feature interactions using SHAP."""
    # Placeholder implementation - replace with actual interaction calculation
    return {
        f"{data.columns[i]}_{data.columns[j]}": 0.0 
        for i in range(5) for j in range(i+1, 5)
    }