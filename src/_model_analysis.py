"""Model analysis module for analyzing model performance and feature importance.

Consolidates functionality from data_3_analysis.py and check_predictions.py for
analyzing model performance, feature importance, and SHAP values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Union
import shap
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import logging
import seaborn as sns
from config import OUTPUT_DIRS

logger = logging.getLogger(__name__)

class ModelAnalyzer:
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = results_dir
        self.analysis_dir = Path(OUTPUT_DIRS['jsons']['model'])
        self.plots_dir = results_dir / "plots"
        self.feature_dir = self.analysis_dir / "features"
        
        # Create directories
        for dir_path in [self.analysis_dir, self.plots_dir, self.feature_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style - using default matplotlib style for robustness
        plt.style.use('default')  # More reliable than seaborn
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def analyze_feature_importance(self, model, feature_names: List[str], 
                                 train_data: pd.DataFrame,
                                 val_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze feature importance, correlations, and SHAP values for train and validation sets."""
        # Get feature importance (works with most tree-based models)
        importance = getattr(model, "feature_importances_", None)
        if importance is None:
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
            
        # Create importance dictionary
        importance_dict = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True))
        
        analysis = {
            'feature_importance': importance_dict,
            'top_features': list(importance_dict.keys())[:10],
            'importance_stats': {
                'mean': float(importance.mean()),
                'std': float(importance.std()),
                'max': float(importance.max()),
                'min': float(importance.min())
            }
        }
        
        # Add correlation analysis for training data
        train_correlations = train_data[feature_names].corr()
        
        # Save training correlation matrix
        train_corr_path = self.feature_dir / 'train_correlation_matrix.csv'
        train_correlations.to_csv(train_corr_path)
        
        # Plot training correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(train_correlations, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations (Training Set)')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'train_correlation_heatmap.png', bbox_inches='tight')
        plt.close()
        
        # Find highly correlated pairs in training
        train_high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr = train_correlations.iloc[i, j]
                if abs(corr) > 0.8:
                    train_high_corr_pairs.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': float(corr)
                    })
        
        analysis['train_correlation_analysis'] = {
            'correlation_matrix_path': str(train_corr_path),
            'high_correlations': train_high_corr_pairs
        }
        
        # Add validation correlation analysis if validation data is provided
        if val_data is not None:
            val_correlations = val_data[feature_names].corr()
            
            # Save validation correlation matrix
            val_corr_path = self.feature_dir / 'val_correlation_matrix.csv'
            val_correlations.to_csv(val_corr_path)
            
            # Plot validation correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(val_correlations, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlations (Validation Set)')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'val_correlation_heatmap.png', bbox_inches='tight')
            plt.close()
            
            # Find highly correlated pairs in validation
            val_high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    corr = val_correlations.iloc[i, j]
                    if abs(corr) > 0.8:
                        val_high_corr_pairs.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(corr)
                        })
            
            analysis['val_correlation_analysis'] = {
                'correlation_matrix_path': str(val_corr_path),
                'high_correlations': val_high_corr_pairs
            }
            
            # Add correlation stability analysis
            correlation_diff = abs(train_correlations - val_correlations)
            unstable_correlations = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    diff = correlation_diff.iloc[i, j]
                    if diff > 0.2:  # Threshold for significant correlation change
                        unstable_correlations.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'train_correlation': float(train_correlations.iloc[i, j]),
                            'val_correlation': float(val_correlations.iloc[i, j]),
                            'difference': float(diff)
                        })
            
            analysis['correlation_stability'] = {
                'unstable_pairs': unstable_correlations,
                'max_difference': float(correlation_diff.max().max()),
                'mean_difference': float(correlation_diff.mean().mean())
            }
            
            # Plot correlation difference heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_diff, annot=True, cmap='YlOrRd')
            plt.title('Feature Correlation Differences (|Train - Val|)')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'correlation_differences.png', bbox_inches='tight')
            plt.close()
        
        # Save analysis
        output_file = self.analysis_dir / "feature_importance.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                              top_n: int = 15):
        """Plot feature importance bar chart."""
        # Sort and get top N features
        sorted_features = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:top_n])
        
        plt.figure(figsize=(10, 6))
        plt.barh(list(sorted_features.keys()), 
                list(sorted_features.values()),
                color='#2ecc71')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_importance.png', bbox_inches='tight')
        plt.close()
    
    def analyze_shap_values(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        """Calculate and analyze SHAP values."""
        try:
            # Create explainer and calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # If output is a list (for multi-class), take first element
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = {
                col: float(np.abs(shap_values[:, i]).mean())
                for i, col in enumerate(X.columns)
            }
            
            # Sort by absolute impact
            mean_abs_shap = dict(sorted(mean_abs_shap.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
            
            # Calculate feature interactions
            interaction_values = self._calculate_interactions(shap_values, X)
            
            analysis = {
                'mean_abs_shap': mean_abs_shap,
                'top_shap_features': list(mean_abs_shap.keys())[:10],
                'shap_stats': {
                    'mean': float(np.abs(shap_values).mean()),
                    'std': float(np.abs(shap_values).std()),
                    'max': float(np.abs(shap_values).max()),
                    'min': float(np.abs(shap_values).min())
                },
                'interaction_effects': interaction_values
            }
            
            # Save analysis
            output_file = self.analysis_dir / "shap_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'shap_summary.png', bbox_inches='tight')
            plt.close()
            
            # Create SHAP interaction plot for top features
            if len(X.columns) > 1:
                plt.figure(figsize=(12, 8))
                top_feature = list(mean_abs_shap.keys())[0]
                shap.dependence_plot(
                    top_feature, shap_values, X, 
                    interaction_index=list(mean_abs_shap.keys())[1],
                    show=False
                )
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'shap_interaction.png', bbox_inches='tight')
                plt.close()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return {}
    
    def _calculate_interactions(self, shap_values: np.ndarray, 
                              data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature interaction strengths."""
        try:
            n_features = shap_values.shape[1]
            interactions = {}
            
            # Calculate interactions for top 5 features only to keep computation manageable
            top_features = 5
            for i in range(min(top_features, n_features)):
                feature1 = data.columns[i]
                for j in range(i+1, min(top_features, n_features)):
                    feature2 = data.columns[j]
                    # Simple interaction strength metric
                    interaction = float(np.abs(
                        np.cov(shap_values[:, i], shap_values[:, j])[0, 1]
                    ))
                    interactions[f"{feature1}_{feature2}"] = interaction
            
            return dict(sorted(interactions.items(), 
                             key=lambda x: abs(x[1]), 
                             reverse=True))
        except Exception as e:
            logger.warning(f"Could not calculate interactions: {str(e)}")
            return {}
    
    def visualize_tree_structure(self, model, feature_names: List[str], 
                               max_depth: int = 3):
        """Visualize the tree structure (for tree-based models)."""
        try:
            plt.figure(figsize=(20, 10))
            plot_tree(model, feature_names=feature_names, 
                     max_depth=max_depth, filled=True, 
                     rounded=True, fontsize=10)
            plt.title('Decision Tree Structure')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'tree_structure.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error visualizing tree structure: {str(e)}")
    
    def analyze_prediction_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 dataset_name: str) -> Dict[str, Any]:
        """Calculate detailed prediction accuracy metrics."""
        metrics = {
            'dataset': dataset_name,
            'basic_metrics': {
                'mse': float(np.mean((y_true - y_pred) ** 2)),
                'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                'mae': float(np.mean(np.abs(y_true - y_pred))),
                'r2': float(1 - np.var(y_true - y_pred) / np.var(y_true))
            },
            'error_distribution': {
                'mean_error': float(np.mean(y_true - y_pred)),
                'std_error': float(np.std(y_true - y_pred)),
                'max_error': float(np.max(np.abs(y_true - y_pred))),
                'percentiles': {
                    str(p): float(np.percentile(np.abs(y_true - y_pred), p))
                    for p in [25, 50, 75, 90, 95, 99]
                }
            },
            'prediction_stats': {
                'true': {
                    'mean': float(np.mean(y_true)),
                    'std': float(np.std(y_true)),
                    'min': float(np.min(y_true)),
                    'max': float(np.max(y_true))
                },
                'predicted': {
                    'mean': float(np.mean(y_pred)),
                    'std': float(np.std(y_pred)),
                    'min': float(np.min(y_pred)),
                    'max': float(np.max(y_pred))
                }
            }
        }
        
        # Save metrics
        output_file = self.analysis_dir / f"{dataset_name}_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics 