"""
Model Evaluation Module for Regression Problem
Handles model evaluation, metrics calculation, and visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from pathlib import Path
from utils import RESULTS_DIR, load_model

class ModelEvaluator:
    """
    Class for evaluating regression models.
    """
    
    def __init__(self, model, X_test, y_test, model_name='Model'):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model to evaluate
        X_test : pd.DataFrame or np.array
            Test feature matrix
        y_test : pd.Series or np.array
            Test target variable
        model_name : str
            Name of the model
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        
    def predict(self):
        """Generate predictions."""
        print(f"Generating predictions for {self.model_name}...")
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred
    
    def calculate_metrics(self):
        """
        Calculate regression metrics.
        
        Returns:
        --------
        dict
            Dictionary of calculated metrics
        """
        if self.y_pred is None:
            self.predict()
        
        print(f"\n{'='*60}")
        print(f"EVALUATION METRICS: {self.model_name}")
        print(f"{'='*60}")
        
        # Regression metrics
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        # Try MAPE (may fail if y_test contains zeros)
        try:
            mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
            self.metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
        except:
            self.metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
        
        # Print metrics
        print("\nRegression Metrics:")
        for metric, value in self.metrics.items():
            if metric == 'MAPE':
                print(f"  {metric}: {value:.4f}%")
            else:
                print(f"  {metric}: {value:.4f}")
        
        return self.metrics
    
    def plot_predictions_vs_actual(self, save=True, figsize=(10, 8)):
        """
        Plot predicted vs actual values.
        
        Parameters:
        -----------
        save : bool
            Whether to save the plot
        figsize : tuple
            Figure size
        """
        if self.y_pred is None:
            self.predict()
        
        plt.figure(figsize=figsize)
        plt.scatter(self.y_test, self.y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual - {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = RESULTS_DIR / f'{self.model_name}_predictions_vs_actual.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Predictions vs Actual plot saved to {filepath}")
        
        plt.show()
    
    def plot_residuals(self, save=True, figsize=(10, 8)):
        """
        Plot residuals (errors).
        
        Parameters:
        -----------
        save : bool
            Whether to save the plot
        figsize : tuple
            Figure size
        """
        if self.y_pred is None:
            self.predict()
        
        residuals = self.y_test - self.y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0].scatter(self.y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = RESULTS_DIR / f'{self.model_name}_residuals.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Residuals plot saved to {filepath}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names=None, top_n=20, save=True, figsize=(12, 8)):
        """
        Plot feature importance (for tree-based models).
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        top_n : int
            Number of top features to display
        save : bool
            Whether to save the plot
        figsize : tuple
            Figure size
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not support feature importance")
            return
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            if hasattr(self.X_test, 'columns'):
                feature_names = self.X_test.columns.tolist()
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save:
            filepath = RESULTS_DIR / f'{self.model_name}_feature_importance.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {filepath}")
        
        plt.show()
    
    def generate_report(self, save=True):
        """
        Generate comprehensive evaluation report.
        
        Parameters:
        -----------
        save : bool
            Whether to save the report
        """
        if not self.metrics:
            self.calculate_metrics()
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION REPORT: {self.model_name}")
        print(f"{'='*60}")
        
        # Generate all plots
        self.plot_predictions_vs_actual(save=save)
        self.plot_residuals(save=save)
        self.plot_feature_importance(save=save)
        
        # Save metrics
        if save:
            import json
            filepath = RESULTS_DIR / f'{self.model_name}_metrics.json'
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            print(f"\nMetrics saved to {filepath}")
        
        return self.metrics

