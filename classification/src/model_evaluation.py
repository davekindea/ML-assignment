"""
Model Evaluation Module for Classification Problem
Handles model evaluation, metrics calculation, and visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from pathlib import Path
from utils import RESULTS_DIR, load_model

class ModelEvaluator:
    """
    Class for evaluating classification models.
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
        
        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)
        else:
            self.y_pred_proba = None
        
        return self.y_pred
    
    def calculate_metrics(self, average='weighted'):
        """
        Calculate classification metrics.
        
        Parameters:
        -----------
        average : str
            Averaging method for multi-class metrics
        
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
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average=average, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, average=average, zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, average=average, zero_division=0)
        
        self.metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        # ROC-AUC (for binary classification or multi-class)
        if self.y_pred_proba is not None:
            try:
                if len(np.unique(self.y_test)) == 2:
                    # Binary classification
                    roc_auc = roc_auc_score(self.y_test, self.y_pred_proba[:, 1])
                else:
                    # Multi-class classification
                    roc_auc = roc_auc_score(self.y_test, self.y_pred_proba, 
                                          multi_class='ovr', average=average)
                self.metrics['ROC-AUC'] = roc_auc
            except Exception as e:
                print(f"Could not calculate ROC-AUC: {str(e)}")
        
        # Print metrics
        print("\nClassification Metrics:")
        for metric, value in self.metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        return self.metrics
    
    def plot_confusion_matrix(self, save=True, figsize=(10, 8)):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        save : bool
            Whether to save the plot
        figsize : tuple
            Figure size
        """
        if self.y_pred is None:
            self.predict()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(self.y_test),
                   yticklabels=np.unique(self.y_test))
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save:
            filepath = RESULTS_DIR / f'{self.model_name}_confusion_matrix.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {filepath}")
        
        plt.show()
    
    def plot_roc_curve(self, save=True, figsize=(10, 8)):
        """
        Plot ROC curve (for binary classification).
        
        Parameters:
        -----------
        save : bool
            Whether to save the plot
        figsize : tuple
            Figure size
        """
        if self.y_pred_proba is None:
            print("Model does not support probability predictions")
            return
        
        if len(np.unique(self.y_test)) != 2:
            print("ROC curve is for binary classification only")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba[:, 1])
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba[:, 1])
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = RESULTS_DIR / f'{self.model_name}_roc_curve.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {filepath}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save=True, figsize=(10, 8)):
        """
        Plot Precision-Recall curve.
        
        Parameters:
        -----------
        save : bool
            Whether to save the plot
        figsize : tuple
            Figure size
        """
        if self.y_pred_proba is None:
            print("Model does not support probability predictions")
            return
        
        if len(np.unique(self.y_test)) != 2:
            print("Precision-Recall curve is for binary classification only")
            return
        
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba[:, 1]
        )
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba[:, 1])
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = RESULTS_DIR / f'{self.model_name}_pr_curve.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {filepath}")
        
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
        self.plot_confusion_matrix(save=save)
        if self.y_pred_proba is not None:
            if len(np.unique(self.y_test)) == 2:
                self.plot_roc_curve(save=save)
                self.plot_precision_recall_curve(save=save)
        self.plot_feature_importance(save=save)
        
        # Save metrics
        if save:
            import json
            filepath = RESULTS_DIR / f'{self.model_name}_metrics.json'
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            print(f"\nMetrics saved to {filepath}")
        
        return self.metrics


