# Model evaluation script
# Updated import
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall}

# TODO: Review: Review implementation

# Last updated: 2025-12-11
