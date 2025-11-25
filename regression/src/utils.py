"""
Utility functions for the regression problem.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def load_data(file_path, file_type='csv'):
    """
    Load data from various file formats.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the data file
    file_type : str
        Type of file ('csv', 'excel', 'json')
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    file_path = Path(file_path)
    
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'excel':
        return pd.read_excel(file_path)
    elif file_type == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def save_model(model, filename):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to save
    filename : str
        Name of the file to save
    """
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filename):
    """
    Load a saved model from disk.
    
    Parameters:
    -----------
    filename : str
        Name of the model file to load
    
    Returns:
    --------
    sklearn model
        Loaded model
    """
    filepath = MODELS_DIR / filename
    return joblib.load(filepath)


def save_results(results_dict, filename='results.json'):
    """
    Save evaluation results to a JSON file.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing evaluation metrics
    filename : str
        Name of the file to save
    """
    import json
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {filepath}")

