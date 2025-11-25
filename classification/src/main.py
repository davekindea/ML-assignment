"""
Main script for Classification Problem
Runs the complete ML pipeline from data loading to model evaluation.
"""
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from utils import load_data, RAW_DATA_DIR

def main():
    """
    Main function to run the complete ML pipeline.
    """
    print("="*70)
    print("CLASSIFICATION PROBLEM - MACHINE LEARNING PIPELINE")
    print("="*70)
    
    # ============================================================
    # STEP 1: Problem Definition
    # ============================================================
    print("\n[STEP 1] PROBLEM DEFINITION")
    print("-" * 70)
    print("Problem Type: Classification")
    print("Problem: Heart Disease Prediction")
    print("Goal: Predict whether a patient has heart disease (binary classification)")
    print("Success Criteria: Accuracy > 0.85, Precision > 0.80, Recall > 0.80, F1-Score > 0.80")
    print("\nDataset: Heart Disease Prediction (from Kaggle)")
    
    # ============================================================
    # STEP 2: Data Collection
    # ============================================================
    print("\n[STEP 2] DATA COLLECTION")
    print("-" * 70)
    
    # TODO: Update these paths with your actual dataset
    data_file = input("Enter path to your dataset (or press Enter to use default): ").strip()
    if not data_file:
        # Look for CSV files in raw data directory
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_files:
            data_file = csv_files[0]
            print(f"Using found dataset: {data_file}")
        else:
            print("No dataset found. Please provide the path to your dataset.")
            print("Place your dataset in: classification/data/raw/")
            print("\nTo download the Heart Disease dataset, run:")
            print("  python download_dataset.py")
            return
    
    target_column = input("Enter the name of the target column: ").strip()
    if not target_column:
        print("Target column is required!")
        return
    
    # ============================================================
    # STEP 3: Data Preprocessing
    # ============================================================
    print("\n[STEP 3] DATA PREPROCESSING")
    print("-" * 70)
    
    preprocessor = DataPreprocessor(data_path=data_file, target_column=target_column)
    preprocessor.load_raw_data()
    preprocessor.explore_data()
    
    # Handle missing values
    missing_strategy = input("\nMissing value strategy (auto/drop/mean/median/mode) [default: auto]: ").strip() or "auto"
    preprocessor.handle_missing_values(strategy=missing_strategy)
    
    # Remove duplicates
    preprocessor.remove_duplicates()
    
    # Handle outliers (optional)
    handle_outliers = input("\nHandle outliers? (y/n) [default: n]: ").strip().lower()
    if handle_outliers == 'y':
        outlier_method = input("Outlier method (iqr/zscore) [default: iqr]: ").strip() or "iqr"
        preprocessor.handle_outliers(method=outlier_method)
    
    # Save processed data
    preprocessor.save_processed_data()
    df_processed = preprocessor.get_processed_data()
    
    # ============================================================
    # STEP 4: Feature Engineering
    # ============================================================
    print("\n[STEP 4] FEATURE ENGINEERING")
    print("-" * 70)
    
    feature_engineer = FeatureEngineer(target_column=target_column)
    
    # Separate features and target
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    
    # Encode categorical variables
    encode_method = input("\nCategorical encoding method (auto/label/onehot) [default: auto]: ").strip() or "auto"
    X = feature_engineer.encode_categorical(X, method=encode_method)
    
    # Scale features
    scale_method = input("Scaling method (standard/minmax/robust/none) [default: standard]: ").strip() or "standard"
    if scale_method != 'none':
        X = feature_engineer.scale_features(X, method=scale_method, fit=True)
    
    # Optional: Create interaction features
    create_interactions = input("\nCreate interaction features? (y/n) [default: n]: ").strip().lower()
    if create_interactions == 'y':
        X = feature_engineer.create_interaction_features(X)
    
    # Optional: Feature selection
    select_features = input("Perform feature selection? (y/n) [default: n]: ").strip().lower()
    if select_features == 'y':
        k = int(input("Number of features to select [default: 20]: ").strip() or "20")
        method = input("Selection method (mutual_info/chi2/f_classif) [default: mutual_info]: ").strip() or "mutual_info"
        X = feature_engineer.select_features(X, y, method=method, k=k)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # ============================================================
    # STEP 5 & 6: Model Training and Hyperparameter Tuning
    # ============================================================
    print("\n[STEP 5 & 6] MODEL TRAINING AND HYPERPARAMETER TUNING")
    print("-" * 70)
    
    trainer = ModelTrainer(X, y, test_size=0.2, val_size=0.2, random_state=42)
    trainer.split_data()
    
    # Train baseline models
    print("\nTraining baseline models...")
    baseline_results = trainer.train_baseline_models(cv=5)
    
    # Hyperparameter tuning
    tune_model = input("\nTune hyperparameters for best model? (y/n) [default: y]: ").strip().lower()
    if tune_model != 'n':
        tuning_method = input("Tuning method (grid/random/optuna) [default: optuna]: ").strip() or "optuna"
        n_iter = int(input("Number of iterations [default: 50]: ").strip() or "50")
        
        best_baseline_name = baseline_results.iloc[0]['Model']
        trainer.tune_hyperparameters(
            best_baseline_name, 
            method=tuning_method, 
            n_iter=n_iter
        )
    
    # Save best model
    trainer.save_best_model()
    
    # ============================================================
    # STEP 7: Model Evaluation
    # ============================================================
    print("\n[STEP 7] MODEL EVALUATION")
    print("-" * 70)
    
    evaluator = ModelEvaluator(
        trainer.best_model, 
        trainer.X_test, 
        trainer.y_test,
        model_name=trainer.best_model_name
    )
    
    metrics = evaluator.calculate_metrics()
    evaluator.generate_report(save=True)
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nBest Model: {trainer.best_model_name}")
    print("\nFinal Test Set Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nModel saved to: classification/models/")
    print(f"Results saved to: classification/results/")
    print("\nNext step: Deploy the model using app.py")

if __name__ == "__main__":
    main()


