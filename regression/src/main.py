"""
Main script for Regression Problem
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
    print("REGRESSION PROBLEM - MACHINE LEARNING PIPELINE")
    print("="*70)
    
    # ============================================================
    # STEP 1: Problem Definition
    # ============================================================
    print("\n[STEP 1] PROBLEM DEFINITION")
    print("-" * 70)
    print("Problem Type: Regression")
    print("Problem: Flight Delays Prediction")
    print("Goal: Predict flight delay time (in minutes) based on flight and weather features")
    print("Success Criteria: R² > 0.70, RMSE < 30 minutes, MAE < 20 minutes")
    print("\nDataset: US DOT Flight Delays (from Kaggle)")
    
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
            # Prefer flights.csv if available (main dataset)
            flights_file = RAW_DATA_DIR / "flights.csv"
            if flights_file.exists():
                data_file = flights_file
                print(f"✅ Using main dataset: {data_file.name}")
            else:
                # Show available files and let user choose
                print("\nAvailable CSV files:")
                for i, f in enumerate(csv_files, 1):
                    print(f"  {i}. {f.name}")
                print(f"\n⚠️  Note: For Flight Delays prediction, use 'flights.csv'")
                choice = input(f"\nSelect file number (1-{len(csv_files)}) or press Enter for first file: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
                    data_file = csv_files[int(choice) - 1]
                else:
                    data_file = csv_files[0]
                print(f"Using: {data_file.name}")
        else:
            print("No dataset found. Please provide the path to your dataset.")
            print("Place your dataset in: regression/data/raw/")
            print("\nTo download the Flight Delays dataset, run:")
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
    
    # Validate that target column exists
    if target_column not in df_processed.columns:
        print(f"\n❌ Error: Target column '{target_column}' not found in processed data!")
        print(f"\nAvailable columns: {list(df_processed.columns)}")
        print(f"\nData shape: {df_processed.shape}")
        print("\nPossible issues:")
        print("1. You may have selected the wrong CSV file")
        print("2. The target column was removed during preprocessing")
        print("3. The target column name is different (check spelling/casing)")
        print("\nFor Flight Delays dataset, use 'flights.csv' (not 'airlines.csv' or 'airports.csv')")
        print("Common target column names: 'ARRIVAL_DELAY', 'DEPARTURE_DELAY'")
        return
    
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
        method = input("Selection method (mutual_info/f_regression/correlation) [default: mutual_info]: ").strip() or "mutual_info"
        X = feature_engineer.select_features(X, y, method=method, k=k)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target statistics:")
    print(f"  Mean: {y.mean():.4f}")
    print(f"  Std: {y.std():.4f}")
    print(f"  Min: {y.min():.4f}")
    print(f"  Max: {y.max():.4f}")
    
    # ============================================================
    # STEP 5 & 6: Model Training and Hyperparameter Tuning
    # ============================================================
    print("\n[STEP 5 & 6] MODEL TRAINING AND HYPERPARAMETER TUNING")
    print("-" * 70)
    
    trainer = ModelTrainer(X, y, test_size=0.2, val_size=0.2, random_state=42)
    trainer.split_data()
    
    # Train best model (XGBoost - best for regression)
    print("\nTraining best model (XGBoost)...")
    model_type = input("Model type (xgboost/lightgbm/random_forest/gradient_boosting) [default: xgboost]: ").strip().lower() or "xgboost"
    trainer.train_best_model(model_type=model_type, cv=5)
    
    # Hyperparameter tuning
    tune_model = input("\nTune hyperparameters? (y/n) [default: y]: ").strip().lower()
    if tune_model != 'n':
        tuning_method = input("Tuning method (grid/random/optuna) [default: optuna]: ").strip() or "optuna"
        n_iter = int(input("Number of iterations [default: 30]: ").strip() or "30")
        
        trainer.tune_hyperparameters(
            trainer.best_model_name, 
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
        if metric == 'MAPE':
            print(f"  {metric}: {value:.4f}%")
        else:
            print(f"  {metric}: {value:.4f}")
    print(f"\nModel saved to: regression/models/")
    print(f"Results saved to: regression/results/")
    print("\nNext step: Deploy the model using app.py")

if __name__ == "__main__":
    main()

