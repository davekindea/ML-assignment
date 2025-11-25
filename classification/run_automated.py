"""
Automated script to run the classification pipeline step by step.
This script runs without user interaction for demonstration.
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from utils import RAW_DATA_DIR

def main():
    """
    Run the complete ML pipeline step by step.
    """
    print("="*70)
    print("CLASSIFICATION PROBLEM - AUTOMATED PIPELINE")
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
    print("Dataset: Heart Disease Prediction (from Kaggle)")
    
    # ============================================================
    # STEP 2: Data Collection
    # ============================================================
    print("\n[STEP 2] DATA COLLECTION")
    print("-" * 70)
    
    # Find dataset
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ No dataset found in data/raw/")
        print("Please run: python download_dataset.py")
        return
    
    data_file = csv_files[0]
    print(f"✅ Found dataset: {data_file.name}")
    
    target_column = "HeartDisease"
    print(f"✅ Target column: {target_column}")
    
    # ============================================================
    # STEP 3: Data Preprocessing
    # ============================================================
    print("\n[STEP 3] DATA PREPROCESSING")
    print("-" * 70)
    
    print("Initializing preprocessor...")
    preprocessor = DataPreprocessor(data_path=data_file, target_column=target_column)
    
    print("\n3.1 Loading raw data...")
    preprocessor.load_raw_data()
    
    print("\n3.2 Exploring data...")
    preprocessor.explore_data()
    
    print("\n3.3 Handling missing values (auto strategy)...")
    preprocessor.handle_missing_values(strategy='auto')
    
    print("\n3.4 Removing duplicates...")
    preprocessor.remove_duplicates()
    
    print("\n3.5 Handling outliers (IQR method, capping)...")
    preprocessor.handle_outliers(method='iqr')
    
    print("\n3.6 Saving processed data...")
    preprocessor.save_processed_data()
    df_processed = preprocessor.get_processed_data()
    
    print(f"✅ Processed data shape: {df_processed.shape}")
    
    # ============================================================
    # STEP 4: Feature Engineering
    # ============================================================
    print("\n[STEP 4] FEATURE ENGINEERING")
    print("-" * 70)
    
    feature_engineer = FeatureEngineer(target_column=target_column)
    
    # Separate features and target
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    
    print(f"\n4.1 Original features: {X.shape[1]}")
    print(f"4.2 Target distribution:")
    print(y.value_counts())
    
    print("\n4.3 Encoding categorical variables (auto method)...")
    X = feature_engineer.encode_categorical(X, method='auto')
    
    print("\n4.4 Scaling features (standard scaling)...")
    X = feature_engineer.scale_features(X, method='standard', fit=True)
    
    # Save preprocessing artifacts for deployment
    print("\n4.5 Saving preprocessing artifacts...")
    feature_engineer.save_scaler()
    feature_engineer.save_encoders()
    
    print(f"✅ Final feature matrix shape: {X.shape}")
    
    # ============================================================
    # STEP 5 & 6: Model Training and Hyperparameter Tuning
    # ============================================================
    print("\n[STEP 5 & 6] MODEL TRAINING AND HYPERPARAMETER TUNING")
    print("-" * 70)
    
    print("\n5.1 Splitting data (Train: 60%, Val: 20%, Test: 20%)...")
    trainer = ModelTrainer(X, y, test_size=0.2, val_size=0.2, random_state=42)
    trainer.split_data()
    
    print("\n5.2 Training baseline models (this may take a few minutes)...")
    baseline_results = trainer.train_baseline_models(cv=5)
    
    print("\n5.3 Hyperparameter tuning for best model (Optuna, 20 trials)...")
    best_baseline_name = baseline_results.iloc[0]['Model']
    print(f"   Tuning: {best_baseline_name}")
    trainer.tune_hyperparameters(
        best_baseline_name, 
        method='optuna', 
        n_iter=20  # Reduced for faster execution
    )
    
    print("\n5.4 Saving best model...")
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
    
    print("\n7.1 Calculating metrics...")
    metrics = evaluator.calculate_metrics()
    
    print("\n7.2 Generating visualizations...")
    evaluator.generate_report(save=True)
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n✅ Best Model: {trainer.best_model_name}")
    print("\n📊 Final Test Set Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print(f"\n📁 Model saved to: classification/models/")
    print(f"📁 Results saved to: classification/results/")
    print(f"\n🚀 Next step: Deploy the model using 'streamlit run app.py'")
    
    return trainer, evaluator, metrics

if __name__ == "__main__":
    try:
        trainer, evaluator, metrics = main()
        print("\n✅ All steps completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Make sure all dependencies are installed:")
        print("   pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm optuna")

