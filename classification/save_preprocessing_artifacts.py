"""
Helper script to save preprocessing artifacts (scaler, encoders) retroactively.
This script recreates the feature engineering pipeline from saved training data.
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR

def main():
    """
    Recreate and save preprocessing artifacts from training data.
    """
    print("="*70)
    print("SAVING PREPROCESSING ARTIFACTS")
    print("="*70)
    
    # Check if we have saved training splits
    X_train_file = SPLITS_DIR / "X_train.csv"
    if X_train_file.exists():
        print("\n✅ Found saved training data splits")
        print("Recreating preprocessing pipeline from training data...")
        
        # Load training data
        X_train = pd.read_csv(X_train_file)
        print(f"Loaded training data: {X_train.shape}")
        
        # Check if we have processed data to get original feature names
        processed_data_file = PROCESSED_DATA_DIR / "processed_data.csv"
        if processed_data_file.exists():
            df_processed = pd.read_csv(processed_data_file)
            print(f"Found processed data: {df_processed.shape}")
            
            # Determine target column (check common names)
            target_col = None
            for col in ['target', 'HeartDisease', 'Target']:
                if col in df_processed.columns:
                    target_col = col
                    break
            
            if target_col:
                print(f"Target column: {target_col}")
                
                # Recreate feature engineering pipeline
                feature_engineer = FeatureEngineer(target_column=target_col)
                
                # Get original features (before encoding/scaling)
                X_original = df_processed.drop(columns=[target_col])
                
                # Apply encoding (this will fit encoders)
                print("\nFitting encoders...")
                X_encoded = feature_engineer.encode_categorical(X_original.copy(), method='auto')
                
                # Apply scaling (this will fit scaler)
                print("Fitting scaler...")
                X_scaled = feature_engineer.scale_features(X_encoded.copy(), method='standard', fit=True)
                
                # Save artifacts
                print("\nSaving preprocessing artifacts...")
                feature_engineer.save_scaler()
                feature_engineer.save_encoders()
                
                print("\n✅ Preprocessing artifacts saved successfully!")
                print("   - scaler.pkl")
                print("   - encoders.pkl")
                return True
            else:
                print("❌ Could not determine target column")
        else:
            print("❌ Processed data not found. Need to recreate from raw data.")
            return recreate_from_raw()
    else:
        print("❌ Training data splits not found.")
        print("Attempting to recreate from raw data...")
        return recreate_from_raw()

def recreate_from_raw():
    """
    Recreate preprocessing pipeline from raw data.
    """
    # Find raw data
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ No raw data found. Please run the training pipeline first.")
        return False
    
    data_file = csv_files[0]
    print(f"\nUsing raw data: {data_file.name}")
    
    # Determine target column
    df = pd.read_csv(data_file)
    target_col = None
    for col in ['target', 'HeartDisease', 'Target']:
        if col in df.columns:
            target_col = col
            break
    
    if not target_col:
        print("❌ Could not determine target column. Please specify manually.")
        return False
    
    print(f"Target column: {target_col}")
    
    # Run preprocessing
    print("\nRunning data preprocessing...")
    preprocessor = DataPreprocessor(data_path=data_file, target_column=target_col)
    preprocessor.load_raw_data()
    preprocessor.handle_missing_values(strategy='auto')
    preprocessor.remove_duplicates()
    preprocessor.handle_outliers(method='iqr')
    df_processed = preprocessor.get_processed_data()
    
    # Run feature engineering
    print("\nRunning feature engineering...")
    feature_engineer = FeatureEngineer(target_column=target_col)
    X = df_processed.drop(columns=[target_col])
    
    # Apply encoding
    print("Fitting encoders...")
    X = feature_engineer.encode_categorical(X, method='auto')
    
    # Apply scaling
    print("Fitting scaler...")
    X = feature_engineer.scale_features(X, method='standard', fit=True)
    
    # Save artifacts
    print("\nSaving preprocessing artifacts...")
    feature_engineer.save_scaler()
    feature_engineer.save_encoders()
    
    print("\n✅ Preprocessing artifacts saved successfully!")
    print("   - scaler.pkl")
    print("   - encoders.pkl")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ You can now use the Streamlit app!")
            print("   Run: streamlit run app.py")
        else:
            print("\n❌ Failed to save preprocessing artifacts.")
            print("   Please run the training pipeline first:")
            print("   python run_automated.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

