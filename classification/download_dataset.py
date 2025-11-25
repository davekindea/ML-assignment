"""
Script to download the Heart Disease Prediction dataset from Kaggle.
"""
import kagglehub
from pathlib import Path
import shutil

def download_heart_disease_dataset():
    """
    Download the Heart Disease Prediction dataset from Kaggle.
    Alternative datasets you can use:
    - "kamilpytlak/heart-disease-prediction" 
    - "johnsmith88/heart-disease-dataset"
    - "fedesoriano/heart-failure-prediction"
    """
    print("="*70)
    print("DOWNLOADING HEART DISEASE PREDICTION DATASET FROM KAGGLE")
    print("="*70)
    
    # Try multiple dataset options
    dataset_options = [
        "fedesoriano/heart-failure-prediction",  # Most popular
        "kamilpytlak/heart-disease-prediction",
        "johnsmith88/heart-disease-dataset"
    ]
    
    raw_data_dir = Path(__file__).parent / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name in dataset_options:
        try:
            print(f"\nTrying dataset: {dataset_name}")
            path = kagglehub.dataset_download(dataset_name)
            
            print(f"Dataset downloaded to: {path}")
            print(f"Copying files to: {raw_data_dir}")
            
            source_path = Path(path)
            
            if source_path.is_dir():
                # Copy all CSV files from the dataset
                csv_files = list(source_path.glob("*.csv"))
                if csv_files:
                    for csv_file in csv_files:
                        dest_file = raw_data_dir / csv_file.name
                        shutil.copy2(csv_file, dest_file)
                        print(f"  ✅ Copied: {csv_file.name}")
                    
                    print(f"\n✅ Dataset files are now in: {raw_data_dir}")
                    print("\nFiles available:")
                    for file in raw_data_dir.glob("*.csv"):
                        print(f"  - {file.name}")
                    
                    return raw_data_dir
                else:
                    # Check subdirectories
                    for item in source_path.iterdir():
                        if item.is_file() and item.suffix == '.csv':
                            dest_file = raw_data_dir / item.name
                            shutil.copy2(item, dest_file)
                            print(f"  ✅ Copied: {item.name}")
                            return raw_data_dir
                        elif item.is_dir():
                            for subitem in item.glob("*.csv"):
                                dest_file = raw_data_dir / subitem.name
                                shutil.copy2(subitem, dest_file)
                                print(f"  ✅ Copied: {subitem.name}")
                                return raw_data_dir
        except Exception as e:
            print(f"  ⚠️  Could not download {dataset_name}: {str(e)}")
            continue
    
    # If all fail, provide manual instructions
    print("\n❌ Could not automatically download dataset.")
    print("\nPlease download manually from one of these Kaggle datasets:")
    print("1. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction")
    print("2. https://www.kaggle.com/datasets/kamilpytlak/heart-disease-prediction")
    print("3. https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    print(f"\nThen place the CSV file(s) in: {raw_data_dir}")
    raise Exception("Dataset download failed. Please download manually.")

if __name__ == "__main__":
    download_heart_disease_dataset()

