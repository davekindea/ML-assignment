"""
Script to download the Flight Delays dataset from Kaggle.
"""
import kagglehub
from pathlib import Path
import shutil

def download_flight_delays_dataset():
    """
    Download the Flight Delays dataset from Kaggle.
    """
    print("="*70)
    print("DOWNLOADING FLIGHT DELAYS DATASET FROM KAGGLE")
    print("="*70)
    
    try:
        # Download latest version
        print("\nDownloading dataset: usdot/flight-delays")
        path = kagglehub.dataset_download("usdot/flight-delays")
        
        print(f"\nDataset downloaded to: {path}")
        
        # Get the raw data directory
        raw_data_dir = Path(__file__).parent / "data" / "raw"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to raw data directory
        print(f"\nCopying files to: {raw_data_dir}")
        source_path = Path(path)
        
        if source_path.is_dir():
            # Copy all CSV files from the dataset
            csv_files = list(source_path.glob("*.csv"))
            if csv_files:
                for csv_file in csv_files:
                    dest_file = raw_data_dir / csv_file.name
                    shutil.copy2(csv_file, dest_file)
                    print(f"  Copied: {csv_file.name}")
            else:
                # If no CSV files in root, check subdirectories
                for item in source_path.iterdir():
                    if item.is_file() and item.suffix == '.csv':
                        dest_file = raw_data_dir / item.name
                        shutil.copy2(item, dest_file)
                        print(f"  Copied: {item.name}")
                    elif item.is_dir():
                        # Check subdirectories
                        for subitem in item.glob("*.csv"):
                            dest_file = raw_data_dir / subitem.name
                            shutil.copy2(subitem, dest_file)
                            print(f"  Copied: {subitem.name}")
        
        print(f"\n✅ Dataset files are now in: {raw_data_dir}")
        print("\nFiles available:")
        for file in raw_data_dir.glob("*.csv"):
            print(f"  - {file.name}")
        
        return raw_data_dir
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed kagglehub: pip install kagglehub")
        print("2. Set up Kaggle API credentials (if required)")
        print("   - Go to https://www.kaggle.com/settings")
        print("   - Create API token and download kaggle.json")
        print("   - Place it in ~/.kaggle/ (or C:/Users/<username>/.kaggle/ on Windows)")
        raise

if __name__ == "__main__":
    download_flight_delays_dataset()


