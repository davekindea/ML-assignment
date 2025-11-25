"""
Script to download the Flight Delays dataset from Kaggle.
"""
import kagglehub
from pathlib import Path
import shutil
import time
import requests

def download_flight_delays_dataset():
    """
    Download the Flight Delays dataset from Kaggle.
    """
    print("="*70)
    print("DOWNLOADING FLIGHT DELAYS DATASET FROM KAGGLE")
    print("="*70)
    
    try:
        # Patch kagglehub's timeout settings before importing
        import kagglehub.clients as kaggle_clients
        
        # Increase timeout values
        kaggle_clients.DEFAULT_READ_TIMEOUT = 300  # 5 minutes (was 5 seconds)
        kaggle_clients.DEFAULT_CONNECT_TIMEOUT = 60  # 1 minute
        
        # Download latest version with increased timeout
        print("\nDownloading dataset: usdot/flight-delays")
        print("⚠️  This may take several minutes depending on your internet connection...")
        print("⚠️  Large datasets can take time to download. Please be patient.")
        print("")
        
        # Try to download with retries
        max_retries = 3
        retry_count = 0
        path = None
        
        while retry_count < max_retries:
            try:
                print(f"Attempt {retry_count + 1}/{max_retries}...")
                path = kagglehub.dataset_download("usdot/flight-delays")
                print("✅ Download successful!")
                break  # Success, exit retry loop
                
            except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 15  # Exponential backoff: 15s, 30s
                    print(f"\n⚠️  Timeout/Connection error: {str(e)}")
                    print(f"⏳ Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ Failed after {max_retries} attempts")
                    raise
            except Exception as e:
                error_str = str(e).lower()
                if "timeout" in error_str or "timed out" in error_str or "connection" in error_str:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_count * 15
                        print(f"\n⚠️  Connection issue: {str(e)}")
                        print(f"⏳ Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    # Non-timeout error, don't retry
                    raise
        
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
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. The dataset might be very large - try again later")
        print("3. Make sure you have:")
        print("   - Installed kagglehub: pip install kagglehub")
        print("   - Set up Kaggle API credentials (if required)")
        print("     * Go to https://www.kaggle.com/settings")
        print("     * Create API token and download kaggle.json")
        print("     * Place it in ~/.kaggle/ (or C:/Users/<username>/.kaggle/ on Windows)")
        print("\nAlternative: Download manually from:")
        print("   https://www.kaggle.com/datasets/usdot/flight-delays")
        print("   Then place the CSV files in: regression/data/raw/")
        raise

if __name__ == "__main__":
    download_flight_delays_dataset()


