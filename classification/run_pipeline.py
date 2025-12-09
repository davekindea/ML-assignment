"""
Convenience script to run the classification pipeline with Heart Disease dataset.
This script automates the dataset download and pipeline execution.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """
    Main function to download dataset and run pipeline.
    """
    print("="*70)
    print("HEART DISEASE CLASSIFICATION - AUTOMATED PIPELINE")
    print("="*70)
    
    # Check if dataset exists
    raw_data_dir = Path(__file__).parent / "data" / "raw"
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    if not csv_files:
        print("\n[STEP 1] Downloading dataset...")
        print("-" * 70)
        try:
            # Run download script
            result = subprocess.run(
                [sys.executable, "download_dataset.py"],
                cwd=Path(__file__).parent,
                check=True
            )
            print("✅ Dataset downloaded successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading dataset: {e}")
            print("\nPlease run manually: python download_dataset.py")
            print("Or download from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction")
            return
        except FileNotFoundError:
            print("❌ download_dataset.py not found!")
            return
    else:
        print("\n[STEP 1] Dataset already exists")
        print(f"Found {len(csv_files)} CSV file(s) in data/raw/")
        for f in csv_files:
            print(f"  - {f.name}")
    
    # Run main pipeline
    print("\n[STEP 2] Running ML pipeline...")
    print("-" * 70)
    print("\nYou will be prompted for:")
    print("  - Dataset file (press Enter to use the first CSV found)")
    print("  - Target column name (e.g., 'HeartDisease', 'target')")
    print("  - Preprocessing options")
    print("\nStarting pipeline...\n")
    
    try:
        subprocess.run(
            [sys.executable, "src/main.py"],
            cwd=Path(__file__).parent,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running pipeline: {e}")
        return
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review results in: classification/results/")
    print("2. Deploy model: streamlit run app.py")

if __name__ == "__main__":
    main()



# Updated: 2025-12-11
