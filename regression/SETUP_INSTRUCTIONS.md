# Flight Delays Regression - Setup Instructions

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r ../requirements.txt
```

### Step 2: Download the Dataset

**Option A: Using the download script (Recommended)**
```bash
cd regression
python download_dataset.py
```

**Option B: Manual download using Python**
```python
import kagglehub
path = kagglehub.dataset_download("usdot/flight-delays")
print("Path to dataset files:", path)
```

**Option C: Download from Kaggle website**
- Go to: https://www.kaggle.com/datasets/usdot/flight-delays
- Download the dataset
- Extract CSV files to `regression/data/raw/`

### Step 3: Run the ML Pipeline

**Option A: Automated (downloads dataset if needed)**
```bash
python run_pipeline.py
```

**Option B: Manual step-by-step**
```bash
python src/main.py
```

When prompted:
1. **Dataset path**: Press Enter to use the first CSV found, or specify the path
2. **Target column**: Enter the name of the target variable
   - Common options: `ARRIVAL_DELAY`, `DEPARTURE_DELAY`, or `DELAY`
   - Check the dataset columns first to see what's available

### Step 4: Deploy the Model
```bash
streamlit run app.py
```

## Expected Dataset Information

The Flight Delays dataset typically contains:

**Main File**: `flights.csv` (or similar)

**Key Columns** (may vary):
- `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`
- `AIRLINE`, `FLIGHT_NUMBER`
- `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`
- `SCHEDULED_DEPARTURE`, `DEPARTURE_TIME`
- `SCHEDULED_ARRIVAL`, `ARRIVAL_TIME`
- `DEPARTURE_DELAY` ← Common target variable
- `ARRIVAL_DELAY` ← Common target variable
- `DISTANCE`
- `CANCELLED`
- Weather-related columns (if included)

**Target Variable Options**:
- `ARRIVAL_DELAY` - Predict arrival delay (recommended)
- `DEPARTURE_DELAY` - Predict departure delay
- `DELAY` - If a combined delay column exists

## Tips

1. **Large Dataset**: If the dataset is very large (>1M rows), consider:
   - Sampling the data for initial experiments
   - Using a subset of features
   - Processing in chunks

2. **Missing Values**: The dataset may have many missing values. The pipeline handles this automatically, but you can choose strategies:
   - `auto` - Recommended (median for numerical, mode for categorical)
   - `drop` - Remove rows with missing values
   - `mean`/`median`/`mode` - Fill with specific statistic

3. **Feature Engineering**: The pipeline automatically:
   - Encodes categorical variables
   - Scales numerical features
   - Can create interaction features (optional)
   - Can perform feature selection (optional)

4. **Model Selection**: The pipeline trains 10+ models and selects the best one automatically.

## Troubleshooting

### Kaggle API Issues
If you get authentication errors:
1. Go to https://www.kaggle.com/settings
2. Create API token (download `kaggle.json`)
3. Place it in:
   - Windows: `C:/Users/<username>/.kaggle/kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

### Memory Issues
If you run out of memory:
- Sample the dataset (e.g., use first 100k rows)
- Reduce number of features
- Use simpler models first

### Column Name Issues
If you get "column not found" errors:
- Check the actual column names in your dataset
- The column names might be different (e.g., `arrival_delay` vs `ARRIVAL_DELAY`)
- Use the exact column name as it appears in the CSV

## Next Steps After Training

1. **Review Results**: Check `regression/results/` for:
   - Evaluation metrics (JSON file)
   - Visualizations (PNG files)
   - Model performance plots

2. **Documentation**: Fill in `docs/ML_Process_Documentation.md` with your findings

3. **Presentation**: Create PowerPoint with:
   - Problem definition
   - Data exploration visualizations
   - Model comparison results
   - Final model performance
   - Key insights

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
cd regression
python download_dataset.py

# 3. Explore the dataset (optional - to find target column name)
python -c "import pandas as pd; df = pd.read_csv('data/raw/flights.csv'); print(df.columns.tolist()); print(df.head())"

# 4. Run pipeline
python src/main.py
# When prompted:
#   - Dataset: press Enter (or specify path)
#   - Target: ARRIVAL_DELAY (or your chosen target)

# 5. Deploy
streamlit run app.py
```

Good luck with your assignment! 🚀


