# Heart Disease Classification - Setup Instructions

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r ../requirements.txt
```

### Step 2: Download the Dataset

**Option A: Using the download script (Recommended)**
```bash
cd classification
python download_dataset.py
```

**Option B: Manual download using Python**
```python
import kagglehub
# Try one of these:
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
# OR
path = kagglehub.dataset_download("kamilpytlak/heart-disease-prediction")
print("Path to dataset files:", path)
```

**Option C: Download from Kaggle website**
- Go to: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- Download the dataset
- Extract CSV files to `classification/data/raw/`

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
   - Common options: `HeartDisease`, `target`, `HeartDiseaseFlag`
   - Check the dataset columns first to see what's available

### Step 4: Deploy the Model
```bash
streamlit run app.py
```

## Expected Dataset Information

The Heart Disease dataset typically contains:

**Main File**: `heart.csv` or `heart_disease.csv` (or similar)

**Key Columns** (may vary by dataset):
- `Age` - Patient age
- `Sex` - Gender (M/F, 0/1, or Male/Female)
- `ChestPainType` - Type of chest pain (categorical)
- `RestingBP` - Resting blood pressure
- `Cholesterol` - Serum cholesterol level
- `FastingBS` - Fasting blood sugar (0/1)
- `RestingECG` - Resting ECG results
- `MaxHR` - Maximum heart rate achieved
- `ExerciseAngina` - Exercise-induced angina (Y/N or 0/1)
- `Oldpeak` - ST depression
- `ST_Slope` - Slope of ST segment
- `HeartDisease` or `target` ← **Target variable** (0 = No disease, 1 = Has disease)

**Target Variable Options**:
- `HeartDisease` - Most common (recommended)
- `target` - Alternative name
- `HeartDiseaseFlag` - Some datasets use this

## Tips

1. **Check Column Names First**:
   ```python
   import pandas as pd
   df = pd.read_csv('data/raw/heart.csv')
   print(df.columns.tolist())
   print(df.head())
   print(df['HeartDisease'].value_counts())  # Check target distribution
   ```

2. **Class Balance**: Check if classes are balanced:
   - If imbalanced, the pipeline handles it, but you may want to use class weights

3. **Categorical Encoding**: The pipeline automatically handles:
   - Label encoding for binary categorical
   - One-hot encoding for multi-class categorical

4. **Feature Scaling**: Recommended for algorithms like SVM, KNN, and Neural Networks

5. **Model Selection**: The pipeline trains 10+ models including:
   - Logistic Regression (interpretable)
   - Random Forest (good performance)
   - XGBoost (often best performance)
   - And more...

## Troubleshooting

### Kaggle API Issues
If you get authentication errors:
1. Go to https://www.kaggle.com/settings
2. Create API token (download `kaggle.json`)
3. Place it in:
   - Windows: `C:/Users/<username>/.kaggle/kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

### Column Name Issues
If you get "column not found" errors:
- Check the actual column names in your dataset
- Column names might be lowercase (e.g., `heartdisease` vs `HeartDisease`)
- Use the exact column name as it appears in the CSV

### Dataset Not Found
If the download script fails:
- Try downloading manually from Kaggle
- Use alternative dataset names (see download script)
- Check Kaggle for the most current dataset

## Alternative Datasets

If Heart Disease doesn't work, try these popular classification datasets:

1. **Titanic Survival**:
   ```python
   path = kagglehub.dataset_download("c/titanic")
   # Target: 'Survived'
   ```

2. **Wine Quality**:
   ```python
   path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
   # Target: 'quality' (binarize: >5 = good, <=5 = bad)
   ```

3. **Customer Churn**:
   ```python
   path = kagglehub.dataset_download("blastchar/telco-customer-churn")
   # Target: 'Churn'
   ```

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
cd classification
python download_dataset.py

# 3. Explore the dataset (optional - to find target column name)
python -c "import pandas as pd; df = pd.read_csv('data/raw/heart.csv'); print(df.columns.tolist()); print(df['HeartDisease'].value_counts())"

# 4. Run pipeline
python src/main.py
# When prompted:
#   - Dataset: press Enter (or specify path)
#   - Target: HeartDisease (or your chosen target)

# 5. Deploy
streamlit run app.py
```

## Expected Results

After running the pipeline, you should see:
- **Baseline model comparison** - Performance of 10+ models
- **Best model selection** - Automatically chosen based on validation performance
- **Hyperparameter tuning** - Optimized parameters for best model
- **Final evaluation** - Test set metrics and visualizations

**Good Performance Indicators**:
- Accuracy: > 0.85
- Precision: > 0.80
- Recall: > 0.80
- F1-Score: > 0.80
- ROC-AUC: > 0.85

Good luck with your assignment! 🚀

