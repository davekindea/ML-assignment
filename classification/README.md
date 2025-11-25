# Classification Problem: Heart Disease Prediction

## Problem Definition

**Business Problem**: Predict whether a patient has heart disease based on medical attributes to assist in early diagnosis and treatment planning.

**Goal**: Classify patients into two categories:
- **Class 0**: No heart disease
- **Class 1**: Has heart disease

**Type**: Binary Classification

**Success Criteria**:
- Accuracy: > 0.85
- Precision: > 0.80
- Recall: > 0.80
- F1-Score: > 0.80
- ROC-AUC: > 0.85

## Dataset

**Dataset**: Heart Disease Prediction / Heart Failure Prediction
- **Source**: Kaggle
- **Popular Options**:
  1. `fedesoriano/heart-failure-prediction` (Recommended)
  2. `kamilpytlak/heart-disease-prediction`
  3. `johnsmith88/heart-disease-dataset`

### Downloading the Dataset

Run the download script:
```bash
python download_dataset.py
```

Or manually download from Kaggle and place CSV files in `data/raw/` directory.

## Running the Pipeline

1. **Download the dataset** (if not already done):
   ```bash
   python download_dataset.py
   ```

2. **Run the main pipeline**:
   ```bash
   python src/main.py
   ```

3. **Deploy the model**:
   ```bash
   streamlit run app.py
   ```

## Expected Dataset Structure

The Heart Disease dataset typically contains features like:

**Key Features** (may vary by dataset):
- `Age` - Patient age
- `Sex` - Gender (M/F or 0/1)
- `ChestPainType` - Type of chest pain
- `RestingBP` - Resting blood pressure
- `Cholesterol` - Serum cholesterol
- `FastingBS` - Fasting blood sugar
- `RestingECG` - Resting electrocardiogram results
- `MaxHR` - Maximum heart rate achieved
- `ExerciseAngina` - Exercise-induced angina (Y/N)
- `Oldpeak` - ST depression induced by exercise
- `ST_Slope` - Slope of peak exercise ST segment
- `HeartDisease` or `target` - Target variable (0/1)

**Target Variable**: 
- `HeartDisease` (0 = No, 1 = Yes)
- OR `target` (depending on dataset)

## Alternative Classification Datasets

If you prefer a different classification problem, here are other popular options:

### 1. **Titanic - Survival Prediction**
   - Dataset: `c/titanic`
   - Target: Survived (0/1)
   - Features: Age, Sex, Class, Fare, etc.

### 2. **Credit Card Fraud Detection**
   - Dataset: `mlg-ulb/creditcardfraud`
   - Target: Class (0 = Normal, 1 = Fraud)
   - Features: Transaction details

### 3. **Customer Churn Prediction**
   - Dataset: `blastchar/telco-customer-churn`
   - Target: Churn (Yes/No)
   - Features: Customer demographics and service usage

### 4. **Wine Quality Classification**
   - Dataset: `uciml/red-wine-quality-cortez-et-al-2009`
   - Target: Quality (0-10, can be binarized)
   - Features: Chemical properties

### 5. **Spam Email Detection**
   - Dataset: Various spam detection datasets
   - Target: Spam (0/1)
   - Features: Email content features

## Notes

- The dataset is typically medium-sized (100-1000 samples), making it perfect for learning
- Some features may need encoding (categorical variables)
- Handle missing values appropriately
- Consider feature scaling for better model performance
- The target variable is binary, making it ideal for binary classification algorithms

## Workflow Tips

1. **Explore the data first**: Check column names and target variable name
2. **Handle categorical features**: The pipeline will encode them automatically
3. **Check for class imbalance**: If present, consider using class weights
4. **Feature importance**: Tree-based models will show which features matter most
5. **Interpretability**: Logistic Regression and Decision Trees are more interpretable

