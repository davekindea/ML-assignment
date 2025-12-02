# Quick Start Guide

## Project Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Your Dataset**
   - Place your classification dataset in: `classification/data/raw/`
   - Place your regression dataset in: `regression/data/raw/`

## Running the ML Pipeline

### For Classification Problem

1. **Run the main pipeline:**
   ```bash
   cd classification
   python src/main.py
   ```

2. **Deploy the model:**
   ```bash
   streamlit run app.py
   ```

### For Regression Problem

1. **Run the main pipeline:**
   ```bash
   cd regression
   python src/main.py
   ```

2. **Deploy the model:**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
ML assignment/
├── classification/          # Classification problem
│   ├── data/
│   │   ├── raw/            # Put your dataset here
│   │   ├── processed/      # Cleaned data
│   │   └── splits/         # Train/val/test splits
│   ├── src/                # Source code
│   ├── models/             # Saved models
│   ├── results/            # Evaluation results
│   └── app.py              # Deployment app
│
├── regression/             # Regression problem
│   ├── data/
│   │   ├── raw/            # Put your dataset here
│   │   ├── processed/
│   │   └── splits/
│   ├── src/
│   ├── models/
│   ├── results/
│   └── app.py
│
└── docs/                   # Documentation
```

## Workflow

1. **Data Collection**: Add your dataset to `data/raw/`
2. **Data Preprocessing**: Handled automatically by `data_preprocessing.py`
3. **Feature Engineering**: Handled automatically by `feature_engineering.py`
4. **Model Training**: Handled automatically by `model_training.py`
5. **Model Evaluation**: Handled automatically by `model_evaluation.py`
6. **Deployment**: Use `app.py` for interactive predictions

## Key Features

- ✅ Complete ML pipeline from data loading to deployment
- ✅ Multiple algorithms (10+ models for each problem type)
- ✅ Hyperparameter tuning (Grid Search, Random Search, Optuna)
- ✅ Comprehensive evaluation metrics
- ✅ Beautiful visualizations
- ✅ Streamlit deployment app
- ✅ Well-documented code

## Next Steps

1. Choose your classification and regression problems
2. Download datasets from Kaggle/UCI/Google Dataset Search
3. Place datasets in respective `data/raw/` folders
4. Run the pipeline and follow the interactive prompts
5. Review results in `results/` folders
6. Deploy models using Streamlit apps
7. Document your process using `docs/ML_Process_Documentation.md`

## Tips

- Start with one problem (classification or regression) first
- Make sure your dataset is in CSV format (or update the code for other formats)
- Know the name of your target column before running
- Review the generated visualizations in the `results/` folder
- Customize the deployment apps (`app.py`) with your actual feature names

## Need Help?

- Check the documentation in `docs/ML_Process_Documentation.md`
- Review the code comments in each module
- The main scripts (`main.py`) guide you through each step interactively



# TODO: Review implementation
