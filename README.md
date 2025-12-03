# Machine Learning Assignment - The ML Process

This project implements the complete Machine Learning process for:
1. **Classification Problem** - [To be selected]
2. **Regression Problem** - [To be selected]

## Project Structure

```
.
├── README.md
├── requirements.txt
├── classification/
│   ├── data/
│   │   ├── raw/           # Original datasets
│   │   ├── processed/     # Cleaned and processed data
│   │   └── splits/        # Train/validation/test splits
│   ├── notebooks/
│   │   └── eda.ipynb      # Exploratory Data Analysis
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── utils.py
│   ├── models/            # Saved models
│   ├── results/           # Evaluation results and visualizations
│   └── app.py             # Deployment application (Flask/FastAPI/Streamlit)
├── regression/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── splits/
│   ├── notebooks/
│   │   └── eda.ipynb
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── utils.py
│   ├── models/
│   ├── results/
│   └── app.py
└── docs/
    └── ML_Process_Documentation.md
```

## ML Process Steps

1. **Problem Definition** - Define business problem, goal, and success criteria
2. **Data Collection** - Gather and load datasets
3. **Data Exploration and Preparation** - EDA, cleaning, feature engineering
4. **Algorithm Selection** - Choose appropriate models
5. **Model Development and Training** - Build and train models
6. **Model Evaluation and Hyperparameter Tuning** - Optimize performance
7. **Model Testing** - Final evaluation on test set
8. **Model Deployment** - Deploy using Flask/FastAPI/Streamlit
9. **Monitoring and Maintenance** - Track performance (optional for assignment)
10. **Documentation** - Document entire process

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Classification Problem
```bash
cd classification
python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py
streamlit run app.py  # or python app.py for Flask/FastAPI
```

### Regression Problem
```bash
cd regression
python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py
streamlit run app.py
```

## Results

Evaluation metrics, visualizations, and model artifacts are saved in respective `results/` directories.




# Updated: 2025-12-11

# Last updated: 2025-12-11

# TODO: Review implementation
