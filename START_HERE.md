# 🚀 START HERE - ML Assignment Quick Guide

## ✅ Everything is Set Up!

Both classification and regression problems are ready to use.

---

## 📊 **Classification: Heart Disease Prediction**

### ✅ Dataset Status
- **File**: `classification/data/raw/heart.csv`
- **Status**: ✅ **DOWNLOADED AND READY**
- **Target Column**: `HeartDisease`
- **Dataset Size**: ~918 rows, 12 features

### 🎯 Quick Start (3 Steps)

1. **Install dependencies** (if not done):
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm optuna streamlit
   ```

2. **Run the pipeline**:
   ```bash
   cd classification
   python src/main.py
   ```
   
   When prompted:
   - **Dataset path**: Press **Enter** (auto-detects `heart.csv`)
   - **Target column**: Type **`HeartDisease`** and press Enter
   - **Other options**: Use defaults (press Enter) or customize

3. **Deploy the model**:
   ```bash
   streamlit run app.py
   ```

---

## 📈 **Regression: Flight Delays Prediction**

### ⚠️ Dataset Status
- **File**: Will be in `regression/data/raw/` after download
- **Status**: ⚠️ **NEEDS DOWNLOAD**
- **Target Column**: `ARRIVAL_DELAY` (recommended) or `DEPARTURE_DELAY`

### 🎯 Quick Start (4 Steps)

1. **Install dependencies** (same as above)

2. **Download dataset**:
   ```bash
   cd regression
   python download_dataset.py
   ```

3. **Run the pipeline**:
   ```bash
   python src/main.py
   ```
   
   When prompted:
   - **Dataset path**: Press **Enter** (auto-detects CSV)
   - **Target column**: Type **`ARRIVAL_DELAY`** and press Enter
   - **Other options**: Use defaults or customize

4. **Deploy the model**:
   ```bash
   streamlit run app.py
   ```

---

## 📋 Dataset Information

### Classification Dataset (Heart Disease)
```
Columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, 
         FastingBS, RestingECG, MaxHR, ExerciseAngina, 
         Oldpeak, ST_Slope, HeartDisease

Target: HeartDisease (0 = No disease, 1 = Has disease)
Size: ~918 rows
```

### Regression Dataset (Flight Delays)
```
Main File: flights.csv (or similar)
Target: ARRIVAL_DELAY or DEPARTURE_DELAY
Size: Large (may need sampling for initial experiments)
```

---

## 🔧 Troubleshooting

### If pandas is not installed:
```bash
pip install pandas numpy scikit-learn
```

### If you get "module not found" errors:
```bash
pip install -r requirements.txt
```

### If dataset download fails:
- Check Kaggle API credentials
- Or download manually from Kaggle and place in `data/raw/`

---

## 📁 Project Files

### Classification
- ✅ `classification/data/raw/heart.csv` - Dataset (downloaded)
- ✅ `classification/src/main.py` - Main pipeline
- ✅ `classification/download_dataset.py` - Download script
- ✅ `classification/run_pipeline.py` - Automated script
- ✅ `classification/app.py` - Streamlit app

### Regression
- ⚠️ `regression/data/raw/` - Place dataset here
- ✅ `regression/src/main.py` - Main pipeline
- ✅ `regression/download_dataset.py` - Download script
- ✅ `regression/run_pipeline.py` - Automated script
- ✅ `regression/app.py` - Streamlit app

---

## 🎓 Assignment Checklist

- [x] Project structure created
- [x] Classification dataset downloaded
- [x] Regression dataset script ready
- [x] All ML pipeline modules created
- [x] Deployment apps created
- [ ] Run classification pipeline
- [ ] Run regression pipeline
- [ ] Document the process
- [ ] Create PowerPoint presentation

---

## 💡 Recommended Order

1. **Start with Classification** (dataset already downloaded)
   ```bash
   cd classification
   python src/main.py
   ```

2. **Then do Regression** (download dataset first)
   ```bash
   cd regression
   python download_dataset.py
   python src/main.py
   ```

3. **Deploy both models**
   ```bash
   # Classification
   cd classification
   streamlit run app.py
   
   # Regression (in another terminal)
   cd regression
   streamlit run app.py
   ```

4. **Document everything** in `docs/ML_Process_Documentation.md`

5. **Create presentation** with results and visualizations

---

## 🎯 Key Points

- **Classification Target**: `HeartDisease`
- **Regression Target**: `ARRIVAL_DELAY` (or `DEPARTURE_DELAY`)
- **Both pipelines are interactive** - they guide you through each step
- **Results are saved** in `results/` folders
- **Models are saved** in `models/` folders

---

## ✅ You're Ready!

Everything is set up. Just run the pipelines and follow the prompts!

**Good luck!** 🚀

