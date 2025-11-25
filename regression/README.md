# Regression Problem: Flight Delays Prediction

## Problem Definition

**Business Problem**: Predict flight delays to help airlines and passengers better plan their travel.

**Goal**: Predict the delay time (in minutes) for flights based on various features such as:
- Flight information (airline, origin, destination)
- Time information (month, day, day of week, scheduled departure time)
- Weather conditions
- Historical delay patterns

**Type**: Regression (predicting continuous delay time)

**Success Criteria**:
- R² Score: > 0.70
- RMSE: < 30 minutes
- MAE: < 20 minutes

## Dataset

**Dataset**: Flight Delays (US DOT)
- **Source**: Kaggle (usdot/flight-delays)
- **Link**: https://www.kaggle.com/datasets/usdot/flight-delays
- **Description**: Contains information about flights in the US, including delays

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

The Flight Delays dataset typically contains:
- `flights.csv` - Main flight data
- Other CSV files with related information

**Key Features** (expected):
- `YEAR`, `MONTH`, `DAY`, `DAY_OF_WEEK`
- `AIRLINE`, `FLIGHT_NUMBER`
- `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`
- `SCHEDULED_DEPARTURE`, `DEPARTURE_TIME`
- `SCHEDULED_ARRIVAL`, `ARRIVAL_TIME`
- `DEPARTURE_DELAY`, `ARRIVAL_DELAY` (target variable)
- `DISTANCE`, `CANCELLED`, etc.

**Target Variable**: `ARRIVAL_DELAY` or `DEPARTURE_DELAY` (depending on the problem)

## Notes

- The dataset might be large, so processing may take time
- You may need to sample the data if it's too large for initial experiments
- Some features may need to be engineered (e.g., time-based features)
- Handle missing values appropriately
- Consider feature selection for better performance


