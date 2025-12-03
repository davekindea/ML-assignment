import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from utils import load_model, MODELS_DIR
from feature_engineering import FeatureEngineer
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Regression Model Deployment",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Flight Delays Prediction Model")
st.markdown("Predict flight arrival delays based on flight information")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_deployment_model():
    try:
        import joblib
        
        # List of possible model file locations to check
        possible_paths = []
        
        # 1. Standard MODELS_DIR path
        model_file = MODELS_DIR / "best_regression_model.pkl"
        possible_paths.append(model_file)
        
        # 2. Relative to app.py location
        app_dir = Path(__file__).parent
        possible_paths.append(app_dir / "models" / "best_regression_model.pkl")
        
        # 3. For Streamlit Cloud deployment (common structure)
        possible_paths.append(app_dir.parent / "regression" / "models" / "best_regression_model.pkl")
        
        # 4. Absolute path from current working directory
        cwd = Path.cwd()
        possible_paths.append(cwd / "regression" / "models" / "best_regression_model.pkl")
        possible_paths.append(cwd / "models" / "best_regression_model.pkl")
        
        # Try each path
        model = None
        for model_path in possible_paths:
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    break
                except Exception as e:
                    continue  # Try next path if loading fails
        
        # If none found, try using load_model function as fallback
        if model is None:
            try:
                model = load_model("best_regression_model.pkl")
            except:
                pass
        
        # If still not found, generate error message
        if model is None:
            checked_paths = "\n".join([f"  - `{p}` (exists: {p.exists()})" for p in possible_paths])
            error_msg = f"""
            **Model Not Found**
            
            The model file `best_regression_model.pkl` was not found in any expected location.
            
            **Checked locations:**
            {checked_paths}
            
            **Current paths:**
            - Working directory: `{Path.cwd()}`
            - App directory: `{app_dir}`
            - MODELS_DIR: `{MODELS_DIR}`
            
            **To fix this:**
            
            1. **For local development:**
               ```bash
               cd regression
               python src/main.py
               ```
            
            2. **For Streamlit Cloud deployment:**
               - Ensure the model file is committed to your repository
               - Check that `regression/models/best_regression_model.pkl` exists in your repo
               - Verify `.gitignore` doesn't exclude it (it may be tracked if added before .gitignore)
               - If file is too large (>100MB), use Git LFS or host elsewhere
            
            3. **Verify the file exists:**
               - Check your repository on GitHub
               - Ensure the file path is: `regression/models/best_regression_model.pkl`
            """
            return None, error_msg
        
        # Model found, return it
        return model, None

    except Exception as e:
        return None, f"Error loading model: {str(e)}\n\nPlease ensure the model file exists and is accessible."

@st.cache_resource
def load_feature_engineer():
    """Load or create feature engineer instance."""
    try:
        # Try to load saved feature engineer if it exists
        feature_engineer_file = MODELS_DIR / "feature_engineer.pkl"
        if feature_engineer_file.exists():
            return joblib.load(feature_engineer_file), None
        else:
            # Create a new instance
            return FeatureEngineer(), "Feature engineer not saved. Using default settings."
    except Exception as e:
        return FeatureEngineer(), f"Could not load feature engineer: {str(e)}"

def encode_categorical_for_inference(df, categorical_features, feature_engineer):
    """
    Encode categorical features for inference using saved encoders or create new ones.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_features : list
        List of categorical feature names
    feature_engineer : FeatureEngineer
        Feature engineer instance with saved encoders (if available)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    
    for col in categorical_features:
        if col not in df_encoded.columns:
            continue
        
        # Check if encoder exists for this column
        if hasattr(feature_engineer, 'encoders') and col in feature_engineer.encoders:
            # Use saved encoder for transformation
            try:
                le = feature_engineer.encoders[col]
                # Handle unseen values by assigning them to a default value
                df_encoded[col] = df_encoded[col].astype(str)
                # Get all unique values in training data
                known_classes = set(le.classes_)
                # Replace unseen values with the first known class
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                df_encoded[col] = le.transform(df_encoded[col])
            except Exception as e:
                # If transformation fails, create new encoder (fallback)
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        else:
            # No saved encoder, create new one (this might cause issues)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded

model, error = load_deployment_model()
feature_engineer, fe_warning = load_feature_engineer()

if error:
    st.error(error)
    st.markdown("---")
    
    # Debug information
    with st.expander("🔍 Debug Information"):
        st.write("**Current working directory:**", Path.cwd())
        st.write("**App file location:**", Path(__file__).parent)
        st.write("**MODELS_DIR path:**", MODELS_DIR)
        st.write("**MODELS_DIR exists:**", MODELS_DIR.exists())
        if MODELS_DIR.exists():
            st.write("**Files in MODELS_DIR:**")
            try:
                model_files = list(MODELS_DIR.glob("*.pkl"))
                if model_files:
                    for f in model_files:
                        st.write(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
                else:
                    st.write("  No .pkl files found")
            except Exception as e:
                st.write(f"  Error listing files: {e}")
    
    st.info("""
    **Quick Start Guide:**
    
    1. Navigate to the regression directory
    2. Run the training pipeline: `python src/main.py`
    3. This will train and save the model to `models/best_regression_model.pkl`
    4. Then restart this Streamlit app
    
    **For Streamlit Cloud deployment:**
    - Ensure the model file is committed to your repository (check .gitignore)
    - Model files are typically excluded by .gitignore - you may need to:
      - Use Git LFS for large files, OR
      - Host the model elsewhere and download it at runtime
    
    For more details, see the README.md file in the regression directory.
    """)
    st.stop()
else:
    st.success("✅ Model loaded successfully!")
    if fe_warning:
        st.warning(fe_warning)

    # ------------------------------------
    # Define expected model features in the EXACT order the model expects
    # ------------------------------------
    FEATURE_COLUMNS = [
        'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
        'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
        'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME',
        'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
        'DIVERTED', 'CANCELLED', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'CANCELLATION_REASON_B', 'CANCELLATION_REASON_C',
        'CANCELLATION_REASON_D'
    ]
    
    # Most important/useful features for user input (available BEFORE departure)
    KEY_FEATURES = [
        'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 'DISTANCE'
    ]
    
    # Optional features available AFTER departure (improves prediction accuracy)
    OPTIONAL_REALTIME_FEATURES = ['DEPARTURE_DELAY']
    
    # Common airlines (US major carriers)
    COMMON_AIRLINES = [
        'AA', 'DL', 'UA', 'WN', 'AS', 'B6', 'F9', 'NK', 'G4', 'SY'
    ]
    
    # Common US airports (major hubs)
    COMMON_AIRPORTS = [
        'ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MIA',
        'CLT', 'PHX', 'EWR', 'MCO', 'MSP', 'DTW', 'PHL', 'LGA', 'BWI', 'BOS',
        'IAD', 'SLC', 'MDW', 'DCA', 'HNL', 'PDX', 'STL', 'MCI', 'AUS', 'SAN'
    ]
    
    # Categorical features that need special handling
    CATEGORICAL_FEATURES = ['AIRLINE', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    
    # Numeric features
    NUMERIC_FEATURES = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATURES]
    
    def get_default_values():
        """Get default values for all features."""
        from datetime import datetime
        now = datetime.now()
        
        return {
            'YEAR': now.year,
            'MONTH': now.month,
            'DAY': now.day,
            'DAY_OF_WEEK': now.weekday() + 1,  # 1-7
            'AIRLINE': 'AA',
            'FLIGHT_NUMBER': 100,
            'TAIL_NUMBER': 'N12345',
            'ORIGIN_AIRPORT': 'JFK',
            'DESTINATION_AIRPORT': 'LAX',
            'SCHEDULED_DEPARTURE': 800,
            'DEPARTURE_TIME': 800,
            'DEPARTURE_DELAY': 0,
            'TAXI_OUT': 15,
            'WHEELS_OFF': 815,
            'SCHEDULED_TIME': 300,
            'ELAPSED_TIME': 300,
            'AIR_TIME': 280,
            'DISTANCE': 2500,
            'WHEELS_ON': 1190,
            'TAXI_IN': 10,
            'SCHEDULED_ARRIVAL': 1200,
            'ARRIVAL_TIME': 1200,
            'DIVERTED': 0,
            'CANCELLED': 0,
            'AIR_SYSTEM_DELAY': 0,
            'SECURITY_DELAY': 0,
            'AIRLINE_DELAY': 0,
            'LATE_AIRCRAFT_DELAY': 0,
            'WEATHER_DELAY': 0,
            'CANCELLATION_REASON_B': 0,
            'CANCELLATION_REASON_C': 0,
            'CANCELLATION_REASON_D': 0
        }

    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("This model predicts flight arrival delays (in minutes) based on flight information.")
    
    with st.sidebar.expander("ℹ️ About DEPARTURE_DELAY"):
        st.markdown("""
        **Why is DEPARTURE_DELAY used?**
        
        - **Strong Predictor**: If a flight departs late, it's very likely to arrive late
        - **Use Case**: Best for real-time predictions AFTER the flight has departed
        - **Before Departure**: Set to 0 if predicting before departure
        - **After Departure**: Enter actual delay for more accurate predictions
        
        The model was trained with this feature because it significantly improves prediction accuracy for in-flight predictions.
        """)

    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction"])

    # -----------------------------
    # SINGLE PREDICTION TAB
    # -----------------------------
    with tab1:
        st.header("Single Prediction")
        st.markdown("Enter the most important flight information to predict arrival delay:")
        st.info("💡 Only the most impactful features are shown. Other features use smart defaults.")

        # Initialize with defaults
        defaults = get_default_values()
        input_values = defaults.copy()

        # Get user inputs for key features only
        st.subheader("📅 Date & Time")
        col1, col2 = st.columns(2)
        with col1:
            month = st.number_input("MONTH (1-12)", min_value=1, max_value=12, value=defaults['MONTH'], step=1, key='month_input')
            input_values['MONTH'] = month
        with col2:
            day_of_week = st.selectbox("DAY OF WEEK", 
                options=[(1, "Monday"), (2, "Tuesday"), (3, "Wednesday"), (4, "Thursday"), 
                        (5, "Friday"), (6, "Saturday"), (7, "Sunday")],
                format_func=lambda x: x[1],
                index=defaults['DAY_OF_WEEK'] - 1,
                key='dow_input')
            input_values['DAY_OF_WEEK'] = day_of_week[0]

        st.subheader("✈️ Flight Route")
        col1, col2 = st.columns(2)
        with col1:
            # Airline dropdown
            airline_options = COMMON_AIRLINES + ['Other (Enter Code)']
            airline_index = COMMON_AIRLINES.index(defaults['AIRLINE']) if defaults['AIRLINE'] in COMMON_AIRLINES else 0
            airline_selection = st.selectbox("AIRLINE", options=airline_options, index=airline_index, key='airline_input')
            
            if airline_selection == 'Other (Enter Code)':
                airline_custom = st.text_input("Enter Airline Code", value="", key='airline_custom_input', max_chars=3)
                input_values['AIRLINE'] = airline_custom.upper() if airline_custom else defaults['AIRLINE']
            else:
                input_values['AIRLINE'] = airline_selection
            
            # Origin airport dropdown
            origin_options = COMMON_AIRPORTS + ['Other (Enter Code)']
            origin_index = COMMON_AIRPORTS.index(defaults['ORIGIN_AIRPORT']) if defaults['ORIGIN_AIRPORT'] in COMMON_AIRPORTS else 0
            origin_selection = st.selectbox("ORIGIN AIRPORT", options=origin_options, index=origin_index, key='origin_input')
            
            if origin_selection == 'Other (Enter Code)':
                origin_custom = st.text_input("Enter Origin Airport Code", value="", key='origin_custom_input', max_chars=3)
                input_values['ORIGIN_AIRPORT'] = origin_custom.upper() if origin_custom else defaults['ORIGIN_AIRPORT']
            else:
                input_values['ORIGIN_AIRPORT'] = origin_selection
        
        with col2:
            # Destination airport dropdown
            dest_options = COMMON_AIRPORTS + ['Other (Enter Code)']
            dest_index = COMMON_AIRPORTS.index(defaults['DESTINATION_AIRPORT']) if defaults['DESTINATION_AIRPORT'] in COMMON_AIRPORTS else COMMON_AIRPORTS.index('LAX') if 'LAX' in COMMON_AIRPORTS else 0
            dest_selection = st.selectbox("DESTINATION AIRPORT", options=dest_options, index=dest_index, key='dest_input')
            
            if dest_selection == 'Other (Enter Code)':
                dest_custom = st.text_input("Enter Destination Airport Code", value="", key='dest_custom_input', max_chars=3)
                input_values['DESTINATION_AIRPORT'] = dest_custom.upper() if dest_custom else defaults['DESTINATION_AIRPORT']
            else:
                input_values['DESTINATION_AIRPORT'] = dest_selection

        st.subheader("🕐 Schedule Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            scheduled_dep = st.number_input("SCHEDULED DEPARTURE TIME (HHMM, e.g., 800 for 8:00 AM)", 
                min_value=0, max_value=2359, value=defaults['SCHEDULED_DEPARTURE'], step=1, key='sched_dep_input')
            input_values['SCHEDULED_DEPARTURE'] = scheduled_dep
        with col2:
            scheduled_time = st.number_input("SCHEDULED FLIGHT TIME (minutes)", 
                min_value=0, value=defaults['SCHEDULED_TIME'], step=1, key='sched_time_input')
            input_values['SCHEDULED_TIME'] = scheduled_time
        with col3:
            distance = st.number_input("DISTANCE (miles)", 
                min_value=0, value=defaults['DISTANCE'], step=1, key='distance_input')
            input_values['DISTANCE'] = distance

        # Show real-time status in expander (only available after departure)
        with st.expander("⏱️ Real-Time Status (Optional - Available After Departure)", expanded=False):
            st.info("💡 **When to use:** If the flight has already departed and you know the current departure delay, enter it here for more accurate predictions.")
            st.markdown("""
            **Why this helps:**
            - If a flight departs late, it's very likely to arrive late
            - This is one of the strongest predictors of arrival delay
            - **Leave as 0 if predicting before departure**
            """)
            departure_delay = st.number_input("CURRENT DEPARTURE DELAY (minutes)", 
                value=defaults['DEPARTURE_DELAY'], step=1, key='dep_delay_input',
                help="Enter 0 if flight hasn't departed yet, or the actual delay if it has already departed")
            input_values['DEPARTURE_DELAY'] = departure_delay

        # Show advanced options in expander
        with st.expander("⚙️ Advanced Options (Optional)"):
            st.markdown("These features use default values but can be customized:")
            col1, col2 = st.columns(2)
            with col1:
                input_values['YEAR'] = st.number_input("YEAR", min_value=2015, max_value=2030, value=defaults['YEAR'], step=1)
                input_values['DAY'] = st.number_input("DAY", min_value=1, max_value=31, value=defaults['DAY'], step=1)
                input_values['FLIGHT_NUMBER'] = st.number_input("FLIGHT_NUMBER", min_value=1, value=defaults['FLIGHT_NUMBER'], step=1)
                input_values['TAIL_NUMBER'] = st.text_input("TAIL_NUMBER", value=defaults['TAIL_NUMBER'])
            with col2:
                input_values['DIVERTED'] = st.selectbox("DIVERTED", [0, 1], index=0)
                input_values['CANCELLED'] = st.selectbox("CANCELLED", [0, 1], index=0)
            
            st.markdown("**Note:** Other operational features (taxi times, wheels off/on, etc.) are calculated automatically.")

        if st.button("🔮 Predict Arrival Delay", type="primary"):
            try:
                # Ensure all required features are present with defaults
                for feature in FEATURE_COLUMNS:
                    if feature not in input_values:
                        input_values[feature] = defaults.get(feature, 0)
                
                # Create DataFrame with features in the EXACT order expected by the model
                input_df = pd.DataFrame([input_values])[FEATURE_COLUMNS]
                
                # Convert numeric columns to proper types
                for col in NUMERIC_FEATURES:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
                # Encode categorical features FIRST (before scaling)
                input_df = encode_categorical_for_inference(input_df, CATEGORICAL_FEATURES, feature_engineer)
                
                # Ensure all columns are numeric (convert any remaining object columns)
                for col in input_df.columns:
                    if input_df[col].dtype == 'object':
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
                # Apply feature engineering (same as training)
                try:
                    # Scale features if scaler was saved
                    if hasattr(feature_engineer, 'scaler') and feature_engineer.scaler:
                        input_df = feature_engineer.scale_features(input_df, method='standard', fit=False)
                except Exception as fe_error:
                    st.warning(f"Feature scaling warning: {str(fe_error)}. Using unscaled features.")

                # Make prediction
                prediction = model.predict(input_df)[0]

                # Display results
                st.success(f"**Predicted Arrival Delay:** {prediction:.2f} minutes")
                
                if prediction > 0:
                    st.info(f"⚠️ The flight is predicted to arrive **{prediction:.2f} minutes late**")
                elif prediction < 0:
                    st.info(f"✅ The flight is predicted to arrive **{abs(prediction):.2f} minutes early**")
                else:
                    st.info(f"🕐 The flight is predicted to arrive **on time**")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Make sure all input features are provided and match the model's expected format.")
                with st.expander("Debug Information"):
                    st.write(f"Error details: {str(e)}")
                    st.write(f"Input data shape: {input_df.shape if 'input_df' in locals() else 'N/A'}")
                    st.write(f"Expected features: {len(FEATURE_COLUMNS)}")
                    if 'input_df' in locals():
                        st.write(f"Input columns: {list(input_df.columns)}")

    # -----------------------------
    # BATCH PREDICTION TAB
    # -----------------------------
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with flight data to get predictions for multiple flights:")
        st.info("💡 **Minimum Required:** Your CSV should include the key features. Missing features will use defaults.")
        
        with st.expander("📋 Key Features (Required)"):
            st.write("**At minimum, include these columns (available before departure):**")
            st.code(", ".join(KEY_FEATURES))
            st.write("\n**Optional Real-Time Features (available after departure):**")
            st.code(", ".join(OPTIONAL_REALTIME_FEATURES))
            st.write("💡 **Note:** DEPARTURE_DELAY is optional but highly recommended if available (significantly improves accuracy)")
            st.write("\n**All Features:** " + ", ".join(FEATURE_COLUMNS))

        uploaded_file = st.file_uploader("Upload CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Preview:")
                st.dataframe(df.head())

                # Check for key features
                missing_key_cols = [c for c in KEY_FEATURES if c not in df.columns]
                missing_realtime_cols = [c for c in OPTIONAL_REALTIME_FEATURES if c not in df.columns]
                missing_all_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]

                if missing_key_cols:
                    st.error(f"❌ Missing key required columns: {missing_key_cols}")
                    st.info(f"Available columns: {', '.join(df.columns)}")
                else:
                    if missing_realtime_cols:
                        st.info(f"💡 **Tip:** Consider including DEPARTURE_DELAY for more accurate predictions (if available)")
                    if missing_all_cols:
                        st.warning(f"⚠️ Some optional features missing ({len(missing_all_cols)}). Using defaults for: {', '.join(missing_all_cols[:5])}{'...' if len(missing_all_cols) > 5 else ''}")
                    else:
                        st.success(f"✅ All features present ({len(FEATURE_COLUMNS)} features)")
                    
                    if st.button("🔮 Predict Batch", type="primary"):
                        try:
                            # Fill missing columns with defaults
                            defaults = get_default_values()
                            for col in FEATURE_COLUMNS:
                                if col not in df.columns:
                                    df[col] = defaults.get(col, 0)
                            
                            # Ensure columns are in the correct order
                            df_processed = df[FEATURE_COLUMNS].copy()
                            
                            # Convert numeric columns
                            for col in NUMERIC_FEATURES:
                                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                            
                            # Encode categorical features FIRST (before scaling)
                            df_processed = encode_categorical_for_inference(df_processed, CATEGORICAL_FEATURES, feature_engineer)
                            
                            # Ensure all columns are numeric (convert any remaining object columns)
                            for col in df_processed.columns:
                                if df_processed[col].dtype == 'object':
                                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                            
                            # Apply feature engineering (same as training)
                            try:
                                # Scale features if scaler was saved
                                if hasattr(feature_engineer, 'scaler') and feature_engineer.scaler:
                                    df_processed = feature_engineer.scale_features(df_processed, method='standard', fit=False)
                            except Exception as fe_error:
                                st.warning(f"Feature scaling warning: {str(fe_error)}. Using unscaled features.")
                            
                            predictions = model.predict(df_processed)
                            df["Predicted_Arrival_Delay"] = predictions

                            st.subheader("Predictions:")
                            # Show key columns plus prediction
                            display_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'MONTH', 'DAY_OF_WEEK', 'DEPARTURE_DELAY', 'Predicted_Arrival_Delay']
                            available_display_cols = [c for c in display_cols if c in df.columns]
                            st.dataframe(df[available_display_cols])
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Delay", f"{predictions.mean():.2f} min")
                            with col2:
                                st.metric("Median Delay", f"{np.median(predictions):.2f} min")
                            with col3:
                                st.metric("Max Delay", f"{predictions.max():.2f} min")
                            with col4:
                                st.metric("Min Delay", f"{predictions.min():.2f} min")

                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="flight_delay_predictions.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                            with st.expander("Debug Information"):
                                st.write(f"Error details: {str(e)}")
                                st.write(f"Data shape: {df_processed.shape if 'df_processed' in locals() else 'N/A'}")
                                if 'df_processed' in locals():
                                    st.write(f"Processed columns: {list(df_processed.columns)}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with st.expander("ℹ️ Model Details"):
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Number of Features:** {len(FEATURE_COLUMNS)}")
        if hasattr(model, "feature_importances_"):
            st.write("**Feature Importance:** Available")
            # Optionally show top features
            try:
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': FEATURE_COLUMNS,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                st.dataframe(feature_importance_df.head(10))
            except:
                pass
        st.write("This model predicts flight arrival delays (in minutes) based on flight information.")
