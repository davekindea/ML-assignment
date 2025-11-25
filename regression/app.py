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
        model_file = MODELS_DIR / "best_regression_model.pkl"
        if not model_file.exists():
            return None, "No model found. Please train a model first."

        model = load_model("best_regression_model.pkl")
        return model, None

    except Exception as e:
        return None, f"Error loading model: {str(e)}"

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

model, error = load_deployment_model()
feature_engineer, fe_warning = load_feature_engineer()

if error:
    st.error(error)
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
    
    # Categorical features that need special handling
    CATEGORICAL_FEATURES = ['AIRLINE', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    
    # Numeric features
    NUMERIC_FEATURES = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATURES]

    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("This model predicts flight arrival delays (in minutes) based on flight information.")

    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction"])

    # -----------------------------
    # SINGLE PREDICTION TAB
    # -----------------------------
    with tab1:
        st.header("Single Prediction")
        st.markdown("Enter flight information to predict arrival delay:")

        input_values = {}

        # Organize inputs into logical groups
        st.subheader("📅 Date & Time Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            input_values['YEAR'] = st.number_input("YEAR", min_value=2015, max_value=2030, value=2023, step=1)
        with col2:
            input_values['MONTH'] = st.number_input("MONTH", min_value=1, max_value=12, value=1, step=1)
        with col3:
            input_values['DAY'] = st.number_input("DAY", min_value=1, max_value=31, value=1, step=1)
        with col4:
            input_values['DAY_OF_WEEK'] = st.number_input("DAY_OF_WEEK", min_value=1, max_value=7, value=1, step=1)

        st.subheader("✈️ Flight Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            input_values['AIRLINE'] = st.text_input("AIRLINE", value="AA")
            input_values['FLIGHT_NUMBER'] = st.number_input("FLIGHT_NUMBER", min_value=1, value=100, step=1)
        with col2:
            input_values['ORIGIN_AIRPORT'] = st.text_input("ORIGIN_AIRPORT", value="JFK")
            input_values['DESTINATION_AIRPORT'] = st.text_input("DESTINATION_AIRPORT", value="LAX")
        with col3:
            input_values['TAIL_NUMBER'] = st.text_input("TAIL_NUMBER", value="N12345")

        st.subheader("🕐 Schedule & Timing")
        col1, col2 = st.columns(2)
        with col1:
            input_values['SCHEDULED_DEPARTURE'] = st.number_input("SCHEDULED_DEPARTURE", min_value=0, max_value=2359, value=800, step=1)
            input_values['DEPARTURE_TIME'] = st.number_input("DEPARTURE_TIME", min_value=0, max_value=2359, value=800, step=1)
            input_values['SCHEDULED_ARRIVAL'] = st.number_input("SCHEDULED_ARRIVAL", min_value=0, max_value=2359, value=1200, step=1)
            input_values['ARRIVAL_TIME'] = st.number_input("ARRIVAL_TIME", min_value=0, max_value=2359, value=1200, step=1)
        with col2:
            input_values['SCHEDULED_TIME'] = st.number_input("SCHEDULED_TIME (minutes)", min_value=0, value=300, step=1)
            input_values['ELAPSED_TIME'] = st.number_input("ELAPSED_TIME (minutes)", min_value=0, value=300, step=1)
            input_values['AIR_TIME'] = st.number_input("AIR_TIME (minutes)", min_value=0, value=280, step=1)

        st.subheader("⏱️ Delays & Taxi Times")
        col1, col2 = st.columns(2)
        with col1:
            input_values['DEPARTURE_DELAY'] = st.number_input("DEPARTURE_DELAY (minutes)", value=0, step=1)
            input_values['TAXI_OUT'] = st.number_input("TAXI_OUT (minutes)", min_value=0, value=15, step=1)
            input_values['TAXI_IN'] = st.number_input("TAXI_IN (minutes)", min_value=0, value=10, step=1)
        with col2:
            input_values['WHEELS_OFF'] = st.number_input("WHEELS_OFF", min_value=0, max_value=2359, value=815, step=1)
            input_values['WHEELS_ON'] = st.number_input("WHEELS_ON", min_value=0, max_value=2359, value=1190, step=1)

        st.subheader("📏 Distance")
        input_values['DISTANCE'] = st.number_input("DISTANCE (miles)", min_value=0, value=2500, step=1)

        st.subheader("⚠️ Status & Delays")
        col1, col2 = st.columns(2)
        with col1:
            input_values['DIVERTED'] = st.selectbox("DIVERTED", [0, 1], index=0)
            input_values['CANCELLED'] = st.selectbox("CANCELLED", [0, 1], index=0)
        with col2:
            input_values['AIR_SYSTEM_DELAY'] = st.number_input("AIR_SYSTEM_DELAY (minutes)", min_value=0, value=0, step=1)
            input_values['SECURITY_DELAY'] = st.number_input("SECURITY_DELAY (minutes)", min_value=0, value=0, step=1)
            input_values['AIRLINE_DELAY'] = st.number_input("AIRLINE_DELAY (minutes)", min_value=0, value=0, step=1)
            input_values['LATE_AIRCRAFT_DELAY'] = st.number_input("LATE_AIRCRAFT_DELAY (minutes)", min_value=0, value=0, step=1)
            input_values['WEATHER_DELAY'] = st.number_input("WEATHER_DELAY (minutes)", min_value=0, value=0, step=1)

        st.subheader("🚫 Cancellation Reasons")
        col1, col2, col3 = st.columns(3)
        with col1:
            input_values['CANCELLATION_REASON_B'] = st.selectbox("CANCELLATION_REASON_B", [0, 1], index=0)
        with col2:
            input_values['CANCELLATION_REASON_C'] = st.selectbox("CANCELLATION_REASON_C", [0, 1], index=0)
        with col3:
            input_values['CANCELLATION_REASON_D'] = st.selectbox("CANCELLATION_REASON_D", [0, 1], index=0)

        if st.button("🔮 Predict Arrival Delay", type="primary"):
            try:
                # Create DataFrame with features in the EXACT order expected by the model
                input_df = pd.DataFrame([input_values])[FEATURE_COLUMNS]
                
                # Convert numeric columns to proper types
                for col in NUMERIC_FEATURES:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
                # Apply feature engineering (same as training)
                try:
                    # Encode categorical variables if feature engineer was saved with encoder
                    if hasattr(feature_engineer, 'label_encoders') and feature_engineer.label_encoders:
                        input_df = feature_engineer.encode_categorical(input_df, method='label')
                    
                    # Scale features if scaler was saved
                    if hasattr(feature_engineer, 'scaler') and feature_engineer.scaler:
                        input_df = feature_engineer.scale_features(input_df, method='standard', fit=False)
                except Exception as fe_error:
                    st.warning(f"Feature engineering warning: {str(fe_error)}. Using raw features.")

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
        st.info("The CSV file should contain all the required features. See the feature list below.")
        
        with st.expander("📋 Required Features"):
            st.write("Your CSV file must include these columns (in any order):")
            st.code(", ".join(FEATURE_COLUMNS))

        uploaded_file = st.file_uploader("Upload CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Preview:")
                st.dataframe(df.head())

                missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]

                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info(f"Available columns: {', '.join(df.columns)}")
                else:
                    st.success(f"✅ All required features present ({len(FEATURE_COLUMNS)} features)")
                    
                    if st.button("🔮 Predict Batch", type="primary"):
                        try:
                            # Ensure columns are in the correct order
                            df_processed = df[FEATURE_COLUMNS].copy()
                            
                            # Convert numeric columns
                            for col in NUMERIC_FEATURES:
                                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                            
                            # Apply feature engineering
                            try:
                                if hasattr(feature_engineer, 'label_encoders') and feature_engineer.label_encoders:
                                    df_processed = feature_engineer.encode_categorical(df_processed, method='label')
                                
                                if hasattr(feature_engineer, 'scaler') and feature_engineer.scaler:
                                    df_processed = feature_engineer.scale_features(df_processed, method='standard', fit=False)
                            except Exception as fe_error:
                                st.warning(f"Feature engineering warning: {str(fe_error)}. Using raw features.")
                            
                            predictions = model.predict(df_processed)
                            df["Predicted_Arrival_Delay"] = predictions

                            st.subheader("Predictions:")
                            # Show key columns plus prediction
                            display_cols = ['AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'Predicted_Arrival_Delay']
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
