"""
Streamlit Deployment App for Classification Model
"""
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
    page_title="Classification Model Deployment",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Classification Model Deployment")
st.markdown("---")

# Load model and preprocessing artifacts
@st.cache_resource
def load_deployment_model():
    """Load the trained model and preprocessing artifacts."""
    try:
        model_files = list(MODELS_DIR.glob("*.pkl"))
        if not model_files:
            return None, None, None, "No model found. Please train a model first."
        
        # Try to load the best model
        model_file = MODELS_DIR / "best_classification_model.pkl"
        if model_file.exists():
            model = load_model("best_classification_model.pkl")
        else:
            # Load the first available model
            model = joblib.load(model_files[0])
        
        # Try to load preprocessing artifacts
        scaler = None
        encoders = None
        
        scaler_file = MODELS_DIR / "scaler.pkl"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
        
        encoders_file = MODELS_DIR / "encoders.pkl"
        if encoders_file.exists():
            encoders = joblib.load(encoders_file)
        
        return model, scaler, encoders, None
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"

model, scaler, encoders, error = load_deployment_model()

if error:
    st.error(error)
    st.info("Please run the training pipeline first using: `python classification/src/main.py`")
else:
    st.success("✅ Model loaded successfully!")
    if scaler is None or encoders is None:
        st.warning("⚠️ Preprocessing artifacts (scaler/encoders) not found. The app will attempt to recreate them, but results may not match training exactly. Consider retraining with saved preprocessing artifacts.")
    
    # Sidebar for input
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("This is a classification model for predicting target classes based on input features.")
    
    # Main content
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction"])
    
    with tab1:
        st.header("Single Prediction")
        st.markdown("Enter feature values to get a prediction:")
        
        # Create input fields based on the heart disease dataset
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=52)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=0)
            trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=125)
        
        with col2:
            chol = st.number_input("Cholesterol (chol)", min_value=0, max_value=600, value=212)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.number_input("Resting ECG (restecg)", min_value=0, max_value=2, value=1)
            thalach = st.number_input("Max Heart Rate (thalach)", min_value=0, max_value=250, value=168)
        
        with col3:
            exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.number_input("Slope", min_value=0, max_value=2, value=2)
            ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=2)
            thal = st.number_input("Thalassemia (thal)", min_value=0, max_value=3, value=3)
        
        if st.button("🔮 Predict", type="primary"):
            try:
                # Create input dataframe with original features
                input_data = pd.DataFrame({
                    'age': [age],
                    'sex': [sex],
                    'cp': [cp],
                    'trestbps': [trestbps],
                    'chol': [chol],
                    'fbs': [fbs],
                    'restecg': [restecg],
                    'thalach': [thalach],
                    'exang': [exang],
                    'oldpeak': [oldpeak],
                    'slope': [slope],
                    'ca': [ca],
                    'thal': [thal]
                })
                
                # Apply the same feature engineering pipeline
                # Note: No target column needed for prediction
                feature_engineer = FeatureEngineer(target_column=None)
                
                # Load saved preprocessing artifacts if available
                if scaler is not None:
                    feature_engineer.scaler = scaler
                if encoders is not None:
                    feature_engineer.encoders = encoders
                
                # Encode categorical variables (same as training)
                input_data = feature_engineer.encode_categorical(input_data, method='auto')
                
                # Scale features (same as training)
                # Note: If scaler is not loaded, this will fail - need to retrain with saved preprocessing
                if feature_engineer.scaler is None:
                    st.error("❌ Preprocessing scaler not found. Please retrain the model with saved preprocessing artifacts.")
                    st.info("The scaler is required to properly scale features for prediction.")
                    st.stop()
                
                input_data = feature_engineer.scale_features(input_data, method='standard', fit=False)
                
                # Create interaction features with age (same as training)
                # Ensure we create them in the same order as during training
                input_data = feature_engineer.create_interaction_features(input_data, max_interactions=10)
                
                # Select only the features the model expects, in the correct order
                expected_features = model.feature_names_in_
                
                # Check if all expected features are present
                missing_features = [f for f in expected_features if f not in input_data.columns]
                if missing_features:
                    st.error(f"❌ Missing features: {', '.join(missing_features)}")
                    st.info("The feature engineering pipeline may not match the training pipeline.")
                    st.stop()
                
                input_data = input_data[expected_features]
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_proba = None
                
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(input_data)[0]
                
                # Display results
                prediction_label = "Heart Disease" if prediction == 1 else "No Heart Disease"
                st.success(f"**Prediction:** {prediction_label} (Class: {prediction})")
                
                if prediction_proba is not None:
                    st.subheader("Prediction Probabilities:")
                    proba_df = pd.DataFrame({
                        'Class': ['No Heart Disease', 'Heart Disease'],
                        'Probability': prediction_proba
                    })
                    st.bar_chart(proba_df.set_index('Class'))
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Make sure the input features match the model's expected features.")
                st.exception(e)
    
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with features to get predictions for multiple samples:")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Preview:")
                st.dataframe(df.head())
                
                # Check if required columns are present
                required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.info("Please ensure your CSV file contains all required columns.")
                else:
                    if st.button("🔮 Predict Batch", type="primary"):
                        try:
                            # Apply the same feature engineering pipeline
                            # Note: No target column needed for prediction
                            feature_engineer = FeatureEngineer(target_column=None)
                            
                            # Load saved preprocessing artifacts if available
                            if scaler is not None:
                                feature_engineer.scaler = scaler
                            if encoders is not None:
                                feature_engineer.encoders = encoders
                            
                            # Encode categorical variables
                            df_processed = feature_engineer.encode_categorical(df.copy(), method='auto')
                            
                            # Scale features
                            # Note: If scaler is not loaded, this will fail - need to retrain with saved preprocessing
                            if feature_engineer.scaler is None:
                                st.error("❌ Preprocessing scaler not found. Please retrain the model with saved preprocessing artifacts.")
                                st.info("The scaler is required to properly scale features for prediction.")
                                st.stop()
                            
                            df_processed = feature_engineer.scale_features(df_processed, method='standard', fit=False)
                            
                            # Create interaction features
                            df_processed = feature_engineer.create_interaction_features(df_processed, max_interactions=10)
                            
                            # Select only the features the model expects, in the correct order
                            expected_features = model.feature_names_in_
                            
                            # Check if all expected features are present
                            missing_features = [f for f in expected_features if f not in df_processed.columns]
                            if missing_features:
                                st.error(f"❌ Missing features: {', '.join(missing_features)}")
                                st.info("The feature engineering pipeline may not match the training pipeline.")
                                st.stop()
                            
                            df_processed = df_processed[expected_features]
                            
                            # Make predictions
                            predictions = model.predict(df_processed)
                            df['Prediction'] = predictions
                            df['Prediction_Label'] = df['Prediction'].map({0: 'No Heart Disease', 1: 'Heart Disease'})
                            
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(df_processed)
                                df['Probability_No_Disease'] = proba[:, 0]
                                df['Probability_Disease'] = proba[:, 1]
                            
                            st.subheader("Predictions:")
                            st.dataframe(df)
                            
                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                            st.exception(e)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Model info
    with st.expander("ℹ️ Model Details"):
        st.write(f"Model Type: {type(model).__name__}")
        if hasattr(model, 'feature_importances_'):
            st.write("This model supports feature importance analysis.")
        if hasattr(model, 'predict_proba'):
            st.write("This model provides prediction probabilities.")


