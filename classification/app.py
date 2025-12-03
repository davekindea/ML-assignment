"""
Streamlit Deployment App for Heart Disease Classification Model
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
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction Model")
st.markdown("Predict the likelihood of heart disease based on patient medical information")

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
    
    # Feature definitions for dropdowns
    CHEST_PAIN_TYPES = {
        0: "Typical Angina",
        1: "Atypical Angina", 
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    }
    
    RESTING_ECG_TYPES = {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    }
    
    SLOPE_TYPES = {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping"
    }
    
    THAL_TYPES = {
        0: "Normal",
        1: "Fixed Defect",
        2: "Reversable Defect",
        3: "Not Available"
    }
    
    # Sidebar for input
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("This model predicts the likelihood of heart disease based on patient medical information.")
    
    with st.sidebar.expander("ℹ️ About the Model"):
        st.markdown("""
        **Model Purpose:**
        - Predicts presence of heart disease (binary classification)
        - Uses medical test results and patient demographics
        - Provides probability scores for better decision-making
        
        **Important Notes:**
        - This is a prediction tool, not a medical diagnosis
        - Always consult healthcare professionals for medical decisions
        - Results should be used as supplementary information only
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction"])
    
    with tab1:
        st.header("Single Prediction")
        st.markdown("Enter patient medical information to predict heart disease risk:")
        st.info("💡 All fields are required. Use dropdowns where available for easier selection.")
        
        # Patient Demographics
        st.subheader("👤 Patient Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=52, step=1, 
                                 help="Patient's age in years")
        with col2:
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
                              help="Patient's biological sex")
        
        # Vital Signs
        st.subheader("💓 Vital Signs")
        col1, col2, col3 = st.columns(3)
        with col1:
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=300, value=125, step=1,
                                      help="Resting blood pressure in mmHg")
        with col2:
            chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=212, step=1,
                                  help="Serum cholesterol level in mg/dl")
        with col3:
            thalach = st.number_input("Max Heart Rate (bpm)", min_value=0, max_value=250, value=168, step=1,
                                     help="Maximum heart rate achieved during exercise")
        
        # Symptoms & Basic Tests
        st.subheader("🩺 Symptoms & Basic Tests")
        col1, col2 = st.columns(2)
        with col1:
            cp = st.selectbox("Chest Pain Type", 
                              options=list(CHEST_PAIN_TYPES.keys()),
                              format_func=lambda x: CHEST_PAIN_TYPES[x],
                              index=0,
                              help="Type of chest pain experienced")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                              options=[0, 1], 
                              format_func=lambda x: "No" if x == 0 else "Yes",
                              help="Fasting blood sugar level")
        with col2:
            restecg = st.selectbox("Resting ECG Results",
                                  options=list(RESTING_ECG_TYPES.keys()),
                                  format_func=lambda x: RESTING_ECG_TYPES[x],
                                  index=1,
                                  help="Resting electrocardiographic results")
            exang = st.selectbox("Exercise Induced Angina",
                                options=[0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes",
                                help="Exercise-induced chest pain")
        
        # Advanced Medical Tests
        st.subheader("🔬 Advanced Medical Tests")
        col1, col2, col3 = st.columns(3)
        with col1:
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                     help="ST depression induced by exercise relative to rest")
        with col2:
            slope = st.selectbox("ST Slope",
                               options=list(SLOPE_TYPES.keys()),
                               format_func=lambda x: SLOPE_TYPES[x],
                               index=2,
                               help="Slope of peak exercise ST segment")
        with col3:
            ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=4, value=0, step=1,
                               help="Number of major vessels colored by fluoroscopy")
        
        thal = st.selectbox("Thalassemia (Thal)",
                            options=list(THAL_TYPES.keys()),
                            format_func=lambda x: THAL_TYPES[x],
                            index=2,
                            help="Thalassemia test result")
        
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
                prediction_label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
                prediction_color = "🔴" if prediction == 1 else "🟢"
                
                st.success(f"{prediction_color} **Prediction:** {prediction_label}")
                
                if prediction_proba is not None:
                    st.subheader("📊 Prediction Probabilities:")
                    proba_df = pd.DataFrame({
                        'Class': ['No Heart Disease', 'Heart Disease'],
                        'Probability': prediction_proba
                    })
                    
                    # Display as metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("No Heart Disease", f"{prediction_proba[0]*100:.1f}%")
                    with col2:
                        st.metric("Heart Disease", f"{prediction_proba[1]*100:.1f}%")
                    
                    # Bar chart
                    st.bar_chart(proba_df.set_index('Class'))
                    
                    # Interpretation
                    if prediction_proba[1] > 0.7:
                        st.warning("⚠️ **High Risk:** The model indicates a high probability of heart disease. Please consult with a healthcare professional.")
                    elif prediction_proba[1] > 0.4:
                        st.info("ℹ️ **Moderate Risk:** The model indicates a moderate probability. Consider further medical evaluation.")
                    else:
                        st.success("✅ **Low Risk:** The model indicates a low probability of heart disease.")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Make sure the input features match the model's expected features.")
                st.exception(e)
    
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with patient data to get predictions for multiple patients:")
        st.info("💡 Your CSV file should include all required features. See the feature list below.")
        
        with st.expander("📋 Required Features"):
            st.write("**Your CSV file must include these columns:**")
            required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            st.code(", ".join(required_cols))
            st.markdown("""
            **Feature Descriptions:**
            - `age`: Patient age (years)
            - `sex`: 0 = Female, 1 = Male
            - `cp`: Chest pain type (0-3)
            - `trestbps`: Resting blood pressure (mmHg)
            - `chol`: Cholesterol (mg/dl)
            - `fbs`: Fasting blood sugar > 120 (0/1)
            - `restecg`: Resting ECG (0-2)
            - `thalach`: Max heart rate (bpm)
            - `exang`: Exercise angina (0/1)
            - `oldpeak`: ST depression
            - `slope`: ST slope (0-2)
            - `ca`: Number of major vessels (0-4)
            - `thal`: Thalassemia (0-3)
            """)
        
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
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.info(f"Available columns: {', '.join(df.columns)}")
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
                            
                            # Summary statistics
                            total_patients = len(df)
                            heart_disease_count = (df['Prediction'] == 1).sum()
                            no_disease_count = (df['Prediction'] == 0).sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Patients", total_patients)
                            with col2:
                                st.metric("Heart Disease Detected", heart_disease_count, 
                                         delta=f"{heart_disease_count/total_patients*100:.1f}%")
                            with col3:
                                st.metric("No Heart Disease", no_disease_count,
                                         delta=f"{no_disease_count/total_patients*100:.1f}%")
                            
                            # Display results
                            display_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'Prediction_Label']
                            if 'Probability_Disease' in df.columns:
                                display_cols.append('Probability_Disease')
                            available_display_cols = [c for c in display_cols if c in df.columns]
                            st.dataframe(df[available_display_cols])
                            
                            # Show full results in expander
                            with st.expander("📋 View Full Results"):
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
        st.write(f"**Model Type:** {type(model).__name__}")
        if hasattr(model, 'feature_importances_'):
            st.write("**Feature Importance:** Available")
            try:
                # Try to show feature importance if available
                if hasattr(model, 'feature_names_in_'):
                    importances = model.feature_importances_
                    feature_names = model.feature_names_in_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    st.dataframe(importance_df.head(10))
            except:
                pass
        if hasattr(model, 'predict_proba'):
            st.write("**Prediction Probabilities:** Available")
        st.write("This model predicts the presence of heart disease based on medical test results and patient demographics.")
        
        st.markdown("""
        **⚠️ Medical Disclaimer:**
        This tool is for educational and research purposes only. It is not a substitute for professional medical advice, 
        diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have 
        regarding a medical condition.
        """)


