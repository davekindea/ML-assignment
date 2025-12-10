"""
Modern Streamlit App for Heart Disease Prediction
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
from sklearn.preprocessing import LabelEncoder

# Modern page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-card-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #f5576c;
        margin: 1rem 0;
    }
    h1 {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_deployment_model():
    """Load the trained model with multiple path checks."""
    try:
        import joblib
        
        possible_paths = [
            MODELS_DIR / "best_classification_model.pkl",
            Path(__file__).parent / "models" / "best_classification_model.pkl",
            Path(__file__).parent.parent / "classification" / "models" / "best_classification_model.pkl",
            Path.cwd() / "classification" / "models" / "best_classification_model.pkl",
            Path.cwd() / "models" / "best_classification_model.pkl"
        ]
        
        model = None
        for model_path in possible_paths:
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    break
                except:
                    continue
        
        if model is None:
            try:
                model = load_model("best_classification_model.pkl")
            except:
                pass
        
        if model is None:
            return None, None, None, "Model file not found in any expected location."
        
        # Load preprocessing artifacts
        scaler = None
        encoders = None
        
        app_dir = Path(__file__).parent
        scaler_paths = [
            MODELS_DIR / "scaler.pkl",
            app_dir / "models" / "scaler.pkl",
            app_dir.parent / "classification" / "models" / "scaler.pkl"
        ]
        for scaler_path in scaler_paths:
            if scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path)
                    break
                except:
                    continue
        
        encoder_paths = [
            MODELS_DIR / "encoders.pkl",
            app_dir / "models" / "encoders.pkl",
            app_dir.parent / "classification" / "models" / "encoders.pkl"
        ]
        for encoder_path in encoder_paths:
            if encoder_path.exists():
                try:
                    encoders = joblib.load(encoder_path)
                    break
                except:
                    continue
        
        return model, scaler, encoders, None
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"

def encode_categorical_string(value, feature_name, feature_engineer):
    """Encode a categorical string value to numeric."""
    if not hasattr(feature_engineer, 'encoders') or feature_name not in feature_engineer.encoders:
        return hash(str(value)) % 1000
    else:
        le = feature_engineer.encoders[feature_name]
        try:
            if str(value) in le.classes_:
                return le.transform([str(value)])[0]
            else:
                return le.transform([le.classes_[0]])[0]
        except:
            return 0

# Load model
model, scaler, encoders, error = load_deployment_model()

# Error page if model not found
if error or model is None:
    st.error("❌ Model Not Found")
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2 style='color: #f5576c;'>Model Not Available</h2>
        <p style='font-size: 1.2rem; color: #666; margin: 2rem 0;'>
            The prediction model could not be loaded. Please ensure the model file exists.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 Setup Instructions"):
        st.markdown("""
        **To fix this issue:**
        
        1. **Train the model first:**
           ```bash
           cd classification
           python src/main.py
           ```
        
        2. **Ensure the model file exists:**
           - Check: `classification/models/best_classification_model.pkl`
           - File should be created after training
        
        3. **For deployment:**
           - Make sure the model file is in your repository
           - Verify `.gitignore` allows model files
        """)
    
    st.stop()

# Main app - Model loaded successfully
st.success("✅ Model Ready")

# Feature mappings
CHEST_PAIN_MAP = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
SEX_MAP = {'Male': 1, 'Female': 0}
EXANG_MAP = {'Yes': 1, 'No': 0}
FBS_MAP = {'No': 0, 'Yes': 1}
RESTECG_MAP = {'Normal': 0, 'ST-T Abnormality': 1, 'LV Hypertrophy': 2}
SLOPE_MAP = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
THAL_MAP = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2, 'Other': 3}

# Feature name mapping: lowercase (app) -> possible model expected names
# Multiple mappings to handle different dataset naming conventions
FEATURE_NAME_MAP = {
    'age': ['Age', 'age'],
    'sex': ['Sex', 'sex'],
    'cp': ['ChestPainType', 'ChestPain', 'cp', 'CP'],
    'trestbps': ['RestingBP', 'RestingBloodPressure', 'trestbps', 'TRESTBPS'],
    'chol': ['Cholesterol', 'chol', 'CHOL'],
    'fbs': ['FastingBS', 'FastingBloodSugar', 'fbs', 'FBS'],
    'restecg': ['RestingECG', 'RestingElectrocardiographic', 'restecg', 'RESTECG'],
    'thalach': ['MaxHR', 'MaximumHeartRate', 'thalach', 'THALACH'],
    'exang': ['ExerciseAngina', 'ExerciseInducedAngina', 'exang', 'EXANG'],
    'oldpeak': ['Oldpeak', 'STDepression', 'oldpeak', 'OLDPEAK'],
    'slope': ['ST_Slope', 'STSlope', 'Slope', 'slope'],
    'ca': ['MajorVessels', 'NumberMajorVessels', 'ca', 'CA'],
    'thal': ['Thalassemia', 'Thal', 'thal', 'THAL']
}

def map_feature_names(input_values, model_expected_features):
    """Map app feature names to model's expected feature names."""
    mapped_values = {}
    
    # Create a lookup for case-insensitive matching
    model_features_lower = {f.lower(): f for f in model_expected_features}
    
    # First, try direct mapping from FEATURE_NAME_MAP
    for app_feature, possible_names in FEATURE_NAME_MAP.items():
        if app_feature in input_values:
            # Find matching feature name in model's expected features
            found = False
            for model_name in possible_names:
                if model_name in model_expected_features:
                    mapped_values[model_name] = input_values[app_feature]
                    found = True
                    break
                # Try case-insensitive match
                elif model_name.lower() in model_features_lower:
                    mapped_values[model_features_lower[model_name.lower()]] = input_values[app_feature]
                    found = True
                    break
            
            # If still not found, try direct case-insensitive match
            if not found:
                app_feature_lower = app_feature.lower()
                if app_feature_lower in model_features_lower:
                    mapped_values[model_features_lower[app_feature_lower]] = input_values[app_feature]
    
    # Also try direct case-insensitive matching for any remaining features
    for app_feature, value in input_values.items():
        app_feature_lower = app_feature.lower()
        if app_feature_lower in model_features_lower and model_features_lower[app_feature_lower] not in mapped_values:
            mapped_values[model_features_lower[app_feature_lower]] = value
    
    return mapped_values

# All required features (using lowercase for app input, will be mapped to model names)
REQUIRED_FEATURES_APP = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Modern UI
st.markdown("<h1>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem;'>Enter patient information to predict heart disease risk</p>", unsafe_allow_html=True)

# Input form - ALL ORIGINAL FEATURES
with st.container():
    input_values = {}
    
    st.markdown("### 👤 Patient Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("**Age** (years)", min_value=0, max_value=120, value=52, step=1)
        input_values['age'] = age
        
        sex_str = st.selectbox("**Sex**", options=['Male', 'Female'], index=0)
        input_values['sex'] = SEX_MAP[sex_str]
    
    st.markdown("---")
    st.markdown("### 💓 Vital Signs & Blood Tests")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trestbps = st.number_input("**Resting Blood Pressure** (mmHg)", min_value=0, max_value=300, value=125, step=1)
        input_values['trestbps'] = trestbps
        
        chol = st.number_input("**Cholesterol** (mg/dl)", min_value=0, max_value=600, value=212, step=1)
        input_values['chol'] = chol
    
    with col2:
        fbs_str = st.selectbox("**Fasting Blood Sugar > 120**", options=['No', 'Yes'], index=0)
        input_values['fbs'] = FBS_MAP[fbs_str]
        
        thalach = st.number_input("**Max Heart Rate** (bpm)", min_value=0, max_value=250, value=168, step=1)
        input_values['thalach'] = thalach
    
    st.markdown("---")
    st.markdown("### 🩺 Symptoms & Medical Tests")
    col1, col2 = st.columns(2)
    
    with col1:
        cp_str = st.selectbox("**Chest Pain Type**", 
                              options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
                              index=0)
        input_values['cp'] = CHEST_PAIN_MAP[cp_str]
        
        restecg_str = st.selectbox("**Resting ECG**", 
                                   options=['Normal', 'ST-T Abnormality', 'LV Hypertrophy'],
                                   index=1)
        input_values['restecg'] = RESTECG_MAP[restecg_str]
        
        exang_str = st.selectbox("**Exercise Induced Angina**", options=['No', 'Yes'], index=0)
        input_values['exang'] = EXANG_MAP[exang_str]
    
    with col2:
        oldpeak = st.number_input("**ST Depression**", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        input_values['oldpeak'] = oldpeak
        
        slope_str = st.selectbox("**ST Slope**", 
                                options=['Upsloping', 'Flat', 'Downsloping'],
                                index=1)
        input_values['slope'] = SLOPE_MAP[slope_str]
        
        ca = st.number_input("**Number of Major Vessels** (0-3)", min_value=0, max_value=3, value=0, step=1)
        input_values['ca'] = ca
        
        thal_str = st.selectbox("**Thalassemia**", 
                               options=['Normal', 'Fixed Defect', 'Reversible Defect', 'Other'],
                               index=2)
        input_values['thal'] = THAL_MAP[thal_str]
    
    if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
        try:
            # Get model's expected feature names
            if hasattr(model, 'feature_names_in_'):
                expected_features = [str(f) for f in model.feature_names_in_]  # Ensure all are strings
            else:
                st.error("❌ Model does not have feature_names_in_ attribute. Cannot determine expected features.")
                st.stop()
            
            # Map input values to ORIGINAL feature names (before one-hot encoding)
            # These are the features that will be fed into the feature engineering pipeline
            original_feature_mapping = {
                'age': 'Age',
                'sex': 'Sex',
                'trestbps': 'RestingBP',
                'chol': 'Cholesterol',
                'fbs': 'FastingBS',
                'thalach': 'MaxHR',
                'exang': 'ExerciseAngina',
                'oldpeak': 'Oldpeak',
                'cp': 'ChestPainType',  # Will be one-hot encoded
                'restecg': 'RestingECG',  # Will be one-hot encoded
                'slope': 'ST_Slope',  # Will be one-hot encoded
                'ca': 'MajorVessel',  # Note: singular, not plural - but might be dropped/encoded
                'thal': 'Thalassemia'  # Might be one-hot encoded
            }
            
            # Create DataFrame with ORIGINAL feature names (before feature engineering)
            original_input_data = {}
            for app_key, app_value in input_values.items():
                if app_key in original_feature_mapping:
                    original_input_data[original_feature_mapping[app_key]] = app_value
            
            # Create DataFrame with original features
            input_data = pd.DataFrame([original_input_data])
            
            # Check if MajorVessel is expected by the model - if not, don't include it
            # The model might have been trained without MajorVessel or it was transformed
            if 'MajorVessel' in input_data.columns and 'MajorVessel' not in expected_features:
                # Check if any MajorVessel-related features are expected (like MajorVessel_0, MajorVessel_1, etc.)
                majorvessel_features = [f for f in expected_features if 'MajorVessel' in f or 'Major' in f]
                if not majorvessel_features:
                    # MajorVessel is not expected at all - remove it
                    input_data = input_data.drop(columns=['MajorVessel'], errors='ignore')
            
            # Ensure all column names are strings
            input_data.columns = input_data.columns.astype(str)
            
            # Apply feature engineering EXACTLY as done during training
            # Step 1: Encode categorical features (this creates one-hot encoded features)
            feature_engineer = FeatureEngineer(target_column=None)
            if encoders is not None:
                feature_engineer.encoders = encoders
            
            # Explicitly specify which columns should be one-hot encoded
            # These are categorical features that need one-hot encoding
            categorical_cols_to_encode = []
            for col in input_data.columns:
                if col in ['ChestPainType', 'RestingECG', 'ST_Slope', 'Thalassemia']:
                    categorical_cols_to_encode.append(col)
            
            # Convert categorical columns to object type so they're detected as categorical
            for col in categorical_cols_to_encode:
                if col in input_data.columns:
                    input_data[col] = input_data[col].astype(str)
            
            # Apply one-hot encoding explicitly for these categorical columns
            # Use pandas get_dummies directly to avoid the unique_count issue
            if categorical_cols_to_encode:
                for col in categorical_cols_to_encode:
                    if col in input_data.columns:
                        # Create one-hot encoded columns
                        dummies = pd.get_dummies(input_data[col], prefix=col, drop_first=True)
                        # Drop the original column and add one-hot encoded columns
                        input_data = pd.concat([input_data.drop(columns=[col]), dummies], axis=1)
            
            # Ensure all column names are strings after encoding
            input_data.columns = input_data.columns.astype(str)
            
            # Verify that original categorical columns were removed
            categorical_originals = ['ChestPainType', 'RestingECG', 'ST_Slope', 'Thalassemia']
            still_present = [col for col in categorical_originals if col in input_data.columns]
            if still_present:
                # Manually remove original categorical columns if they still exist
                input_data = input_data.drop(columns=still_present, errors='ignore')
            
            # Remove MajorVessel if it's not expected by the model
            # The model might have been trained without MajorVessel or it was transformed/removed
            if 'MajorVessel' in input_data.columns:
                # Check if MajorVessel or any MajorVessel-related features are expected
                majorvessel_expected = any('MajorVessel' in f or 'Major' in f for f in expected_features)
                if not majorvessel_expected:
                    # MajorVessel is not expected - remove it
                    input_data = input_data.drop(columns=['MajorVessel'], errors='ignore')
            
            # Ensure all column names are still strings
            input_data.columns = input_data.columns.astype(str)
            
            # Step 2: Scale features (using the fitted scaler)
            if scaler is not None:
                feature_engineer.scaler = scaler
                # Scale features - this preserves all columns and only scales numerical ones
                input_data = feature_engineer.scale_features(input_data, method='standard', fit=False)
            else:
                st.warning("⚠️ Scaler not found, but continuing with prediction...")
            
            # Ensure column names are still strings after scaling
            input_data.columns = input_data.columns.astype(str)
            
            # Convert expected features to strings
            expected_features_str = [str(f) for f in expected_features]
            
            # Create interaction features (if model expects them)
            if hasattr(model, 'feature_names_in_'):
                # Check if model expects interaction features
                interaction_features = [f for f in expected_features if '_x_' in f or '*' in f]
                if interaction_features:
                    input_data = feature_engineer.create_interaction_features(input_data, max_interactions=10)
                    # Ensure column names are still strings after interaction features
                    input_data.columns = input_data.columns.astype(str)
            
            # Final check: ensure all expected features are present and in correct order
            if hasattr(model, 'feature_names_in_'):
                # Convert expected_features to strings for comparison
                expected_features_str = [str(f) for f in expected_features]
                missing_features = [f for f in expected_features_str if f not in input_data.columns]
                
                if missing_features:
                    # Add missing features with default value 0
                    for feat in missing_features:
                        input_data[feat] = 0
                
                # Reorder columns to match expected order
                input_data = input_data[expected_features_str]
            
            # Final safety check: ensure all column names are strings before prediction
            input_data.columns = input_data.columns.astype(str)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = None
            
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_data)[0]
            
            # Display result
            st.markdown("---")
            if prediction == 1:
                st.markdown(f"""
                <div class='prediction-card'>
                    <h2 style='color: white; margin: 0;'>🔴 Heart Disease Detected</h2>
                    <p style='font-size: 2rem; font-weight: bold; margin: 1rem 0;'>High Risk</p>
                    <p style='font-size: 1.2rem; opacity: 0.9;'>Please consult with a healthcare professional</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-card-safe'>
                    <h2 style='color: white; margin: 0;'>🟢 No Heart Disease</h2>
                    <p style='font-size: 2rem; font-weight: bold; margin: 1rem 0;'>Low Risk</p>
                    <p style='font-size: 1.2rem; opacity: 0.9;'>Continue maintaining a healthy lifestyle</p>
                </div>
                """, unsafe_allow_html=True)
            
            if prediction_proba is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Heart Disease", f"{prediction_proba[0]*100:.1f}%")
                with col2:
                    st.metric("Heart Disease", f"{prediction_proba[1]*100:.1f}%")
                
                # Risk interpretation
                if prediction_proba[1] > 0.7:
                    st.warning("⚠️ **High Risk:** The model indicates a high probability of heart disease. Please consult with a healthcare professional.")
                elif prediction_proba[1] > 0.4:
                    st.info("ℹ️ **Moderate Risk:** Consider further medical evaluation.")
                else:
                    st.success("✅ **Low Risk:** The model indicates a low probability of heart disease.")
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            with st.expander("Debug Info"):
                st.write(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #999; font-size: 0.9rem;'>
    Heart Disease Prediction Model | Powered by Machine Learning<br>
    <small>⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.</small>
</p>
""", unsafe_allow_html=True)

# TODO: Review implementation
