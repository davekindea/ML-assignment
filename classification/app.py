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

# Most important features only
KEY_FEATURES = {
    'age': {'type': 'number', 'label': 'Age', 'default': 52, 'min': 0, 'max': 120},
    'sex': {'type': 'string', 'label': 'Sex', 'default': 'Male', 'options': ['Male', 'Female']},
    'cp': {'type': 'string', 'label': 'Chest Pain Type', 'default': 'Typical Angina', 
           'options': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']},
    'trestbps': {'type': 'number', 'label': 'Resting Blood Pressure (mmHg)', 'default': 125, 'min': 0, 'max': 300},
    'chol': {'type': 'number', 'label': 'Cholesterol (mg/dl)', 'default': 212, 'min': 0, 'max': 600},
    'thalach': {'type': 'number', 'label': 'Max Heart Rate (bpm)', 'default': 168, 'min': 0, 'max': 250},
    'exang': {'type': 'string', 'label': 'Exercise Induced Angina', 'default': 'No', 'options': ['No', 'Yes']},
    'oldpeak': {'type': 'number', 'label': 'ST Depression', 'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.1}
}

# Feature mappings
CHEST_PAIN_MAP = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
SEX_MAP = {'Male': 1, 'Female': 0}
EXANG_MAP = {'Yes': 1, 'No': 0}

# All required features with defaults
REQUIRED_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Modern UI
st.markdown("<h1>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem;'>Enter patient information to predict heart disease risk</p>", unsafe_allow_html=True)

# Input form
with st.container():
    input_values = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Patient Demographics")
        age = st.number_input("**Age** (years)", min_value=0, max_value=120, value=52, step=1)
        input_values['age'] = age
        
        sex_str = st.selectbox("**Sex**", options=['Male', 'Female'], index=0)
        input_values['sex'] = SEX_MAP[sex_str]
        
        st.markdown("### 💓 Vital Signs")
        trestbps = st.number_input("**Resting Blood Pressure** (mmHg)", min_value=0, max_value=300, value=125, step=1)
        input_values['trestbps'] = trestbps
        
        chol = st.number_input("**Cholesterol** (mg/dl)", min_value=0, max_value=600, value=212, step=1)
        input_values['chol'] = chol
        
        thalach = st.number_input("**Max Heart Rate** (bpm)", min_value=0, max_value=250, value=168, step=1)
        input_values['thalach'] = thalach
    
    with col2:
        st.markdown("### 🩺 Symptoms & Tests")
        cp_str = st.selectbox("**Chest Pain Type**", 
                              options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
                              index=0)
        input_values['cp'] = CHEST_PAIN_MAP[cp_str]
        
        fbs = st.selectbox("**Fasting Blood Sugar > 120**", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
        input_values['fbs'] = fbs
        
        restecg = st.selectbox("**Resting ECG**", 
                               options=[0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
                               index=1)
        input_values['restecg'] = restecg
        
        exang_str = st.selectbox("**Exercise Induced Angina**", options=['No', 'Yes'], index=0)
        input_values['exang'] = EXANG_MAP[exang_str]
        
        oldpeak = st.number_input("**ST Depression**", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        input_values['oldpeak'] = oldpeak
    
    # Defaults for other required features
    input_values['slope'] = 2
    input_values['ca'] = 0
    input_values['thal'] = 3
    
    if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
        try:
            # Create input dataframe
            input_data = pd.DataFrame([input_values])[REQUIRED_FEATURES]
            
            # Apply feature engineering
            feature_engineer = FeatureEngineer(target_column=None)
            if scaler is not None:
                feature_engineer.scaler = scaler
            if encoders is not None:
                feature_engineer.encoders = encoders
            
            # Encode categorical
            input_data = feature_engineer.encode_categorical(input_data, method='auto')
            
            # Scale features
            if feature_engineer.scaler is None:
                st.error("❌ Preprocessing scaler not found.")
                st.stop()
            
            input_data = feature_engineer.scale_features(input_data, method='standard', fit=False)
            
            # Create interaction features
            input_data = feature_engineer.create_interaction_features(input_data, max_interactions=10)
            
            # Select expected features
            expected_features = model.feature_names_in_
            missing_features = [f for f in expected_features if f not in input_data.columns]
            if missing_features:
                st.error(f"❌ Missing features: {', '.join(missing_features)}")
                st.stop()
            
            input_data = input_data[expected_features]
            
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
