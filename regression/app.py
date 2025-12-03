"""
Modern Streamlit App for Flight Delays Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from utils import load_model, MODELS_DIR
from feature_engineering import FeatureEngineer
from sklearn.preprocessing import LabelEncoder

# Modern page config
st.set_page_config(
    page_title="Flight Delays Prediction",
    page_icon="✈️",
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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
            MODELS_DIR / "best_regression_model.pkl",
            Path(__file__).parent / "models" / "best_regression_model.pkl",
            Path(__file__).parent.parent / "regression" / "models" / "best_regression_model.pkl",
            Path.cwd() / "regression" / "models" / "best_regression_model.pkl",
            Path.cwd() / "models" / "best_regression_model.pkl"
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
                model = load_model("best_regression_model.pkl")
            except:
                pass
        
        if model is None:
            return None, None, "Model file not found in any expected location."
        
        return model, None, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

@st.cache_resource
def load_feature_engineer():
    """Load feature engineer with encoders."""
    try:
        app_dir = Path(__file__).parent
        possible_paths = [
            MODELS_DIR / "feature_engineer.pkl",
            app_dir / "models" / "feature_engineer.pkl",
            app_dir.parent / "regression" / "models" / "feature_engineer.pkl"
        ]
        
        for fe_path in possible_paths:
            if fe_path.exists():
                try:
                    return joblib.load(fe_path), None
                except:
                    continue
        
        return FeatureEngineer(), "Feature engineer not saved. Using default settings."
    except Exception as e:
        return FeatureEngineer(), f"Could not load feature engineer: {str(e)}"

def encode_categorical_string(value, feature_name, feature_engineer):
    """Encode a categorical string value to numeric."""
    if not hasattr(feature_engineer, 'encoders') or feature_name not in feature_engineer.encoders:
        # Create new encoder if not found
        le = LabelEncoder()
        # For now, just return a simple hash-based encoding
        return hash(str(value)) % 1000
    else:
        le = feature_engineer.encoders[feature_name]
        try:
            if str(value) in le.classes_:
                return le.transform([str(value)])[0]
            else:
                # Return first class if unseen
                return le.transform([le.classes_[0]])[0]
        except:
            return 0

# Load model and feature engineer
model, fe_error, model_error = load_deployment_model()
feature_engineer, fe_warning = load_feature_engineer()

# Error page if model not found
if model_error or model is None:
    st.error("❌ Model Not Found")
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2 style='color: #667eea;'>Model Not Available</h2>
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
           cd regression
           python src/main.py
           ```
        
        2. **Ensure the model file exists:**
           - Check: `regression/models/best_regression_model.pkl`
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
    'AIRLINE': {'type': 'string', 'label': 'Airline Code', 'default': 'AA'},
    'ORIGIN_AIRPORT': {'type': 'string', 'label': 'Origin Airport', 'default': 'JFK'},
    'DESTINATION_AIRPORT': {'type': 'string', 'label': 'Destination Airport', 'default': 'LAX'},
    'MONTH': {'type': 'number', 'label': 'Month', 'default': datetime.now().month, 'min': 1, 'max': 12},
    'DAY_OF_WEEK': {'type': 'number', 'label': 'Day of Week (1=Mon, 7=Sun)', 'default': datetime.now().weekday() + 1, 'min': 1, 'max': 7},
    'SCHEDULED_DEPARTURE': {'type': 'number', 'label': 'Scheduled Departure (HHMM)', 'default': 800, 'min': 0, 'max': 2359},
    'DISTANCE': {'type': 'number', 'label': 'Distance (miles)', 'default': 2500, 'min': 0},
    'DEPARTURE_DELAY': {'type': 'number', 'label': 'Current Departure Delay (minutes)', 'default': 0, 'min': -60, 'max': 300}
}

# All required features for model (with defaults)
FEATURE_COLUMNS = [
    'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
    'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME',
    'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME',
    'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
    'DIVERTED', 'CANCELLED', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
    'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'CANCELLATION_REASON_B', 'CANCELLATION_REASON_C',
    'CANCELLATION_REASON_D'
]

CATEGORICAL_FEATURES = ['AIRLINE', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']

def get_defaults():
    """Get default values for all features."""
    now = datetime.now()
    return {
        'YEAR': now.year,
        'MONTH': now.month,
        'DAY': now.day,
        'DAY_OF_WEEK': now.weekday() + 1,
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

# Modern UI
st.markdown("<h1>✈️ Flight Delays Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem;'>Enter flight details to predict arrival delay</p>", unsafe_allow_html=True)

# Input form with modern design
with st.container():
    defaults = get_defaults()
    input_values = defaults.copy()
    
    # Common airlines and airports
    COMMON_AIRLINES = ['AA', 'DL', 'UA', 'WN', 'AS', 'B6', 'F9', 'NK', 'G4', 'SY']
    COMMON_AIRPORTS = [
        'ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MIA',
        'CLT', 'PHX', 'EWR', 'MCO', 'MSP', 'DTW', 'PHL', 'LGA', 'BWI', 'BOS',
        'IAD', 'SLC', 'MDW', 'DCA', 'HNL', 'PDX', 'STL', 'MCI', 'AUS', 'SAN'
    ]
    
    # Key features input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛫 Flight Information")
        
        # Airline dropdown
        airline_options = COMMON_AIRLINES + ['Other (Enter Code)']
        airline_index = COMMON_AIRLINES.index(defaults['AIRLINE']) if defaults['AIRLINE'] in COMMON_AIRLINES else 0
        airline_selection = st.selectbox("**Airline Code**", options=airline_options, index=airline_index, help="Select airline or enter custom code")
        
        if airline_selection == 'Other (Enter Code)':
            airline_custom = st.text_input("Enter Airline Code", value="", max_chars=3, help="e.g., AA, DL, UA")
            input_values['AIRLINE'] = airline_custom.upper() if airline_custom else defaults['AIRLINE']
        else:
            input_values['AIRLINE'] = airline_selection
        
        # Origin airport dropdown
        origin_options = COMMON_AIRPORTS + ['Other (Enter Code)']
        origin_index = COMMON_AIRPORTS.index(defaults['ORIGIN_AIRPORT']) if defaults['ORIGIN_AIRPORT'] in COMMON_AIRPORTS else 0
        origin_selection = st.selectbox("**Origin Airport**", options=origin_options, index=origin_index, help="Select airport or enter custom code")
        
        if origin_selection == 'Other (Enter Code)':
            origin_custom = st.text_input("Enter Origin Airport Code", value="", max_chars=3, help="e.g., JFK, LAX, SFO")
            input_values['ORIGIN_AIRPORT'] = origin_custom.upper() if origin_custom else defaults['ORIGIN_AIRPORT']
        else:
            input_values['ORIGIN_AIRPORT'] = origin_selection
        
        # Destination airport dropdown
        dest_options = COMMON_AIRPORTS + ['Other (Enter Code)']
        dest_index = COMMON_AIRPORTS.index(defaults['DESTINATION_AIRPORT']) if defaults['DESTINATION_AIRPORT'] in COMMON_AIRPORTS else COMMON_AIRPORTS.index('LAX') if 'LAX' in COMMON_AIRPORTS else 0
        dest_selection = st.selectbox("**Destination Airport**", options=dest_options, index=dest_index, help="Select airport or enter custom code")
        
        if dest_selection == 'Other (Enter Code)':
            dest_custom = st.text_input("Enter Destination Airport Code", value="", max_chars=3, help="e.g., LAX, SFO, JFK")
            input_values['DESTINATION_AIRPORT'] = dest_custom.upper() if dest_custom else defaults['DESTINATION_AIRPORT']
        else:
            input_values['DESTINATION_AIRPORT'] = dest_selection
    
    with col2:
        st.markdown("### 📅 Schedule & Timing")
        month = st.number_input("**Month**", min_value=1, max_value=12, value=defaults['MONTH'], step=1)
        input_values['MONTH'] = month
        
        day_of_week = st.number_input("**Day of Week** (1=Monday, 7=Sunday)", min_value=1, max_value=7, value=defaults['DAY_OF_WEEK'], step=1)
        input_values['DAY_OF_WEEK'] = day_of_week
        
        scheduled_dep = st.number_input("**Scheduled Departure** (HHMM)", min_value=0, max_value=2359, value=defaults['SCHEDULED_DEPARTURE'], step=1)
        input_values['SCHEDULED_DEPARTURE'] = scheduled_dep
        
        distance = st.number_input("**Distance** (miles)", min_value=0, value=defaults['DISTANCE'], step=1)
        input_values['DISTANCE'] = distance
    
    st.markdown("### ⏱️ Current Status")
    departure_delay = st.number_input("**Current Departure Delay** (minutes)", value=defaults['DEPARTURE_DELAY'], step=1, 
                                     help="Enter 0 if flight hasn't departed yet")
    input_values['DEPARTURE_DELAY'] = departure_delay
    
    if st.button("🔮 Predict Arrival Delay", type="primary", use_container_width=True):
        try:
            # Ensure all features present
            for feature in FEATURE_COLUMNS:
                if feature not in input_values:
                    input_values[feature] = defaults.get(feature, 0)
            
            # Create DataFrame
            input_df = pd.DataFrame([input_values])[FEATURE_COLUMNS]
            
            # Encode categorical strings to numeric
            for col in CATEGORICAL_FEATURES:
                if col in input_df.columns and input_df[col].dtype == 'object':
                    input_df[col] = input_df[col].apply(lambda x: encode_categorical_string(x, col, feature_engineer))
            
            # Convert all to numeric
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            
            # Apply scaling if available
            if hasattr(feature_engineer, 'scaler') and feature_engineer.scaler:
                try:
                    input_df = feature_engineer.scale_features(input_df, method='standard', fit=False)
                except:
                    pass
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display result with modern design
            st.markdown("---")
            if prediction > 0:
                st.markdown(f"""
                <div class='prediction-card'>
                    <h2 style='color: white; margin: 0;'>⚠️ Delayed</h2>
                    <p style='font-size: 3rem; font-weight: bold; margin: 1rem 0;'>{prediction:.1f} minutes</p>
                    <p style='font-size: 1.2rem; opacity: 0.9;'>The flight is predicted to arrive late</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction < 0:
                st.markdown(f"""
                <div class='prediction-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                    <h2 style='color: white; margin: 0;'>✅ Early</h2>
                    <p style='font-size: 3rem; font-weight: bold; margin: 1rem 0;'>{abs(prediction):.1f} minutes</p>
                    <p style='font-size: 1.2rem; opacity: 0.9;'>The flight is predicted to arrive early</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                    <h2 style='color: white; margin: 0;'>🕐 On Time</h2>
                    <p style='font-size: 3rem; font-weight: bold; margin: 1rem 0;'>0 minutes</p>
                    <p style='font-size: 1.2rem; opacity: 0.9;'>The flight is predicted to arrive on time</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            with st.expander("Debug Info"):
                st.write(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 0.9rem;'>Flight Delays Prediction Model | Powered by Machine Learning</p>", unsafe_allow_html=True)
