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

st.title("📈 Regression Model Deployment")
st.markdown("---")

# Load model
@st.cache_resource
def load_deployment_model():
    """Load the trained model."""
    try:
        model_files = list(MODELS_DIR.glob("*.pkl"))
        if not model_files:
            return None, "No model found. Please train a model first."
        
        # Try to load the best model
        model_file = MODELS_DIR / "best_regression_model.pkl"
        if model_file.exists():
            model = load_model("best_regression_model.pkl")
            return model, None
        else:
            # Load the first available model
            model = joblib.load(model_files[0])
            return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

model, error = load_deployment_model()

if error:
    st.error(error)
    st.info("Please run the training pipeline first using: `python classification/src/main.py`")
else:
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for input
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("This is a regression model for predicting continuous values based on input features.")
    
    # Main content
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction"])
    
    with tab1:
        st.header("Single Prediction")
        st.markdown("Enter feature values to get a prediction:")
        
        # Create input fields (this is a template - update based on your features)
        st.warning("⚠️ Update this section with your actual feature inputs based on your dataset.")
        
        # Example inputs (replace with your actual features)
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.number_input("Feature 1", value=0.0)
            feature2 = st.number_input("Feature 2", value=0.0)
        
        with col2:
            feature3 = st.number_input("Feature 3", value=0.0)
            feature4 = st.number_input("Feature 4", value=0.0)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Feature1': [feature1],
            'Feature2': [feature2],
            'Feature3': [feature3],
            'Feature4': [feature4]
        })
        
        if st.button("🔮 Predict", type="primary"):
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display results
                st.success(f"**Predicted Value:** {prediction:.4f}")
                
                # Show prediction with confidence interval (if applicable)
                st.info(f"The model predicts: **{prediction:.2f}**")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Make sure the input features match the model's expected features.")
    
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with features to get predictions for multiple samples:")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Preview:")
                st.dataframe(df.head())
                
                if st.button("🔮 Predict Batch", type="primary"):
                    try:
                        predictions = model.predict(df)
                        df['Prediction'] = predictions
                        
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
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Model info
    with st.expander("ℹ️ Model Details"):
        st.write(f"Model Type: {type(model).__name__}")
        if hasattr(model, 'feature_importances_'):
            st.write("This model supports feature importance analysis.")
        st.write("This is a regression model that predicts continuous values.")

