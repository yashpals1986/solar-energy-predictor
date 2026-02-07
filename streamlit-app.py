import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime

# Page config
st.set_page_config(page_title="Solar Energy Predictor", page_icon="☀️", layout="wide")

@st.cache_resource
def download_model_from_huggingface():
    """Download model from Hugging Face"""
    if os.path.exists('model.pkl'):
        return joblib.load('model.pkl')
    
    url = "https://huggingface.co/ysuwansia/solar/resolve/main/final_production_model.pkl"
    
    with st.spinner('⏳ Downloading model (first time only)...'):
        response = requests.get(url, stream=True)
        with open('model.pkl', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return joblib.load('model.pkl')

model = download_model_from_huggingface()

# ✅ CHECK WHAT FEATURES MODEL EXPECTS
try:
    expected_features = model.feature_names_in_
    st.sidebar.write("**Model expects these features:**")
    st.sidebar.write(list(expected_features))
except:
    st.sidebar.warning("Could not extract feature names from model")

# Header
st.title("☀️ Solar Energy Prediction System")
st.markdown("Predict solar energy generation using Machine Learning")

# Sidebar info
st.sidebar.title("📊 Model Info")
st.sidebar.metric("Model", "Random Forest")
st.sidebar.metric("Tuned Test MAE", "10.83")
st.sidebar.metric("Tuned Accuracy", "93.4%")

# Main prediction interface
st.header("🔮 Energy Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌦️ Weather")
    temperature = st.slider("Temperature (°C)", -20.0, 50.0, 25.0)
    
with col2:
    st.subheader("🌩️ Atmosphere")
    aerosol = st.slider("Aerosol Optical Depth", 0.0, 2.0, 0.1)
    hour = st.slider("Hour", 0, 23, 12)
    minute = st.slider("Minute", 0, 59, 0)
     
with col3:
    st.subheader("🔆 Solar Position")
    zenith = st.slider("Zenith Angle (°)", 0.0, 90.0, 30.0)
    azimuth = st.slider("Azimuth Angle (°)", 0.0, 360.0, 180.0)
    elevation = st.slider("Elevation Angle (°)", 0.0, 90.0, 60.0)
    
    st.subheader("⚙️ Panel Config")
    best_tilt = st.slider("Best Tilt (°)", 0.0, 90.0, 30.0)
    azimuth_bin = st.selectbox("Azimuth Bin", list(range(8)))
    zenith_bin = st.selectbox("Zenith Bin", list(range(9)))


# Predict button
if st.button("🔮 Predict Energy", type="primary", use_container_width=True):
    # ✅ PROPER WAY: Create input with ALL features model expects
    # Based on your training notebook, the model likely expects these features:
    
    input_data = pd.DataFrame({
        'Clearsky DHI': [0],  # Default value - adjust as needed
        'Clearsky DNI': [800],  # Default value
        'Clearsky GHI': [600],  # Default value
        'Cloud Type': [0],  # Default value
        'Dew Point': [15],  # Default value
        'Fill Flag': [0],  # Default value
        'Relative Humidity': [50],  # Default value
        'Solar Zenith Angle': [zenith],
        'Temperature': [temperature],
        'Pressure': [1013],  # Default value
        'Wind Direction': [180],  # Default value
        'Wind Speed': [5],  # Default value
        'Precipitable Water': [1.5],  # Default value
        'Ozone': [300],  # Default value
        'Aerosol Optical Depth': [aerosol],
        'azimuth': [azimuth],
        'elevation': [elevation],
        'zenith': [zenith],
        'Best_Tilt': [best_tilt],
        'Azimuth_Bin': [azimuth_bin],
        'Zenith_Bin': [zenith_bin],
        'Hour': [hour],
        'Minute': [minute],
    })
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Energy", f"{prediction:.2f} Wh/m²")
        col2.metric("For 1kW System", f"{prediction * 0.001:.2f} Wh")
        
        if prediction > 800:
            quality = "Excellent ⭐⭐⭐"
        elif prediction > 500:
            quality = "Good ⭐⭐"
        elif prediction > 200:
            quality = "Fair ⭐"
        else:
            quality = "Poor ❌"
        col3.metric("Quality", quality)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Input data columns:", list(input_data.columns))
        if hasattr(model, 'feature_names_in_'):
            st.write("Model expects:", list(model.feature_names_in_))

# Footer
st.markdown("---")
st.markdown("**Developed by Yashpal Suwansia** | Powered by Streamlit & Scikit-learn")
