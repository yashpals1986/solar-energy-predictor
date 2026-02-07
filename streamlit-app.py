import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime

# Page config
st.set_page_config(page_title="Solar Energy Predictor", page_icon="☀️", layout="wide")

# At the top, replace the download function:

@st.cache_resource
def download_model_from_huggingface():
    """Download model from Hugging Face"""
    if os.path.exists('model.pkl'):
        return joblib.load('model.pkl')
    
    # ⚠️ REPLACE WITH YOUR HUGGING FACE URL
    url = "https://huggingface.co/ysuwansia/solar/resolve/main/final_production_model.pkl"
    
    with st.spinner('⏳ Downloading model (first time only)...'):
        response = requests.get(url, stream=True)
        with open('model.pkl', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return joblib.load('model.pkl')

model = download_model_from_huggingface()


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
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
    pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0)
    wind_direction = st.slider("Wind Direction (°)", 0, 360, 180)
    
with col2:
    st.subheader("🌩️ Atmosphere")
    aerosol = st.slider("Aerosol Optical Depth", 0.0, 2.0, 0.1)
         
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
    # Create input dataframe
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Aerosol Optical Depth': [aerosol],
        'Dew Point': [dew_point],
        'Cloud Type': [cloud_type],
        'Relative Humidity': [humidity],
        'Pressure': [pressure],
        'Wind Speed': [wind_speed],
        'Wind Direction': [wind_direction],
        'Precipitable Water': [precipitable_water],
        'zenith': [zenith],
        'azimuth': [azimuth],
        'elevation': [elevation],
        'Best_Tilt': [best_tilt],
        'Azimuth_Bin': [azimuth_bin],
        'Zenith_Bin': [zenith_bin],
        })
    
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

# Footer
st.markdown("---")
st.markdown("**Developed by Yashpal Suwansia** | Powered by Streamlit & Scikit-learn")


