import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime




# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Solar Energy Predictor",
    page_icon="☀️",
    layout="wide"
)
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

# ===============================
# Header
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;">☀️ Solar Energy Production Predictor</h1>
    <p style="text-align:center;color:gray;">
    RF Model | 93% R² | Rajasthan Dataset
    </p>
    """,
    unsafe_allow_html=True
)
# ===============================
# Sidebar Inputs
# ===============================
   

st.sidebar.header("⚙️ Input Parameters")
hour = st.sidebar.number_input("hour", 0.0, 23, 12)
temperature = st.sidebar.number_input("temperature(degree)", 0.0, 50, 25)
elevation = st.sidebar.number_input("elevation", -90.0, 90.0, 0.0)
aerosol = st.sidebar.slider("Aerosol Optical Depth (°)", 0.0, 1.5, 0.7)
azimuth = st.sidebar.slider("azimuth", 0, 59, 0)
azimuth_bin = st.sidebar.slider("azimuth_bin (°)", 0.0, 180.0, 30.0)
zenith = st.sidebar.slider("zenith", 0, 23, 12)
zenith_bin = st.sidebar.slider("Zenith_Bin (°)", 0.0, 360.0, 180.0)
best_tilt = st.sidebar.slider("best_tilt", 0.0, 360.0, 180.0, step=0.5)

# ===============================
# Prediction
# ===============================
if st.sidebar.button("🔮 Predict"):

    # Time
    time_display = f"{hour:02d}:{minute:02d}"

    # Defaults
    input_dict = DEFAULTS.copy()

    # Update
    input_dict.update({
        'temperature': [temperature],
        'Aerosol Optical Depth': [aerosol],
        'zenith': [zenith],
        'azimuth': [azimuth],
        'elevation': [elevation],
        'best_tilt': [best_tilt],
        'azimuth_bin': [azimuth_bin],
        'zenith_bin': [zenith_bin],
        'hour': [hour]
    })


    
    # Predict
    input_df = pd.DataFrame([input_dict])[FEATURES]
    pred = model.predict(input_df)[0]

    # ===============================
    # KPIs
    # ===============================
    c1, c2, c3 = st.columns(3)

    c1.metric("⚡ Energy (kWh)", f"{pred:.2f}")
    c2.metric("🕒 Time", time_display)
    
# ===============================
# Footer - PERFECTLY CENTERED
# ===============================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 14px;'>"
        "**Developed by Yashpal Suwansia | IIT Bombay 2010**"
        "</p>",
        unsafe_allow_html=True
    )
st.markdown("---")





