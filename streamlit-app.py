import streamlit as st
import joblib
import pandas as pd
import requests
import os


# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Solar Energy Predictor",
    page_icon="☀️",
    layout="wide"
)


# ===============================
# Download & Load Model
# ===============================
@st.cache_resource
def download_model_from_huggingface():

    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    url = "https://huggingface.co/ysuwansia/solar/resolve/main/final_production_model.pkl"

    try:
        with st.spinner("⏳ Downloading model (first time only)..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open("model.pkl", "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)

        return joblib.load("model.pkl")

    except Exception as e:
        st.error(f"❌ Model download failed: {e}")
        return None


model = download_model_from_huggingface()


# ===============================
# Load Feature Names From Model
# ===============================
if model is not None and hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    st.error("❌ Model does not contain feature names.")
    st.stop()


# ===============================
# Header
# ===============================
st.markdown("""
<h1 style="text-align:center;">☀️ Solar Energy Production Predictor</h1>
<p style="text-align:center;color:gray;">
RF Model | Rajasthan Dataset
</p>
""", unsafe_allow_html=True)


# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("⚙️ Input Parameters")

hour = st.sidebar.number_input("Hour", 0, 23, 12)
minute = st.sidebar.number_input("Minute", 0, 59, 0)

temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
elevation = st.sidebar.number_input("Elevation", -90.0, 90.0, 0.0)

aerosol = st.sidebar.slider("Aerosol Optical Depth", 0.0, 1.5, 0.7)

azimuth = st.sidebar.slider("Azimuth", 0, 360, 180)
azimuth_bin = st.sidebar.slider("Azimuth Bin", 0.0, 180.0, 30.0)

zenith = st.sidebar.slider("Zenith", 0, 90, 45)
zenith_bin = st.sidebar.slider("Zenith Bin", 0.0, 360.0, 180.0)

best_tilt = st.sidebar.slider("Best Tilt", 0.0, 360.0, 180.0)


# =============
