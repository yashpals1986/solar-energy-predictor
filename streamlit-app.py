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
# Model Download
# ===============================
@st.cache_resource
def download_model_from_huggingface():

    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    url = "https://huggingface.co/ysuwansia/solar/resolve/main/final_production_model.pkl"

    try:
        with st.spinner("⏳ Downloading model..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open("model.pkl", "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)

        return joblib.load("model.pkl")

    except Exception as e:
        st.error(f"Model download failed: {e}")
        return None


model = download_model_from_huggingface()


# ===============================
# Feature List (VERY IMPORTANT)
# ===============================
FEATURES = [
    "temperature",
    "Aerosol Optical Depth",
    "zenith",
    "azimuth",
    "elevation",
    "best_tilt",
    "azimuth_bin",
    "zenith_bin",
    "hour"
]


# ===============================
# Header
# ===============================
st.markdown("""
<h1 style="text-align:center;">☀️ Solar Energy Production Predictor</h1>
<p style="text-align:center;color:gray;">
RF Model | 93% R² | Rajasthan Dataset
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


# ===============================
# Prediction
# ===============================
if st.sidebar.button("🔮 Predict"):

    if model is None:
        st.error("Model not loaded!")
        st.stop()

    # Time display
    time_display = f"{hour:02d}:{minute:02d}"

    # Input dictionary
    input_data = {
        "temperature": [temperature],
        "Aerosol Optical Depth": [aerosol],
        "zenith": [zenith],
        "azimuth": [azimuth],
        "elevation": [elevation],
        "best_tilt": [best_tilt],
        "azimuth_bin": [azimuth_bin],
        "zenith_bin": [zenith_bin],
        "hour": [hour]
    }

    # DataFrame
    input_df = pd.DataFrame(input_data)[FEATURES]

    # Prediction
    prediction = model.predict(input_df)[0]


    # ===============================
    # KPIs
    # ===============================
    c1, c2 = st.columns(2)

    c1.metric("⚡ Energy (kWh)", f"{prediction:.2f}")
    c2.metric("🕒 Time", time_display)


# ===============================
# Footer
# ===============================
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <p style='text-align:center;color:gray;font-size:14px;'>
    Developed by Yashpal Suwansia | IIT Bombay 2010
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")
