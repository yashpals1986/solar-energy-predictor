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
# Download Model
# ===============================
@st.cache_resource
def load_model():

    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    url = "https://huggingface.co/ysuwansia/solar/resolve/main/final_production_model.pkl"

    try:
        with st.spinner("⏳ Downloading model..."):

            r = requests.get(url, stream=True)
            r.raise_for_status()

            with open("model.pkl", "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        return joblib.load("model.pkl")

    except Exception as e:
        st.error(f"❌ Model download failed: {e}")
        return None


model = load_model()


# ===============================
# Load Feature Names
# ===============================
if model and hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    st.error("❌ Model has no feature names")
    st.stop()


# ===============================
# Header
# ===============================
st.markdown("""
<h1 style="text-align:center;">☀️ Solar Energy Production Predictor</h1>
<p style="text-align:center;color:gray;">
Random Forest Based Forecast System
</p>
""", unsafe_allow_html=True)


# ===============================
# Sidebar Inputs (Important Features)
# ===============================
st.sidebar.header("⚙️ Main Input Parameters")


hour = st.sidebar.slider("Hour", 0, 23, 12)

temperature = st.sidebar.number_input("Temperature (°C)", -10.0, 60.0, 25.0)

aerosol = st.sidebar.slider("Aerosol Optical Depth", 0.0, 2.0, 0.6)

azimuth = st.sidebar.slider("Azimuth (°)", 0, 360, 180)

azimuth_bin = st.sidebar.slider("Azimuth Bin (°)", 0.0, 360.0, 180.0)

zenith = st.sidebar.slider("Zenith (°)", 0, 90, 45)

zenith_bin = st.sidebar.slider("Zenith Bin (°)", 0.0, 360.0, 180.0)

elevation = st.sidebar.number_input("Elevation (°)", -90.0, 90.0, 20.0)

best_tilt = st.sidebar.slider("Best Tilt (°)", 0.0, 360.0, 180.0)


# ===============================
# Default Values (Low Importance)
# ===============================
DEFAULTS = {

    "Pressure": 987,
    "Wind Speed": 3,
    "WeekOfYear": 25,
    "Year": 2017,
    "Wind Direction": 250,
    "Dew Point": 10,
    "Relative Humidity": 35,
    "Precipitable Water": 3,
    "Cloud Type": 2,
    "DayOfYear": 200,
    "Day": 4,
    "DayOfWeek": 4,
    "Month": 5
}


# ===============================
# Predict
# ===============================
if st.sidebar.button("🔮 Predict"):

    if not model:
        st.error("Model not loaded")
        st.stop()


    # Main Inputs
    input_data = {

        "azimuth": [azimuth],
        "Azimuth_Bin": [azimuth_bin],
        "Hour": [hour],
        "Best_Tilt": [best_tilt],
        "elevation": [elevation],
        "zenith": [zenith],
        "Zenith_Bin": [zenith_bin],
        "Temperature": [temperature],
        "Aerosol Optical Depth": [aerosol]
    }


    # Add Default Inputs
    for key, value in DEFAULTS.items():
        input_data[key] = [value]


    # Check Missing
    missing = set(FEATURES) - set(input_data.keys())

    if missing:
        st.error(f"❌ Missing features: {missing}")
        st.stop()


    # Create DataFrame
    df = pd.DataFrame(input_data)[FEATURES]


    # Predict
    prediction = model.predict(df)[0]


    # ===============================
    # Output
    # ===============================
    st.success("✅ Prediction Successful")

    col1, col2 = st.columns(2)

    col1.metric("⚡ Energy (kWh)", f"{prediction:.2f}")
    col2.metric("🕒 Hour", f"{hour}:00")


# ===============================
# Footer
# ===============================
st.markdown("---")

st.markdown("""
<p style="text-align:center;color:gray;">
Developed by Yashpal Suwansia | IIT Bombay 2010
</p>
""", unsafe_allow_html=True)

st.markdown("---")
