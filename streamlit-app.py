import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Solar Energy Predictor",
    page_icon="☀️",
    layout="wide"
)

# ===============================
# Load Resources
# ===============================
@st.cache_resource
def load_resources():
    model = joblib.load("xgboost_v1.0.pkl")
    train_df = pd.read_csv("train.csv")

    ALL_FEATURES = [
        'Temperature', 'Aerosol Optical Depth', 'Clearsky DNI', 'Dew Point', 'Cloud Type',
        'Clearsky GHI', 'DHI', 'Clearsky DHI', 'DNI', 'Relative Humidity',
        'Pressure', 'Wind Speed', 'Wind Direction', 'Precipitable Water', 'zenith',
        'azimuth', 'elevation', 'Best_Tilt', 'Azimuth_Bin', 'Zenith_Bin',
        'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 'WeekOfYear'
    ]

    # Defaults (Median)
    DEFAULT_DICT = train_df[ALL_FEATURES].median().to_dict()

    # Error stats
    train_df["Predicted"] = model.predict(train_df[ALL_FEATURES])
    train_df["Error"] = abs(train_df["Predicted"] - train_df["Predicted_Energy"])
    train_df["Error_Pct"] = (train_df["Error"] / (train_df["Predicted_Energy"] + 0.1)) * 100

    ERROR_BY_HOUR = train_df.groupby("Hour")["Error"].mean().to_dict()
    ERROR_PCT_BY_HOUR = train_df.groupby("Hour")["Error_Pct"].mean().to_dict()

    return model, train_df, ALL_FEATURES, DEFAULT_DICT, ERROR_BY_HOUR, ERROR_PCT_BY_HOUR

model, train_df, FEATURES, DEFAULTS, ERR_HOUR, ERR_PCT = load_resources()

# ===============================
# Header
# ===============================
st.markdown(
    """
    <h1 style="text-align:center;">☀️ Solar Energy Production Predictor</h1>
    <p style="text-align:center;color:gray;">
    XGBoost Model | 93% R² | Rajasthan Dataset
    </p>
    """,
    unsafe_allow_html=True
)

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("⚙️ Input Parameters")

ghi = st.sidebar.number_input("Clearsky GHI (W/m²)", 0.0, 1200.0, 800.0)
dni = st.sidebar.number_input("Clearsky DNI (W/m²)", 0.0, 1200.0, 850.0)
dhi = st.sidebar.number_input("Clearsky DHI (W/m²)", 0.0, 544.0, 120.0)

tilt = st.sidebar.slider("Best Tilt (°)", 0.0, 90.0, 30.0)

hour = st.sidebar.slider("Hour", 0, 23, 12)
minute = st.sidebar.slider("Minute", 0, 59, 0)

zenith = st.sidebar.slider("Zenith (°)", 0.0, 180.0, 30.0)
azimuth = st.sidebar.slider("Azimuth (°)", 0.0, 360.0, 180.0)

azimuth_bin = st.sidebar.slider("Azimuth Bin", 0.0, 360.0, 180.0, step=0.5)
zenith_bin = st.sidebar.slider("Zenith Bin", 0.0, 180.0, 30.0, step=2.0)

elevation = st.sidebar.slider("Elevation (°)", -90.0, 90.0, 60.0, step=0.5)


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
        "Clearsky GHI": ghi,
        "Clearsky DNI": dni,
        "Clearsky DHI": dhi,
        "Best_Tilt": tilt,
        "Hour": hour,
        "zenith": zenith,
        "azimuth": azimuth,
        "Azimuth_Bin": azimuth_bin,
        "elevation": elevation,
        "Zenith_Bin": zenith_bin
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
    # Error Info
    # ===============================
    if hour in ERR_HOUR:
        err = ERR_HOUR[hour]
        acc = 100 - ERR_PCT[hour]
    else:
        err = 7.5
        acc = 92.0

    st.info(
        f"""
        📊 **Expected Accuracy**

        • Typical Error: ± {err:.1f} kWh  
        • Accuracy: {acc:.1f} %  
        • Samples: {len(train_df[train_df["Hour"] == hour])}
        """
    )

    # ===============================
    # Charts SIDE BY SIDE
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Hourly Prediction Error")
        hours = list(ERR_HOUR.keys())
        errors = list(ERR_HOUR.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hours, errors, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean Error (kWh)")
        ax.set_title("Error vs Hour")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("📉 Energy Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(train_df["Predicted_Energy"], bins=40, alpha=0.7, color='#ff9500')
        ax2.set_xlabel("Energy (kWh)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Training Data Distribution")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

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



