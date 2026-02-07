
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

# Set page configuration
st.set_page_config(page_title="Solar Energy Prediction", layout="wide")

# Title
st.title("☀️ Solar Energy Production Predictor")

# Function to download model from GitHub releases
@st.cache_resource
def download_and_load_model():
    model_path = 'final_production_model.pkl'

    # Check if model already exists locally
    if not os.path.exists(model_path):
        try:
            st.info("📥 Downloading model from GitHub releases... (85 MB, may take a moment)")
            model_url = "https://github.com/yashpals1986/solar-energy-predictor/releases/download/v1.0.0/final_production_model.pkl"

            # Download with progress
            urllib.request.urlretrieve(model_url, model_path)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            st.info("The app will continue in demo mode.")
            return None

    # Load the model
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
with st.spinner("Loading model..."):
    model = download_and_load_model()

# Load the data with error handling
@st.cache_data
def load_data():
    csv_path = 'data_with_features.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df, True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, False
    else:
        st.warning("⚠️ Data file 'data_with_features.csv' not found. Using default ranges.")
        st.info("""
        📁 **To upload your data file:**
        - Add `data_with_features.csv` to your GitHub repository root
        - Then redeploy the app
        """)
        return None, False

df, data_loaded = load_data()

# Default ranges if data is not loaded
if data_loaded and df is not None:
    temp_min, temp_max, temp_mean = df['Temperature'].min(), df['Temperature'].max(), df['Temperature'].mean()
    dni_min, dni_max, dni_mean = df['DNI'].min(), df['DNI'].max(), df['DNI'].mean()
    dhi_min, dhi_max, dhi_mean = df['DHI'].min(), df['DHI'].max(), df['DHI'].mean()
    clearsky_dni_min, clearsky_dni_max, clearsky_dni_mean = df['Clearsky DNI'].min(), df['Clearsky DNI'].max(), df['Clearsky DNI'].mean()
    clearsky_ghi_min, clearsky_ghi_max, clearsky_ghi_mean = df['Clearsky GHI'].min(), df['Clearsky GHI'].max(), df['Clearsky GHI'].mean()
    rh_min, rh_max, rh_mean = df['Relative Humidity'].min(), df['Relative Humidity'].max(), df['Relative Humidity'].mean()
    ws_min, ws_max, ws_mean = df['Wind Speed'].min(), df['Wind Speed'].max(), df['Wind Speed'].mean()
    date_min, date_max = df['Date'].min(), df['Date'].max()
    azimuth_min, azimuth_max = df['azimuth'].min(), df['azimuth'].max()
    zenith_min, zenith_max = df['zenith'].min(), df['zenith'].max()
else:
    # Default values for demo mode
    temp_min, temp_max, temp_mean = -10.0, 50.0, 25.0
    dni_min, dni_max, dni_mean = 0.0, 1000.0, 400.0
    dhi_min, dhi_max, dhi_mean = 0.0, 500.0, 150.0
    clearsky_dni_min, clearsky_dni_max, clearsky_dni_mean = 0.0, 1000.0, 500.0
    clearsky_ghi_min, clearsky_ghi_max, clearsky_ghi_mean = 0.0, 1000.0, 400.0
    rh_min, rh_max, rh_mean = 0.0, 100.0, 50.0
    ws_min, ws_max, ws_mean = 0.0, 20.0, 3.0
    date_min, date_max = "2017-01-01", "2017-12-31"
    azimuth_min, azimuth_max = 8.4, 352.6
    zenith_min, zenith_max = 7.1, 174.1

# Sidebar for input features
st.sidebar.header("Input Features")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("☀️ Solar Position Parameters")

    # Azimuth - continuous value
    azimuth = st.slider(
        "Azimuth (degrees)", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0, 
        step=1.0,
        help=f"Solar azimuth angle (0-360°). Data range: {azimuth_min:.1f}° to {azimuth_max:.1f}°"
    )

    # Zenith - continuous value
    zenith = st.slider(
        "Zenith (degrees)", 
        min_value=0.0, 
        max_value=180.0, 
        value=45.0, 
        step=1.0,
        help=f"Solar zenith angle (0-180°). Data range: {zenith_min:.1f}° to {zenith_max:.1f}°"
    )

    # Calculate elevation from zenith
    elevation = 90 - zenith
    st.info(f"**Calculated Elevation:** {elevation:.1f}°")

    # Create bins similar to the training data
    azimuth_bin = int(round(azimuth / 5) * 5)  # Round to nearest 5
    zenith_bin = int(round(zenith / 2) * 2)     # Round to nearest 2

    st.caption(f"Azimuth Bin: {azimuth_bin}° | Zenith Bin: {zenith_bin}°")

with col2:
    st.subheader("🕐 Time Parameters")

    # Hour (0-23, representing measurements at XX:30:00)
    hour = st.slider(
        "Hour of Day", 
        min_value=0, 
        max_value=23, 
        value=12,
        help="Hour of day (0-23). Actual measurements at XX:30 minutes"
    )

    # Month
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=5,  # June
        format_func=lambda x: [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ][x-1]
    )

    # Day
    day = st.slider("Day of Month", min_value=1, max_value=31, value=15)

    # Day of Week
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(0, 7)),
        index=2,
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", 
                               "Friday", "Saturday", "Sunday"][x]
    )

# Additional Features Section
st.sidebar.subheader("🌡️ Weather & Environmental Parameters")

# Temperature
temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=float(temp_min),
    max_value=float(temp_max),
    value=float(temp_mean),
    step=0.1
)

# DNI (Direct Normal Irradiance)
dni = st.sidebar.slider(
    "DNI (W/m²)",
    min_value=float(dni_min),
    max_value=float(dni_max),
    value=float(dni_mean),
    step=10.0,
    help="Direct Normal Irradiance"
)

# DHI (Diffuse Horizontal Irradiance)
dhi = st.sidebar.slider(
    "DHI (W/m²)",
    min_value=float(dhi_min),
    max_value=float(dhi_max),
    value=float(dhi_mean),
    step=10.0,
    help="Diffuse Horizontal Irradiance"
)

# Clearsky DNI
clearsky_dni = st.sidebar.slider(
    "Clearsky DNI (W/m²)",
    min_value=float(clearsky_dni_min),
    max_value=float(clearsky_dni_max),
    value=float(clearsky_dni_mean),
    step=10.0
)

# Clearsky GHI
clearsky_ghi = st.sidebar.slider(
    "Clearsky GHI (W/m²)",
    min_value=float(clearsky_ghi_min),
    max_value=float(clearsky_ghi_max),
    value=float(clearsky_ghi_mean),
    step=10.0,
    help="Clearsky Global Horizontal Irradiance"
)

# Relative Humidity
relative_humidity = st.sidebar.slider(
    "Relative Humidity (%)",
    min_value=float(rh_min),
    max_value=float(rh_max),
    value=float(rh_mean),
    step=0.1
)

# Wind Speed
wind_speed = st.sidebar.slider(
    "Wind Speed (m/s)",
    min_value=float(ws_min),
    max_value=float(ws_max),
    value=float(ws_mean),
    step=0.1
)

# Create input dataframe with default values
if data_loaded and df is not None:
    default_aod = df['Aerosol Optical Depth'].mean()
    default_dew = df['Dew Point'].mean()
    default_cloud = df['Cloud Type'].mode()[0]
    default_clearsky_dhi = df['Clearsky DHI'].mean()
    default_pressure = df['Pressure'].mean()
    default_wind_dir = df['Wind Direction'].mean()
    default_precip = df['Precipitable Water'].mean()
    default_tilt = df['Best_Tilt'].mean()
    default_doy = df['DayOfYear'].mean()
    default_woy = df['WeekOfYear'].mean()
else:
    default_aod = 0.3
    default_dew = 10.0
    default_cloud = 0
    default_clearsky_dhi = 100.0
    default_pressure = 1013.0
    default_wind_dir = 180.0
    default_precip = 15.0
    default_tilt = 30.0
    default_doy = 180
    default_woy = 26

input_data = pd.DataFrame({
    'Temperature': [temperature],
    'Aerosol Optical Depth': [default_aod],
    'Clearsky DNI': [clearsky_dni],
    'Dew Point': [default_dew],
    'Cloud Type': [default_cloud],
    'Clearsky GHI': [clearsky_ghi],
    'DHI': [dhi],
    'Clearsky DHI': [default_clearsky_dhi],
    'DNI': [dni],
    'Relative Humidity': [relative_humidity],
    'Pressure': [default_pressure],
    'Wind Speed': [wind_speed],
    'Wind Direction': [default_wind_dir],
    'Precipitable Water': [default_precip],
    'zenith': [zenith],
    'azimuth': [azimuth],
    'elevation': [elevation],
    'Best_Tilt': [default_tilt],
    'Azimuth_Bin': [azimuth_bin],
    'Zenith_Bin': [zenith_bin],
    'Year': [2017],
    'Month': [month],
    'Day': [day],
    'Hour': [hour],
    'DayOfWeek': [day_of_week],
    'DayOfYear': [int(default_doy)],
    'WeekOfYear': [int(default_woy)]
})

# Display input summary
st.subheader("📊 Input Summary")
col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Azimuth", f"{azimuth:.1f}°", f"Bin: {azimuth_bin}°")
    st.metric("Zenith", f"{zenith:.1f}°", f"Bin: {zenith_bin}°")
    st.metric("Elevation", f"{elevation:.1f}°")

with col4:
    st.metric("Hour", f"{hour}:30", "Half-hourly data")
    st.metric("Month-Day", f"{month:02d}-{day:02d}")
    st.metric("Temperature", f"{temperature:.1f}°C")

with col5:
    st.metric("DNI", f"{dni:.0f} W/m²")
    st.metric("DHI", f"{dhi:.0f} W/m²")
    st.metric("Humidity", f"{relative_humidity:.1f}%")

# Prediction button
if st.button("🔮 Predict Energy Production", type="primary"):
    if model is not None:
        try:
            with st.spinner("Making prediction..."):
                # Make prediction
                prediction = model.predict(input_data)

                st.success("✅ Prediction Complete!")
                st.subheader("⚡ Predicted Energy Production")

                # Display prediction with large font
                st.markdown(f"# **{prediction[0]:.3f} kWh**")

                # Additional info
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"**Daily Equivalent:** ~{prediction[0] * 24:.2f} kWh/day")
                with col_b:
                    st.info(f"**Monthly Estimate:** ~{prediction[0] * 24 * 30:.2f} kWh/month")

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.write("**Debug Info:**")
            st.write("Input data shape:", input_data.shape)
            st.write("Input columns:", input_data.columns.tolist())

            with st.expander("See detailed error"):
                st.code(str(e))

            # Show expected vs actual
            if data_loaded and df is not None:
                st.write("**Expected columns:**", df.columns.tolist())
    else:
        # Dummy prediction for demonstration
        # Simple heuristic based on solar position and irradiance
        if zenith > 90:  # Sun below horizon
            dummy_prediction = 0.0
        else:
            # Basic energy estimation
            cos_zenith = np.cos(np.radians(zenith))
            dummy_prediction = max(0, (dni * cos_zenith * 0.15 + dhi * 0.1) * 0.001)

        st.warning("⚠️ Using dummy prediction (model not loaded)")
        st.markdown(f"### **{dummy_prediction:.2f} kWh** (Estimated)")
        st.caption("This is a simplified estimate. The model will be downloaded on first run.")

# Show data statistics (only if data is loaded)
if data_loaded and df is not None:
    with st.expander("📈 View Data Statistics"):
        st.write("**Dataset Overview:**")
        st.write(f"- Total Records: {len(df):,}")
        st.write(f"- Date Range: {date_min} to {date_max}")
        st.write(f"- Azimuth Range: {azimuth_min:.2f}° to {azimuth_max:.2f}°")
        st.write(f"- Zenith Range: {zenith_min:.2f}° to {zenith_max:.2f}°")
        st.write(f"- Temperature Range: {temp_min:.1f}°C to {temp_max:.1f}°C")

        st.write("\n**Key Statistics:**")
        st.dataframe(df[['Temperature', 'DNI', 'DHI', 'azimuth', 'zenith', 'Predicted_Energy']].describe())

    # Show sample data
    with st.expander("🔍 View Sample Data"):
        st.dataframe(df.head(20))
else:
    with st.expander("💡 Setup Instructions"):
        st.markdown("""
        ### To enable full functionality:

        1. **Upload your data file** to GitHub:
           - File: `data_with_features.csv`
           - Location: Repository root

        2. **Model file** (already configured):
           - ✅ Automatically downloads from GitHub releases
           - File: `final_production_model.pkl` (85 MB)
           - URL: https://github.com/yashpals1986/solar-energy-predictor/releases/download/v1.0.0/final_production_model.pkl

        3. **Redeploy** your Streamlit app after uploading the CSV
        """)

# Footer
st.markdown("---")
st.caption("🌞 Solar Energy Prediction System | Data-driven renewable energy forecasting")
st.caption("Built with Streamlit • Powered by Machine Learning • Developed by Yashpal Suwansia")
