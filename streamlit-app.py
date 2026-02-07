
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(page_title="Solar Energy Prediction", layout="wide")

# Title
st.title("☀️ Solar Energy Production Predictor")

# Load the trained model (make sure to save your model first)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('solar_energy_model.pkl')  # Update with your model path
        return model
    except:
        st.warning("Model not loaded. Using dummy predictions.")
        return None

model = load_model()

# Load the data to get feature statistics
@st.cache_data
def load_data():
    df = pd.read_csv('data_with_features.csv')
    return df

df = load_data()

# Sidebar for input features
st.sidebar.header("Input Features")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Solar Position Parameters")

    # Azimuth - continuous value (original data range: 8.448 to 352.599)
    azimuth = st.slider(
        "Azimuth (degrees)", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0, 
        step=1.0,
        help="Solar azimuth angle (0-360°). Range in data: 8.4° to 352.6°"
    )

    # Zenith - continuous value (original data range: 7.093 to 174.052)
    zenith = st.slider(
        "Zenith (degrees)", 
        min_value=0.0, 
        max_value=180.0, 
        value=45.0, 
        step=1.0,
        help="Solar zenith angle (0-180°). Range in data: 7.1° to 174.1°"
    )

    # Calculate elevation from zenith
    elevation = 90 - zenith
    st.info(f"**Calculated Elevation:** {elevation:.1f}°")

    # Create bins similar to the training data
    azimuth_bin = int(round(azimuth / 5) * 5)  # Round to nearest 5
    zenith_bin = int(round(zenith / 2) * 2)     # Round to nearest 2

    st.caption(f"Azimuth Bin: {azimuth_bin}° | Zenith Bin: {zenith_bin}°")

with col2:
    st.subheader("Time Parameters")

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
st.sidebar.subheader("Weather & Environmental Parameters")

# Temperature
temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=float(df['Temperature'].min()),
    max_value=float(df['Temperature'].max()),
    value=float(df['Temperature'].mean()),
    step=0.1
)

# DNI (Direct Normal Irradiance)
dni = st.sidebar.slider(
    "DNI (W/m²)",
    min_value=float(df['DNI'].min()),
    max_value=float(df['DNI'].max()),
    value=float(df['DNI'].mean()),
    step=10.0
)

# DHI (Diffuse Horizontal Irradiance)
dhi = st.sidebar.slider(
    "DHI (W/m²)",
    min_value=float(df['DHI'].min()),
    max_value=float(df['DHI'].max()),
    value=float(df['DHI'].mean()),
    step=10.0
)

# Clearsky DNI
clearsky_dni = st.sidebar.slider(
    "Clearsky DNI (W/m²)",
    min_value=float(df['Clearsky DNI'].min()),
    max_value=float(df['Clearsky DNI'].max()),
    value=float(df['Clearsky DNI'].mean()),
    step=10.0
)

# Clearsky GHI
clearsky_ghi = st.sidebar.slider(
    "Clearsky GHI (W/m²)",
    min_value=float(df['Clearsky GHI'].min()),
    max_value=float(df['Clearsky GHI'].max()),
    value=float(df['Clearsky GHI'].mean()),
    step=10.0
)

# Relative Humidity
relative_humidity = st.sidebar.slider(
    "Relative Humidity (%)",
    min_value=float(df['Relative Humidity'].min()),
    max_value=float(df['Relative Humidity'].max()),
    value=float(df['Relative Humidity'].mean()),
    step=0.1
)

# Wind Speed
wind_speed = st.sidebar.slider(
    "Wind Speed (m/s)",
    min_value=float(df['Wind Speed'].min()),
    max_value=float(df['Wind Speed'].max()),
    value=float(df['Wind Speed'].mean()),
    step=0.1
)

# Create input dataframe
input_data = pd.DataFrame({
    'Temperature': [temperature],
    'Aerosol Optical Depth': [df['Aerosol Optical Depth'].mean()],
    'Clearsky DNI': [clearsky_dni],
    'Dew Point': [df['Dew Point'].mean()],
    'Cloud Type': [df['Cloud Type'].mode()[0]],
    'Clearsky GHI': [clearsky_ghi],
    'DHI': [dhi],
    'Clearsky DHI': [df['Clearsky DHI'].mean()],
    'DNI': [dni],
    'Relative Humidity': [relative_humidity],
    'Pressure': [df['Pressure'].mean()],
    'Wind Speed': [wind_speed],
    'Wind Direction': [df['Wind Direction'].mean()],
    'Precipitable Water': [df['Precipitable Water'].mean()],
    'zenith': [zenith],
    'azimuth': [azimuth],
    'elevation': [elevation],
    'Best_Tilt': [df['Best_Tilt'].mean()],
    'Azimuth_Bin': [azimuth_bin],
    'Zenith_Bin': [zenith_bin],
    'Year': [2017],
    'Month': [month],
    'Day': [day],
    'Hour': [hour],
    'DayOfWeek': [day_of_week],
    'DayOfYear': [df['DayOfYear'].mean()],
    'WeekOfYear': [df['WeekOfYear'].mean()]
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
            # Make prediction
            prediction = model.predict(input_data)

            st.success("✅ Prediction Complete!")
            st.subheader("⚡ Predicted Energy Production")

            # Display prediction with large font
            st.markdown(f"### **{prediction[0]:.2f} kWh**")

            # Display confidence interval (if available)
            if hasattr(model, 'predict_proba'):
                st.info("📈 Model confidence metrics available")

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.write("Input data shape:", input_data.shape)
            st.write("Columns:", input_data.columns.tolist())
    else:
        # Dummy prediction for demonstration
        dummy_prediction = max(0, dni * 0.15 + dhi * 0.1 - zenith * 0.5 + hour * 2)
        st.warning("⚠️ Using dummy prediction (model not loaded)")
        st.markdown(f"### **{dummy_prediction:.2f} kWh** (Estimated)")

# Show data statistics
with st.expander("📈 View Data Statistics"):
    st.write("**Dataset Overview:**")
    st.write(f"- Total Records: {len(df)}")
    st.write(f"- Date Range: {df['Date'].min()} to {df['Date'].max()}")
    st.write(f"- Azimuth Range: {df['azimuth'].min():.2f}° to {df['azimuth'].max():.2f}°")
    st.write(f"- Zenith Range: {df['zenith'].min():.2f}° to {df['zenith'].max():.2f}°")
    st.write(f"- Temperature Range: {df['Temperature'].min():.1f}°C to {df['Temperature'].max():.1f}°C")

    st.write("\n**Key Statistics:**")
    st.dataframe(df[['Temperature', 'DNI', 'DHI', 'azimuth', 'zenith', 'Predicted_Energy']].describe())

# Show sample data
with st.expander("🔍 View Sample Data"):
    st.dataframe(df.head(10))

# Footer
st.markdown("---")
st.caption("Solar Energy Prediction System | Data-driven renewable energy forecasting")
