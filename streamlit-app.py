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

# Header
st.title("☀️ Solar Energy Prediction System")
st.markdown("Predict solar energy generation using Machine Learning")

# Sidebar info
st.sidebar.title("📊 Model Info")
st.sidebar.metric("Model", "Random Forest")
st.sidebar.metric("Test MAE", "10.83")
st.sidebar.metric("Accuracy", "93.4%")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Top 9 Important Features")
st.sidebar.markdown("""
1. **azimuth** (35.7%)
2. **Azimuth_Bin** (19.3%)
3. **Hour** (16.5%)
4. **Best_Tilt** (9.4%)
5. **elevation** (8.0%)
6. **zenith** (6.2%)
7. **Zenith_Bin** (2.3%)
8. **Temperature** (1.0%)
9. **Aerosol** (0.6%)

*Other features: auto-filled*
""")

# Main prediction interface
st.header("🔮 Energy Prediction")
st.info("💡 Enter only the 9 most important features. Others are auto-filled with optimal defaults.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔆 Solar Position (Most Important!)")
    azimuth = st.slider("Azimuth Angle (°)", 0.0, 360.0, 180.0, 1.0, 
                        help="Most important feature! Sun's horizontal position")
    azimuth_bin = st.selectbox("Azimuth Bin", list(range(8)), index=4,
                               help="Binned azimuth (0-7)")
    elevation = st.slider("Elevation Angle (°)", 0.0, 90.0, 45.0, 1.0,
                         help="Sun's height above horizon")
    zenith = st.slider("Zenith Angle (°)", 0.0, 90.0, 45.0, 1.0,
                      help="Angle from directly overhead")
    
with col2:
    st.subheader("⏰ Time")
    hour = st.slider("Hour", 0, 23, 12,
                    help="Hour of day (0-23)")
    
    st.subheader("🌡️ Weather")
    temperature = st.slider("Temperature (°C)", -20.0, 50.0, 25.0, 0.5)
    aerosol = st.slider("Aerosol Optical Depth", 0.0, 2.0, 0.15, 0.01,
                       help="Air clarity measure")
    
with col3:
    st.subheader("⚙️ Panel Config")
    best_tilt = st.slider("Best Tilt (°)", 0.0, 90.0, 30.0, 1.0,
                         help="Optimal panel angle")
    zenith_bin = st.selectbox("Zenith Bin", list(range(9)), index=4,
                             help="Binned zenith (0-8)")

# Predict button
if st.button("🔮 Predict Energy", type="primary", use_container_width=True):
    
    # Get current date for auto-filled features
    now = datetime.now()
    day_of_year = now.timetuple().tm_yday
    
    # ✅ Create input with ALL 22 features in EXACT order model expects
    input_data = pd.DataFrame({
        'Temperature': [temperature],                    # User input (important)
        'Aerosol Optical Depth': [aerosol],             # User input (important)
        'Dew Point': [15.0],                            # Auto: default
        'Cloud Type': [0],                              # Auto: clear sky
        'Relative Humidity': [50.0],                    # Auto: default
        'Pressure': [1013.25],                          # Auto: standard pressure
        'Wind Speed': [3.0],                            # Auto: light wind
        'Wind Direction': [180.0],                      # Auto: south
        'Precipitable Water': [1.5],                    # Auto: default
        'zenith': [zenith],                             # User input (important)
        'azimuth': [azimuth],                           # User input (MOST important!)
        'elevation': [elevation],                        # User input (important)
        'Best_Tilt': [best_tilt],                       # User input (important)
        'Azimuth_Bin': [azimuth_bin],                   # User input (important)
        'Zenith_Bin': [zenith_bin],                     # User input (important)
        'Year': [now.year],                             # Auto: current year
        'Month': [now.month],                           # Auto: current month
        'Day': [now.day],                               # Auto: current day
        'Hour': [hour],                                 # User input (important)
        'DayOfWeek': [now.weekday()],                  # Auto: current day of week
        'DayOfYear': [day_of_year],                    # Auto: current day of year
        'WeekOfYear': [now.isocalendar()[1]],          # Auto: current week
    })
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result with nice formatting
        st.success("✅ Prediction Complete!")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ Predicted Energy", f"{prediction:.2f} Wh/m²", 
                   help="Energy per square meter")
        col2.metric("🏠 For 1kW System", f"{prediction * 0.001:.2f} kWh",
                   help="Multiply by your system size")
        
                
        # Show what was predicted
        with st.expander("📋 See Input Details"):
            st.write("**User Inputs (9 important features):**")
            important_features = {
                'azimuth': azimuth,
                'Azimuth_Bin': azimuth_bin,
                'Hour': hour,
                'Best_Tilt': best_tilt,
                'elevation': elevation,
                'zenith': zenith,
                'Zenith_Bin': zenith_bin,
                'Temperature': temperature,
                'Aerosol Optical Depth': aerosol
            }
            st.json(important_features)
            
            st.write("**Auto-filled features (13 less important):**")
            st.caption("These have minimal impact on prediction (<1% each)")
            
    except Exception as e:
        st.error(f"❌ Prediction Error!")
        st.write("Debug info:")
        st.write(f"Input columns: {list(input_data.columns)}")
        st.write(f"Model expects: {list(model.feature_names_in_)}")
        st.write(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Developed by Yashpal Suwansia</strong></p>
    <p>Powered by Streamlit & Scikit-learn | Based on SHAP Feature Analysis</p>
</div>
""", unsafe_allow_html=True)

# Tips section
with st.expander("💡 Tips for Best Predictions"):
    st.markdown("""
    **Most Important Settings:**
    1. **Azimuth (35.7% importance)**: Sun's compass direction
       - 90° = East (morning), 180° = South (noon), 270° = West (evening)
    2. **Hour (16.5%)**: Time of day matters a lot!
    3. **Best_Tilt (9.4%)**: Panel angle optimization
    
    **Quick Scenarios:**
    - **Morning:** azimuth=90°, hour=8, elevation=30°
    - **Noon:** azimuth=180°, hour=12, elevation=60°
    - **Evening:** azimuth=270°, hour=18, elevation=20°
    """)

