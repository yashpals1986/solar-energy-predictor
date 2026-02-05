import streamlit as st
import joblib
import pandas as pd

# Load model (NO SCALER!)
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_v1.0.pkl')
    features = model.get_booster().feature_names
    
    # Load defaults
    try:
        train = pd.read_csv('train.csv')
        exclude = ['Timestamp', 'Date', 'Predicted Energy']
        feature_cols = [c for c in train.columns if c not in exclude]
        defaults = train.iloc[500][feature_cols].to_dict()
    except:
        defaults = {f: 0 for f in features}
    
    return model, features, defaults

model, FEATURES, defaults = load_model()

st.title("☀️ Solar Energy Predictor")
st.caption("XGBoost ML Model | 93% R² Accuracy")

# Inputs
col1, col2 = st.columns(2)
with col1:
    ghi = st.number_input("Clearsky GHI (W/m²)", 0, 1000, 800)
    temp = st.number_input("Temperature (°C)", -10, 50, 32)
    humidity = st.number_input("Humidity (%)", 0, 100, 35)
with col2:
    dni = st.number_input("Clearsky DNI (W/m²)", 0, 1000, 850)
    hour = st.slider("Hour", 0, 23, 12)
    dew = st.number_input("Dew Point (°C)", -20, 30, 15)

if st.button("🔮 Predict Energy"):
    # Update defaults with inputs
    input_dict = defaults.copy()
    updates = {
        'Clearsky GHI': ghi, 'Clearsky DNI': dni,
        'Temperature': temp, 'Hour': hour,
        'Relative Humidity': humidity, 'Dew Point': dew,
        'Clearsky DHI': ghi * 0.15, 'DHI': ghi * 0.12,
        'DNI': dni - 30
    }
    for k, v in updates.items():
        if k in input_dict:
            input_dict[k] = v
    
    # Predict (NO SCALING!)
    input_df = pd.DataFrame([input_dict])[FEATURES]
    prediction = model.predict(input_df)[0]
    
    st.success(f"### ⚡ {prediction:.2f} kWh")
    
    if 6 <= hour <= 18:
        st.balloons()
        st.info("☀️ Daytime production")
    else:
        st.warning("🌙 Nighttime - minimal production")
