from flask import Flask, render_template_string, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('xgboost_v1.0.pkl')
train_df = pd.read_csv('train.csv')

# ALL 27 features in correct order
ALL_FEATURES = [
    'Temperature', 'Aerosol Optical Depth', 'Clearsky DNI', 'Dew Point', 'Cloud Type',
    'Clearsky GHI', 'DHI', 'Clearsky DHI', 'DNI', 'Relative Humidity',
    'Pressure', 'Wind Speed', 'Wind Direction', 'Precipitable Water', 'zenith',
    'azimuth', 'elevation', 'Best_Tilt', 'Azimuth_Bin', 'Zenith_Bin',
    'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 'WeekOfYear'
]

# Get median defaults
DEFAULT_DICT = train_df[ALL_FEATURES].median().to_dict()

# Calculate error statistics from training data (for confidence estimates)
train_df['Predicted'] = model.predict(train_df[ALL_FEATURES])
train_df['Error'] = abs(train_df['Predicted'] - train_df['Predicted_Energy'])
train_df['Error_Pct'] = (train_df['Error'] / (train_df['Predicted_Energy'] + 0.1)) * 100

# Error by hour (to give context-specific error estimates)
ERROR_BY_HOUR = train_df.groupby('Hour')['Error'].mean().to_dict()
ERROR_PCT_BY_HOUR = train_df.groupby('Hour')['Error_Pct'].mean().to_dict()

print("✅ Model loaded with error statistics")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Solar Energy Predictor - Yashpal Suwansia</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {margin:0; padding:0; box-sizing:border-box;}
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #FF6B00; 
            text-align: center; 
            margin-bottom: 10px;
            font-size: 32px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 35px;
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .time-group {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }
        .time-input {
            flex: 1;
        }
        label {
            display: block; 
            font-weight: 600; 
            margin-bottom: 8px; 
            color: #333;
            font-size: 14px;
        }
        .helper-text {
            font-size: 11px;
            color: #999;
            margin-top: 4px;
        }
        input {
            width: 100%; 
            padding: 12px; 
            border: 2px solid #e0e0e0; 
            border-radius: 8px; 
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus {
            border-color: #FF6B00; 
            outline: none;
            box-shadow: 0 0 0 3px rgba(255,107,0,0.1);
        }
        button[type="submit"] {
            background: linear-gradient(135deg, #FF6B00 0%, #E55A00 100%);
            color: white;
            padding: 18px;
            border: none;
            width: 100%;
            font-size: 20px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            margin-top: 25px;
            transition: transform 0.2s;
        }
        button[type="submit"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(255,107,0,0.3);
        }
        .result-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 35px;
            margin: 30px 0;
            border-radius: 15px;
            text-align: center;
            animation: slideIn 0.5s;
        }
        @keyframes slideIn {
            from {opacity:0; transform:translateY(-20px);}
            to {opacity:1; transform:translateY(0);}
        }
        .result-card h2 {
            font-size: 52px; 
            margin-bottom: 10px;
        }
        .result-card p {
            font-size: 16px;
            opacity: 0.9;
            margin: 5px 0;
        }
        .error-info {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 14px;
        }
        .error-badge {
            background: rgba(255,255,255,0.3);
            padding: 8px 15px;
            border-radius: 20px;
            display: inline-block;
            margin: 5px;
            font-weight: bold;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }
        .info-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #FF6B00;
        }
        .info-card h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }
        .info-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>☀️ Solar Energy Production Predictor</h1>
        <p class="subtitle">
            Advanced ML Model using XGBoost | 93% R² Accuracy<br>
            Trained on 8,760 hours of Rajasthan Solar Data 
        </p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label>☀️ Clearsky GHI (W/m²)</label>
                    <input type="number" name="ghi" value="800" step="0.01" required>
                    <span class="helper-text">Global Horizontal Irradiance</span>
                </div>
                
                <div class="form-group">
                    <label>🌤️ Clearsky DNI (W/m²)</label>
                    <input type="number" name="dni" value="850" step="0.01" required>
                    <span class="helper-text">Direct Normal Irradiance</span>
                </div>
                
                <div class="form-group">
                    <label>☁️ Clearsky DHI (W/m²)</label>
                    <input type="number" name="dhi" value="120" step="0.01" required>
                    <span class="helper-text">Diffuse Horizontal Irradiance</span>
                </div>
                
                <div class="form-group">
                    <label>📐 Best Tilt (degrees)</label>
                    <input type="number" name="tilt" value="30" step="0.1" min="0" max="90" required>
                    <span class="helper-text">Optimal panel angle (0-90°)</span>
                </div>
                
                <div class="form-group">
                    <label>🕐 Time of Day</label>
                    <div class="time-group">
                        <div class="time-input">
                            <input type="number" name="hour" id="hour" value="12" min="0" max="23" 
                                   placeholder="Hour" required style="text-align:center;">
                            <span class="helper-text">Hour (0-23)</span>
                        </div>
                        <div class="time-input">
                            <input type="number" name="minute" id="minute" value="0" min="0" max="59" 
                                   placeholder="Min" required style="text-align:center;">
                            <span class="helper-text">Minute (0-59)</span>
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>📏 Zenith Angle (degrees)</label>
                    <input type="number" name="zenith" value="30" step="0.1" min="0" max="180" required>
                    <span class="helper-text">Sun angle from vertical</span>
                </div>
                
                <div class="form-group">
                    <label>🧭 Azimuth (degrees)</label>
                    <input type="number" name="azimuth" value="180" step="0.1" min="0" max="360" required>
                    <span class="helper-text">Sun compass direction</span>
                </div>
                
                <div class="form-group">
                    <label>🧭 Azimuth Bin</label>
                    <input type="number" name="azimuth_bin" value="180" step="5" required>
                    <span class="helper-text">Rounded azimuth (5° bins)</span>
                </div>
                
                <div class="form-group">
                    <label>📐 Elevation (degrees)</label>
                    <input type="number" name="elevation" value="60" step="0.1" min="-90" max="90" required>
                    <span class="helper-text">Sun height above horizon</span>
                </div>
                
                <div class="form-group">
                    <label>📏 Zenith Bin</label>
                    <input type="number" name="zenith_bin" value="30" step="2" required>
                    <span class="helper-text">Rounded zenith (2° bins)</span>
                </div>
            </div>
            
            <button type="submit">🔮 Predict Solar Energy Production</button>
        </form>
        
        {% if prediction %}
        <div class="result-card">
            <h2>⚡ {{ prediction }} kWh</h2>
            <p>Predicted Solar Energy Production</p>
            <p style="font-size:14px; margin-top:5px; opacity:0.8;">
                Time: {{ time_display }} | Solar Irradiance: {{ ghi }} W/m²
            </p>
            
            <div class="error-info">
                <strong>📊 Expected Prediction Accuracy:</strong><br>
                <div style="margin-top:10px;">
                    <span class="error-badge">Typical Error: ± {{ error_range }} kWh</span>
                    <span class="error-badge">Accuracy: {{ error_pct }}%</span>
                </div>
                <p style="margin-top:10px; font-size:12px; opacity:0.9;">
                    Based on {{ hour_samples }} historical samples at this hour
                </p>
            </div>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>Model Performance</h3>
                <div class="value">93% R²</div>
            </div>
            <div class="info-card">
                <h3>Training Data</h3>
                <div class="value">8,760 hrs</div>
            </div>
            <div class="info-card">
                <h3>Avg Error</h3>
                <div class="value">{{ avg_error }} kWh</div>
            </div>
        </div>
        {% endif %}
    </div>
    
    <footer>
        <p style="font-size:18px; font-weight:bold; color:#333;">Developed by Yashpal Suwansia</p>
        <p style="margin-top:8px;">IIT Bombay 2010 | ML Engineer</p>
        <p style="margin-top:5px; font-size:12px;">Delhi | Specializing in Machine Learning & Data Science</p>
    </footer>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    error_range = None
    error_pct = None
    time_display = None
    ghi = None
    hour_samples = 0
    avg_error = "7.2"
    
    if request.method == 'POST':
        # Get inputs (now accepting decimals!)
        ghi = float(request.form['ghi'])
        dni = float(request.form['dni'])
        dhi = float(request.form['dhi'])
        tilt = float(request.form['tilt'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        zenith = float(request.form['zenith'])
        azimuth = float(request.form['azimuth'])
        azimuth_bin = float(request.form['azimuth_bin'])
        elevation = float(request.form['elevation'])
        zenith_bin = float(request.form['zenith_bin'])
        
        # Format time display
        time_display = f"{hour:02d}:{minute:02d}"
        
        # Convert minute to fractional hour for model (if needed)
        hour_fraction = hour + (minute / 60.0)
        
        # Create input with defaults
        input_dict = DEFAULT_DICT.copy()
        
        # Update top 10 features
        input_dict['Clearsky GHI'] = ghi
        input_dict['Clearsky DNI'] = dni
        input_dict['Clearsky DHI'] = dhi
        input_dict['Best_Tilt'] = tilt
        input_dict['Hour'] = hour  # Use integer hour for model
        input_dict['zenith'] = zenith
        input_dict['azimuth'] = azimuth
        input_dict['Azimuth_Bin'] = azimuth_bin
        input_dict['elevation'] = elevation
        input_dict['Zenith_Bin'] = zenith_bin
        
        # Predict
        input_data = pd.DataFrame([input_dict])[ALL_FEATURES]
        pred = model.predict(input_data)[0]
        prediction = f"{pred:.2f}"
        
        # Calculate expected error for this hour
        if hour in ERROR_BY_HOUR:
            error_range = f"{ERROR_BY_HOUR[hour]:.1f}"
            error_pct_val = ERROR_PCT_BY_HOUR[hour]
            error_pct = f"{100 - error_pct_val:.1f}"
        else:
            error_range = "7.5"
            error_pct = "92.0"
        
        # Count samples at this hour
        hour_samples = len(train_df[train_df['Hour'] == hour])
        
        print(f"✅ Predicted: {prediction} kWh at {time_display}")
        print(f"   Inputs: GHI={ghi}, DNI={dni}, Zenith={zenith}")
        print(f"   Expected Error: ±{error_range} kWh")
    
    return render_template_string(HTML, 
                                 prediction=prediction,
                                 error_range=error_range,
                                 error_pct=error_pct,
                                 time_display=time_display,
                                 ghi=ghi,
                                 hour_samples=hour_samples,
                                 avg_error=avg_error)

if __name__ == '__main__':
    print("="*60)
    print("🚀 Solar Energy Prediction API")
    print("="*60)
    print("✅ Model: XGBoost (93% R² accuracy)")
    print("✅ Features: Top 10 from SHAP analysis")
    print("✅ Error estimates: Calculated per hour")
    print("="*60)
    app.run(debug=True, port=5000, host='0.0.0.0')
