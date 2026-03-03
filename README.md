# 🌤️ Trivandrum Weather Prediction System

Real-time 7-Day Weather Forecasting for Trivandrum, Kerala using XGBoost and Streamlit

🔗 **Live App:** https://trivandrumweather-prediction-dtbg4x4sl4zande6ou4egl.streamlit.app

---

## Project Overview

This project is an end-to-end machine learning application that combines:

- Real-time weather data fetched daily from Open-Meteo API
- 6-variable simultaneous prediction (temperature, rainfall, humidity, wind)
- 7-Day recursive weather forecasting with confidence intervals
- XGBoost regression model validated on 2025 held-out data
- Bidirectional LSTM as a secondary reference model
- Production-style Streamlit dashboard with interactive charts and CSV export

It demonstrates practical integration of weather APIs, time-series feature engineering, machine learning model training, performance evaluation, and deployment as a live web application.

---

## Machine Learning Architecture

**Primary Model — XGBoost Regressor**

- One XGBoost model trained per target variable (6 models total)
- 60+ engineered features: lag features, rolling statistics, monsoon phase flags
- Trained on 2020–2024 historical data (1,800+ days)
- Validated on full-year 2025 held-out data (no leakage)
- Metrics: MAE, RMSE, R²

**Secondary Model — Bidirectional LSTM**

- Single multi-output LSTM with shared encoder + Dense(6) output
- 60-day sequence length
- Metrics: MAE, RMSE, R²

**Forecast Strategy**

- Recursive multi-step prediction (Day+1 → Day+7)
- Confidence intervals widen per day: ±1×MAE on Day 1 to ±2.2×MAE on Day 7
- XGBoost used as primary forecast; LSTM kept as reference

---

## Model Performance (2025 Validation)

| Variable | MAE | R² | Grade |
|---|---|---|---|
| 🌡️ Max Temperature | 0.378°C | 0.941 | A+ |
| ❄️ Min Temperature | 0.364°C | 0.801 | A  |
| 🌤️ Mean Temperature | 0.123°C | 0.977 | A+ |
| 🌧️ Rainfall | 4.173mm | 0.447 | B  |
| 💧 Humidity | 2.135% | 0.865 | A  |
| 💨 Wind Speed | 1.075 km/h | 0.927 | A+ |

> Note: Rainfall R² < 0.5 is expected for daily tropical rainfall prediction. Even professional NWP models face this challenge in Kerala's monsoon climate.

---

## Tech Stack

- **Frontend & Deployment:** Streamlit
- **Primary ML Model:** XGBoost
- **Secondary ML Model:** TensorFlow / Keras (Bidirectional LSTM)
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Weather Data:** Open-Meteo Archive API (free, no API key required)

---

## Features

- 📍 Live location header with current conditions
- 📊 6 metric cards showing latest observed weather
- 🗓️ 7-day forecast cards with weather icons
- 📈 Interactive charts — Temperature / Rainfall & Humidity / Wind Speed
- 🗺️ Forecast heatmap across all 6 variables
- 📐 Model performance table with grades
- ⬇️ Download forecast CSV and 90-day historical CSV
- ⚙️ Sidebar controls — forecast days, training start year, CI toggle

---

## Performance & Optimization

- API response caching (`st.cache_data`) — refreshes every hour
- Model caching (`st.cache_resource`) — trains once per session
- Confidence intervals based on actual 2025 validation MAE
- Clean dark UI with metric cards and responsive layout

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Jebin-A/Trivandrum_Weather-prediction.git
cd Trivandrum_Weather-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

> No API key needed. Open-Meteo is completely free with no registration.

---

## Project Structure

```
Trivandrum_Weather-prediction/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
└── README.md            ← Project documentation
```

---

## How It Works

```
Open-Meteo Archive API (free, no key)
            ↓
  Fetch 2020–present daily data
            ↓
  Engineer 60+ features
  (lags, rolling stats, monsoon flags,
   heat index, diurnal range)
            ↓
  Train 6 XGBoost models
  (one per target variable)
            ↓
  Validate on 2025 held-out data
            ↓
  Recursive 7-day forecast
  with widening confidence intervals
            ↓
  Streamlit dashboard — live at URL
```

---

## Conclusion

This project demonstrates how historical weather API data can be combined with machine learning to build a practical, accurate forecasting system for a specific location.

By combining time-series feature engineering, XGBoost regression, LSTM deep learning, evaluation metrics, and an interactive Streamlit interface, the application delivers both analytical insights and user-friendly visualization in a production-ready deployment.

It reflects a complete end-to-end ML workflow — from raw API data ingestion through feature engineering, model training, validation, and live web deployment — applied to a real-world problem specific to Trivandrum's tropical monsoon climate.

---

## About

7-Day Weather Prediction for Trivandrum, Kerala using XGBoost, LSTM, and Streamlit.

**Topics:** `python` `machine-learning` `time-series` `xgboost` `lstm` `weather-forecast` `streamlit` `open-meteo` `kerala` `monsoon`
