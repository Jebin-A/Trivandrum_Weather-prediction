# 🌤️ Trivandrum Weather Forecast

A live 7-day weather prediction web app for Trivandrum (Thiruvananthapuram), Kerala, India.

Built with XGBoost + Open-Meteo API. Validated on 2025 held-out data.

## 📊 What It Predicts

| Variable | 2025 MAE | 2025 R² |
|---|---|---|
| 🌡️ Max Temperature | ~0.38°C | ~0.941 |
| ❄️ Min Temperature | ~0.36°C | ~0.801 |
| 🌤️ Mean Temperature | ~0.12°C | ~0.977 |
| 🌧️ Rainfall | ~4.2mm | ~0.447 |
| 💧 Humidity | ~2.1% | ~0.865 |
| 💨 Wind Speed | ~1.1 km/h | ~0.927 |

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
# Create a new repo on github.com, then:
git init
git add app.py requirements.txt README.md
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/trivandrum-weather.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy**

Done! Your app will be live at:
`https://YOUR_USERNAME-trivandrum-weather-app-XXXXX.streamlit.app`

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🏗️ How It Works

```
Open-Meteo API (free)
      ↓
Fetch 2020–present daily weather data
      ↓
Engineer 60+ features (lags, rolling stats, monsoon flags)
      ↓
Train 6 XGBoost models (one per variable)
      ↓
Validate on 2025 held-out data
      ↓
Recursive 7-day forecast
      ↓
Streamlit dashboard with charts + download
```

## 📁 Files

```
app.py            ← Main Streamlit app
requirements.txt  ← Python dependencies
README.md         ← This file
```

## 🛠️ Tech Stack

- **ML**: XGBoost
- **Data**: Open-Meteo Archive API
- **App**: Streamlit
- **Plots**: Matplotlib, Seaborn
- **Deploy**: Streamlit Cloud (free)
