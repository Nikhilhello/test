import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import os

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="EV Forecast", layout="wide")

# ---------------------------
# Header Styling
# ---------------------------
st.markdown("""
    <style>
        body { background-color: #f2f4f7; font-family: 'Helvetica Neue', sans-serif; }
        .main-container { max-width: 1300px; margin: auto; padding: 20px 30px; }
        .app-header {
            background-color: #003566; padding: 10px 20px; border-radius: 10px;
            color: white; text-align: center; margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .app-header h1 { font-size: 52px; letter-spacing: 1px; margin: 0; font-weight: 600; }
        .app-header p { font-size: 22px; letter-spacing: 1.5px; margin-top: 8px; color: #e0e0e0; }
        @media only screen and (max-width: 768px) {
            .app-header h1 { font-size: 25px; }
            .app-header p { font-size: 14px; }
        }
    </style>
    <div class="main-container">
        <div class="app-header">
            <h1>🔋 EV Demand Forecasting System <b><small style="color: white;">Forecast tool</small></b></h1>
            <p>📈 Predicting future electric vehicle demand across regions</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------
# Check files
# ---------------------------
required_files = ["forecasting_ev_model.pkl", "preprocessed_ev_data.csv"]
for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Required file '{file}' is missing! Please upload it to the app folder.")
        st.stop()

# ---------------------------
# Load Model
# ---------------------------
try:
    model = joblib.load("forecasting_ev_model.pkl")
except Exception as e:
    st.error("❌ Failed to load model")
    st.exception(e)
    st.stop()

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

try:
    df = load_data()
except Exception as e:
    st.error("❌ Failed to load data")
    st.exception(e)
    st.stop()

# ---------------------------
# County Selection
# ---------------------------
county_list = sorted(df["County"].dropna().unique().tolist())
st.markdown("---")
st.header("Select a county and see the forecasted EV adoption trend for the next 3 years.")
st.markdown("---")

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("<div style='font-size:20px; font-weight:600;'>🔽 County:</div>", unsafe_allow_html=True)
with col2:
    county = st.selectbox("U.S. counties", county_list)

if county not in df["County"].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df["County"] == county].sort_values("Date")
county_code = county_df["county_encoded"].iloc[0]

# ---------------------------
# Forecast Logic
# ---------------------------
historical_ev = list(county_df["Electric Vehicle (EV) Total"].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df["months_since_start"].max()
latest_date = county_df["Date"].max()

future_rows = []
forecast_horizon = 36
feature_names = model.feature_names_in_  # ensure correct features

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) >= 2 else 0

    new_row = {
        "months_since_start": months_since_start,
        "county_encoded": county_code,
        "ev_total_lag1": lag1,
        "ev_total_lag2": lag2,
        "ev_total_lag3": lag3,
        "ev_total_roll_mean_3": roll_mean,
        "ev_total_pct_change_1": pct_change_1,
        "ev_total_pct_change_3": pct_change_3,
        "ev_growth_slope": ev_growth_slope
    }

    # ✅ Safe predict
    input_df = pd.DataFrame([new_row], columns=feature_names).fillna(0)
    pred = float(model.predict(input_df)[0])

    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# ---------------------------
# Combine Data for Plot
# ---------------------------
historical_cum = county_df[["Date", "Electric Vehicle (EV) Total"]].copy()
historical_cum["Source"] = "Historical"
historical_cum["Cumulative EV"] = historical_cum["Electric Vehicle (EV) Total"].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df["Source"] = "Forecast"
forecast_df["Cumulative EV"] = forecast_df["Predicted EV Total"].cumsum() + historical_cum["Cumulative EV"].iloc[-1]

combined = pd.concat([
    historical_cum[["Date", "Cumulative EV", "Source"]],
    forecast_df[["Date", "Cumulative EV", "Source"]]
], ignore_index=True)

# ---------------------------
# Plot
# ---------------------------
st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby("Source"):
    ax.plot(data["Date"], data["Cumulative EV"], label=label, marker="o")
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=14, color="white")
ax.set_xlabel("Date", color="white")
ax.set_ylabel("Cumulative EV Count", color="white")
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor("#1c1c1c")
ax.tick_params(colors="white")
ax.legend()
st.pyplot(fig)

# ---------------------------
# Forecast Summary
# ---------------------------
historical_total = historical_cum["Cumulative EV"].iloc[-1]
forecasted_total = forecast_df["Cumulative EV"].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.markdown(f"""
    <div style='background-color:#d1f2eb;border-left:6px solid #117a65;
        padding:20px 25px;border-radius:10px;font-size:20px;
        font-family:"Segoe UI",sans-serif;margin:20px 0;color:#004d40;'>
        <strong>🔍 Forecast Summary:</strong><br><br>
        Based on the graph, EV adoption in <strong>{county}</strong> is expected
        to show a <strong>{trend} of {forecast_growth_pct:.2f}%</strong> over the next 3 years.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<center>© 2025 EV Forecasting App | Nikhil Kuchana</center>", unsafe_allow_html=True)
