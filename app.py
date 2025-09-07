import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt


import skops.io as sio

# Save
sio.dump(model, "forecasting_ev_model.skops")

# Load
model = sio.load("forecasting_ev_model.skops", trusted=True)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="EV Forecast", layout="wide")

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
    input_df = pd.DataFrame([new_row]).astype(float).fillna(0)
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
# Summary Box
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
# Multi-County Comparison
# ---------------------------
st.markdown("---")
st.header("Compare EV Adoption Trends for up to 3 Counties")
multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

if multi_counties:
    comparison_data = []
    for cty in multi_counties:
        cty_df = df[df["County"] == cty].sort_values("Date")
        cty_code = cty_df["county_encoded"].iloc[0]

        hist_ev = list(cty_df["Electric Vehicle (EV) Total"].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df["months_since_start"].max()
        last_date = cty_df["Date"].max()

        future_rows_cty = []
        for i in range(1, forecast_horizon + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            recent_cum = cum_ev[-6:]
            ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) >= 2 else 0

            new_row = {
                "months_since_start": months_since,
                "county_encoded": cty_code,
                "ev_total_lag1": lag1,
                "ev_total_lag2": lag2,
                "ev_total_lag3": lag3,
                "ev_total_roll_mean_3": roll_mean,
                "ev_total_pct_change_1": pct_change_1,
                "ev_total_pct_change_3": pct_change_3,
                "ev_growth_slope": ev_slope
            }

            # ✅ Safe predict
            input_df = pd.DataFrame([new_row]).astype(float).fillna(0)
            pred = float(model.predict(input_df)[0])
            future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

            hist_ev.append(pred)
            if len(hist_ev) > 6:
                hist_ev.pop(0)
            cum_ev.append(cum_ev[-1] + pred)
            if len(cum_ev) > 6:
                cum_ev.pop(0)

        hist_cum = cty_df[["Date", "Electric Vehicle (EV) Total"]].copy()
        hist_cum["Cumulative EV"] = hist_cum["Electric Vehicle (EV) Total"].cumsum()

        fc_df = pd.DataFrame(future_rows_cty)
        fc_df["Cumulative EV"] = fc_df["Predicted EV Total"].cumsum() + hist_cum["Cumulative EV"].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[["Date", "Cumulative EV"]],
            fc_df[["Date", "Cumulative EV"]]
        ], ignore_index=True)
        combined_cty["County"] = cty
        comparison_data.append(combined_cty)

    if comparison_data:
        comp_df = pd.concat(comparison_data, ignore_index=True)
        st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
        fig, ax = plt.subplots(figsize=(14, 7))
        for cty, group in comp_df.groupby("County"):
            ax.plot(group["Date"], group["Cumulative EV"], marker="o", label=cty)
        ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color="white")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Cumulative EV Count", color="white")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1c1c1c")
        fig.patch.set_facecolor("#1c1c1c")
        ax.tick_params(colors="white")
        ax.legend(title="County")
        st.pyplot(fig)

st.markdown("---")
st.markdown("<center>© 2025 EV Forecasting App | Nikhil Kuchana</center>", unsafe_allow_html=True)
