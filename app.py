import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Set Streamlit page config first thing
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load model ===
model = joblib.load('forecasting_ev_model.pkl')

st.markdown("""
    <style>
        /* Global Styles */
        body {
            background-color: #f2f4f7;
            font-family: 'Helvetica Neue', sans-serif;
        }

        .main-container {
            max-width: 1300px;
            margin: auto;
            padding: 20px 30px;
        }

        .app-header {
            background-color: #003566;
            padding: 10px 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .app-header h1 {
            font-size: 52px;
            letter-spacing: 1px; 
            margin: 0;
            font-weight: 600;
        }

        .app-header p {
            font-size: 22px;
            letter-spacing: 1.5px; 
            margin-top: 8px;
            color: #e0e0e0; 
        }

        /* Widget Styling */
        .stButton>button, .stSelectbox, .stTextInput>div>input {
            border-radius: 6px !important;
            font-size: 16px !important;
        }

        /* Responsive Layout */
        @media only screen and (max-width: 768px) {
            .app-header h1 {
                font-size: 25px;
            }
            .app-header p {
                font-size: 14px;
            }
        }
    </style>

    <div class="main-container">
        <div class="app-header">
            <h1>🔋 EV Demand Forecasting System <b><small style="color: white; ">Forecast tool</small></b></h1>
        <!--   <p>Interactive Dashboard for Future Electric Vehicle Predictions</p>   -->
            <p>📈 Predicting future electric vehicle demand across regions</p>
        </div>
    </div>
""", unsafe_allow_html=True)



# Display image after config and styles
# Stylized title using markdown + HTML
st.markdown("""
    <div style='text-align: center; font-size: 26px; font-weight: bold; color: black; margin-top: 1px; margin-bottom: 10px; letter-spacing: 2px; '>
        🔮 EV Adoption Forecaster for a County in Washington State
    </div>
""", unsafe_allow_html=True)

# Welcome subtitle
# st.markdown("""
#     <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: #blue;'>
#         Welcome to the Electric Vehicle (EV) Adoption Forecast tool.
#     </div>
# """, unsafe_allow_html=True)

#imges
# Create three columns
col1, col2, col3 = st.columns(3)

# Define a uniform style block to control the image size
image_style = """
    <style>
    .equal-img img {
        height: 300px;
        width: 250%;
        object-fit: cover;
        border-radius: 10px;
    }
    </style>
"""

# Inject the style once
st.markdown(image_style, unsafe_allow_html=True)

# Show images inside each column
with col1:
    st.markdown(
        '<div class="equal-img">'
        '<img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZTVld2lhbGFqeGdjNmd3bWRkMDloamVnOWN0MTF2a21ubm9vY2NtMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pgRCrzUkze8lGnz8zu/giphy.gif">'
        '</div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        '<div class="equal-img">'
        '<img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExdmpueXI4bnhod2dwd3VuMmd3bzEyY2gwbW00cHFnZ2cwZHBsc2V4NSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/u3usCBH7molssu8Y2M/giphy.gif">'
        '</div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        '<div class="equal-img">'
        '<img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWsxYWI0dXA5bXVtMWx4N3AydnVhYXFodHdzMm1xNzFkdzZiOXptYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o85xujGNkLT8iBaeY/giphy.gif">'
        '</div>',
        unsafe_allow_html=True
    )





# Image
# col1, col2 ,col3= st.columns(3)
# with col1:
#     st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZTVld2lhbGFqeGdjNmd3bWRkMDloamVnOWN0MTF2a21ubm9vY2NtMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pgRCrzUkze8lGnz8zu/giphy.gif", use_container_width=True)
# with col2:
#     st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWsxYWI0dXA5bXVtMWx4N3AydnVhYXFodHdzMm1xNzFkdzZiOXptYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o85xujGNkLT8iBaeY/giphy.gif", use_container_width=True)
# with col3:
#     st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExdmpueXI4bnhod2dwd3VuMmd3bzEyY2gwbW00cHFnZ2cwZHBsc2V4NSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/u3usCBH7molssu8Y2M/giphy.gif", use_container_width=True)
# # st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZTVld2lhbGFqeGdjNmd3bWRkMDloamVnOWN0MTF2a21ubm9vY2NtMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pgRCrzUkze8lGnz8zu/giphy.gif", use_container_width=True)
# st.image("ev-car-factory.jpg", use_container_width=True)



# Instruction line
# st.markdown("""
#     <div style='text-align: left; font-size: 33px; padding-top: 10px; color: black; font-family: 'Poppins', 'Segoe UI', sans-serif;'>
#         Select a county and see the forecasted EV adoption trend for the next 3 years.
#     </div>
# """, unsafe_allow_html=True)      #sir 
st.markdown("---")
st.header("Select a county and see the forecasted EV adoption trend for the next 3 years.")
# st.markdown("""
#     <style>
#         .instruction-msg {
#             text-align: left;
#             font-size: 30px;
#             font-family: 'Poppins', 'Segoe UI', sans-serif;
#             color: #003566;
#             padding: 25px 15px;
#             margin-bottom: 20px;
#           /*  background-color: #eaf3ff;
#             border-left: 6px solid #003566;
#             border-radius: 10px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.08); */
#         }

#         @media only screen and (max-width: 768px) {
#             .instruction-msg {
#                 font-size: 18px;
#             }
#         }
#     </style>   

#     <div class="instruction-msg"><b>
#         Select a county and see the forecasted EV adoption trend for the next 3 years.</b>
#     </div>
# """, unsafe_allow_html=True)
st.markdown("---")


# === Load data (must contain historical values, features, etc.) ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()



# === County dropdown ===
# Custom style for selectbox using internal classes
county_list = sorted(df['County'].dropna().unique().tolist())
# county = st.selectbox("Select a County", county_list)

#----------------
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<div style='font-size:20px; font-weight:600; padding-top: 1px;'>🔽 County:</div>", unsafe_allow_html=True)

with col2:
    county = st.selectbox("U.S. counties", county_list)
#--------------
if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Forecasting ===
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

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
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# === Combine Historical + Forecast for Cumulative Plot ===
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# === Plot Cumulative Graph ===
st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=14, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)

# === Compare historical and forecasted cumulative EVs ===
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    # st.success(f"Based on the graph, EV adoption in **{county}** is expected to show a **{trend} of {forecast_growth_pct:.2f}%** over the next 3 years.")
    st.markdown(f"""
    <div style='
        background-color: #d1f2eb;
        border-left: 6px solid #117a65;
        padding: 20px 25px;
        border-radius: 10px;
        font-size: 20px;
        font-family: "Segoe UI", sans-serif;
        margin: 20px 0;
        color: #004d40;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.50);
    '>
        <strong>🔍 Forecast Summary:</strong><br><br>
        Based on the graph, EV adoption in <strong>{county}</strong> is expected to show a <strong>{trend} of {forecast_growth_pct:.2f}%</strong> over the next 3 years.
    </div>
""", unsafe_allow_html=True)

else:
    # st.warning("Historical EV total is zero, so percentage forecast change can't be computed.")
    st.markdown("""
    <div style='
        background-color: #fff3cd;
        border-left: 6px solid #ffae42;
        padding: 16px 22px;
        border-radius: 10px;
        font-size: 18px;
        font-family: "Segoe UI", sans-serif;
        margin: 20px 0;
        color: #856404;
    '>
        ⚠️ Historical EV total is zero, so percentage forecast change can't be computed.
    </div>
""", unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

# === New: Compare up to 3 counties ===
st.markdown("---")

st.header("Compare EV Adoption Trends for up to 3 Counties")
st.markdown("---")
multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

if multi_counties:
    comparison_data = []

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        cty_code = cty_df['county_encoded'].iloc[0]

        hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df['months_since_start'].max()
        last_date = cty_df['Date'].max()

        future_rows_cty = []
        for i in range(1, forecast_horizon + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            recent_cum = cum_ev[-6:]
            ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0

            new_row = {
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_slope
            }
            pred = model.predict(pd.DataFrame([new_row]))[0]
            future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

            hist_ev.append(pred)
            if len(hist_ev) > 6:
                hist_ev.pop(0)

            cum_ev.append(cum_ev[-1] + pred)
            if len(cum_ev) > 6:
                cum_ev.pop(0)

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(future_rows_cty)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ], ignore_index=True)

        combined_cty['County'] = cty
        comparison_data.append(combined_cty)

    # Combine all counties data for plotting
    comp_df = pd.concat(comparison_data, ignore_index=True)

    # Plot
    st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby('County'):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
    ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Cumulative EV Count", color='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='white')
    ax.legend(title="County")
    st.pyplot(fig)
    
    # Display % growth for selected counties ===
    growth_summaries = []
    for cty in multi_counties:
        cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
        historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
        forecasted_total = cty_df['Cumulative EV'].iloc[-1]

        if historical_total > 0:
            growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
            growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
        else:
            growth_summaries.append(f"{cty}: N/A (no historical data)")

    # Join all in one sentence and show with st.success
    growth_sentence = " | ".join(growth_summaries)

    st.markdown(f"""
    <div style='
        background-color: #d1f2eb;
        border-left: 6px solid #117a65;
        padding: 20px 25px;
        border-radius: 10px;
        font-size: 18px;
        font-family: "Segoe UI", sans-serif;
        margin: 20px 0;
        color: #004d40;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    '>
        <strong>📊 Forecasted EV adoption growth over next 3 years:</strong><br><br>
        {growth_sentence}
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='
        background-color: #d4edda;
        border-left: 10px solid #28a745;
        border-right: 10px solid #28a745;
        padding: 16px 20px;
        text-align: center;
        border-radius: 20px;
        font-size: 19px;
        font-family: "Segoe UI", sans-serif;
        letter-spacing: 2px; 
        margin-top: 10px;
        color: #155724;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.50);
    '>
        ✅ Forecast completed.
    </div>
""", unsafe_allow_html=True)
    # st.success(f"Forecasted EV adoption growth over next 3 years — {growth_sentence}")
# st.success("Forecast complete")
st.markdown("---")
st.markdown("<center>© 2025 EV Forecasting App | Nikhil Kuchana </center>", unsafe_allow_html=True)
