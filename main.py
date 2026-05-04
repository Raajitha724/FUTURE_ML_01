import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📊 Sales & Demand Forecasting Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Superstore.csv", encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

df = load_data()

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
st.sidebar.header("⚙️ Settings")

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(df['Category'].unique())
)

forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

p = st.sidebar.slider("ARIMA p", 0, 3, 1)
d = st.sidebar.slider("ARIMA d", 0, 2, 1)
q = st.sidebar.slider("ARIMA q", 0, 3, 1)

# Apply filter
if category != "All":
    df = df[df['Category'] == category]

# -------------------------------
# PREPROCESSING
# -------------------------------
sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()
sales_data.set_index('Order Date', inplace=True)

# Ensure daily frequency (IMPORTANT)
sales_data = sales_data.asfreq('D')
sales_data['Sales'].fillna(method='ffill', inplace=True)

# Smooth data
sales_data['Sales_Smoothed'] = sales_data['Sales'].rolling(7).mean()

# -------------------------------
# KPI METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{round(sales_data['Sales'].sum(), 2)}")
col2.metric("Avg Daily Sales", f"{round(sales_data['Sales'].mean(), 2)}")
col3.metric("Max Sales", f"{round(sales_data['Sales'].max(), 2)}")

# -------------------------------
# SHOW DATA
# -------------------------------
st.subheader("📄 Recent Sales Data")
st.dataframe(sales_data.tail())

# -------------------------------
# HISTORICAL TREND
# -------------------------------
st.subheader("📈 Historical Sales Trend")

fig1 = plt.figure(figsize=(10,5))
plt.plot(sales_data['Sales'], alpha=0.2, label="Actual Sales")
plt.plot(sales_data['Sales_Smoothed'], linewidth=2, label="Trend (7-day Avg)")
plt.legend()
plt.title("Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()

st.pyplot(fig1)

# -------------------------------
# ARIMA FORECAST
# -------------------------------
st.subheader("🤖 ARIMA Forecast")

forecast = None

try:
    model = ARIMA(sales_data['Sales_Smoothed'].dropna(), order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    model_success = True

except Exception:
    model_success = False
    st.warning("⚠️ Model unstable. Try parameters like (1,1,1)")

# -------------------------------
# FORECAST PLOT
# -------------------------------
fig2 = plt.figure(figsize=(10,5))

plt.plot(sales_data['Sales'], alpha=0.2, label="Actual Sales")
plt.plot(sales_data['Sales_Smoothed'], linewidth=2, label="Trend")

if model_success:
    future_dates = pd.date_range(
        start=sales_data.index[-1],
        periods=forecast_days+1,
        freq='D'
    )[1:]

    plt.plot(future_dates, forecast, linewidth=3, label="Forecast")

plt.legend()
plt.title("Sales Forecast (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()

st.pyplot(fig2)

# -------------------------------
# BUSINESS INSIGHTS
# -------------------------------
st.subheader("💡 Business Insights")

if model_success:
    avg_forecast = np.mean(forecast)
    last_value = sales_data['Sales'].iloc[-1]

    growth = ((avg_forecast - last_value) / last_value) * 100

    st.write(f"📊 Avg predicted sales (next {forecast_days} days): **{round(avg_forecast,2)}**")
    st.write(f"📈 Expected growth: **{round(growth,2)}%**")

    if growth > 5:
        st.success("📈 Strong growth expected → Increase inventory & staffing")
    elif growth < -5:
        st.warning("📉 Demand may drop → Plan promotions")
    else:
        st.info("📊 Stable demand expected")

else:
    st.info("⚠️ Forecast not available due to unstable model.")