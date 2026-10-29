import calendar
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.forecast import forecast_next_days

BASE = Path(__file__).parent

st.set_page_config(page_title="WattWise", page_icon="⚡", layout="wide")


@st.cache_data
def load_daily():
    return pd.read_csv(BASE / "data" / "daily.csv", index_col=0, parse_dates=True)


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


@st.cache_data
def load_metrics():
    with open(BASE / "metrics.json") as f:
        return json.load(f)


daily = load_daily()
art = load_model()
metrics = load_metrics()
model, features = art["model"], art["features"]

# ---------- Header ----------
st.title("⚡ WattWise")
st.caption(
    "Household electricity consumption analysis and next-day forecasting "
    "(UCI Individual Household Electric Power Consumption dataset)."
)
st.info(
    "The data comes from a single household, 2006-2010. This is an analytical "
    "demo, not a validated system for guaranteeing savings or for other households."
)

# ---------- Sidebar ----------
st.sidebar.header("Settings")
days = st.sidebar.slider("Forecast horizon (days)", 1, 7, 3)
history_days = st.sidebar.slider("History shown (days)", 30, 365, 90)
st.sidebar.caption(
    "Multi-day forecasts are recursive: each predicted day feeds the next, "
    "so errors accumulate with a longer horizon."
)

# ---------- Forecast ----------
forecast = forecast_next_days(model, features, daily, days=days)
last_date = daily.index.max()
last_value = daily["kwh"].iloc[-1]
avg_30 = daily["kwh"].iloc[-30:].mean()
next_value = forecast.iloc[0]
delta = next_value - avg_30

c1, c2, c3 = st.columns(3)
c1.metric(f"Last recorded day ({last_date.date()})", f"{last_value:.1f} kWh")
c2.metric(f"Forecast for {forecast.index[0].date()}", f"{next_value:.1f} kWh",
          f"{delta:+.1f} vs last-30-day avg", delta_color="inverse")
c3.metric("Last 30-day average", f"{avg_30:.1f} kWh")

# ---------- Chart ----------
st.subheader("Recent consumption and forecast")
recent = daily["kwh"].iloc[-history_days:].rename("Actual")
chart_df = pd.concat([recent, forecast.rename("Forecast")], axis=1)
# نوصل خط التوقع بآخر قيمة فعلية عشان الرسمة تبان متصلة
chart_df.loc[last_date, "Forecast"] = last_value
st.line_chart(chart_df)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Usage patterns", "Model performance", "About"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.markdown("**Average daily consumption by month**")
        by_month = daily["kwh"].groupby(daily.index.month).mean()
        by_month.index = [calendar.month_abbr[m] for m in by_month.index]
        st.bar_chart(by_month)
    with right:
        st.markdown("**Average daily consumption by weekday**")
        by_dow = daily["kwh"].groupby(daily.index.dayofweek).mean()
        by_dow.index = [calendar.day_abbr[d] for d in by_dow.index]
        st.bar_chart(by_dow)
    st.caption(
        "Winter months use roughly twice as much as summer, and weekends run "
        "higher than weekdays."
    )

with tab2:
    st.markdown(
        f"Chronological split. Train: **{metrics['train_period'][0]} → {metrics['train_period'][1]}**, "
        f"Test: **{metrics['test_period'][0]} → {metrics['test_period'][1]}**."
    )
    st.dataframe(pd.DataFrame(metrics["results"]).set_index("Model"))
    st.caption(
        "MAE = average error in kWh per day (lower is better). "
        "R² = share of variance explained (higher is better). "
        f"Selected model: {metrics['best_model']}."
    )
    st.warning(
        "The model improves on the simple baselines, but the gain is moderate. "
        "Daily consumption of one household is noisy (guests, travel, weather "
        "are not in the data)."
    )

with tab3:
    st.markdown(
        """
**How it works**
- Minute-level readings are aggregated into hourly and then daily consumption.
- Features: yesterday, 2 days ago, same weekday last week, 7- and 30-day rolling averages, weekday, month.
- Models are compared with persistence baselines on a later, held-out period.

**Limitations**
- One household only; may not generalize.
- Unusual days (e.g. a trip) cannot be predicted from past consumption.
- Recursive multi-day forecasts accumulate error.
"""
    )