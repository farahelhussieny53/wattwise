import calendar
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import build_features, chrono_split
from src.forecast import forecast_next_days, predict_from_inputs, prepare_history

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

# مستوى الاستهلاك نسبي: التوقع ÷ متوسط آخر 30 يوم (حدود من فترة التدريب بس)
feat_df = build_features(daily)
train_part, _ = chrono_split(feat_df)
ratio = train_part["kwh"] / train_part["roll_30"]
low_cut, high_cut = ratio.quantile([0.33, 0.67])

# ---------- Header ----------
st.title("⚡ WattWise")
st.caption(
    "Household electricity consumption analysis and next-day forecasting "
    "(UCI Individual Household Electric Power Consumption dataset)."
)
st.info(
    "The model learned weekday and seasonal patterns from a single household "
    "(2006-2010). It works with relative changes, so it accepts any consumption "
    "scale, but it is an analytical demo, not a validated system for guaranteeing "
    "savings or for other households."
)

tab_try, tab_fc, tab_pat, tab_perf, tab_about = st.tabs(
    ["🔮 Try it yourself", "📈 Forecast", "📊 Usage patterns", "🧪 Model performance", "ℹ️ About"]
)

# ======================================================
# TAB 1: Try it yourself (input -> output)
# ======================================================
with tab_try:
    st.subheader("Predict a day's electricity consumption")
    st.write(
        "Enter recent consumption values (in kWh per day) and the date you want "
        "to predict, then press **Predict**. The fields are pre-filled with the "
        "latest values from the dataset."
    )

    hist = prepare_history(daily)
    default_date = hist.index[-1] + pd.Timedelta(days=1)
    d_lag1 = float(round(hist.iloc[-1], 1))
    d_lag2 = float(round(hist.iloc[-2], 1))
    d_lag7 = float(round(hist.loc[default_date - pd.Timedelta(days=7)], 1))
    d_roll7 = float(round(hist.iloc[-7:].mean(), 1))
    d_roll30 = float(round(hist.iloc[-30:].mean(), 1))

    with st.form("predict_form"):
        target_date = st.date_input("Day to predict", value=default_date.date())
        c1, c2, c3 = st.columns(3)
        lag_1 = c1.number_input("Yesterday (kWh)", 0.0, 1000.0, d_lag1, 0.5)
        lag_2 = c2.number_input("2 days ago (kWh)", 0.0, 1000.0, d_lag2, 0.5)
        lag_7 = c3.number_input("Same weekday last week (kWh)", 0.0, 1000.0, d_lag7, 0.5)
        c4, c5 = st.columns(2)
        roll_7 = c4.number_input("Average of last 7 days (kWh)", 0.0, 1000.0, d_roll7, 0.5)
        roll_30 = c5.number_input("Average of last 30 days (kWh)", 0.1, 1000.0, d_roll30, 0.5)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        pred = predict_from_inputs(
            model, features, lag_1, lag_2, lag_7, roll_7, roll_30, target_date
        )
        rel = pred / roll_30
        level = "Low" if rel < low_cut else ("High" if rel > high_cut else "Medium")
        diff = pred - roll_30

        r1, r2, r3 = st.columns(3)
        r1.metric(f"Predicted for {target_date}", f"{pred:.1f} kWh")
        r2.metric("vs your 30-day average", f"{diff:+.1f} kWh", delta_color="inverse")
        r3.metric("Consumption level", level)
        st.caption(
            f"Level is relative to your own 30-day average: Low below {low_cut:.2f}x, "
            f"Medium up to {high_cut:.2f}x, High above that. "
            f"Typical model error on the test period was about ±{metrics['best_mae_pct']:.0f}%."
        )
        st.warning(
            "The weekday and seasonal patterns were learned from one French household. "
            "A home with very different habits (e.g. gas heating, frequent travel) may "
            "behave differently, so treat the result as an estimate."
        )

# ======================================================
# TAB 2: Forecast
# ======================================================
with tab_fc:
    s1, s2 = st.columns(2)
    days = s1.slider("Forecast horizon (days)", 1, 7, 3)
    history_days = s2.slider("History shown (days)", 30, 365, 90)
    st.caption(
        "Multi-day forecasts are recursive: each predicted day feeds the next, "
        "so errors accumulate with a longer horizon."
    )

    forecast = forecast_next_days(model, features, daily, days=days)
    last_date = daily.index.max()
    last_value = daily["kwh"].iloc[-1]
    avg_30 = daily["kwh"].iloc[-30:].mean()
    next_value = forecast.iloc[0]
    delta = next_value - avg_30

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Last recorded day ({last_date.date()})", f"{last_value:.1f} kWh")
    m2.metric(
        f"Forecast for {forecast.index[0].date()}",
        f"{next_value:.1f} kWh",
        f"{delta:+.1f} vs last-30-day avg",
        delta_color="inverse",
    )
    m3.metric("Last 30-day average", f"{avg_30:.1f} kWh")

    st.subheader("Recent consumption and forecast")
    recent = daily["kwh"].iloc[-history_days:].rename("Actual")
    chart_df = pd.concat([recent, forecast.rename("Forecast")], axis=1)
    chart_df.loc[last_date, "Forecast"] = last_value
    st.line_chart(chart_df)

# ======================================================
# TAB 3: Patterns
# ======================================================
with tab_pat:
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

# ======================================================
# TAB 4: Model performance
# ======================================================
with tab_perf:
    st.markdown(
        f"Chronological split. Train: **{metrics['train_period'][0]} → {metrics['train_period'][1]}**, "
        f"Test: **{metrics['test_period'][0]} → {metrics['test_period'][1]}**."
    )
    st.dataframe(pd.DataFrame(metrics["results"]).set_index("Model"))
    st.caption(
        "MAE = average error in kWh per day (lower is better). "
        "RMSE penalizes large errors more. "
        "R² = share of variance explained (higher is better). "
        f"Selected model (lowest RMSE): {metrics['best_model']}."
    )
    st.warning(
        "The models improve on the simple baselines, but the gain is moderate, and "
        "the two models are close. Daily consumption of one household is noisy "
        "(guests, travel, weather are not in the data)."
    )

# ======================================================
# TAB 5: About
# ======================================================
with tab_about:
    st.markdown(
        """
**How it works**
- Minute-level readings are aggregated into hourly and then daily consumption.
- Features: yesterday, 2 days ago, same weekday last week and the 7-day average, all expressed
  relative to the 30-day average, plus weekday and month.
- The model predicts today's consumption as a multiple of the 30-day average, so it adapts
  to different consumption scales.
- Models are compared with persistence baselines on a later, held-out period.

**Limitations**
- Trained on one household; habits of other homes may differ.
- Unusual days (e.g. a trip) cannot be predicted from past consumption.
- Recursive multi-day forecasts accumulate error.
"""
    )