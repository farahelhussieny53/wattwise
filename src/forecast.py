import numpy as np
import pandas as pd


def prepare_history(daily):
    """سلسلة يومية كاملة؛ الأيام الناقصة بتتملى بالـ interpolation."""
    s = daily["kwh"].asfreq("D")
    return s.interpolate(limit_direction="both")


def make_row(history, date, features):
    """نبني خصائص يوم معين من الأيام اللي قبله بس."""
    past = history[history.index < date]
    row = {
        "lag_1": past.get(date - pd.Timedelta(days=1), np.nan),
        "lag_2": past.get(date - pd.Timedelta(days=2), np.nan),
        "lag_7": past.get(date - pd.Timedelta(days=7), np.nan),
        "roll_7": past.iloc[-7:].mean(),
        "roll_30": past.iloc[-30:].mean(),
        "dow": date.dayofweek,
        "month": date.month,
    }
    return pd.DataFrame([row])[features]


def forecast_next_days(model, features, daily, days=1):
    """توقع متكرر (Recursive): كل يوم متوقع بيدخل في خصائص اليوم اللي بعده."""
    history = prepare_history(daily)
    out = {}
    for _ in range(days):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        x = make_row(history, next_date, features)
        pred = float(model.predict(x)[0])
        history.loc[next_date] = pred
        out[next_date] = pred
    return pd.Series(out, name="forecast")