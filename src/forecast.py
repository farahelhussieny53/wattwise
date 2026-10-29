import numpy as np
import pandas as pd

from src.features import to_relative


def prepare_history(daily):
    """سلسلة يومية كاملة؛ الأيام الناقصة بتتملى بالـ interpolation."""
    s = daily["kwh"].asfreq("D")
    return s.interpolate(limit_direction="both")


def make_row(history, date):
    """نبني خصائص يوم معين (بالـ kWh) من الأيام اللي قبله بس."""
    past = history[history.index < date]
    return {
        "lag_1": past.get(date - pd.Timedelta(days=1), np.nan),
        "lag_2": past.get(date - pd.Timedelta(days=2), np.nan),
        "lag_7": past.get(date - pd.Timedelta(days=7), np.nan),
        "roll_7": past.iloc[-7:].mean(),
        "roll_30": past.iloc[-30:].mean(),
        "dow": date.dayofweek,
        "month": date.month,
    }


def _predict(model, features, row):
    """الموديل بيتوقع نسبة، فبنضربها في متوسط الـ 30 يوم ونرجع kWh."""
    if row["roll_30"] <= 0:
        raise ValueError("30-day average must be greater than zero")
    x = to_relative(pd.DataFrame([row]))[features]
    return float(model.predict(x)[0] * row["roll_30"])


def forecast_next_days(model, features, daily, days=1):
    """توقع متكرر (Recursive): كل يوم متوقع بيدخل في خصائص اليوم اللي بعده."""
    history = prepare_history(daily)
    out = {}
    for _ in range(days):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        pred = _predict(model, features, make_row(history, next_date))
        history.loc[next_date] = pred
        out[next_date] = pred
    return pd.Series(out, name="forecast")


def predict_from_inputs(model, features, lag_1, lag_2, lag_7, roll_7, roll_30, date):
    """توقع استهلاك يوم من أرقام يدخلها المستخدم."""
    date = pd.Timestamp(date)
    row = {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_7": lag_7,
        "roll_7": roll_7,
        "roll_30": roll_30,
        "dow": date.dayofweek,
        "month": date.month,
    }
    return _predict(model, features, row)