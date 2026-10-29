import pandas as pd

def load_minute_data(path):
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.set_index("datetime").sort_index()
    return df[["Global_active_power"]]

def to_hourly(minute_df, min_valid_minutes=30):
    g = minute_df["Global_active_power"].resample("h")
    hourly = pd.DataFrame({"kwh": g.mean(), "valid_minutes": g.count()})
    hourly.loc[hourly["valid_minutes"] < min_valid_minutes, "kwh"] = float("nan")
    return hourly

def to_daily(hourly_df, min_valid_hours=20):
    g = hourly_df["kwh"].resample("D")
    daily = pd.DataFrame({"kwh": g.sum(min_count=1), "valid_hours": g.count()})
    daily = daily[daily["valid_hours"] >= min_valid_hours]
    return daily
    