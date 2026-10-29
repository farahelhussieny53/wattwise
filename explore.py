from src.data import load_minute_data, to_hourly, to_daily

minute = load_minute_data("data/household_power_consumption.txt")
print("minute rows:", len(minute), "| missing:", minute["Global_active_power"].isna().sum())

hourly = to_hourly(minute)
print("hourly rows:", len(hourly), "| missing hours:", hourly["kwh"].isna().sum())

daily = to_daily(hourly)
print("daily rows kept:", len(daily))
print(daily.describe().T)
print(daily.head())

hourly.to_csv("data/hourly_cache.csv")
daily.to_csv("data/daily.csv")

