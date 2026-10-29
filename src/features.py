import pandas as pd

FEATURES = ["lag_1", "lag_2", "lag_7", "roll_7", "roll_30", "dow", "month"]

def build_features(daily):
    # نكمّل الأيام الناقصة عشان lag_1 يبقى فعلاً "امبارح"
    df = daily[["kwh"]].asfreq("D")
    df["lag_1"] = df["kwh"].shift(1)
    df["lag_2"] = df["kwh"].shift(2)
    df["lag_7"] = df["kwh"].shift(7)
    # shift(1) قبل rolling عشان اليوم نفسه ميدخلش في متوسطه
    df["roll_7"] = df["kwh"].shift(1).rolling(7, min_periods=5).mean()
    df["roll_30"] = df["kwh"].shift(1).rolling(30, min_periods=20).mean()
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    # نشيل الصفوف اللي ناقصها هدف أو خصائص
    return df.dropna()

def chrono_split(df, train_ratio=0.8):
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut], df.iloc[cut:]