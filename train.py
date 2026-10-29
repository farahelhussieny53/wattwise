import json
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import build_features, chrono_split, FEATURES

daily = pd.read_csv("data/daily.csv", index_col=0, parse_dates=True)
df = build_features(daily)
train, test = chrono_split(df)

print("train:", train.index.min().date(), "->", train.index.max().date(), len(train))
print("test :", test.index.min().date(), "->", test.index.max().date(), len(test))

results = []

def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<22} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:6.3f}")
    results.append({"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3)})
    return mae

y_test = test["kwh"]
print("\n--- Results on test set ---")
report("Baseline: yesterday", y_test, test["lag_1"])
report("Baseline: last week", y_test, test["lag_7"])

lr = LinearRegression().fit(train[FEATURES], train["kwh"])
mae_lr = report("Linear Regression", y_test, lr.predict(test[FEATURES]))

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(train[FEATURES], train["kwh"])
mae_rf = report("Random Forest", y_test, rf.predict(test[FEATURES]))

best_name, best = ("Linear Regression", lr) if mae_lr <= mae_rf else ("Random Forest", rf)
print("\nBest model:", best_name)

# نعيد التدريب على كل الداتا قبل الحفظ للنشر
best.fit(df[FEATURES], df["kwh"])
joblib.dump({"model": best, "features": FEATURES, "name": best_name}, "model.joblib")

info = {
    "best_model": best_name,
    "train_period": [str(train.index.min().date()), str(train.index.max().date())],
    "test_period": [str(test.index.min().date()), str(test.index.max().date())],
    "results": results,
}
with open("metrics.json", "w") as f:
    json.dump(info, f, indent=2)
print("saved model.joblib and metrics.json")