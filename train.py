import json
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import build_features, chrono_split, to_relative, REL_FEATURES

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
    return mae, rmse

y_test = test["kwh"]
print("\n--- Results on test set ---")
report("Baseline: yesterday", y_test, test["lag_1"])
report("Baseline: last week", y_test, test["lag_7"])

# الموديلات بتتدرب على النسبة: kwh / متوسط آخر 30 يوم
X_train, X_test = to_relative(train)[REL_FEATURES], to_relative(test)[REL_FEATURES]
y_train_ratio = train["kwh"] / train["roll_30"]

lr = LinearRegression().fit(X_train, y_train_ratio)
mae_lr, rmse_lr = report("Linear Regression", y_test, lr.predict(X_test) * test["roll_30"])

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_ratio)
mae_rf, rmse_rf = report("Random Forest", y_test, rf.predict(X_test) * test["roll_30"])

# نختار بالـ RMSE (بيعاقب الأخطاء الكبيرة، ودي مهمة في الاستهلاك)
best_name, best, best_mae = (
    ("Linear Regression", lr, mae_lr) if rmse_lr <= rmse_rf else ("Random Forest", rf, mae_rf)
)
print("\nBest model:", best_name)

# نعيد التدريب على كل الداتا قبل الحفظ للنشر
best.fit(to_relative(df)[REL_FEATURES], df["kwh"] / df["roll_30"])
joblib.dump({"model": best, "features": REL_FEATURES, "name": best_name}, "model.joblib")

info = {
    "best_model": best_name,
    "best_mae_pct": round(best_mae / y_test.mean() * 100, 1),
    "train_period": [str(train.index.min().date()), str(train.index.max().date())],
    "test_period": [str(test.index.min().date()), str(test.index.max().date())],
    "results": results,
}
with open("metrics.json", "w") as f:
    json.dump(info, f, indent=2)
print("saved model.joblib and metrics.json")