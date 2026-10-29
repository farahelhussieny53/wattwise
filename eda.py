import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

daily = pd.read_csv("data/daily.csv", index_col=0, parse_dates=True)

print(daily["kwh"].describe())

# 1) الاستهلاك اليومي عبر السنين
plt.figure(figsize=(12, 4))
daily["kwh"].plot()
plt.title("Daily consumption (kWh)")
plt.ylabel("kWh")
plt.tight_layout()
plt.savefig("plot_daily.png")

# 2) المتوسط حسب الشهر (الموسمية)
plt.figure(figsize=(8, 4))
daily.groupby(daily.index.month)["kwh"].mean().plot(kind="bar")
plt.title("Average consumption by month")
plt.xlabel("Month")
plt.ylabel("kWh")
plt.tight_layout()
plt.savefig("plot_month.png")

# 3) المتوسط حسب يوم الأسبوع
plt.figure(figsize=(8, 4))
daily.groupby(daily.index.dayofweek)["kwh"].mean().plot(kind="bar")
plt.title("Average consumption by day of week (0=Mon)")
plt.xlabel("Day of week")
plt.ylabel("kWh")
plt.tight_layout()
plt.savefig("plot_dow.png")

# 4) توزيع الأيام
plt.figure(figsize=(8, 4))
daily["kwh"].hist(bins=40)
plt.title("Distribution of daily consumption")
plt.xlabel("kWh")
plt.tight_layout()
plt.savefig("plot_hist.png")

plt.show()