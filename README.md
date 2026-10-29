# ⚡ WattWise: Household Electricity Consumption Forecasting

**Live demo:** https://wattwise-farah.streamlit.app

A machine learning web app that analyses a household's electricity usage and forecasts daily consumption. Enter recent readings, pick a date, and get a prediction with a consumption level (Low / Medium / High) relative to your own average.

> **Note:** The app may go to sleep when it has not been visited for a while. If you see a "wake up" button, click it and wait a minute.

## What it does

- **Try it yourself:** enter recent daily consumption (kWh) and a target date to get a predicted value, the difference from your 30-day average, and a relative consumption level.
- **Forecast:** recursive forecast of the next 1 to 7 days, plotted after the recent history.
- **Usage patterns:** average consumption by month and by weekday.
- **Model performance:** transparent comparison against simple baselines on a held-out period.

## Dataset

[Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from the UCI Machine Learning Repository: one household, about 2.07 million minute-level readings from December 2006 to November 2010.

Preprocessing:

- Missing values (`?` in the raw file, about 1.25% of minutes) are converted to NaN.
- Minute readings are averaged into hourly values. Hours with fewer than 30 valid minutes are treated as missing.
- Hours are summed into daily consumption. Days with fewer than 20 valid hours are dropped, leaving **1,420 clean days**.

## Method

**Features** (built only from information available before the target day):

- consumption 1, 2 and 7 days earlier
- 7-day and 30-day rolling averages (shifted so the target day never leaks into its own features)
- day of week and month

All consumption features are divided by the 30-day average, and the model predicts the target as a **multiple of that average**. This makes the model scale-free: it learns weekday and seasonal patterns instead of one household's absolute kWh level.

**Evaluation:** chronological 80/20 split (no shuffling). Train: 2007-01-06 to 2010-01-29 (1,086 days). Test: 2010-01-30 to 2010-11-26 (272 days).

## Results (test period)

| Model | MAE (kWh) | RMSE (kWh) | R² |
|---|---|---|---|
| Baseline: same as yesterday | 4.99 | 6.91 | 0.224 |
| Baseline: same weekday last week | 5.83 | 7.95 | -0.026 |
| Linear Regression | 4.30 | 5.91 | 0.432 |
| **Random Forest** (selected, lowest RMSE) | 4.34 | **5.74** | **0.465** |

- The selected model reduces error by about **13% (MAE) and 17% (RMSE)** compared with the "same as yesterday" baseline, and by about 26% (MAE) compared with the weekly baseline.
- Average error is roughly **17.6% of daily consumption**.
- Linear Regression and Random Forest are close. Random Forest was chosen for its lower RMSE and because it can capture non-linear weekday and seasonal effects.
- Error is higher on weekends (MAE about 5.8 kWh) than on weekdays (about 3.8 kWh).

## Findings from the data

- Strong seasonality: December and January average about 35 kWh/day, versus about 14 kWh/day in August.
- Weekends are about 4 to 5 kWh/day higher than weekdays.
- A sharp drop in August 2008 (about 4 kWh/day for several days) looks like an empty home. It sets the household's base load.

## Limitations

- **One household only.** The model learned patterns from a single French home. Homes with very different habits (for example gas heating or frequent travel) may behave differently.
- Weather, guests and travel are not in the data, so about half of the day-to-day variation is not explained.
- Unusual days (such as a holiday trip) cannot be predicted from past consumption.
- Multi-day forecasts are recursive, so errors accumulate with the horizon.
- This is an analytical demo, not a validated system for guaranteeing savings.

## Run it locally

```bash
git clone https://github.com/farahelhussieny53/wattwise.git
cd wattwise
pip install -r requirements.txt
python -m streamlit run app.py
```

The processed file `data/daily.csv` and the trained model are included, so the app runs without the raw dataset. To rebuild everything from scratch:

1. Download `household_power_consumption.txt` from UCI and place it in `data/`.
2. `python explore.py` builds the hourly and daily files.
3. `python eda.py` creates exploratory plots (optional).
4. `python train.py` trains and evaluates the models and saves `model.joblib` and `metrics.json`.

## Project structure

```
wattwise/
├── app.py              # Streamlit app
├── train.py            # training and evaluation
├── explore.py          # raw minute data -> hourly and daily data
├── eda.py              # exploratory plots
├── model.joblib        # trained model
├── metrics.json        # evaluation results shown in the app
├── requirements.txt
├── data/daily.csv      # processed daily consumption
└── src/
    ├── data.py         # loading, cleaning, aggregation
    ├── features.py     # feature engineering, relative transform, split
    └── forecast.py     # prediction and recursive forecasting
```

## Built with

Python, pandas, NumPy, scikit-learn, Streamlit.

## Possible next steps

- Let users upload their own CSV and retrain the model on their data.
- Evaluate on multiple households.
- Add weather and holiday features.
- Add uncertainty intervals to multi-day forecasts.

## Author

Farah Awad, [GitHub](https://github.com/farahelhussieny53)