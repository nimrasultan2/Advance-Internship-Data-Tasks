import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# 1. LOAD DATA 
df = pd.read_csv(".venv\household_power_consumption.csv");

# 2. PARSE DATETIME
# Combine Date and Time columns into a single datetime object
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df = df.set_index('datetime')

# Convert power column to numeric (in case of remaining strings)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['Global_active_power'])

# 3. RESAMPLE TO HOURLY (As required by instructions)
df_hourly = df['Global_active_power'].resample('H').mean()

# 4. FEATURE ENGINEERING
def create_features(df_series):
    features = pd.DataFrame(index=df_series.index)
    features['target'] = df_series.values
    features['hour'] = features.index.hour
    features['dayofweek'] = features.index.dayofweek
    # Instructions ask for weekend feature
    features['is_weekend'] = features.index.dayofweek.isin([5, 6]).astype(int)
    # Lag feature (24 hours ago) helps XGBoost see patterns
    features['lag_24h'] = features['target'].shift(24)
    return features.dropna()

data = create_features(df_hourly)

# Split data (Example: use last 20% for testing)
split_idx = int(len(data) * 0.8)
train, test = data.iloc[:split_idx], data.iloc[split_idx:]

# 5. XGBOOST MODEL
X_train, y_train = train.drop('target', axis=1), train['target']
X_test, y_test = test.drop('target', axis=1), test['target']

xgb_model = XGBRegressor(n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# 6. QUICK VISUALIZATION (Actual vs Forecasted)
plt.figure(figsize=(12, 5))
plt.plot(y_test.index[:100], y_test[:100], label='Actual Usage', color='blue')
plt.plot(y_test.index[:100], xgb_preds[:100], label='XGBoost Forecast', color='orange', linestyle='--')
plt.title('Short-term Household Energy Usage Forecast')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.show()

print(f"XGBoost MAE: {mean_absolute_error(y_test, xgb_preds):.4f}")