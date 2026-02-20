import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv(".venv\household_power_consumption.csv")

# Combine the separate Date and Time columns into one proper datetime column
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df = df.set_index('datetime')

# Make sure power readings are numbers — some may have slipped in as text
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Remove any rows where power data is missing
df = df.dropna(subset=['Global_active_power'])

# Compress the data from per-minute readings down to hourly averages
df_hourly = df['Global_active_power'].resample('H').mean()

# Build extra columns the model can learn from
def create_features(df_series):
    features = pd.DataFrame(index=df_series.index)
    features['target'] = df_series.values
    features['hour'] = features.index.hour
    features['dayofweek'] = features.index.dayofweek
    # Weekends tend to have different usage patterns, so we flag them
    features['is_weekend'] = features.index.dayofweek.isin([5, 6]).astype(int)
    # What was the power usage exactly 24 hours ago? That's a strong signal
    features['lag_24h'] = features['target'].shift(24)
    return features.dropna()

data = create_features(df_hourly)

# Use the first 80% of the timeline for training, the rest for testing
split_idx = int(len(data) * 0.8)
train, test = data.iloc[:split_idx], data.iloc[split_idx:]

# Separate inputs from the value we're trying to predict
X_train, y_train = train.drop('target', axis=1), train['target']
X_test, y_test = test.drop('target', axis=1), test['target']

# Train the XGBoost model on historical energy patterns
xgb_model = XGBRegressor(n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Plot the first 100 hours of actual vs predicted usage to see how close we got
plt.figure(figsize=(12, 5))
plt.plot(y_test.index[:100], y_test[:100], label='Actual Usage', color='blue')
plt.plot(y_test.index[:100], xgb_preds[:100], label='XGBoost Forecast', color='orange', linestyle='--')
plt.title('Short-term Household Energy Usage Forecast')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.show()

# Print the average error so we know how far off our predictions are
print(f"XGBoost MAE: {mean_absolute_error(y_test, xgb_preds):.4f}")