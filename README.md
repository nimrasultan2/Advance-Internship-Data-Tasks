HOSEHOLD ENERGY USAGE FORCASTER


This project forecasts short-term household electricity consumption using an XGBoost regression model trained on hourly power readings. Raw per-minute data is resampled to hourly averages, and features like hour of day, day of week, weekend flag, and a 24-hour lag are added to help the model catch daily usage patterns. 
The last 20% of the timeline is held out for testing, and performance is measured using Mean Absolute Error (MAE).


1. Resamples minute-level data into hourly averages for cleaner modeling
2. Engineers time-based features including a 24-hour lag as a strong predictor
3. Trains XGBoost on 80% of the data and evaluates on the remaining 20%
4. Plots the first 100 hours of actual vs. predicted power usage for a quick visual check
