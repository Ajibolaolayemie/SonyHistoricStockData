import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
import numpy as np
import seaborn as sns
import pandas_datareader.data as web
import datetime
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import sklearn as skl
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('SONY.csv')
print(df.head())

# Display the first few rows
print(df.head())

# Display the last few rows
print(df.tail())

# Display data types of each column
print(df.dtypes)

# Group by a column and calculate aggregate statistics
print(df.columns)

# Group by a column and calculate aggregate statistics
print(df.groupby('Date')['High'].describe())

# Plot a time series'
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Close', data=df)
plt.title('Closed trade Over Time')
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()

# Columns to include in correlation matrix
columns_corr = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Select these columns from the DataFrame
df_selected = df[columns_corr]

# Create correlation matrix
corr_mat = df_selected.corr()

# Plot correlation matrix
plt.matshow(corr_mat, cmap='coolwarm')
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()

# creating datetime index
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.index = df['Date']
df = df.drop(columns=['Date'])
print(df.index)
print(df.columns)

# plotting variables to see trend
sns.set()
plt.ylabel('Close')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(df.index, df['Close'], )
plt.show()

# training and testing data
cutoff_date = pd.to_datetime("2020-11-01", format='%Y-%m-%d')
train = df[df.index < cutoff_date]
test = df[df.index > cutoff_date]

plt.plot(train.index, train['Close'], color="black", label="train")
plt.plot(test.index, test['Close'], color="red", label="test")
plt.ylabel('ClosePrice')
plt.xlabel('Date')
plt.xticks(rotation=55)
plt.title("Train/Test split for SonyData")
plt.show()

# Predicting using ARMA Model
y = train['Close']
arima_model = ARIMA(y, order=(1, 0, 1))
arima_model_result = arima_model.fit()
arima_forecast = arima_model_result.forecast(steps=len(test))
forecast_index = test.index

# Predicting using SARIMA Model
sarima_model = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
sarima_model_result = sarima_model.fit()
sarima_forecast = sarima_model_result.forecast(steps=len(test))

plt.figure(figsize=(12, 6))

# Plot the train data
plt.plot(train.index, train['Close'], label='Train', color='blue')

# Plot the test data
plt.plot(test.index, test['Close'], label='Test', color='orange')

# Plot the ARIMA forecast
plt.plot(forecast_index, arima_forecast, label='ARIMA Forecast', color='red')

# Plot the SARIMA forecast
plt.plot(forecast_index, sarima_forecast, label='SARIMA Forecast', color='green')

# Improve x-axis formatting
plt.gca().xaxis.set_major_locator(YearLocator())  # Adjusts the x-axis ticks to show every year
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))  # Formats the ticks to 'YYYY-MM'

# Rotate date labels for better readability
plt.gcf().autofmt_xdate(rotation=45)

plt.title('ARIMA v SARIMA Forecasting')
plt.ylabel('Close Price')
plt.xlabel('Date')

plt.legend()
plt.show()

arima_residuals = arima_model_result.resid
sarima_residuals = sarima_model_result.resid

# Plot ARIMA residuals
plt.figure(figsize=(10, 4))
plt.plot(arima_residuals)
plt.title('ARIMA Residuals')
plt.show()

# Plot SARIMA residuals
plt.figure(figsize=(10, 4))
plt.plot(sarima_residuals)
plt.title('SARIMA Residuals')
plt.show()

# AIC and BIC for ARIMA
arima_aic = arima_model_result.aic
arima_bic = arima_model_result.bic

# AIC and BIC for SARIMA
sarima_aic = sarima_model_result.aic
sarima_bic = sarima_model_result.bic

print(f'ARIMA AIC: {arima_aic}, BIC: {arima_bic}')
print(f'SARIMA AIC: {sarima_aic}, BIC: {sarima_bic}')

# y_pred = ARMAmodel.get_forecast(len(test.index))
# y_pred_df = y_pred.conf_int(alpha=0.05)
# y_pred_df["Predictions"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
# y_pred_df.index = test.index
# y_pred_out = y_pred_df["Predictions"]
# plt.plot(y_pred_out, color='green', label = 'Predictions')
# plt.legend()
# plt.show()
