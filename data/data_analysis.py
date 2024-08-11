import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas_datareader.data as web
import datetime


df = pd.read_csv('SONY.csv')
print (df.head())

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
columns_corr = ['Open', 'High', 'Low','Close', 'Adj Close']

# Select these columns from the DataFrame
df_selected = df[columns_corr]

# Create correlation matrix
corr_mat = df_selected.corr()

# Plot correlation matrix
plt.matshow(corr_mat, cmap='coolwarm')
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()


# df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# train = df[df.Close < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
# test = df[df.Close > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
#
# print(train)
# plt.plot(train)  # Basic plot without additional parameters
# plt.show()
# print(test)

