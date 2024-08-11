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

df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
training_data = df[df.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
test_data = df[df.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
# Load the CSV file into a DataFrame
# df = pd.read_csv(
#     'SONY.csv', parse_dates=['Date'])

# Display the first few rows of the DataFrame
# print(sony_data)
#
# sony_data.head()

# print(sony_data)

# print(df)
#
# corr_matrix = df.corr()
#
# print(corr_matrix)


# print(df.head())
#
# print(df.isnull().sum())
#
# print(df['Open'].value_counts())
# print(df['High'].value_counts())
# print(df['Low'].value_counts())
#
# correlation = df[['Value']].corr()
# print(correlation)
#
# # Plot a histogram of 'Value'
# plt.figure(figsize=(10, 6))
# sns.histplot(df['High'], bins=30, kde=True)
# plt.title('Distribution of Values')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()
#
# Plot a time series'
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Close', data=df)
plt.title('Closed trade Over Time')
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()

df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
train = df[df.Close < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
test = df[df.Close > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]

print(train)
plt.plot(train)  # Basic plot without additional parameters
plt.show()
print(test)

# Plot a time series'
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Close', data=df)
plt.xticks(rotation=45)
plt.title("Train for Sony Closing Stock")
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()