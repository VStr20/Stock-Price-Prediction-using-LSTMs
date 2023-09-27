#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
#importing the necessary libaries


# In[48]:


from numpy import loadtxt


# In[49]:


stock1 = pandas.read_csv("AAL_1min.txt",
                                   sep="\s+|,| ",
                                   header=None, 
                                   engine="python")

stock2 = pandas.read_csv("AAP_1min.txt",
                                   sep="\s+|,| ",
                                   header=None, 
                                   engine="python")
stock3 = pandas.read_csv("AAPL_1min.txt",
                                   sep="\s+|,| ",
                                   header=None, 
                                   engine="python")
# The files are imported using the read_csv command with the required delimiters.


# In[50]:


print(stock1.shape)


# Q1

# In[51]:


name = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
stock1.columns = name
stock2.columns = name
stock3.columns = name


# In[52]:


stock1


# In[53]:


stock1_closing = stock1[stock1['Date'] == stock1['Date'][0]]
n1 = len(stock1_closing)

stock2_closing = stock2[stock2['Date'] == stock2['Date'][0]]
n2 = len(stock1_closing)

stock3_closing = stock3[stock3['Date'] == stock3['Date'][0]]
n3 = len(stock1_closing)

# Filtering rows based on matching dates in the 'Date' column for three DataFrames.
# Calculating the number of rows in each filtered DataFrame.


# Q1 a)

# In[54]:


plt.plot(pandas.DatetimeIndex(data=stock1_closing['Time'][int(0.05*n3):int(0.95*n3)]).hour+pandas.DatetimeIndex(data=stock1_closing['Time'][int(0.05*n3):int(0.95*n3)]).minute/60 ,stock1_closing['Close'][int(0.05*n3):int(0.95*n3)])
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.show()

# Plotting the closing price of 'stock1_closing' within a specified time range.
# X-axis represents time (in hours and minutes), Y-axis represents closing price.
# Setting labels for the X and Y axes.


# Here the 5 % entries in the beginning and the end are removed so as to count only the prices during the open trade time.

# In[55]:


plt.plot(pandas.DatetimeIndex(data=stock2_closing['Time'][int(0.05*n3):int(0.95*n3)]).hour+pandas.DatetimeIndex(data=stock2_closing['Time'][int(0.05*n3):int(0.95*n3)]).minute/60 ,stock2_closing['Close'][int(0.05*n3):int(0.95*n3)])
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.show()

# Plotting the closing price of 'stock1_closing' within a specified time range.
# X-axis represents time (in hours and minutes), Y-axis represents closing price.
# Setting labels for the X and Y axes.


# In[56]:


plt.plot(pandas.DatetimeIndex(data=stock3_closing['Time'][int(0.05*n3):int(0.95*n3)]).hour+pandas.DatetimeIndex(data=stock3_closing['Time'][int(0.05*n3):int(0.95*n3)]).minute/60 ,stock3_closing['Close'][int(0.05*n3):int(0.95*n3)])
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.show()

# Plotting the closing price of 'stock1_closing' within a specified time range.
# X-axis represents time (in hours and minutes), Y-axis represents closing price.
# Setting labels for the X and Y axes.


# Q1 c)

# In[57]:


import plotly.graph_objects as go
from datetime import datetime

figure=go.Figure(go.Candlestick(close=stock3_closing['Close'][int(0.05*n3):int(0.95*n3)],open=stock3_closing['Open'][int(0.05*n3):int(0.95*n3)],high=stock3_closing['High'][int(0.05*n3):int(0.95*n3)],low=stock3_closing['Low'][int(0.05*n3):int(0.95*n3)],x=pandas.DatetimeIndex(data=stock3_closing['Time'][int(0.05*n3):int(0.95*n3)]).hour+pandas.DatetimeIndex(data=stock3_closing['Time'][int(0.05*n3):int(0.95*n3)]).minute/60, increasing_line_color= 'cyan', decreasing_line_color= 'gray'))
figure.show()

#https://plotly.com/python/candlestick-charts/

# Importing the necessary modules for creating a candlestick chart.
# Creating a candlestick chart using the 'stock3_closing' data within a specified time range.
# Customizing candlestick colors for increasing and decreasing trends.
# Displaying the candlestick chart using Plotly.


# In[58]:


figure=go.Figure(go.Candlestick(close=stock2_closing['Close'][int(0.05*n2):int(0.95*n2)],open=stock2_closing['Open'][int(0.05*n2):int(0.95*n2)],high=stock2_closing['High'][int(0.05*n2):int(0.95*n2)],low=stock2_closing['Low'][int(0.05*n2):int(0.95*n2)],x=pandas.DatetimeIndex(data=stock2_closing['Time'][int(0.05*n2):int(0.95*n2)]).hour+pandas.DatetimeIndex(data=stock2_closing['Time'][int(0.05*n2):int(0.95*n2)]).minute/60, increasing_line_color= 'cyan', decreasing_line_color= 'gray'))
figure.show()

# Importing the necessary modules for creating a candlestick chart.
# Creating a candlestick chart using the 'stock3_closing' data within a specified time range.
# Customizing candlestick colors for increasing and decreasing trends.
# Displaying the candlestick chart using Plotly.


# In[59]:


figure=go.Figure(go.Candlestick(close=stock1_closing['Close'][int(0.05*n1):int(0.95*n1)],open=stock1_closing['Open'][int(0.05*n1):int(0.95*n1)],high=stock1_closing['High'][int(0.05*n1):int(0.95*n1)],low=stock1_closing['Low'][int(0.05*n1):int(0.95*n1)],x=pandas.DatetimeIndex(data=stock1_closing['Time'][int(0.05*n1):int(0.95*n1)]).hour+pandas.DatetimeIndex(data=stock1_closing['Time'][int(0.05*n1):int(0.95*n1)]).minute/60, increasing_line_color= 'cyan', decreasing_line_color= 'gray'))
figure.show()

# Importing the necessary modules for creating a candlestick chart.
# Creating a candlestick chart using the 'stock3_closing' data within a specified time range.
# Customizing candlestick colors for increasing and decreasing trends.
# Displaying the candlestick chart using Plotly.


# In[60]:


stock1_ = stock1.copy()


# Q1 b

# In[61]:


# Convert the date column to datetime format
stock1_['Date'] = pandas.to_datetime(stock1_['Date'])

# Print the first five rows of the DataFrame
print(stock1_.head())


# In[62]:


# Extract unique dates
unique_dates = stock1_['Date'].dt.date.unique()
unique_dates

# https://saturncloud.io/blog/how-to-extract-unique-dates-from-time-series-using-python-pandas/


# In[63]:


stock1


# Q1 b)
# 
# Extract unique dates for inter day trading.

# In[64]:


# Initialize empty lists to store daily data for each stock.
stock1_day = []
stock2_day = []
stock3_day = []

# Initialize a count variable to limit the number of records processed per stock.
count = 0

# Loop through unique dates in 'stock1' and extract the last record for each day.
for date in stock1['Date'].unique():
    st1 = stock1.loc[stock1['Date'] == date]
    stock1_day.append(st1.iloc[len(st1) - 1, :])
    count += 1
    if count == 365:
        break

# Repeat the same process for 'stock2' and 'stock3'.
count = 0
for date in stock2['Date'].unique():
    st2 = stock2.loc[stock2['Date'] == date]
    stock2_day.append(st2.iloc[len(st2) - 1, :])
    count += 1
    if count == 365:
        break

count = 0
for date in stock3['Date'].unique():
    st3 = stock3.loc[stock3['Date'] == date]
    stock3_day.append(st3.iloc[len(st3) - 1, :])
    count += 1
    if count == 365:
        break


# In[65]:


# Initialize empty lists to store closing values for each stock.
closing_values1 = []
closing_values2 = []
closing_values3 = []

# Extract closing values from 'stock1_day' and store them in 'closing_values1'.
for series in stock1_day:
    closing_values1.append(series['Close'])

# Extract closing values from 'stock2_day' and store them in 'closing_values2'.
for series in stock2_day:
    closing_values2.append(series['Close'])

# Extract closing values from 'stock3_day' and store them in 'closing_values3'.
for series in stock3_day:
    closing_values3.append(series['Close'])


# In[66]:


x = np.arange(0, 365)
plt.plot(x, closing_values1)
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.show()

# Create an array 'x' representing days from 0 to 364.
# Plot the closing values for 'stock1' over the course of 365 days.
# Set labels for the X and Y axes.
# Display the plot.


# In[67]:


x = np.arange(0, 365)
plt.plot(x, closing_values2)
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.show()

# Create an array 'x' representing days from 0 to 364.
# Plot the closing values for 'stock1' over the course of 365 days.
# Set labels for the X and Y axes.
# Display the plot.


# In[68]:


x = np.arange(0, 365)
plt.plot(x, closing_values3)
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.show()

# Create an array 'x' representing days from 0 to 364.
# Plot the closing values for 'stock1' over the course of 365 days.
# Set labels for the X and Y axes.
# Display the plot.


# Q1 d) 
# 
# Note down your observations, e.g. are there any data issues, unexpected jumps,
# unexpected missing data etc.
# 
# -In some places the sampling is not even. Such Inconsistent sampling can lead to issues in modeling and analysis if not properly standardized.
# 
# -Some flat lines can be observed in the plots which indicates that the data for some time has not been recorded (missing).
# 
# -Some data is recorded during non-trading hours and some intra-day transactions are made. Since this is outside of the non-trading
# hours, there are inconsistencies within the data on daily basis.

# Q2) Data normalization

# In[69]:


# Z-Score normalization:
# Z = (X - X_mean) / X_std

stock1.describe()


# In[70]:


stock1_ = stock1.iloc[: 24*60*30].copy()
stock1_.columns = name
stock1_.describe()

# Extract the first 30 days of data (24 hours * 60 minutes * 30 days) from 'stock1' and create a copy.


# In[71]:


import math


# In[72]:


window_size = 24*60  # Number of previous entries to consider for mean and standard deviation

rolling_mean = stock1_['Open'].rolling(window=window_size).mean()
rolling_std = stock1_['Open'].rolling(window=window_size).std()
stock1_['Z_Open'] = (stock1_['Open'] - rolling_mean) / rolling_std

rolling_mean = stock1_['High'].rolling(window=window_size).mean()
rolling_std = stock1_['High'].rolling(window=window_size).std()
stock1_['Z_High'] = (stock1_['High'] - rolling_mean) / rolling_std

rolling_mean = stock1_['Low'].rolling(window=window_size).mean()
rolling_std = stock1_['Low'].rolling(window=window_size).std()
stock1_['Z_Low'] = (stock1_['Low'] - rolling_mean) / rolling_std

rolling_mean = stock1_['Close'].rolling(window=window_size).mean()
rolling_std = stock1_['Close'].rolling(window=window_size).std()
stock1_['Z_Close'] = (stock1_['Close'] - rolling_mean) / rolling_std

rolling_mean = stock1_['Volume'].rolling(window=window_size).mean()
rolling_std = stock1_['Volume'].rolling(window=window_size).std()
stock1_['Z_Volume'] = (stock1_['Volume'] - rolling_mean) / rolling_std

#

rolling_mean = stock1_['Open'].rolling(window=window_size).mean()
rolling_std = stock1_['Open'].rolling(window=window_size).std()
stock1_['T_Open'] = 0.5*(np.tanh(0.01*(stock1_['Open'] - rolling_mean) / rolling_std)+1)

rolling_mean = stock1_['High'].rolling(window=window_size).mean()
rolling_std = stock1_['High'].rolling(window=window_size).std()
stock1_['T_High'] = 0.5*(np.tanh(0.01*(stock1_['High'] - rolling_mean) / rolling_std)+1)

rolling_mean = stock1_['Low'].rolling(window=window_size).mean()
rolling_std = stock1_['Low'].rolling(window=window_size).std()
stock1_['T_Low'] = 0.5*(np.tanh(0.01*(stock1_['Low'] - rolling_mean) / rolling_std)+1)

rolling_mean = stock1_['Close'].rolling(window=window_size).mean()
rolling_std = stock1_['Close'].rolling(window=window_size).std()
stock1_['T_Close'] = 0.5*(np.tanh(0.01*(stock1_['Close'] - rolling_mean) / rolling_std)+1)

rolling_mean = stock1_['Volume'].rolling(window=window_size).mean()
rolling_std = stock1_['Volume'].rolling(window=window_size).std()
stock1_['T_Volume'] = 0.5*(np.tanh(0.01*(stock1_['Volume'] - rolling_mean) / rolling_std)+1)

#

rolling_max = stock1_['Open'].rolling(window=window_size).max()
rolling_min = stock1_['Open'].rolling(window=window_size).min()
stock1_['X_Open'] = (stock1_['Open'] - rolling_min) / (rolling_max - rolling_min)

rolling_max = stock1_['High'].rolling(window=window_size).max()
rolling_min = stock1_['High'].rolling(window=window_size).min()
stock1_['X_High'] = (stock1_['High'] - rolling_min) / (rolling_max - rolling_min)

rolling_max = stock1_['Low'].rolling(window=window_size).max()
rolling_min = stock1_['Low'].rolling(window=window_size).min()
stock1_['X_Low'] = (stock1_['Low'] - rolling_min) / (rolling_max - rolling_min)

rolling_max = stock1_['Close'].rolling(window=window_size).max()
rolling_min = stock1_['Close'].rolling(window=window_size).min()
stock1_['X_Close'] = (stock1_['Close'] - rolling_min) / (rolling_max - rolling_min)

rolling_max = stock1_['Volume'].rolling(window=window_size).max()
rolling_min = stock1_['Volume'].rolling(window=window_size).min()
stock1_['X_Volume'] = (stock1_['Open'] - rolling_min) / (rolling_max - rolling_min)


# filling the empty spaces
stock1_['T_Open'].fillna(0.5*(np.tanh(0.01*(stock1_['Open'] - stock1_['Open'].mean()) / stock1_['Open'])+1), inplace=True)
stock1_['T_High'].fillna(0.5*(np.tanh(0.01*(stock1_['High'] - stock1_['High'].mean()) / stock1_['High'])+1), inplace=True)
stock1_['T_Low'].fillna(0.5*(np.tanh(0.01*(stock1_['Low'] - stock1_['Low'].mean()) / stock1_['Low'])+1), inplace=True)
stock1_['T_Close'].fillna(0.5*(np.tanh(0.01*(stock1_['Close'] - stock1_['Close'].mean()) / stock1_['Close'])+1), inplace=True)
stock1_['T_Volume'].fillna(0.5*(np.tanh(0.01*(stock1_['Volume'] - stock1_['Volume'].mean()) / stock1_['Volume'])+1), inplace=True)

stock1_['Z_Open'].fillna((stock1_['Open']-stock1_['Open'].mean())/stock1_['Open'].std(), inplace=True)
stock1_['Z_High'].fillna((stock1_['High']-stock1_['High'].mean())/stock1_['High'].std(), inplace=True)
stock1_['Z_Low'].fillna((stock1_['Low']-stock1_['Low'].mean())/stock1_['Low'].std(), inplace=True)
stock1_['Z_Close'].fillna((stock1_['Close']-stock1_['Close'].mean())/stock1_['Close'].std(), inplace=True)
stock1_['Z_Volume'].fillna((stock1_['Volume']-stock1_['Volume'].mean())/stock1_['Volume'].std(), inplace=True)

stock1_['X_Open'].fillna((stock1_['Open']-stock1_['Open'].min())/(stock1_['Open'].max()-stock1_['Open'].min()), inplace=True)
stock1_['X_High'].fillna((stock1_['High']-stock1_['High'].min())/(stock1_['High'].max()-stock1_['High'].min()), inplace=True)
stock1_['X_Low'].fillna((stock1_['Low']-stock1_['Low'].min())/(stock1_['Low'].max()-stock1_['Low'].min()), inplace=True)
stock1_['X_Close'].fillna((stock1_['Close']-stock1_['Close'].min())/(stock1_['Close'].max()-stock1_['Close'].min()), inplace=True)
stock1_['X_Volume'].fillna((stock1_['Volume']-stock1_['Volume'].min())/(stock1_['Volume'].max()-stock1_['Volume'].min()), inplace=True)

stock1_

# Using mean of the past few days
# Calculate rolling mean and standard deviation for 'Open', 'High', 'Low', 'Close', and 'Volume' columns.
# Normalize 'Open', 'High', 'Low', 'Close', and 'Volume' using z-score (Z_*) transformation.
# Normalize 'Open', 'High', 'Low', 'Close', and 'Volume' using min-max (X_*) scaling.
# Fill missing values in 'T_*', 'Z_*', and 'X_*' columns with corresponding transformations or default values.
# This code prepares and preprocesses the data for further analysis.


# In[73]:


stock1closing = stock1_[stock1_['Date'] == stock1_['Date'][0]]
# stock1[5].size()
n = len(stock1closing)


# In[74]:


stock1_.describe()


# In[75]:


plt.plot(pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).hour+pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).minute/60 ,stock1_['X_Open'][int(0.05*n):int(0.95*n)], label='X')

plt.xlabel("Time")
plt.ylabel("Prices")
plt.show()

# Plotting a subset of 'X_Open' values from 'stock1_' against time.
# X-axis represents time (in hours and minutes), Y-axis represents 'X_Open' values.


# In[76]:


plt.plot(pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).hour+pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).minute/60 ,stock1_['Z_Open'][int(0.05*n):int(0.95*n)])
plt.xlabel("Time")
plt.ylabel("Prices")
plt.show()

# Plotting a subset of 'X_Open' values from 'stock1_' against time.
# X-axis represents time (in hours and minutes), Y-axis represents 'X_Open' values.


# In[77]:


plt.plot(pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).hour+pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).minute/60 ,stock1_['T_Open'][int(0.05*n):int(0.95*n)])
plt.xlabel("Time")
plt.ylabel("Prices")
plt.show()

# Plotting a subset of 'X_Open' values from 'stock1_' against time.
# X-axis represents time (in hours and minutes), Y-axis represents 'X_Open' values.


# In[78]:


plt.plot(pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).hour+pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).minute/60 ,stock1_['X_Open'][int(0.05*n):int(0.95*n)], label='X')
plt.plot(pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).hour+pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).minute/60 ,stock1_['Z_Open'][int(0.05*n):int(0.95*n)], label='Z')
plt.plot(pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).hour+pandas.DatetimeIndex(data=stock1_['Time'][int(0.05*n):int(0.95*n)]).minute/60 ,stock1_['T_Open'][int(0.05*n):int(0.95*n)], label='T')
plt.show()

# Plotting 'X_Open', 'Z_Open', and 'T_Open' values from 'stock1_' against time.
# X-axis represents time (in hours and minutes), Y-axis represents corresponding values.


# Z-normalization
# Z-normalization transforms data to have a mean of 0 and a standard deviation of 1, making it centered around zero.
# This preserves the data's original distribution
# 
# Max-Min normalization
# Min-Max scaling scales data to a specified range, typically between 0 and 1.
# 
# Tanh estimator
# This method is one of the most powerful and efficient normalization technique.
# 
# From the plot above, it can be concluded that tanh estimator is the most efficient technique followed by Max-min scaling and Z-normalization
# For the observations from the data and reading various sources (attached below), I concluded to use Max-Min scaling 
# for data normalization considering the tradeoff between eficiency and computation.
# 
# 
# https://arxiv.org/ftp/arxiv/papers/1812/1812.05519.pdf

# In[79]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Q3. 
# 
# Make some scenario decisions: [1]
# a) high-frequency trading or intra-day swing, or inter-day trade, or long-term (multi-day ormulti-week or multi-month).
# b) Assume a buy-ask spread (inversely related to volume and directly related to price) andtrade commission based on a quick market research. Your trade will lose the buy-askspread and commissions every time you trade.
# c) Decide if you will trade only one stock, or have a model to trade a basket from a particularindustry, or any stock.

# Q3)
# 
# a)
# High-Frequency Trading (HFT):
# 
# Trading Strategy: High-frequency trading involves making a large number of trades within a short time, holding market positions only for seconds or minutes.
# Real-time market data and complex trading algorithms capable of executing quick trades and making rapid decisions corresponding to teh market signals.
# Higher trading costs due to frequent trading and narrow profit margins as trades are occuring in fraction of seconds.
# These types are carried by large financial firms having access to fast computation systems.
# 
# Intra-Day Swing Trading:
# 
# Trading Strategy: Intra-day swing trading involves holding positions for a few hours within a single trading day.
# For this purpose charts and technical analysis are required to identify market signals. Fundamental factors can also play a role.
# Traders look for intraday price swings and trends. Intraday trades aim for larger price movements compared to HFT since hence trades are carried oevr a longer time frame.
# Trading Costs: Trading costs are moderate, as we make fewer trades compared to HFT.
#     
# Inter-Day Trading:
# 
# Trading Strategy: Inter-day trading involves holding positions overnight but closing them before the next trading session.
# Data and Research: Analyze daily and longer-term charts, use technical and fundamental analysis, and consider news and events.
# Have a close look at multi-hour to multi-day trends to make profits from trades. Swing trades aim for more significant price movements.
# Trading costs are lower than HFT but may still be moderate due to longer holding periods.
# 
# Long-Term Trading:
# 
# Trading Strategy: Long-term trading involves holding positions for an extended period, ranging from several days to months or even years.
# Analyze daily, weekly, or monthly charts, use fundamental analysis, and consider macroeconomic factors to make good returns for this investment.
# Captures larger trends and major price movements.
# Trading costs are relatively low as you make fewer trades and hold positions for extended periods.
# 
# 
# Considering all the scenarios, the type of trading implemented here is intra day trading considering the computational limitations and efficiency. 
# 
# b)
# 
# A bid-ask spread is the amount by which the ask price exceeds the bid price for an asset in the market. The bid-ask spread is essentially the difference between the highest price that a buyer is willing to pay for an asset and the lowest price that a seller is willing to accept.
# The spread is the transaction cost. Price takers buy at the ask price and sell at the bid price, but the market maker buys at the bid price and sells at the ask price.
# The bid represents demand and the ask represents supply for an asset.
# 
# c)
# 
# Single Stock:
# 
# Focus on trading a specific stock or asset. This approach simplifies analysis.
# Ideal for those with a deep understanding of a specific company or asset.
# 
# Diversified Portfolio:
# 
# Trade a diverse range of assets from various industries or sectors. This provides more diversification but requires a broader analysis.
# Suitable for those who want to spread risk across multiple assets and industries.
# Your choice of trading universe should align with your trading strategy, expertise, and risk tolerance.
# 
# Ultimately, trading decisions should be driven by your financial goals, risk appetite, access to resources, and your ability to implement your chosen strategy effectively. It's crucial to thoroughly research and plan your approach before executing any trades.

# To simplify things only the trading of 1 stock is shown here. However the same can be done for the other stocks as well

# References
# 
# Types of trading - Investopedia
# 
# https://www.investopedia.com/terms/b/bid-askspread.asp

# Q4)  Write a pytorch module for defining an LSTM model. Keep it flexible so that the input dimension,
# number of units, number of layers can easily be changed.

# In[80]:


pip install --upgrade pip


# In[81]:


pip install torch


# In[82]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# 
# https://arxiv.org/pdf/1402.1128.pdf
# 
# https://cnvrg.io/pytorch-lstm/
# 
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
# 
# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/

# In[87]:


window_size = 4*24*60  # Number of previous entries to consider for mean and standard deviation
odata = stock1.copy()
tdata = stock1.copy()


# Checking for Max-Min normalization

# In[88]:


import torch
import torch.nn as nn

rolling_max = odata['Open'].rolling(window=window_size).max()
rolling_min = odata['Open'].rolling(window=window_size).min()
odata['X_Open'] = (odata['Open'] - rolling_min) / (rolling_max - rolling_min)

rolling_max = odata['High'].rolling(window=window_size).max()
rolling_min = odata['High'].rolling(window=window_size).min()
odata['X_High'] = (odata['High'] - rolling_min) / (rolling_max - rolling_min)

rolling_max = odata['Low'].rolling(window=window_size).max()
rolling_min = odata['Low'].rolling(window=window_size).min()
odata['X_Low'] = (odata['Low'] - rolling_min) / (rolling_max - rolling_min)

rolling_max = odata['Close'].rolling(window=window_size).max()
rolling_min = odata['Close'].rolling(window=window_size).min()
odata['X_Close'] = (odata['Close'] - rolling_min) / (rolling_max - rolling_min)

rolling_max = odata['Volume'].rolling(window=window_size).max()
rolling_min = odata['Volume'].rolling(window=window_size).min()
odata['X_Volume'] = (odata['Open'] - rolling_min) / (rolling_max - rolling_min)

odata['X_Open'].fillna((odata['Open']-odata['Open'].min())/(odata['Open'].max()-odata['Open'].min()), inplace=True)
odata['X_High'].fillna((odata['High']-odata['High'].min())/(odata['High'].max()-odata['High'].min()), inplace=True)
odata['X_Low'].fillna((odata['Low']-odata['Low'].min())/(odata['Low'].max()-odata['Low'].min()), inplace=True)
odata['X_Close'].fillna((odata['Close']-odata['Close'].min())/(odata['Close'].max()-odata['Close'].min()), inplace=True)
odata['X_Volume'].fillna((odata['Volume']-odata['Volume'].min())/(odata['Volume'].max()-odata['Volume'].min()), inplace=True)

odata

# Calculate rolling max and min values for 'Open', 'High', 'Low', 'Close', and 'Volume' columns.
# Normalize 'Open', 'High', 'Low', 'Close', and 'Volume' using min-max (X_*) scaling.
# Fill missing values in 'X_*' columns with corresponding transformations or default values.
# This code prepares and preprocesses the data for further analysis using PyTorch.


# Checking for tanh normalization

# In[137]:


import torch
import torch.nn as nn

rolling_mean = tdata['Open'].rolling(window=window_size).mean()
rolling_std = tdata['Open'].rolling(window=window_size).std()
tdata['T_Open'] = 0.5*(np.tanh(0.01*(tdata['Open'] - rolling_mean) / rolling_std)+1)

rolling_mean = tdata['High'].rolling(window=window_size).mean()
rolling_std = tdata['High'].rolling(window=window_size).std()
tdata['T_High'] = 0.5*(np.tanh(0.01*(tdata['High'] - rolling_mean) / rolling_std)+1)

rolling_mean = tdata['Low'].rolling(window=window_size).mean()
rolling_std = tdata['Low'].rolling(window=window_size).std()
tdata['T_Low'] = 0.5*(np.tanh(0.01*(tdata['Low'] - rolling_mean) / rolling_std)+1)

rolling_mean = tdata['Close'].rolling(window=window_size).mean()
rolling_std = tdata['Close'].rolling(window=window_size).std()
tdata['T_Close'] = 0.5*(np.tanh(0.01*(tdata['Close'] - rolling_mean) / rolling_std)+1)

rolling_mean = tdata['Volume'].rolling(window=window_size).mean()
rolling_std = tdata['Volume'].rolling(window=window_size).std()
tdata['T_Volume'] = 0.5*(np.tanh(0.01*(tdata['Volume'] - rolling_mean) / rolling_std)+1)

tdata['T_Open'].fillna(0.5*(np.tanh(0.01*(tdata['Open'] - tdata['Open'].mean()) / tdata['Open'])+1), inplace=True)
tdata['T_High'].fillna(0.5*(np.tanh(0.01*(tdata['High'] - tdata['High'].mean()) / tdata['High'])+1), inplace=True)
tdata['T_Low'].fillna(0.5*(np.tanh(0.01*(tdata['Low'] - tdata['Low'].mean()) / tdata['Low'])+1), inplace=True)
tdata['T_Close'].fillna(0.5*(np.tanh(0.01*(tdata['Close'] - tdata['Close'].mean()) / tdata['Close'])+1), inplace=True)
tdata['T_Volume'].fillna(0.5*(np.tanh(0.01*(tdata['Volume'] - tdata['Volume'].mean()) / tdata['Volume'])+1), inplace=True)

tdata

# Calculate rolling max and min values for 'Open', 'High', 'Low', 'Close', and 'Volume' columns.
# Normalize 'Open', 'High', 'Low', 'Close', and 'Volume' using min-max (X_*) scaling.
# Fill missing values in 'X_*' columns with corresponding transformations or default values.
# This code prepares and preprocesses the data for further analysis using PyTorch.


# Q4

# In[197]:


# Define a custom LSTM model using PyTorch's nn.Module.
# The model takes 'input_dim' features, outputs 'output_dim', and has 'num_of_unit' hidden units and 'num_of_layer' layers.

class LSTMmodel(nn.Module):
    def __init__(self, input_dim, output_dim, num_of_unit, num_of_layer):
        super(LSTMmodel, self).__init__()
        self.num_of_layers = num_of_layer
        self.hidden_states = num_of_unit
        
        # Define an LSTM layer with specified input dimensions, hidden units, and layers.
        self.lstm = nn.LSTM(input_dim, num_of_unit, num_of_layer, batch_first=True)
        
        # Define a fully connected (linear) layer to output the final results.
        self.fc = nn.Linear(num_of_unit, output_dim)
    
    def forward(self, input_x):
        # Initialize initial hidden and cell states.
        h0=torch.zeros(self.num_of_layers,input_x.size(0),self.hidden_states,dtype=torch.float64)
        c0=torch.zeros(self.num_of_layers,input_x.size(0),self.hidden_states,dtype=torch.float64)
        out,(hn,cn)=self.lstm(input_x.to(torch.float32),(h0.to(torch.float32),c0.to(torch.float32)))
        out=self.fc(out[:,-1,:])
        return out


# In[140]:


import pandas as pd
ndata = pd.DataFrame(odata)

# Select specific columns to create a new DataFrame
selected_columns = ['Date', 'Time', 'X_Open', 'X_High', 'X_Low', 'X_Close', 'X_Volume']
data = ndata[selected_columns]

odata['X_Close']=pd.to_numeric(odata['X_Close'])
odata['X_Open']=pd.to_numeric(odata['X_Open'])
odata['X_Low']=pd.to_numeric(odata['X_Low'])
odata['X_High']=pd.to_numeric(odata['X_High'])
odata['X_Volume']=pd.to_numeric(odata['X_Volume'])

# Create a new DataFrame 'ndata' from the preprocessed 'odata' DataFrame.
# Select specific columns ('Date', 'Time', 'X_Open', 'X_High', 'X_Low', 'X_Close', 'X_Volume') to create 'data'.

# Convert columns 'X_Close', 'X_Open', 'X_Low', 'X_High', and 'X_Volume' to numeric data types in 'odata'.

print(data)


# In[141]:


import pandas as pd
n1data = pd.DataFrame(tdata)

# Select specific columns to create a new DataFrame
selected_columns = ['Date', 'Time', 'T_Open', 'T_High', 'T_Low', 'T_Close', 'T_Volume']
data1 = n1data[selected_columns]

tdata['T_Close']=pd.to_numeric(tdata['T_Close'])
tdata['T_Open']=pd.to_numeric(tdata['T_Open'])
tdata['T_Low']=pd.to_numeric(tdata['T_Low'])
tdata['T_High']=pd.to_numeric(tdata['T_High'])
tdata['T_Volume']=pd.to_numeric(tdata['T_Volume'])

# Create a new DataFrame 'ndata' from the preprocessed 'odata' DataFrame.
# Select specific columns ('Date', 'Time', 'X_Open', 'X_High', 'X_Low', 'X_Close', 'X_Volume') to create 'data'.

# Convert columns 'X_Close', 'X_Open', 'X_Low', 'X_High', and 'X_Volume' to numeric data types in 'odata'.

print(data1)


# Q5

# In[142]:


stock1_day = pandas.DataFrame()
count = 0
i = 0
for date in data['Date'].unique():
    st1 = data.loc[data['Date'] == date]
    stock1_day = pandas.concat([stock1_day, st1.iloc[len(st1)-1, :]], axis=1)
    count+=1

stock1_day = stock1_day.transpose()
stock1_day

# Create an empty DataFrame 'stock1_day' to store daily data.
# Initialize counters 'count' and 'i'.

# Loop through unique dates in the 'Date' column of the 'data' DataFrame.
# Select data for each date and append it to 'stock1_day'.
# Increment the 'count' variable.

# Transpose 'stock1_day' to have rows represent days and columns represent data features.
# This code aggregates daily data from the 'data' DataFrame into 'stock1_day'.


# In[143]:


stock1_dayt = pandas.DataFrame()
count = 0
i = 0
for date in data1['Date'].unique():
    st2 = data1.loc[data1['Date'] == date]
    stock1_dayt = pandas.concat([stock1_dayt, st2.iloc[len(st2)-1, :]], axis=1)
    count+=1

stock1_dayt = stock1_dayt.transpose()
stock1_dayt


# In[144]:


stock1_d = stock1_day.copy()
stock1_dt = stock1_dayt.copy()


# In[145]:


stock1_d = stock1_d.drop(['Date'], axis=1)
# Remove the 'Date' column from the 'stock1_d' DataFrame.
# This code drops the specified column to exclude it from the DataFrame.
stock1_dt = stock1_dt.drop(['Date'], axis=1)


# In[146]:


stock1_d = stock1_d.drop(['Time'], axis=1)
# Remove the 'Time' column from the 'stock1_d' DataFrame.
# This code drops the specified column to exclude it from the DataFrame.
stock1_dt = stock1_dt.drop(['Time'], axis=1)


# In[147]:


stock1_d.reset_index(drop=True, inplace=True)
stk1_d = stock1_d.drop(['X_Close'], axis=1)

stock1_dt.reset_index(drop=True, inplace=True)
stk1_dt = stock1_dt.drop(['T_Close'], axis=1)

# Reset the index of the 'stock1_d' DataFrame, dropping the previous index and modifying it in-place.
# Remove the 'X_Close' column from the 'stk1_d' DataFrame.
# This code resets the index and removes the specified column.


# In[148]:


stock1_train = stk1_d[:1764]
stock1_test = stock1_d['X_Close'][1764:]

stock1t_train = stk1_dt[:1764]
stock1t_test = stock1_dt['T_Close'][1764:]

y_train = stk1_d[:1764]
y_test = stock1_d['X_Close'][1764:]

yt_train = stk1_dt[:1764]
yt_test = stock1_dt['T_Close'][1764:]


# In[149]:


print(stock1_train.shape, stock1_test.shape, stock1t_train.shape, stock1t_test.shape)
print(y_train.shape, y_test.shape, yt_train.shape, yt_test.shape)


# In[229]:


from torch.autograd import Variable 


# In[230]:


input_size = 4
hidden_layer_size = 64
num_layers = 2
output_size = 1

model = LSTMmodel(input_size, output_size, hidden_layer_size, num_layers)
print(model)

# Create an instance of the custom LSTM model ('LSTMmodel') with the specified hyperparameters.
# Print the model to display its architecture.


# In[231]:


batch_size = 16
sequence_length = 8
print(stk1_d)
print(stk1_dt)


# Q5

# In[232]:


import pandas as pd
import numpy as np
get_ipython().system('pip install torchvision')

from torch.utils.data import DataLoader,Dataset

def create_dataset(dataset,time_care=10):
        x_data=[]
        for i in range(len(dataset)-time_care-1):
            x=np.array(dataset.iloc[i:i+time_care+1])
            x_data.append(x)
        return x_data
    
    
class CustomDataset(Dataset):
    def __init__(self,data,Close):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert data to a tensor
        self.label = Close
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx],self.label[idx]

# Define a function 'create_dataset' that takes a dataset and a time window size as input.
# The function creates sequences of data samples with the specified time window size.
# It returns a list of data sequences.

# Define a custom dataset class 'CustomDataset' that inherits from 'Dataset'.
# In the constructor '__init__', the class takes 'data' and


# In[234]:


# stock1_d=stock1_d.reset_index(drop=True)
# pp=stock1_d.drop(['X_Close'],axis=1)
# data=create_dataset(pp,10)

# df=CustomDataset(data,stock1_d['X_Close'])
# print(df.__getitem__(1))
# print(df.__len__())

# # Reset the index of the 'stock1_d' DataFrame, dropping the previous index and modifying it in-place.
# # Create a new DataFrame 'pp' by dropping the 'X_Close' column from 'stock1_d'.

# # Create a dataset 'data' using the 'create_dataset' function with 'pp' as input and a time window of 10.
# # 'data' contains sequences of data samples.

# # Create an instance of the 'CustomDataset' class with 'data' as data samples and 'stock1_d['X_Close']' as labels.


# In[233]:


# stock1_dt=stock1_dt.reset_index(drop=True)
# pp1=stock1_dt.drop(['T_Close'],axis=1)
# data2=create_dataset(pp1,10)

# df1=CustomDataset(data2,stock1_dt['T_Close'])
# print(df1.__getitem__(1))
# print(df1.__len__())


# Q7

# In[235]:


class Trading_model():
    def __init__(self,model):
        self.model=model
        self.capital=1000
        self.portfolio=0
        self.trades=0
        
    def strategy(self,inputs,data,i):
        prediction=self.model(inputs)
        closing_price=data[i]
        predict=(np.max(data[i-7:i])-np.min(data[i-7:i]))*prediction + np.sum(data[i-7:i])/4
        if(closing_price>predict+np.std(data[i-7:i])):
            self.capital=self.capital+closing_price
            self.portfolio=self.portfolio-1
            self.trades=self.trades+1
            print("Sold one share at:",closing_price)
        elif(abs(predict-closing_price)<=np.std(data[i-7:i])):
            self.trades=self.trades
            print("No Transactions done")            
        else:
            self.capital=self.capital-closing_price
            self.portfolio=self.portfolio+1
            self.trades=self.trades+1
            print("Bought one share at:",closing_price)
            
    def test(self,dataloader,data,start_point):
        i=start_point
        for X,y in dataloader:
            self.strategy(X,data,i)
            i=i+1
        print("Ending Capital:",self.capital)
        print("No. of trades:",self.trades)
        print("Current Share holding:",self.portfolio)
        print("Current asset price(market):",data[i])

# In the constructor '__init__', initialize the trading model with initial capital, portfolio, and trades count.

# Define a trading strategy 'strategy' that takes 'inputs', 'data', and 'i' as inputs.
# 'inputs' are the inputs to the neural network model.
# 'data' is the historical stock price data.
# 'i' is the current time step.
# The strategy calculates a prediction using the model and compares it with the closing price.
# If the closing price is higher than the prediction plus one standard deviation of historical prices, it sells one share.
# If the difference between the prediction and closing price is within one standard deviation, no transactions are made.
# If the closing price is lower than the prediction minus one standard deviation, it buys one share.

# Define a 'test' method that takes a dataloader, historical data, and a starting point as inputs.
# The 'test' method applies the trading strategy to the historical data starting from the specified point.
# It keeps track of capital, portfolio, and the number of trades made.
# It prints the ending capital, number of trades, current share holding, and the current asset price (market) at the end.


# Q6

# In[261]:


data3=create_dataset(stk1_d,24)
dataset3=CustomDataset(data3[:1764],np.array(stock1_d.loc[:1764,"X_Close"]))
dataloader3=DataLoader(dataset3,batch_size=1,shuffle=True)
# Create a dataset 'data' by calling the 'create_dataset' function on the 'stk1_d' dataframe.
# Each sample in the dataset consists of 24 consecutive data points.

# Create a 'CustomDataset' object 'dataset' using 'data' as the input data and 'np.array(stock1_d.loc[:1764,"X_Close"])'
# as the labels (closing prices) for the dataset.

# Create a 'DataLoader' object 'dataloader' to load batches of data from the 'dataset'.
# The batch size is set to 8, and the data is shuffled during loading.


# In[262]:


data4=create_dataset(stk1_dt,24)
dataset4=CustomDataset(data4[:1764],np.array(stock1_dt.loc[:1764,"T_Close"]))
dataloader4=DataLoader(dataset4,batch_size=1,shuffle=True)


# In[263]:


import torch.optim as optim

criterion=nn.MSELoss();
optimizer=optim.Adam(model.parameters(),lr=0.01)

for i in range(0,30):
    for _,(inputs,targets) in enumerate(dataloader3):
        optimizer.zero_grad()
        outputs=model(inputs)
#         targets = targets.view(-1, 1)
        
        loss=criterion(outputs.to(torch.float32),targets.to(torch.float32))
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1,30,loss.item()))
    
# Import the Adam optimizer from torch.optim.
# Create a mean squared error loss function 'criterion' using nn.MSELoss().
# Create an Adam optimizer 'optimizer' for the model's parameters with a learning rate of 0.005.
# Train the model for 30 epochs:
# This loop trains the model using the specified optimizer and loss function for a total of 30 epochs.


# In[264]:


targets


# In[265]:


with torch.no_grad():
    print(model(inputs))
    
# Use 'torch.no_grad()' to temporarily disable gradient computation.
# Pass the 'inputs' through the 'model' to get predictions.
# This line makes predictions without updating the model's parameters.


# In[266]:


import torch.optim as optim

criterion=nn.MSELoss();
optimizer=optim.Adam(model.parameters(),lr=0.005)

for i in range(0,30):
    for _,(inputs,targets) in enumerate(dataloader4):
        optimizer.zero_grad()
        outputs=model(inputs)
#         targets = targets.view(-1, 1)
        
        loss=criterion(outputs.to(torch.float32),targets.to(torch.float32))
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1,30,loss.item()))
    
# Import the Adam optimizer from torch.optim.
# Create a mean squared error loss function 'criterion' using nn.MSELoss().
# Create an Adam optimizer 'optimizer' for the model's parameters with a learning rate of 0.005.
# Train the model for 30 epochs:
# This loop trains the model using the specified optimizer and loss function for a total of 30 epochs.


# In[267]:


targets


# In[268]:


with torch.no_grad():
    print(model(inputs))


# Q8
# 
# Part A) Testing

# In[269]:


# Create a CustomDataset for testing using the remaining data.
# 'data[1764:]' contains the testing data, and 'stock1_d.loc[1764:,"X_Close"]' contains the corresponding labels.
dataset_test = CustomDataset(data3[1764:], np.array(stock1_d.loc[1764:, "X_Close"]))

# Create a DataLoader for testing with a batch size of 8 and shuffle the data.
test_dataloader1 = DataLoader(dataset_test, batch_size=1, shuffle=True)

dataset_test1 = CustomDataset(data4[1764:], np.array(stock1_dt.loc[1764:, "T_Close"]))

# Create a DataLoader for testing with a batch size of 8 and shuffle the data for tanh normalization
test_dataloader2 = DataLoader(dataset_test1, batch_size=1, shuffle=True)


# Q8

# In[270]:


# Set the model to evaluation mode (no gradient computation)
model.eval()

with torch.no_grad():
    # Initialize arrays to store the predicted values
    train_plot = np.ones_like(stock1_d['X_Close']) * np.nan
    test_plot = np.ones_like(stock1_d['X_Close']) * np.nan
    i = 7

    # Generate predictions for the training data
    for X, y in dataloader3:
        output = model(X)
#         print(output, i)
        train_plot[i] = (np.max(stock1_d["X_Close"][i-7:i]) - np.min(stock1_d["X_Close"][i-7:i])) * float(output) + np.sum(stock1_d["X_Close"][i-7:i]) / 7
        i = i + 1

    # Store the initial value of 'i'
    initial = i
    near = 0
    far = 0

    # Generate predictions for the test data
    for X, y in test_dataloader1:
        output = model(X)
        test_plot[i] = (np.max(stock1_d["X_Close"][i-7:i]) - np.min(stock1_d["X_Close"][i-7:i])) * float(output) + np.sum(stock1_d["X_Close"][i-7:i]) / 7
        i = i + 1

        # Calculate error metrics for near and far predictions
        if i < 1700:
            near = near + (stock1_d["X_Close"][i-1] - test_plot[i-1]) ** 2
        else:
            far = far + (stock1_d["X_Close"][i-1] - test_plot[i-1]) ** 2

    # Print the error metrics
    print("Near to training data for test gives:", near / (1700 - initial))
    print("Far to training data for test gives:", far / (i - 1700))


# In[271]:


# Set the model to evaluation mode (no gradient computation)
model.eval()

with torch.no_grad():
    # Initialize arrays to store the predicted values
    train_plot = np.ones_like(stock1_d['X_Close']) * np.nan
    test_plot = np.ones_like(stock1_d['X_Close']) * np.nan
    i = 7

    # Generate predictions for the training data
    for X, y in dataloader3:
        output = model(X)
        train_plot[i] = (np.max(stock1_d["X_Close"][i-7:i]) - np.min(stock1_d["X_Close"][i-7:i])) * float(output) + np.sum(stock1_d["X_Close"][i-7:i]) / 7
        i = i + 1

    # Store the initial value of 'i'
    initial = i
    near = 0
    far = 0

    # Generate predictions for the test data
    for X, y in test_dataloader1:
        output = model(X)
        test_plot[i] = (np.max(stock1_d["X_Close"][i-7:i]) - np.min(stock1_d["X_Close"][i-7:i])) * float(output) + np.sum(stock1_d["X_Close"][i-7:i]) / 7
        i = i + 1

        # Calculate error metrics for near and far predictions
        if i < 1900:
            near = near + (stock1_d["X_Close"][i-1] - test_plot[i-1]) ** 2
        else:
            far = far + (stock1_d["X_Close"][i-1] - test_plot[i-1]) ** 2

    # Print the error metrics
    print("Near to training data for test gives:", near / (1900 - initial))
    print("Far to training data for test gives:", far / (i - 1900))


# In[272]:


# Set the model to evaluation mode (no gradient computation)
model.eval()

with torch.no_grad():
    # Initialize arrays to store the predicted values
    train_plot = np.ones_like(stock1_d['X_Close']) * np.nan
    test_plot = np.ones_like(stock1_d['X_Close']) * np.nan
    i = 7

    # Generate predictions for the training data
    for X, y in dataloader3:
        output = model(X)
        train_plot[i] = (np.max(stock1_d["X_Close"][i-7:i]) - np.min(stock1_d["X_Close"][i-7:i])) * float(output) + np.sum(stock1_d["X_Close"][i-7:i]) / 7
        i = i + 1

    # Store the initial value of 'i'
    initial = i
    near = 0
    far = 0

    # Generate predictions for the test data
    for X, y in test_dataloader1:
        output = model(X)
        test_plot[i] = (np.max(stock1_d["X_Close"][i-7:i]) - np.min(stock1_d["X_Close"][i-7:i])) * float(output) + np.sum(stock1_d["X_Close"][i-7:i]) / 7
        i = i + 1

        # Calculate error metrics for near and far predictions
        if i < 2000:
            near = near + (stock1_d["X_Close"][i-1] - test_plot[i-1]) ** 2
        else:
            far = far + (stock1_d["X_Close"][i-1] - test_plot[i-1]) ** 2

    # Print the error metrics
    print("Near to training data for test gives:", near / (2000 - initial))
    print("Far to training data for test gives:", far / (i - 2000))


# From the 3 code cells below it can be seen that as test set moves away from the training set (go further from the last time on which it was trained), the error value increases as shown in the code output above, which is justified and expected

# In[273]:


# Define the dataset for the entire available data
dataset5 = CustomDataset(data3, np.array(stock1_d.loc[:, "X_Close"]))

# Create a data loader for the entire dataset with a batch size of 1
total_dataloader = DataLoader(dataset5, batch_size=1, shuffle=True)

# Initialize the trading agent with the trained model
trading_agent = Trading_model(model)

# Test the trading agent's strategy on the entire dataset starting from index 7
trading_agent.test(total_dataloader, stock1_d['X_Close'], 7)


# In[274]:


# Define the dataset for the entire available data
dataset6 = CustomDataset(data4, np.array(stock1_dt.loc[:, "T_Close"]))

# Create a data loader for the entire dataset with a batch size of 1
total_dataloader1 = DataLoader(dataset6, batch_size=1, shuffle=True)

# Initialize the trading agent with the trained model
trading_agent1 = Trading_model(model)

# Test the trading agent's strategy on the entire dataset starting from index 7
trading_agent1.test(total_dataloader1, stock1_dt['T_Close'], 7)


# Part C

# In[275]:


#This strategy fails as compared to buy-and-hold as the prise almost grew 1400% over the time frame in consideration
(stock1_d['X_Close'][len(stock1_d)-1]-stock1_d['X_Close'][0])/stock1_d['X_Close'][0]


# Q8 c) 
# 
# How does your profitability compare to a simple buy-and-hold strategy over long term (e.g.
# one or two years)?
# 
# For stocks like apple which has increased over 1000% during this time, the model can perform poorly as it does not consider such long terms gains and mor ecomplex market signls.

# In[283]:


# Set the model in evaluation mode
model.eval()

# Initialize the test loss
test_loss = 0.0

# Loop through the test data using the test data loader
with torch.no_grad():
    for test_input, test_target in test_dataloader1:
        # Forward pass to get test predictions
        test_output = model(test_input)
        
        # Calculate the test loss using the specified loss criterion
        test_loss += criterion(test_output, test_target)

# Print the test loss
print(f"Test Loss: {test_loss.item():}")


# In[282]:


# Set the model in evaluation mode
model.eval()

# Initialize the test loss
test_loss = 0.0

# Loop through the test data using the test data loader
with torch.no_grad():
    for test_input, test_target in test_dataloader2:
        # Forward pass to get test predictions
        test_output = model(test_input)
        
        # Calculate the test loss using the specified loss criterion
        test_loss += criterion(test_output, test_target)

# Print the test loss
print(f"Test Loss: {test_loss.item():}")


# In[280]:


correlation_matrix1 = stock2.corrwith(stock1)
correlation_matrix2 = stock2.corrwith(stock3)
correlation_matrix3 = stock1.corrwith(stock3)

# This line calculates the correlation between the columns of stock1 and stock3 and stores the result in correlation_matrix3.
# It measures how each column in stock1 correlates with the corresponding columns in stock3.

print(correlation_matrix1, correlation_matrix2, correlation_matrix3)


# Q9)
# 
# Negative correaltion suggests that this stock data cannot be used to predict the desired result.
# 
# Positive correaltion suggests that this data can precisly predict the desired result.

# Using stock 2 with stock 3 can help

# In[ ]:


# window_size = 4*24*60  # Number of previous entries to consider for mean and standard deviation
# o1data = stock3.copy()

# # Calculate and normalize the rolling statistics for 'Open', 'High', 'Low', 'Close', and 'Volume' columns

# rolling_max = o1data['Open'].rolling(window=window_size).max()
# rolling_min = o1data['Open'].rolling(window=window_size).min()
# o1data['X_Open1'] = (o1data['Open'] - rolling_min) / (rolling_max - rolling_min)

# # Similar calculations for other columns: 'High', 'Low', 'Close', and 'Volume'

# rolling_max = o1data['High'].rolling(window=window_size).max()
# rolling_min = o1data['High'].rolling(window=window_size).min()
# o1data['X_High1'] = (o1data['High'] - rolling_min) / (rolling_max - rolling_min)

# rolling_max = o1data['Low'].rolling(window=window_size).max()
# rolling_min = o1data['Low'].rolling(window=window_size).min()
# o1data['X_Low1'] = (o1data['Low'] - rolling_min) / (rolling_max - rolling_min)

# rolling_max = o1data['Close'].rolling(window=window_size).max()
# rolling_min = o1data['Close'].rolling(window=window_size).min()
# o1data['X_Close1'] = (o1data['Close'] - rolling_min) / (rolling_max - rolling_min)

# rolling_max = o1data['Volume'].rolling(window=window_size).max()
# rolling_min = o1data['Volume'].rolling(window=window_size).min()
# o1data['X_Volume1'] = (o1data['Open'] - rolling_min) / (rolling_max - rolling_min)

# # Fill missing values with min-max scaling for each column

# o1data['X_Open1'].fillna((o1data['Open']-o1data['Open'].min())/(o1data['Open'].max()-o1data['Open'].min()), inplace=True)
# o1data['X_High1'].fillna((o1data['High']-o1data['High'].min())/(o1data['High'].max()-o1data['High'].min()), inplace=True)
# o1data['X_Low1'].fillna((o1data['Low']-o1data['Low'].min())/(o1data['Low'].max()-o1data['Low'].min()), inplace=True)
# o1data['X_Close1'].fillna((o1data['Close']-o1data['Close'].min())/(o1data['Close'].max()-o1data['Close'].min()), inplace=True)
# o1data['X_Volume1'].fillna((o1data['Volume']-o1data['Volume'].min())/(o1data['Volume'].max()-o1data['Volume'].min()), inplace=True)

# n1data = pd.DataFrame(o1data)

# # Select specific columns to create a new DataFrame
# selected_columns = ['Date', 'Time', 'X_Open1', 'X_High1', 'X_Low1', 'X_Close1', 'X_Volume1']
# data1 = n1data[selected_columns]

# stock3_day = pandas.DataFrame()
# count = 0
# i = 0

# # Loop through unique dates in 'data1' DataFrame

# for date in data1['Date'].unique():
#     st3 = data1.loc[data1['Date'] == date]
#     stock3_day = pandas.concat([stock3_day, st3.iloc[len(st3)-1, :]], axis=1)
#     count+=1
# # Concatenate daily summary data to 'stock3_day' DataFrame

# # Transpose 'stock3_day' to have the summary data for each day as rows

# stock3_day = stock3_day.transpose()
# stock3_day


# In[ ]:


# # Define the input size, hidden layer size, number of layers, output size, and number of epochs for the LSTM model
# input_size = 9
# hidden_layer_size = 48
# num_layers = 2
# output_size = 1
# num_epochs = 100

# # Create an instance of the LSTM model with the specified architecture
# model3 = LSTMmodel(input_size, output_size, hidden_layer_size, num_layers)
# print(model3)

# # Define batch size and sequence length for data processing
# batch_size = 16
# sequence_length = 8

# # Create a copy of the 'stock3_day' DataFrame for data preprocessing
# stock3_d = stock3_day.copy()

# # Remove 'Time' and 'Date' columns from the DataFrame as they may not be needed for modeling
# stock3_d = stock3_d.drop(['Time'], axis=1)
# stock3_d = stock3_d.drop(['Date'], axis=1)


# In[ ]:


# stock3_d.reset_index(drop=True, inplace=True)
# stock3_d


# In[ ]:


# stock1_d.reset_index(drop=True, inplace=True)
# stock1_d


# In[ ]:


# # Reset the index of 'stock3_d' DataFrame and store the result in 'stock3_d'
# stock3_d = stock3_d.reset_index(drop=True)

# # Concatenate 'stock3_d' and 'stock1_d' DataFrames along the rows to create 'concatenated_df'
# concatenated_df = pd.concat([stock3_d, stock1_d], axis=1, join='inner')

# concatenated_df


# In[ ]:


# # Drop the 'X_Close' column from 'concatenated_df' and store the result in 'pp1'
# pp1 = concatenated_df.drop(['X_Close'], axis=1)

# # Create a dataset 'data2' using 'pp1' with a time window of 10
# data2 = create_dataset(pp1, 10)

# # Create a custom dataset 'df1' using 'data2' and the 'X_Close' column from 'concatenated_df'
# df1 = CustomDataset(data2, concatenated_df['X_Close'])

# # Print an example item from the dataset and the total number of items in 'df1'
# print(df1.__getitem__(1))
# print(df1.__len__())


# In[ ]:


# concatenated_df.reset_index(drop=True, inplace=True)
# concat_df = concatenated_df.drop(['X_Close'], axis=1)

# data3=create_dataset(concat_df,24)
# dataset1=CustomDataset(data3[:1764],np.array(concatenated_df.loc[:1764,"X_Close"]))
# dataloader1=DataLoader(dataset1,batch_size=8,shuffle=True)

# criterion=nn.MSELoss();
# optimizer=optim.Adam(model3.parameters(),lr=0.005)


# In[ ]:


# for i in range(0,30):
#     for _,(inputs,targets) in enumerate(dataloader1):
#         optimizer.zero_grad()
#         outputs=model3(inputs)
# #         targets = targets.view(-1, 1)
        
#         loss=criterion(outputs.to(torch.float32),targets.to(torch.float32))
#         loss.backward()
#         optimizer.step()
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1,30,loss.item()))


# In[ ]:


# #Part A)Testing
# dataset1=CustomDataset(data3[1600:],np.array(concatenated_df.loc[1600:,"X_Close"]))
# test_dataloader1=DataLoader(dataset1,batch_size=8,shuffle=True)
# #Part B
# dataset2=CustomDataset(data3,np.array(concatenated_df.loc[:,"X_Close"]))
# total_dataloader1=DataLoader(dataset2,batch_size=8,shuffle=True)

# trading_agent1=Trading_model(model3)
# trading_agent1.test(total_dataloader1,concatenated_df['X_Close'],7)


# It is observed that since stock 1 and stock3 are correlated so the overall profit/prediction performs poorly.
# Using a positively correlated stock performs better.

# Q9 b)
# 
# Adding day of the week, day in the year, and time as inputs to a predictive model can potentially improve its performance, especially if there are patterns or dependencies in the data related to these features. These additional features can capture time-related trends or patterns that may be useful for making predictions.
# 
# Day of the Week - can extract the day of the week from the date and add it as a numerical feature. This feature can help capture weekly patterns.
# 
# Day in the Year - can calculate the day in the year. This can capture annual seasonality.
# 
# Time - can extract it from the time column
