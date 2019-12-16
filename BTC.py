import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
import time
from pylab import rcParams
import datetime
from selenium import webdriver
import statistics

#executable_path = ('/Users/Kaiden Thrailkill/Desktop/Environment/chromedriver_win32/chromedriver.exe')
#driver = webdriver.Chrome(executable_path=executable_path)
#driver.get('https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD')
#time_period = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[1]/span[2]/span/input')
#time_period.click()
#Max = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[1]/span[2]/div/div[1]/span[8]')
#Max.click()
#done = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[1]/span[2]/div/div[3]/button[1]')
#done.click()
#data = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[2]/span[2]/a')
#data.click()
#time.sleep(10)



warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df = pd.read_csv("/Users/Kaiden Thrailkill/Downloads/BTC-USD.csv")

#df.rename(columns={' Open ': 'Open'}, inplace=True)

#df['Date'] = df['Date'].astype('datetime64[ns]')


df.Open = df.Open.astype(int)

print(df.dtypes)
#df.rename(columns={' BIT_COIN ': 'BIT_COIN'}, inplace=True)
#df.BIT_COIN = df.BIT_COIN.astype(str)

#BTC = df.loc[df['BIT_COIN'] == 'BTC']
#print(BTC)
#BTC['Date'].min(), BTC['Date'].max()



BTC = df.groupby('Date')['Open'].sum().reset_index()
print(BTC)
BTC = BTC.set_index('Date')
BTC.index = pd.to_datetime(BTC.index)
print(BTC.index)
BTC.plot(figsize=(15, 6))
plt.show()



y = BTC['Open'].resample('MS').mean()
y['2011':]
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
	for param_seasonal in seasonal_pdq:
		try:
			mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
			results = mod.fit()
			print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
		except:

			continue 
mod = sm.tsa.statespace.SARIMAX(y, order=(1, 1, 0), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2015':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('BTC Open')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = y['2015-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Open')
plt.legend()
plt.show()
print(" ")
print("Predicted BTC Open by month: ")
print(" ")
print(pred_ci)
print(" ")
print("Predicted BTC Open Lower And Upper Mean: ")
print(pred_uc.predicted_mean)


