# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 23:54:53 2022

@author: abhij
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf 

sns.set()

#tickerSymbolsCSV=pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
tickerSymbol = 'MSFT'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(interval='1d', start='2019-1-1', end=date.today())    

high_data=tickerDf['High']

#check for stationarity
sns.lineplot(data=high_data)
plt.show()

#remove the trend
first_diff=high_data.diff().dropna()
sns.lineplot(data=first_diff)
plt.show()

#recheck for stationarity
result=adfuller(first_diff)
print('p value for first difference of series {}'.format(result[1]))

plot_pacf(first_diff)
plot_acf(first_diff)

lags_pacf,conf_int_pacf=pacf(first_diff, alpha=0.05,nlags=10)
conf_int_pacf-=np.vstack((lags_pacf, lags_pacf)).T

lags_acf,conf_int_acf=acf(first_diff, alpha=0.05,nlags=10)
conf_int_acf-=np.vstack((lags_acf, lags_acf)).T

imp_lags_pacf=[]
for i in range(lags_pacf.size):
    if lags_pacf[i]>conf_int_pacf[i,1] or lags_pacf[i]<conf_int_pacf[i,0]:
        imp_lags_pacf.append(i)
        

imp_lags_acf=[]
for i in range(lags_acf.size):
    if lags_acf[i]>conf_int_acf[i,1] or lags_acf[i]<conf_int_acf[i,0]:
        imp_lags_acf.append(i) 

fitted_models={}

for i in imp_lags_acf:
    for j in imp_lags_pacf:
        model=ARIMA(high_data,order=(j,1,i))
        model=model.fit()
        fitted_models[i,j]=model

lowest_aic=10000000
best_lag_aic=[]

for i in imp_lags_acf:
    for j in imp_lags_pacf:
        if(fitted_models[i,j].aic<lowest_aic):
            lowest_aic=fitted_models[i,j].aic
            best_lag_aic=[i,j]    
        #print('AIC for ARIMA {} {} is {}'.format(i,j,fitted_models[i,j].aic))


test_size = 30

rolling_predictions_arima = []

for i in range(0,test_size,5):
    train = high_data[:-(test_size-i)]
    model = ARIMA(train, order=(best_lag_aic[0],1,best_lag_aic[1]))
    model = model.fit()
    pred = model.forecast(steps=5)[0]
    rolling_predictions_arima.append(pred)

rolling_predictions_arima=np.concatenate( rolling_predictions_arima, axis=0 )
rolling_predictions_arima = pd.Series(rolling_predictions_arima, index=high_data.index[-test_size:])


print('MAE ',mean_absolute_error(high_data[-test_size:].values, rolling_predictions_arima))
print('RMSE ',np.sqrt(mean_squared_error(high_data[-test_size:].values, rolling_predictions_arima)))

plt.figure(figsize=(10,4))
plt.plot(high_data[-test_size:])
plt.plot(rolling_predictions_arima)
plt.show()

residuals=model.resid

plot_pacf(residuals**2,lags=10)

lags_pacf_garch,conf_int_pacf_garch=pacf(residuals**2, alpha=0.05,nlags=6)
conf_int_pacf_garch-=np.vstack((lags_pacf_garch, lags_pacf_garch)).T

imp_lags_pacf_garch=[]
for i in range(lags_pacf_garch.size):
    if lags_pacf_garch[i]>conf_int_pacf_garch[i,1] or lags_pacf_garch[i]<conf_int_pacf_garch[i,0]:
        imp_lags_pacf_garch.append(i)


rolling_predictions_garch = []
for i in range(test_size):
    train = residuals[:-(test_size-i)]
    garch_model = arch_model(train,vol='GARCH', p=6, q=6)
    garch_model = garch_model.fit(disp='off')
    pred = garch_model.forecast(horizon=1)
    rolling_predictions_garch.append(np.sqrt(pred.variance.values[-1,:][0]))

rolling_predictions_garch = pd.Series(rolling_predictions_garch, index=high_data.index[-test_size:])

plt.plot(residuals[-test_size:])
plt.plot(rolling_predictions_garch)

plt.fill_between(x=residuals.index[-test_size:], y1=rolling_predictions_garch,
                 y2=-rolling_predictions_garch,alpha=0.5)
plt.plot(residuals[-test_size:])
plt.show()

test_size = 30

rolling_predictions_arima = []
rolling_predictions_garch = []

for i in range(0,test_size,5):
    train = high_data[:-(test_size-i)]
    model = ARIMA(train, order=(best_lag_aic[0],1,best_lag_aic[1]))
    model = model.fit()
    pred = model.forecast(steps=5)[0]
    rolling_predictions_arima.append(pred)
    train_garch = residuals[:-(test_size-i)]
    garch_model = arch_model(train_garch,vol='GARCH', p=6, q=6)
    garch_model = garch_model.fit(disp='off')
    pred = garch_model.forecast(horizon=5)
    rolling_predictions_garch.append(np.sqrt(pred.variance.values[-1,:]))

rolling_predictions_arima=np.concatenate( rolling_predictions_arima, axis=0 )
rolling_predictions_arima = pd.Series(rolling_predictions_arima, index=high_data.index[-test_size:])

rolling_predictions_garch=np.concatenate( rolling_predictions_garch, axis=0 )
rolling_predictions_garch = pd.Series(rolling_predictions_garch, index=high_data.index[-test_size:])


plt.plot(rolling_predictions_arima)
plt.plot(high_data[-test_size:],color='red')
plt.fill_between(x=residuals.index[-test_size:], y1=rolling_predictions_arima+rolling_predictions_garch,
                 y2=rolling_predictions_arima-rolling_predictions_garch,alpha=0.5)
plt.xticks(rotation=90)
plt.show()

import tensorflow as tf

def windowed_dataset(series,window_size,batch_size,shuffle_buffer):
    dataset=tf.data.Dataset.from_tensor_slices(series)
    dataset=dataset.window(window_size+1,shift=1,drop_remainder=True)
    dataset=dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset=dataset.shuffle(shuffle_buffer).map(lambda window:(window[:-5],window[-5:]))
    dataset=dataset.batch(batch_size).prefetch(1)
    return dataset

df=windowed_dataset(high_data,14,32,200)

from keras.layers import GRU, Lambda, Dense, LSTM, Bidirectional
from keras.models import Sequential

ml_model=Sequential()
ml_model.add(Lambda(lambda x: tf.expand_dims(x,axis=-1),input_shape=[None]))
ml_model.add(Bidirectional(LSTM(50,return_sequences=True)))
ml_model.add(GRU(50))
ml_model.add(Dense(5))
ml_model.summary()

ml_model.compile(loss=tf.keras.losses.MeanSquaredError(),metrics='mse',optimizer='Adam')

history=ml_model.fit(df,epochs=500)

test_size = 30

rolling_predictions_lstm = []

for i in range(0,test_size,5):
    train = high_data[-(test_size-i+10):-(test_size-i)].values
    pred = ml_model.predict(train[np.newaxis])
    rolling_predictions_lstm.append(pred)

rolling_predictions_lstm=np.concatenate( rolling_predictions_lstm, axis=0 )
rolling_predictions_lstm=np.concatenate( rolling_predictions_lstm, axis=0 )
rolling_predictions_lstm = pd.Series(rolling_predictions_lstm, index=high_data.index[-test_size:])

print('RMSE ',np.sqrt(mean_squared_error(high_data[-test_size:].values, rolling_predictions_lstm)))

plt.figure(figsize=(10,4))
plt.plot(high_data[-test_size:])
plt.plot(rolling_predictions_lstm)
plt.show()


















