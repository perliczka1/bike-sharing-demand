# -*- coding: utf-8 -*-
# %reset
import os
import pandas as pd
import datetime as dt
import seaborn as sn
import importlib as il
os.chdir('D:\\Kaggle\\bike-sharing-demand')
import functions.preprocess as pr
import functions.atemp_impute as ai
import functions.plot_by_time as pbt
il.reload(pr)
il.reload(ai)
il.reload(pbt)
#### LOADING AND CHECKING DATA ##############################################################################
data = pd.read_csv('data\\train.csv', parse_dates = ['datetime'])
data_test = pd.read_csv('data\\test.csv', parse_dates = ['datetime'])
data.dtypes
data.head()
## first preprocessing and plotting variables by season/month
data = pr.preprocess_and_check(data,True,19)
data_test = pr.preprocess_and_check(data_test,False,19)
## scatterplot matrix
cols_disp =  ['holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'total',
        'hour', 'temp_diff']
sn.pairplot(data[cols_disp])
sn.pairplot(data_test[['holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed','hour', 'temp_diff']])

#### CORRECTING SOME MINOR ISSUES ############################################################################################################
## humidity & temp_diff from train data 
## humidity can not be 0
data[data.humidity == 0]
tmp = data[data.date <= dt.date(2011,3,19)] # I wont us future data

sn.boxplot(x = 'weather', y = 'humidity', data = tmp[(tmp.humidity != 0)]) # depends on weather
tmp = tmp[(tmp.weather == 3)] # use humudity from rows with the same value of weather variable

# input humidity 2011-03-10 16:00:00
hum_ind = data.humidity == 0
data.loc[hum_ind & (data.datetime >= dt.datetime(2011, 3, 10,16)), 'humidity'] = 93
data.loc[hum_ind & (data.datetime <= dt.datetime(2011, 3, 10,6)), 'humidity'] = 93
data.loc[data.humidity == 0, 'humidity'] = 100

## atemp for train data
## constant and strange value for one day

data[data.temp_diff < -10]
tmp = data[(data.date <= dt.date(2012,8,19)) & (data.temp_diff > -10)]
## from scatterplot we know that atemp depend on temp, humidity and windspeed
temp_diff_new = ai.atemp_impute(tmp[['temp','humidity', 'windspeed']], 
                                tmp.temp_diff,
                                data.loc[data.temp_diff < -10, ['temp','humidity', 'windspeed']])
data.loc[data.temp_diff < -10, 'temp_diff'] = temp_diff_new.y
data.atemp = data.temp + data.temp_diff
sn.boxplot(x="season2", y="temp_diff", data=data )

#### PLOTS BY TIME ##############################################################################################
# quick plots by time
pbt.plot_by_time(data,'holiday', fmt='o')
pbt.plot_by_time(data,'weather', fmt='.')
pbt.plot_by_time(data,'temp')
pbt.plot_by_time(data,'atemp')
pbt.plot_by_time(data,'humidity')
pbt.plot_by_time(data,'windspeed')
pbt.plot_by_time(data,'temp_diff')
pbt.plot_by_time(data,'casual')
## plotting first month
pbt.plot_by_time(data,'casual',fmt='-', datetime_to = dt.datetime(2011,1,20), granular = True)
pbt.plot_by_time(data,'casual',color ='temp', datetime_to = dt.datetime(2011,1,20), granular = True)
pbt.plot_by_time(data,'total',color ='temp', datetime_to = dt.datetime(2011,1,20), granular = True)
pbt.plot_by_time(data,'registered',datetime_to = dt.datetime(2011,1,20), granular = True)
pbt.plot_by_time(data,'registered',color ='temp', datetime_to = dt.datetime(2011,1,20), granular = True)
## even in the first month we see that distribution of casual and registered is different 
pbt.plot_by_time(data,'total',color ='temp')
pbt.plot_by_time(data,'registered')
pbt.plot_by_time(data,'casual')
## 
sn.pairplot(data[cols_disp])
sn.pairplot(data[['temp', 'atemp']])
#### SAVING DATA ###############################################################################################
data.to_csv('data\\train_clean.csv', index = False)
data_test.to_csv('data\\test_clean.csv', index = False)