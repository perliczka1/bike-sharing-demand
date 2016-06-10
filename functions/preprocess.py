# -*- coding: utf-8 -*-
import datetime as dt
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import calendar as c
from dateutil.relativedelta import relativedelta
import pandas as pd
#### FIRST PREPROCESSING AND SOME SIMPLE CHECKS ###########################################
def preprocess_and_check(data,is_train,days_in_month_train = 19):
    ## rename count variable
    data.rename(columns={'count': 'total'}, inplace=True)
    ## add time related variables 
    data['year'] = data.datetime.apply(lambda x: x.year)
    data['month'] = data.datetime.apply(lambda x: x.month)
    data['day'] = data.datetime.apply(lambda x: x.day)
    data['hour'] = data.datetime.apply(lambda x: x.hour)
    data['date'] = data.datetime.apply(dt.datetime.date)

    ## add ys
    if is_train:
        data['lcasual'] = np.log(data.casual+1)
        data['lregistered'] = np.log(data.registered+1)
        data['ltotal'] = np.log(data.total+1)
    ## check weather variables by time
    sn.set(style='whitegrid')
    plt.figure()
    sn.boxplot(x="season", y="month", data=data)
    data['season2'] = (data.month % 12 ) // 3 + 1
    plt.figure()
    sn.boxplot(x="season2", y="month", data=data)
    plt.figure()
    sn.boxplot(x="month", y="temp", data=data)
    data['temp_diff'] = data.atemp - data.temp
    plt.figure()
    sn.boxplot(x="season2", y="temp_diff", data=data )
    plt.figure()
    sn.countplot(x = 'season2', hue="weather", data=data)
    plt.figure()
    sn.boxplot(x="season2", y="humidity", data=data )
    plt.figure()
    sn.boxplot(x="season2", y="windspeed", data=data )
    
    ## check missing rows
    if is_train:
        rows_all = 12 * days_in_month_train * 24 
    else:
        rows_all = ((data.year.drop_duplicates().apply(c.isleap) + 365 - 12 * days_in_month_train)*24).values
    print('Missing rows:')
    print(-data.groupby( by = 'year').size() + rows_all)
    return data
    
#### ADDING TO DATA FRAME LAGGED WEATHER VARIABLES ######################################################################     
def add_weather_lag(data_all,weather_var, n):
    weath_lag_names = [var+'_'+str(n)+'hlag' for var in weather_var]
    weather_lag = data_all[weather_var].shift(periods=n)
    weather_lag.columns = weath_lag_names
    data_all_new = pd.concat([data_all,weather_lag],axis = 1)  
    # inserting missing values from first days of January 2011
    print(data_all_new[weath_lag_names].isnull().sum())
    data_all_new[weath_lag_names] = data_all_new[weath_lag_names].fillna(method = 'backfill') # only n observation 
    return data_all_new
    
#### NEW DATA FRAME WITH PAST VALUES OF DEPENDENT VAR ###################################################################    
# modify data_all adding one useful column
def add_y_lag(data_all, n):
    if not 'first_day_of_month' in data_all.columns:
        data_all['first_day_of_month'] = data_all.datetime.apply(lambda x: (x-dt.timedelta(days=x.day-1)).date())
    avg_rent_by_date = data_all.groupby(['first_day_of_month'])['total','casual','registered'].agg([np.mean, np.median])  
    avg_rent_by_date.index = avg_rent_by_date.index +relativedelta(months=n)   
    avg_rent_by_date.columns = [c[0]+'_'+c[1]+'_'+str(n)+'mlag' for c in avg_rent_by_date.columns]
    data_all_new = pd.merge(data_all,avg_rent_by_date, how='left', left_on = 'first_day_of_month',right_index = True)
    return data_all_new
    
#### NEW DATA FRAME WITH PAST VALUES OF DEPENDENT VAR ##################################################################### 
## needs dataframe with train & test data
def add_new_variables(data_all):
    ## if there are holidays this week 
    data_all=data_all.copy() #to avoid modification of the argument
    data_all['first_day_of_week'] = data_all.datetime.apply(lambda x: (x-dt.timedelta(days=x.weekday())).date())
    tmp_hol = data_all.groupby('first_day_of_week').holiday.sum() 
    tmp_hol = tmp_hol[tmp_hol>0]
    data_all['holiday_this_week'] = data_all.first_day_of_week.isin(tmp_hol.index)
    # weekday
    data_all['week_day'] = data_all.datetime.apply(lambda x: x.isoweekday())
    # month numbers
    data_all['i'] = (data_all.year-2011)*12 + data_all.month
    data_all.reset_index(drop = True, inplace = True)
    # average weather yesterday
    data_all.date = pd.to_datetime(data_all.date)
    weather_var = ['weather','temp', 'humidity', 'atemp','windspeed']
    avg_weath_by_date = data_all.groupby('date')[weather_var].mean()
    avg_weath_by_date.index = avg_weath_by_date.index +dt.timedelta(days=1)
    data_all = pd.merge(data_all,avg_weath_by_date, how='left', left_on = 'date',right_index = True, suffixes=['','_1dlag'])
    for var in weather_var:
        ## for missing values insert current weather from the same day
        data_all.loc[data_all[var+'_1dlag'].isnull(),var+'_1dlag'] = data_all.loc[data_all[var+'_1dlag'].isnull(),var] 
    # average & median rental from previous months
    for i in range(1,7):    
        data_all = add_y_lag(data_all, i)    
    # weather lag
    for i in range(1,9):    
        data_all = add_weather_lag(data_all,weather_var, i)
    # data abour sunset and sunrise    
    sun_data = pd.read_csv('data\\sun_data.csv', sep=';')
    sun_data['hour_rise'] = sun_data.h1.apply(lambda x: int(x.split(':')[0]))
    sun_data['min_rise'] = sun_data.h1.apply(lambda x: 1-float(x.split(':')[1])/60)
    
    sun_data['hour_set'] = sun_data.h2.apply(lambda x: int(x.split(':')[0]))
    sun_data['min_set'] = sun_data.h2.apply(lambda x: float(x.split(':')[1])/60)
    
    sun_data.drop(['h1','h2'], axis = 1,inplace = True)
    data_all = pd.merge(data_all,sun_data,how = 'left', left_on = ['day','month'], right_on = ['day','month'])
    data_all['daylight'] = ((data_all.hour > data_all.hour_rise) &
                           (data_all.hour < data_all.hour_set)) + \
                           ((data_all.hour == data_all.hour_rise)*data_all.min_rise) + \
                           ((data_all.hour == data_all.hour_set)*data_all.min_set) 
   
    ## remove unnecessary columns
    data_all.drop(['first_day_of_month','first_day_of_week','hour_rise', 'min_rise', 'hour_set', 'min_set'],axis=1,inplace = True)
    return data_all
    
#### NEW DATA FRAME WITH PAST VALUES OF DEPENDENT VAR ##################################################################### 
# requires all data, saves new files
def get_train_test_data_with_new_var():
    data_train = pd.read_csv('data\\train_clean.csv', parse_dates = ['datetime'])  
    data_test = pd.read_csv('data\\test_clean.csv', parse_dates = ['datetime'])  
    data_all = pd.concat([data_train,data_test],axis = 0)
    data_all.sort_values('datetime', inplace = True)
    data_all = add_new_variables(data_all)
    train_result = data_all[~data_all.total.isnull()]
    test_result = data_all[data_all.total.isnull()]
    train_result.to_csv('data\\train_clean_new_more_var.csv',index = False)
    test_result.to_csv('data\\test_clean_new_more_var.csv',index = False)
    return train_result, test_result
    
       

