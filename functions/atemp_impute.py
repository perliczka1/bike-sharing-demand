# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import pandas as pd
def atemp_impute(X,y, X_missing):
    scaler = preprocessing.StandardScaler()
    X2 = scaler.fit_transform(X)
    nn = KNeighborsRegressor(1)
    nn.fit(X2, y)
    X_missing2 = scaler.transform(X_missing)
    dist, ind = nn.kneighbors(X=X_missing2, n_neighbors=1, return_distance=True)
    res = X.iloc[ind[:,0]].copy()
    res['dist'] = dist
    res['y'] = y[ind[:,0]]
    res.index = X_missing.index
    res = pd.concat([res,X_missing], axis = 1)
    return res
    
#X = tmp[['temp','humidity', 'windspeed']] 
#y = tmp.temp_diff
#X_missing =  data.loc[data.temp_diff < -10, ['temp','humidity', 'windspeed']]