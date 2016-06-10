# -*- coding: utf-8 -*-
# %reset
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
os.chdir('D:\\Kaggle\\bike-sharing-demand')
import functions.rf_functions as rff
import functions.xgb_functions as xf
#### preparation of data #############################################################################
data = pd.read_csv('data//train_clean_new_more_var.csv', parse_dates = ['datetime'])
cols_drop = ['casual', 'date', 'datetime', 'day','lcasual', 'lregistered', 'ltotal',
             'registered', 'season2', 'total'] 
cols_X = [c for c in data.columns if c not in cols_drop]

X,dates,y = rff.prepare_RF_2(data, True, cols_X, False)

#### manual tuning (only for registered) ##########################################################
param = {'objective':'reg:linear', 
             'eval_metric':'rmse'}
param['eta'] = 0.04
param['gamma'] =0.01
param['max_depth'] = 4
param['alpha'] = 2#0.1
param['lambda'] =10
param['colsample_bytree'] =0.7
param['p']=0.1

r,d = xf.test_one_XGB(2012,6,'lregistered', param,X, y, dates,10,8,3,param['p'] )
### plotting evaluation metrics
n =10
plt.plot(d['eval']['rmse'][n:],'-r',d['train']['rmse'][n:],'-b')
### plotting features importance
_, ax = plt.subplots(figsize=(20, 20))
xgb.plot_importance(r , ax = ax )

### saving results for each month
param_final = {}
best_ntree_limits = {}
result_final = {}

param_final[(2012,6)] = param
result_final[(2011,5)] =0.443361

####################################################################################################
with open('results//xgboost//params.pkl','wb') as f:
    pickle.dump(param_final,f)
    
with open('results//xgboost//results.pkl','wb') as f:
    pickle.dump(result_final,f)

### I forgot ntree_limits
best_ntree_limits = {}
for y_m,param in param_final.items():
    b, _= xf.test_one_XGB(y_m[0],y_m[1],'lregistered', param,X, y, dates,10,8,3,param['p'] )
    best_ntree_limits[y_m] = b.best_ntree_limit   
    
with open('results//xgboost//best_ntree_limits.pkl','wb') as f:
    pickle.dump(best_ntree_limits,f)

## train final models ##############################################################################
with open('results//xgboost//params.pkl','rb') as f:
    param_final = pickle.load(f)
    
with open('results//xgboost//best_ntree_limits.pkl','rb') as f:
    best_ntree_limits = pickle.load(f)

xf.train_XGB('xgb_models', X,y,dates,10,param_final, best_ntree_limits) 
xf.train_XGB('xgb_models_on_all_data', X,y,dates,21,param_final, best_ntree_limits)   
  
