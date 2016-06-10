# %reset
import os
os.chdir('D:\\Kaggle\\bike-sharing-demand')
import xgboost as xgb
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import functions.subsample_sel as ss


#### train one model using train and validation sample and return results ####################################################
def test_one_XGB(year,month,y_type, param, X_tv, y_tv, dates_tv,day_test, day_valid_small, day_valid, p):
    num_round = 1000    
    if year == 2012 or month >=4:
        day_valid_curr = day_valid
    else:
        day_valid_curr = day_valid_small
    train_ind = ss.get_train(dates_tv, year,month,day_test, day_valid_curr)
    valid_ind = ss.get_valid(dates_tv, year,month,day_test, day_valid_curr)
    print('year {}, month {}'.format(year,month), flush = True)
    print('train size: {}, validation size: {}'.format(train_ind.sum(),valid_ind.sum()))
    print('learning from {} to {}'.format(dates_tv[train_ind].min(), dates_tv[train_ind].max())) 
    print('validation from {} to {}'.format(dates_tv[valid_ind].min(), dates_tv[valid_ind].max()))
    weights = np.power(np.arange(train_ind.sum(),1,-1),p)
    print (weights)
    param = param.copy()
    param['seed'] = 13
    dtrain = xgb.DMatrix(X_tv[train_ind], label = y_tv.loc[train_ind,y_type],weight = weights )      
    dvalid =  xgb.DMatrix(X_tv[valid_ind], label = y_tv.loc[valid_ind,y_type])   
    evallist  = [(dtrain,'train'),(dvalid,'eval')]
    evals_result = {}
    bst = xgb.train(param, dtrain, num_round, evallist, evals_result =  evals_result, early_stopping_rounds=50)	
    return bst,evals_result

### traing final models #########################################################################################################
def train_XGB(path, X_tv,y_tv,dates_tv,day_test,param_final, best_ntree_limits):
    for y_m,param in param_final.items():
        train_ind = ss.get_train(dates_tv, y_m[0],y_m[1],day_test, day_test)
        print('year {}, month {}'.format(y_m[0],y_m[1]), flush = True)
        print('learning from {} to {}'.format(dates_tv[train_ind].min(), dates_tv[train_ind].max()))
        param['seed'] = 13
        weights = np.power(np.arange(train_ind.sum(),1,-1),param['p'])
        dtrain = xgb.DMatrix(X_tv[train_ind], label = y_tv.loc[train_ind,'lregistered'],weight = weights)      
        bst = xgb.train(param, dtrain, num_boost_round = best_ntree_limits[y_m])	
        bst.save_model(os.path.join(path,str(y_m[0])+"_"+str(y_m[1])))
 
#### predict values for X data using trained models #########################################################################################
def predict_XGB(X, dates, path):
    result = None
    for year in [2011,2012]:
        for month in range(1,13):
            print(year, month, flush = True)
            ind_curr = (dates >= dt.datetime(year,month,1)) & (dates < dt.datetime(year,month,1)+relativedelta(months=1))
            data = xgb.DMatrix(X[ind_curr])
            bst = xgb.Booster()            
            bst.load_model(os.path.join(path,'{}_{}'.format(year,month)))
                
            ypred=bst.predict(data)
            ypred_e = (np.exp(ypred)-1) 
            if result is None:
                result = ypred_e
            else:
                result = np.concatenate((result, ypred_e))
    return result
### 







