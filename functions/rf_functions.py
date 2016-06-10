# -*- coding: utf-8 -*-
import os
os.chdir('D:\\Kaggle\\bike-sharing-demand')
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import importlib as il
import pickle
import datetime as dt
import itertools as it
import numpy as np
from dateutil.relativedelta import relativedelta
import functions.subsample_sel as ss
import functions.results_functions as resf
il.reload(ss)

##### prepare data without new variables ################################################################################
def prepare_RF(data, is_y, cols_X):
    X = pd.get_dummies(data[cols_X], columns = ['season'],prefix = ['season'])
    if is_y:
        y= data[['lcasual', 'lregistered', 'ltotal', 'total']]
    else:
        y = None
    return X, data.datetime, y
    
##### prepare data with new variables ##################################################################################
def prepare_RF_2(data, is_y, cols_X, if_fillna = True):
    X = pd.get_dummies(data[cols_X], columns = ['season'],prefix = ['season'])
    mlag_col = [c for c in X.columns if ('mlag' in c)]
    if if_fillna:    
        X[mlag_col] = X[mlag_col].fillna(-1)
    if is_y:
        y= data[['lcasual', 'lregistered', 'ltotal', 'total']]
    else:
        y = None
    return X, data.datetime, y
    
#### test parameters ############################################################################################################
def test_RF(X_tv, y_tv, dates_tv,day_test, day_valid_small, day_valid, m_f_opt,m_d_opt, n_e_opt):
    # preparing data
    n_rows = 2 * 12 * len(m_f_opt) * len(m_d_opt) * len(n_e_opt) 
    rf_results = pd.DataFrame(np.zeros([n_rows,11]), 
                          columns = ['year', 'month', 'max_features','max_depth',
                                     'n_estimators', 'rmsle_tot', 
                                     'rmsle_cas', 'rmsle_reg','train_rmsle_tot', 
                                     'train_rmsle_cas', 'train_rmsle_reg'])
    rf_results.loc[:,['year', 'month', 'max_features','max_depth','n_estimators']] = list(it.product([2011, 2012],range(1,13), m_f_opt,m_d_opt, n_e_opt))           

    i = 0
    for year in [2011,2012]:
        for month in range(1,13):
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
            for m in m_f_opt:
                for md in m_d_opt:
                    rf_c = RandomForestRegressor(n_jobs=-1, max_features = m, max_depth = md, warm_start=True)
                    rf_r = RandomForestRegressor(n_jobs=-1, max_features = m, max_depth = md, warm_start=True)
                    for n in n_e_opt:
                        ## casual
                        rf_c.n_estimators= n
                        rf_c.fit(X_tv[train_ind], y_tv.loc[train_ind,'lcasual'])
                        pred_cas = rf_c.predict(X_tv[valid_ind])
                        rf_results.ix[i, 'rmsle_cas'] = resf.rmsle_of_logs(pred_cas, y_tv.loc[valid_ind,'lcasual'])
                        pred_cas_train = rf_c.predict(X_tv[train_ind])
                        rf_results.ix[i, 'train_rmsle_cas'] = resf.rmsle_of_logs(pred_cas_train, y_tv.loc[train_ind,'lcasual'])
                        ## registered
                        rf_r.n_estimators= n
                        rf_r.fit(X_tv[train_ind], y_tv.loc[train_ind,'lregistered'])
                        pred_reg = rf_r.predict(X_tv[valid_ind])
                        rf_results.ix[i, 'rmsle_reg'] = resf.rmsle_of_logs(pred_reg, y_tv.loc[valid_ind,'lregistered'])
                        pred_reg_train = rf_r.predict(X_tv[train_ind])
                        rf_results.ix[i, 'train_rmsle_reg'] = resf.rmsle_of_logs(pred_reg_train, y_tv.loc[train_ind,'lregistered'])                    
                        ## total
                        pred_total = np.log(np.exp(pred_cas)+np.exp(pred_reg)-1) #np.log(resf.total_from_log(pred_cas, pred_reg)+1)
                        pred_total_train = np.log(resf.total_from_log(pred_cas_train, pred_reg_train)+1)
                        rf_results.ix[i, 'rmsle_tot'] = resf.rmsle_of_logs(pred_total, y_tv.loc[valid_ind,'ltotal'])
                        rf_results.ix[i, 'train_rmsle_tot'] = resf.rmsle_of_logs(pred_total_train, y_tv.loc[train_ind,'ltotal'])                    
                        print('Done: ', flush = True)
                        print(rf_results.ix[i], flush = True)
                                     
                        i+=1
        
    return rf_results
    
#### train and save forest on data without specified test set ###############################################################################################
def train_RF(path, X_tv,y_tv,dates_tv,day_test, n_estimators, max_features = None, max_depth = None):
    for year in [2011,2012]:
        for month in range(1,13):
            train_ind = ss.get_train(dates_tv, year,month,day_test, day_test) # I use whole train-validation dataset up to current year & month (but without test dates)
            print('year {}, month {}'.format(year,month), flush = True)
            print('learning from {} to {}'.format(dates_tv[train_ind].min(), dates_tv[train_ind].max()))
            rf_c = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, n_jobs=-1)
            rf_r = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, n_jobs=-1)
            ## casual
            rf_c.fit(X_tv[train_ind], y_tv.loc[train_ind,'lcasual'])       
            ## registered
            rf_r.fit(X_tv[train_ind], y_tv.loc[train_ind,'lregistered'])
            ## save
            with open(os.path.join(path,'{}_{}_c.pkl'.format(year,month)),'wb') as f:
                pickle.dump(rf_c, f)
            with open(os.path.join(path,'{}_{}_r.pkl'.format(year,month)),'wb') as f:
                pickle.dump(rf_r, f)     
                
#### predict using saved forests #################################################################################################################
def predict_RF(X, dates, path):
    result_cas = None
    result_reg = None
    for year in [2011,2012]:
        for month in range(1,13):
            print(year, month, flush = True)
            ind_curr = ((dates >= dt.datetime(year,month,1)) & (dates < dt.datetime(year,month,1)+relativedelta(months=1))).values
            with open(os.path.join(path,'{}_{}_c.pkl'.format(year,month)),'rb') as f:
                rf_c = pickle.load(f)
            with open(os.path.join(path,'{}_{}_r.pkl'.format(year,month)),'rb') as f:
                rf_r = pickle.load(f) 
                
            pred_cas = rf_c.predict(X[ind_curr])
            pred_reg = rf_r.predict(X[ind_curr])
            ## total
            pred_cas_e = (np.exp(pred_cas)-1) 
            pred_reg_e = (np.exp(pred_reg)-1) 
            if result_cas is None:
                result_cas = pred_cas_e
            else:
                result_cas = np.concatenate((result_cas, pred_cas_e))
            if result_reg is None:
                result_reg = pred_reg_e
            else:
                result_reg = np.concatenate((result_reg, pred_reg_e))
    
    return result_reg, result_cas