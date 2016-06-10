# %reset
import os
os.chdir('D:\\Kaggle\\bike-sharing-demand')
import pandas as pd
import importlib as il
import numpy as np
import functions.preprocess as pr
import functions.results_functions as resf
import functions.subsample_sel as ss
import functions.rf_functions as rff
import functions.xgb_functions as xf
import datetime as dt
il.reload(pr)
il.reload(resf)
il.reload(ss)
il.reload(rff)
il.reload(xf)
# prepare datasets ################################################################################
## data with all columns
data,test_leaderboard = pr.get_train_test_data_with_new_var()
cols_drop = ['casual', 'date', 'datetime', 'day','lcasual', 'lregistered', 'ltotal',
             'registered', 'season2', 'total'] 
cols_X = [c for c in data.columns if c not in cols_drop]
X,dates,y = rff.prepare_RF_2(data, True, cols_X)
X_xgb, _ ,_= rff.prepare_RF_2(data, True, cols_X, False)
## data only with 35 columns
data_less_f = pd.read_csv('data\\train_clean_new.csv', parse_dates = ['datetime'])
cols_X_less_f = [c for c in data_less_f.columns if c not in cols_drop]
X_less_f,_,_ = rff.prepare_RF_2(data_less_f, True, cols_X_less_f) 

## check on test data ##########################################################################
test_ind = (data.day >= 10).values
## random forest on all columns ##
X_rf_test = X[test_ind]
RF_test_reg,RF_test_cas = rff.predict_RF(X_rf_test, dates[test_ind], 'trees_new_feat')

## random forest on 35 variables ##
X_rf_lf_test = X_less_f[test_ind]
RF_lf_test_reg,RF_lf_test_cas = rff.predict_RF(X_rf_lf_test, dates[test_ind], 'trees_less_f')

## xgboost ##
X_xbg_test = X_xgb[test_ind]
XGB_test_reg = xf.predict_XGB(X_xbg_test, dates[test_ind], 'xgb_models')
np.isnan(XGB_test_reg).sum()

summary = pd.concat([
resf.rmsle_by_month_true_log(RF_test_reg,y.loc[test_ind,'lregistered'], dates[test_ind]),
resf.rmsle_by_month_true_log(RF_test_cas,y.loc[test_ind,'lcasual'], dates[test_ind]),
resf.rmsle_by_month_true_log(RF_lf_test_reg,y.loc[test_ind,'lregistered'], dates[test_ind]),
resf.rmsle_by_month_true_log(RF_lf_test_cas,y.loc[test_ind,'lcasual'], dates[test_ind]),
resf.rmsle_by_month_true_log(XGB_test_reg,y.loc[test_ind,'lregistered'], dates[test_ind])],
axis = 1)
summary.columns = ['RF_reg', 'RF_tcas', 'RF_lf_reg', 'RF_lf_cas', 'XGB_reg']
summary['RF_sum_cas'] = resf.rmsle_by_month_true_log((RF_test_cas+RF_lf_test_cas)/2,y.loc[test_ind,'lcasual'], dates[test_ind])
summary['RF_sum_reg'] = resf.rmsle_by_month_true_log((RF_test_reg+RF_lf_test_reg)/2,y.loc[test_ind,'lregistered'], dates[test_ind])
summary['RF_XGB_reg'] = resf.rmsle_by_month_true_log((RF_test_reg+XGB_test_reg)/2,y.loc[test_ind,'lregistered'], dates[test_ind])
summary['all_reg'] = resf.rmsle_by_month_true_log((RF_test_reg+XGB_test_reg+RF_lf_test_reg)/3,y.loc[test_ind,'lregistered'], dates[test_ind])
   

##  PREDICT FOR DATA FROM LEADERBOARD ##############################################################################################
## prepare data 
X_rf_lb,dates_lb,_ = rff.prepare_RF_2(test_leaderboard, False, cols_X)
X_xgb_lb,_,_ = rff.prepare_RF_2(test_leaderboard, False, cols_X, True)
test_leaderboard_lf = pd.read_csv('data\\test_clean_new.csv', parse_dates = ['datetime', 'date'])
X_rf_lf_lb,_,_  = rff.prepare_RF_2(test_leaderboard_lf, False, cols_X_less_f)
## predict
RF_lb_reg,RF_lb_cas = rff.predict_RF(X_rf_lb, dates_lb, 'trees_on_all_data')
RF_lf_lb_reg,RF_lf_lb_cas = rff.predict_RF(X_rf_lf_lb, dates_lb, 'tree_on_all_data_less_f')
XGB_lb_reg = xf.predict_XGB(X_xgb_lb, dates_lb, 'xgb_models_on_all_data')
np.isnan(XGB_lb_reg).sum()
dates_lb[np.isnan(XGB_lb_reg)]
## weights 
RF_cas = (dates_lb >= dt.datetime(2011,3,1))*0.5
RF_lf_cas = (dates_lb < dt.datetime(2011,3,1))*1 + (dates_lb >= dt.datetime(2011,3,1))*0.5
RF_reg = (dates_lb < dt.datetime(2011,5,1))*1 + (dates_lb >= dt.datetime(2011,5,1))*0.5
XBG_reg= (dates_lb >= dt.datetime(2011,5,1))*0.5 
((RF_cas + RF_lf_cas) !=1 ).sum()     
((RF_reg + XBG_reg) !=1 ).sum()  
pred_cas = RF_cas * RF_lb_cas +  RF_lf_cas * RF_lf_lb_cas
pred_reg = XBG_reg * XGB_lb_reg + RF_reg * RF_lb_reg 
## make submission 
sub = pd.concat([dates_lb, pred_cas+pred_reg], axis = 1, ignore_index = True)
sub.to_csv('submissions//3.csv', index = False, header = ['datetime','count'])

