# %reset
import os
os.chdir('D:\\Kaggle\\bike-sharing-demand')
import pandas as pd
import importlib as il
import seaborn as sn
import functions.preprocess as pr
import functions.results_functions as resf
import functions.subsample_sel as ss
import functions.rf_functions as rff

il.reload(pr)
il.reload(resf)
il.reload(ss)
il.reload(rff)

# prepare datasets ###############################################################################
data,_ = pr.get_train_test_data_with_new_var()
cols_drop = ['casual', 'date', 'datetime', 'day','lcasual', 'lregistered', 'ltotal',
             'registered', 'season2', 'total'] 
cols_X = [c for c in data.columns if c not in cols_drop]

X,dates,y = rff.prepare_RF_2(data, True, cols_X)

## parameters to test ###########################################################################
m_f_max = X.shape[1]
m_f_opt = [11,20]
m_d_opt = [None]
n_e_opt = [300]

## building forests ##############################################################################
rf_results = rff.test_RF(X, y, dates,10, 8, 3, m_f_opt,m_d_opt, n_e_opt)

rf_results.to_csv('results\\rf_6-max_11-20_300_new_fea_more.csv') 
rf_results_old = pd.read_csv('results\\rf_5-max_4-None_300_new_fea.csv') 

rf_results.loc[rf_results.max_depth.isnull(), 'max_depth'] = 1000

## plotting results #####################################################################################
sn.factorplot(x="max_features", y="rmsle_cas", hue="max_depth",palette = sn.color_palette("RdBu_r", 7),
                   col="year", row = 'month', data=rf_results)
## comparing to previous ones ###########################################################################
rf_results['type'] = 'more f'
rf_results_old['type'] = 'less_f'
rf_results_all = pd.concat([rf_results[(rf_results.n_estimators == 300) & (rf_results.max_depth == 1000)], rf_results_old])
sn.factorplot(x="max_features", y="rmsle_reg", hue="type",palette = sn.color_palette("RdBu_r", 7),
                   col="year", row = 'month', data=rf_results_all, ci=None)
                   
sn.factorplot(x="max_features", y="rmsle_cas", hue="type",palette = sn.color_palette("RdBu_r", 7),
                   col="year", row = 'month', data=rf_results_all, ci=None)
                  
## training with selected parameters using validation data##############################################################################          
rff.train_RF('trees_new_feat', X,y,dates,10,150) 
data_less_f = pd.read_csv('data\\train_clean_new.csv', parse_dates = ['datetime'])
cols_X = [c for c in data_less_f.columns if c not in cols_drop]
X_less_f,dates_less_f,y_less_f = rff.prepare_RF_2(data_less_f, True, cols_X) 
rff.train_RF('trees_less_f', X_less_f,y_less_f,dates_less_f,10,150, max_features = 11)  

rff.train_RF('trees_on_all_data', X,y,dates,21,700) 
rff.train_RF('tree_on_all_data_less_f', X_less_f,y_less_f,dates_less_f,21,700,max_features = 11)   
## check features importance ##########################################################################
path = 'trees'
import pickle as pickle
with open(os.path.join(path,'{}_{}_c.pkl'.format(2012,12)),'rb') as f:
    rf_c = pickle.load(f)
with open(os.path.join(path,'{}_{}_r.pkl'.format(2012,12)),'rb') as f:
    rf_r = pickle.load(f) 
    
features = pd.DataFrame({'feat': X.columns, 'cas': rf_c.feature_importances_,
                         'reg': rf_r.feature_importances_})
features.sort_values('reg',inplace = True, ascending = False)
##########################################################################################################


