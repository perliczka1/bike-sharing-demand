# -*- coding: utf-8 -*-
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
#### RMSLE CALCULATED ON LOGARITHMS + 1 #######################################################
def rmsle_of_logs(y_pred_log, y_true_log):
    return pow(mean_squared_error(y_true_log, y_pred_log),0.5)
    
#### RMSLE CALCULATED ON LOGARITHMS + 1 OF TRUE VAL AND PLAIN PRED VAL ###########################
def rmsle_true_log(y_pred, y_true_log):
    return rmsle_of_logs(np.log(y_pred+1), y_true_log)

#### AS ABOVE BUT BY MONTH ##########################################################################
def rmsle_by_month_true_log(y_pred, y_true_log,dates):
    d = pd.DataFrame({'y_log_diff': np.power(np.log(y_pred+1)-y_true_log,2),
                      'month': dates.apply(lambda x: x.month),
                      'year': dates.apply(lambda x: x.year)})
    res = d.groupby(['year','month']).y_log_diff.aggregate(lambda x: pow(sum(x)/len(x),0.5))
    return res
    
#### GET PREDICTION FROM LOGARITHMS +1 ###########################################################    
def total_from_log(pred_cas, pred_reg):
    return (np.exp(pred_cas)-1) +(np.exp(pred_reg)-1) 

    
