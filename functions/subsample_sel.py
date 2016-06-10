# -* coding: utf-8 -*
import datetime as dt
from dateutil.relativedelta import relativedelta
#### GET INDEXES FOR VALIDATION SAMPLE ##################################################
def get_valid(dates,year,month,day_test,day_valid):
    if day_test < day_valid:
        raise Exception("day_test < day_valid")
    return ((dates >= dt.datetime(year,month,day_valid)) & #validation days
            (dates < dt.datetime(year,month,1)+relativedelta(months=1)) & #only this month  
            (dates < dt.datetime(year,month,day_test)))    #smaller than test days
            
#### GET INDEXES FOR TEST SAMPLE #######################################################
def get_test(dates,year,month,day_test,day_valid):
    if day_test < day_valid:
        raise Exception("day_test < day_valid")
    return ((dates >= dt.datetime(year,month,day_test)) & #validation days
            (dates < dt.datetime(year,month,1)+relativedelta(months=1))) #only this month
            
#### GET INDEXES FOR TEST SAMPLE #######################################################
# returns all data from previous months and keep only validation & test for current month
def get_train(dates,year,month, day_test,day_valid):
    if day_test < day_valid:
        raise Exception("day_test < day_valid")
    return dates < dt.datetime(year,month,min(day_test,day_valid)) 
       