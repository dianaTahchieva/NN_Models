# -*- coding: utf-8 -*-

"""
The dataset contains aggregated profile features for each indicator (kpi) at each timestamp. 
Features are normalized and detrended.
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from utils.data_processing import DataProcessing
import datetime
#from datatable import dt
from matplotlib.pyplot import spring



class split_to_train_test_valid(object):
    
    def __init__(self):
        
        super().__init__()
               
    
        
    def processing(self, file,train_len, test_len, valid_len, df_column):
        dp = DataProcessing()
        df = pd.read_csv(file,index_col=0)
                
        keys = list(df.columns[df_column])
        print(df_column, "keys",list(keys))
        if "timestamp" not in keys:
            keys.insert(0, "timestamp")

        print(keys)
        sub_df = dict()
        for key in keys:
            sub_df[key] = np.array(df[key])
        
        #print(np.shape(list(sub_df[keys[0]])))
        
        #x = range((train_len + valid_len + test_len))
       
        #sub_df[keys[0]] = x
        #print("sub_df", sub_df)
        
        training_set = pd.DataFrame()
        test_set = pd.DataFrame()
        valid_set = pd.DataFrame()
        df_min_max = pd.DataFrame()
        df_trend = pd.DataFrame()

        for col in keys:
            if col != "timestamp":
                #print("sub_df", sub_df[col])
                detrend_y, trend = dp.detrend(sub_df["timestamp"], sub_df[col])
                
                if col not in df_trend.keys(): 
                    df_trend[col] = trend
                
    
                #fig,ax = plt.subplots()
                range_ = range(len(detrend_y))
                #plt.plot(range_,df[col])
                #plt.plot(range_,detrend_y)
                #print("total len ",len(sub_df[col]), "train_len", train_len , 
                #      "train_len + valid ", (train_len + valid_len), "train+valid+test",(train_len + valid_len + test_len))
                min_ = np.array(detrend_y[:train_len]).min()
                max_ = np.array(detrend_y[:train_len]).max()
                
                if col not in df_min_max.keys(): 
                    df_min_max[col] = [min_, max_]
                    
                
                training_set[col] = dp.min_max_norm(detrend_y[:train_len], min_, max_)
                valid_set[col] =  dp.min_max_norm(detrend_y[train_len:(train_len + valid_len)], min_, max_)
                test_set[col] = dp.min_max_norm(detrend_y[(train_len + valid_len):(train_len + valid_len + test_len)], min_, max_)
                            
                #plt.plot(range_[:train_len],training_set[col])
                #plt.plot(range_[train_len:(train_len + valid_len)],valid_set[col])
                #plt.plot(range_[(train_len + valid_len):(train_len + valid_len + test_len)],test_set[col])
                
                #plt.xlabel("time [h]")
                #plt.ylabel("KPI [s]")
                #plt.legend(loc="best")
                #plt.show()
            
        """    
        df_trend.to_csv("logs/trend.csv")            
        df_min_max.to_csv("logs/min_max.csv")
        training_set.to_csv("logs/training_set.csv",index=False)
        test_set.to_csv("logs/test_set.csv",index=False)
        valid_set.to_csv("logs/valid_set.csv",index=False)
        """
        return df_trend, df_min_max, training_set, test_set, valid_set