# -*- coding: utf-8 -*-

"""
The dataset contains aggregated profile features for each indicator (kpi) at each timestamp. 
Features are normalized and detrended.
"""

import numpy as np
import pandas as pd
from utils.data_processing import DataProcessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler,QuantileTransformer  
import matplotlib.pylab as plt

class Energy_data_processing(object):
    
    def __init__(self):
        
        super().__init__()
               
    
        
    def processing(self, file, train_len, test_len, valid_len, df_column):
        
        dp = DataProcessing()
        df = pd.read_csv(file,index_col=0)
        
        keys = df.columns[df_column]
        print("keys",keys)
        sub_df = df[keys]#['Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3',
           #'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
           #'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed']]
        
        """
        dt = []
        for line in df["date"]:
            dt.append(datetime.datetime.strptime(line, '%Y-%m-%d %H:%M:%S').timestamp())
       
        df["date"] = dt
        
        print(df)
        df.to_csv(file)
        """
        
        training_set = pd.DataFrame()
        test_set = pd.DataFrame()
        valid_set = pd.DataFrame()

        df_min_max = pd.DataFrame()
        
        for i,col in enumerate(sub_df.keys()):
 
            min_ = (sub_df[col].iloc[:train_len]).min()
            max_ = (sub_df[col].iloc[:train_len]).max()
            
            if col not in df_min_max.keys(): 
                df_min_max[col] = [min_, max_]
            
            print(train_len, (train_len + valid_len), (train_len + valid_len + test_len))
            training = dp.min_max_norm(sub_df[col].iloc[:train_len], min_, max_)
            valid = dp.min_max_norm(sub_df[col].iloc[train_len:(train_len + valid_len)], min_, max_)
            test = dp.min_max_norm(sub_df[col].iloc[(train_len + valid_len): (train_len + valid_len + test_len)], min_, max_)
           
            #print("training.shape ",training.shape)
            #print("test.shape ",test.shape)
            #print("valid.shape ",valid.shape)
            
            #ss = MinMaxScaler()
            #ss.fit(training)

            #training = np.squeeze(ss.transform(training))
            #valid = np.squeeze(ss.transform(valid))
            #test = np.squeeze(ss.transform(test))
            #print("training", np.any(np.isnan(training)), np.any(np.isinf(training)), 
            #      "test", np.any(np.isnan(test)), np.any(np.isinf(test)),
            #      "valid", np.any(np.isnan(valid)), np.any(np.isinf(valid)))
            training_set[col] =  training
            valid_set[col] = valid
            test_set[col] = test
            
            print("training min",np.min(training), "max", np.max(training), "mean", np.mean(training), "std", np.std(training) )
            print("test min",np.min(test), "max", np.max(test), "mean", np.mean(test), "std", np.std(test) )
            print("valid min",np.min(valid), "max", np.max(valid), "mean", np.mean(valid), "std", np.std(valid) )
            
            #range_ = range(train_len + valid_len + test_len)
            #plt.title(str(i) + " " +col )
            #plt.plot(range_[:train_len],training)
            #plt.plot(range_[train_len:(train_len + valid_len)],test)
            #plt.plot(range_[(train_len + valid_len): (train_len + valid_len + test_len)],valid)
            #plt.show()
            
         
        df_min_max.to_csv("logs/min_max.csv")
        training_set.to_csv("logs/training_set.csv")
        test_set.to_csv("logs/test_set.csv")
        valid_set.to_csv("logs/valid_set.csv")
        