# -*- coding: utf-8 -*-

"""
The dataset contains aggregated profile features for each indicator (kpi) at each timestamp. 
Features are normalized and detrended.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
#from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import configparser
import os
import pandas as pd


class DataProcessing(object):
    
    def __init__(self):
        
        super().__init__()
               
    
        
    def fillNaN(self, d):
        #replace strings with NaN
        d = d.replace(r'^([A-Za-z]|[0-9]|_)+$', np.NaN, regex=True)
        #replace NaN with 0
        d = d.fillna(0)
        #print([d[x] for x in list(d.keys())])
        return d
    
    
    def read_data(self, file):

        df = pd.read_csv(file,index_col=0)
        
        return df
        
    """
    * Estimates linear trend = > solves the equation ax + b = y
    * 
    * @param ixValues
    *          X-Values of time series = timestamps
    * @param iyValues
    *          Y-Values of the time series
    * a - slope
    * b - intercept
    * @return
    """
    
    def identifyLinearTrend(self, ixValues, iyValues):

        counter = 0;
        denominator = 0;
        avgX = np.mean(ixValues)
        avgY = np.mean(iyValues)

        nrValues = len(ixValues);

        for i in range(nrValues):
            tmpXiXavg = ixValues[i] - avgX;
            tmpYiYavg = iyValues[i] - avgY;
            
            counter += tmpXiXavg * tmpYiYavg;
            denominator += tmpXiXavg * tmpXiXavg;
 
        if (denominator == 0): 
          slope = sys.float_info.max
        else:
          slope = counter / denominator;

        intercept = avgY - slope * avgX;
    
        return  slope, intercept 
      

    def detrend(self, x, y):
        x = np.array(x)
        y = np.array(y)
        #slope, intercept = self.identifyLinearTrend(x, y)
        #print(slope, intercept)
        slope_intercept = np.polyfit(x,y,1)
        trend = slope_intercept[0]*x + slope_intercept[1]
        return y - trend, trend
    
    def saveDetrendToCsv(self, df, file):
        keys = list(df.keys())
        df_detrended =  pd.DataFrame({key:[] for key in keys})
        df_detrended["timestamp"] = df["timestamp"]
        for i in range(1,len(keys)):
            detrend_y = self.detrend(df["timestamp"], df[df.columns[i]])
            df_detrended[keys[i]] = detrend_y
        
        df_detrended.to_csv(file)
            
    
    def min_max_norm(self,ts, min_, max_):
        ts = np.array(ts)
        return (ts - min_)/(max_ - min_)
    
                
    def timeseries_to_supervised(self, data, seq_length):
    
        data = np.array(data).reshape(-1,1)
        #print(len(data), seq_length ) 
        x = np.zeros((len(data)-seq_length,seq_length,data.shape[1]))
        y = np.zeros(len(data)-seq_length)
    
        for i in range(seq_length, len(data)):
            #print("i", i, "i-seq_length", i-seq_length, data[i-seq_length:i])
            x[i-seq_length] = data[i-seq_length:i]
            y[i-seq_length] = data[i,0]
        x = x.reshape(-1,seq_length,data.shape[1])
        y = y.reshape(-1,1)
        
        return x,y
    
