# -*- coding: utf-8 -*-

"""
The dataset contains aggregated profile features for each indicator (kpi) at each timestamp. 
Features are normalized and detrended.
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import configparser
import os, sys
import time
from utils.data_processing import DataProcessing
from utils.slighding_window import SlighdingWindow
from gru import GRU_NN
import utils.localizer_log as localizer_log
from utils.energy_data_processing import Energy_data_processing
from utils.split_to_train_test_valid import split_to_train_test_valid

import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)    

print(torch.__version__)

def plot_results(true, pred):
    
    print("mean error: ",np.mean(true - pred))
    sMAPE = np.mean(abs(pred-true)/(true+pred)/2)/len(pred)
    print("sMAPE: {}%".format(sMAPE*100))
    
    fig,ax = plt.subplots()

    plt.plot(np.concatenate(pred, axis=0), c= "b", label = "pred")
    plt.plot(np.concatenate(true, axis=0), c= "k", label = "true")
    
    plt.xlabel("time [h]")
    plt.ylabel("KPI [s]")
    plt.legend(loc="best")
    plt.show()    
    

if __name__ == '__main__':
    
    config = configparser.RawConfigParser()
    rest_mode = False
    config.optionxform = str
    if os.path.isfile('config'):
        config.read('config')
    else:
        localizer_log.stdout("config file not found")

    
    file = config.get('default', 'file')
    df_column = [int(config.get('default', 'df_column'))]
        
    slighding_window = int(config.get('NN', 'slighding_window_num_pints'))
    hidden_dim = int(config.get('NN', 'hidden_dim'))
    dropout = float(config.get('NN', 'dropout'))
    num_layers = int(config.get('NN', 'num_layers'))
    num_epochs = int(config.get('NN', 'number_epoches'))
    regularisation_parameter = float(config.get('NN', 'regularisation_parameter'))
    learning_rate = float(config.get('NN', 'learning_rate'))

    batch_size = int(config.get('NN', 'batch_size'))
    model_type = str(config.get('NN', 'model_type'))
    model_name = str(config.get('NN', 'model_name'))

    points_per_hour = int(config.get('NN', 'points_per_hour'))
    trainingset_size_num_weeks = float(config.get('NN', 'trainingset_size_num_weeks'))
    testset_size_num_weeks = float(config.get('NN', 'testset_size_num_weeks'))
    validset_size_num_weeks = float(config.get('NN', 'validset_size_num_weeks'))
    
    
    train_len = int(points_per_hour*24*7*trainingset_size_num_weeks)
    test_len = int(points_per_hour*24*7*testset_size_num_weeks)
    valid_len = int(points_per_hour*24*7*validset_size_num_weeks)
    
    
    split = split_to_train_test_valid()
    df_trend, df_min_max, training_df, test_df, valid_df = split.processing(file, train_len, test_len, valid_len, df_column)
    
    """
    training_df = pd.read_csv('logs/training_set.csv', index_col=False)
    test_df = pd.read_csv('logs/test_set.csv', index_col=False)
    valid_df = pd.read_csv('logs/valid_set.csv', index_col=False)
    """    
    dp = DataProcessing()
    x_train, y_train  = dp.timeseries_to_supervised(training_df, slighding_window)
    x_test, y_test  = dp.timeseries_to_supervised(test_df, slighding_window)
    x_valid, y_valid  = dp.timeseries_to_supervised(valid_df, slighding_window)   
       
    #print("x_train",x_train)
    #print("y_train",y_train)
    #input tensor with the size of [N, input_shape] where N is the number of examples, 
    #and input_shape is the number of features in one example
    x_train = torch.tensor(np.array(x_train), dtype=torch.float,  device=device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float,  device=device)
    x_test = torch.tensor(np.array(x_test), dtype=torch.float,  device=device)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float,  device=device)
    x_valid = torch.tensor(np.array(x_valid), dtype=torch.float,  device=device)
    y_valid = torch.tensor(np.array(y_valid), dtype=torch.float,  device=device)
    
    
    # must be (batch_size, seq_len, numb_features/input_size) 
    localizer_log.stdout("train_data.shape: " + str(x_train.shape))
    localizer_log.stdout("test_data.shape: " + str(x_test.shape))
    localizer_log.stdout("valid_data.shape: " + str(x_valid.shape))
    
    
    
    training_loader = (DataLoader(TensorDataset(x_train, y_train), 
                               batch_size=batch_size, shuffle=True, drop_last=True))
    
    #for i, d in enumerate(training_loader):
    #    print(i, d[0].shape, d[1].shape)
    #    print(d)
    localizer_log.stdout("training_loader.shape: " + str(next(iter(training_loader))[0].shape)+"  " \
                          + str(next(iter(training_loader))[1].shape))

    #the feature size of the input tensor 
    input_size = next(iter(training_loader))[0].shape[2]
    # the feature size of the output tensor 
    output_dim = 1
    #numb_layers = int(config.get('autoencoder', 'numb_layers'))

    is_cuda = torch.cuda.is_available()
    localizer_log.stdout("cuda avaialbel: " +  str(is_cuda) )


    #Initialize the model 
    # load it to the specified device, either gpu or cpu    
    model = GRU_NN(dropout, input_size, output_dim, hidden_dim, num_layers).to(device)
    
    model.to(device)
    
    #Define loss criterion
    criterion = nn.MSELoss() 
   
    #Define the optimizer
    optimizer = optim.Shampoo(model.parameters(), lr=learning_rate,
                        weight_decay=1e-4 )

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.01, max_lr=0.1,
                                                  step_size_up=5,mode="exp_range",gamma=0.85)
    if os.path.isfile(model_name) :
        localizer_log.info(model_name + " found. Restaring the trainig.")
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

    start_time = time.time()
    model.train_model(criterion,optimizer,scheduler, num_epochs, model, training_loader, x_valid, y_valid,  model_name, batch_size)
    localizer_log.info("--- %s seconds ---" % (time.time() - start_time))
    
    result, target = model.run_predict(x_test, y_test, model_name, model)
    
    min_, max_ = df_min_max[key]
    true = (true + min_ ) * (max_ - min_)
    pred = (pred + min_ ) * (max_ - min_)
    
    plot_results(key, true, pred)
    
    


    
    
    


