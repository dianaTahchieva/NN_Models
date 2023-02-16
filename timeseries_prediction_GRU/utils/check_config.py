# -*- coding: utf-8 -*-

import os
import configparser
import utils.localizer_log as localizer_log

config = configparser.RawConfigParser()
config.optionxform = str
if os.path.isfile('config'):
    config.read('config')
else:
    print("config file not found")

def check_config_variables_restful():
    try:
        config.get('restful', 'url')
    except:
        localizer_log.error("Missing variable: \'url\' under \'restful\'; e.g., \'http://localhost:5000\' ")

    try:
        config.get('restful', 'service_port')
    except:
        localizer_log.error("Missing variable: \'service_port\' under \'restful\'; e.g., \'5000\' ")
        
        
def check_config_variables_nn():
    
    try:
        config.get('NN', 'num_layers')
    except:
        localizer_log.error("Missing variable: \'num_layers\' under \'restful\'; e.g., \'3\' ")

    try:
        config.get('NN', 'dropout')
    except:
        localizer_log.error("Missing variable: \'dropout\' under \'restful\'; e.g., \'0.2\' ")
        
    try:
        config.get('NN', 'number_epoches')
    except:
        localizer_log.error("Missing variable: \'number_epoches\' under \'restful\'; e.g., \'5000\' ")
        
    try:
        config.get('NN', 'hidden_dim')
    except:
        localizer_log.error("Missing variable: \'hidden_dim\' under \'restful\'; e.g., \'100\' ")
        
    try:
        config.get('NN', 'regularisation_parameter')
    except:
        localizer_log.error("Missing variable: \'regularisation_parameter\' under \'restful\'; e.g., \'10E-5\' ")
        
    try:
        config.get('NN', 'learning_rate')
    except:
        localizer_log.error("Missing variable: \'learning_rate\' under \'restful\'; e.g., \'10E-3\' ")
        
    try:
        config.get('NN', 'slighding_window_num_pints')
    except:
        localizer_log.error("Missing variable: \'slighding_window_num_pints\' under \'restful\'; e.g., \'144\' ")
        
    try:
        config.get('NN', 'batch_size')
    except:
        localizer_log.error("Missing variable: \'batch_size\' under \'restful\'; e.g., \'64\' ")
        

        
    
    
        
    


