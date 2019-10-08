import keras
from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
#import required to load the attacked model
import sys
sys.path.append('../')
from Attacked_Model.autoencoder_BATADAL import load_AEED
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, confusion_matrix

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os  # We need this module
import datetime

#this function fills the data with the 'window' seconds before attack starts, this gives correct results about the detection
def fill_window(original_data, attack_data, att_number,window):
    att_timings = {'1':{'date':'2017-10-9', 'start': '19:25:00'},
               '2':{'date':'2017-10-10', 'start': '10:24:10'},
               '3':{'date':'2017-10-10', 'start': '10:55:00'},
               #'4':{'date':'2017-10-10', 'start': '11:07:46'},
               '5':{'date':'2017-10-10', 'start': '11:30:40'},
               '6':{'date':'2017-10-10', 'start': '13:39:30'},
               '7':{'date':'2017-10-10', 'start': '14:48:17'},
               '8':{'date':'2017-10-10', 'start': '17:40:00'},
               '9':{'date':'2017-10-11', 'start': '10:55:00'},
               '10':{'date':'2017-10-11', 'start': '11:17:54'},
               '11':{'date':'2017-10-11', 'start': '11:36:31'},
               '12':{'date':'2017-10-11', 'start': '11:59:00'},
               '13':{'date':'2017-10-11', 'start': '12:07:30'},
               '14':{'date':'2017-10-11', 'start': '12:16:00'},
               '15':{'date':'2017-10-11', 'start': '15:26:30'}
              }
    window_start = pd.to_datetime(att_timings[str(att_number)]['date']+' '+att_timings[str(att_number)]['start'], format='%Y-%m-%d %H:%M:%S') - datetime.timedelta(0,window) # days, seconds, then other fields.
    end = pd.to_datetime(att_timings[str(att_number)]['date']+' '+att_timings[str(att_number)]['start'], format='%Y-%m-%d %H:%M:%S')
    window_data = original_data[window_start:end].reset_index()
    unified_data = pd.DataFrame(columns = attack_data.columns, data = pd.concat((window_data, attack_data), ignore_index = True))
    return unified_data