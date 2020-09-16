import time
import keras
from keras.layers import Input, Dense, Activation, BatchNormalization, Lambda
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf
import os
# import required to load the attacked model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simplejson as json
import pickle
from str2bool import str2bool

from adversarial_AE import Adversarial_AE
"""
Select wich dataset are you considering
(we are not allowed to publish WADI data, please request them itrust Singapore website)
-----------
Set options for computation

    pretrain_generator : bool
        if the adversarial network needs to be pretrained. (if False, the model should be already stored accordingly)
    measure_time : bool
       True: measure the required computational time
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='BATADAL')
parser.add_argument('-p', '--pretrain', type=str2bool, default=False)
args = parser.parse_args()
print(args)

dataset = args.data
data_folder = '../../Data/'+dataset
pretrain_generator = args.pretrain

if dataset == 'BATADAL':
    attack_ids = range(1, 15)
    att_data = pd.read_csv(data_folder+'/attack_1_from_test_dataset.csv')
    xset = [col for col in att_data.columns if col not in [
        'Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
    plcs = ['PLC_1', 'PLC_2', 'PLC_3', 'PLC_4',
            'PLC_5', 'PLC_6', 'PLC_7', 'PLC_8', 'PLC_9']
    feature_dims = 43
    hide_layers = 128

if dataset == 'WADI':
    attack_ids = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    att_data = pd.read_csv(data_folder+'/attack_1_from_test_dataset.csv')
    xset = [col for col in att_data.columns if col not in [
        'Row', 'DATETIME', 'ATT_FLAG', '2_MV_001_STATUS', '2_LT_001_PV', '2_MV_002_STATUS']]
    plcs = ['PLC_1', 'PLC_2']
    feature_dims = 82
    hide_layers = 256
yset = ['ATT_FLAG']


if __name__ == '__main__':
    
    ben_data = pd.read_csv(data_folder+'/train_dataset.csv')
    variables = {}
    f = open("./constraints/"+dataset+"/constraint_PLC.txt", 'r').read()
    variables = eval(f)

    for plc in plcs:
        K.clear_session()
        if pretrain_generator:
            advAE = Adversarial_AE(len(variables[plc]), len(variables[plc])*4)
            advAE.attacker_scaler = advAE.attacker_scaler.fit(ben_data[variables[plc]])
            advAE.train_advAE(ben_data, variables[plc])
            advAE.generator.save('adversarial_models/' +
                                 dataset+'/generator_'+plc+'.h5')
        else:
            advAE = Adversarial_AE(len(variables[plc]), len(variables[plc])*4)
            advAE.generator = load_model(
                'adversarial_models/'+dataset+'/generator_'+plc+'.h5')
            advAE.attacker_scaler = advAE.attacker_scaler.fit(ben_data[variables[plc]])
    
        
        for i in attack_ids:
            print('\nAttack: ' + str(i))
            constraints = variables[plc]
            print('Imposed constraint:' + str(constraints))
            att_data = pd.read_csv(data_folder+'/attack_' +
                                   str(i)+'_from_test_dataset.csv')
            
            X = pd.DataFrame(index=att_data.index,
                             columns=xset, data=att_data[xset])

            binary_dataframe = pd.DataFrame(columns=xset)
            gen_examples = advAE.generator.predict(
                advAE.attacker_scaler.transform(X[constraints]))
            print(gen_examples.shape)
            gen_examples = advAE.fix_sample(pd.DataFrame(
                columns=constraints, data=advAE.attacker_scaler.inverse_transform(gen_examples)), dataset)

            gen_examples = advAE.conceal_fixed(constraints, pd.DataFrame(columns=constraints, data=gen_examples),
                                               pd.DataFrame(columns=xset, data=X))

            if not os.path.exists('results/'+dataset+'/AE_PLC_constraint/'):
                os.makedirs(
                    'results/'+dataset+'/AE_PLC_constraint/')
            pd.DataFrame(columns=xset, data=gen_examples).to_csv(
                'results/'+dataset+'/AE_PLC_constraint/new_constrained_plc_AE_attack_'+str(i)+'_from_test_dataset_'+plc+'.csv')
