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
from sklearn.preprocessing import MinMaxScaler


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
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='BATADAL')
parser.add_argument('-p', '--pretrain', type=str2bool, default=False)
args = parser.parse_args()
print(args)

dataset = args.data
data_folder = '../../Data/'+dataset
pretrain_generator =  args.pretrain

if dataset == 'BATADAL':
    attack_ids = range(1, 15)
    att_data = pd.read_csv(data_folder+'/attack_1_from_test_dataset.csv')
    xset = [col for col in att_data.columns if col not in [
        'Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
    feature_dims = 43
    hide_layers = 128

if dataset == 'WADI':
    attack_ids = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    att_data = pd.read_csv(data_folder+'/attack_1_from_test_dataset.csv')
    xset = [col for col in att_data.columns if col not in [
        'Row', 'DATETIME', 'ATT_FLAG', '2_MV_001_STATUS', '2_LT_001_PV', '2_MV_002_STATUS']]
    feature_dims = 82
    hide_layers = 256
yset = ['ATT_FLAG']

if __name__ == '__main__':
    
    #seeds for random sampling of data
    for seed in [0, 1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789]:
        
        # 25 means train the adversarial network with 25% of data
        for percentage_reduced in [5, 10, 25, 50, 75]:
            print('Experiment seed = '+str(seed))
            print('percentage = '+str(percentage_reduced))
            advAE = Adversarial_AE(feature_dims, hide_layers)
            ben_data = pd.read_csv(data_folder+'/train_dataset.csv')
            if percentage_reduced != 100:
                ben_data = ben_data.sample(
                    frac=((percentage_reduced)/100), random_state=seed)
            advAE.attacker_scaler = advAE.attacker_scaler.fit(ben_data[xset])
            if pretrain_generator:
                print('Training generator')
                advAE.train_advAE(ben_data, xset)
                advAE.generator.save('adversarial_models/'+dataset+'/generator_' +
                            str(percentage_reduced)+'_percent_seed_'+str(seed)+'.h5')

            advAE.generator = load_model(
                'adversarial_models/'+dataset+'/generator_'+str(percentage_reduced)+'_percent_seed_'+str(seed)+'.h5')

            for i in attack_ids:
                att_number = i
                att_data = pd.read_csv(
                    data_folder+'/attack_'+str(i)+'_from_test_dataset.csv')
                y_att = att_data[yset]
                X = pd.DataFrame(index=att_data.index,
                                 columns=xset, data=att_data[xset])
                
                gen_examples = advAE.generator.predict(
                    advAE.attacker_scaler.transform(X))
                gen_examples = advAE.fix_sample(pd.DataFrame(
                    columns=xset, data=advAE.attacker_scaler.inverse_transform(gen_examples)), dataset)

                if not os.path.exists('results/'+dataset+'/AE_'+str(percentage_reduced)+'_percent/'):
                    os.makedirs(
                        'results/'+dataset+'/AE_'+str(percentage_reduced)+'_percent/')
                if not os.path.exists('results/'+dataset+'/AE_'+str(percentage_reduced)+'_percent/seed_'+str(seed)):
                    os.makedirs(
                        'results/'+dataset+'/AE_'+str(percentage_reduced)+'_percent/seed_'+str(seed))
                
                pd.DataFrame(columns=xset, data=gen_examples).to_csv(
                    'results/'+dataset+'/AE_'+str(percentage_reduced)+'_percent/seed_'+str(seed)+'/new_advAE_attack_'+str(i)+'_from_test_dataset.csv')
            print('Results Saved \n -------------')
