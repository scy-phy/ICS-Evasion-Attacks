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
    conceal_up_to_n: bool
        True: if you would like to perform the black box constrained attack
        False: perform unconstrained black box attack
    fixed_constraints: bool
       True: perform the experiment of a defined set of constraints that are stored in a dictionary
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', type=str, default=['BATADAL'])
parser.add_argument('-p', '--pretrain', type=str2bool, default = False)
parser.add_argument('-f', '--fixed_constraints', type=str2bool, default=True)
args = parser.parse_args()
print(args)

dataset = args.data[0]
data_folder = '../../Data/'+dataset

pretrain_generator = args.pretrain[0]
fixed_constraints = args.fixed_constraints[0]
conceal_up_to_n = True

if dataset == 'BATADAL':
    attack_ids = range(1, 15)
    att_data = pd.read_csv(data_folder+'/attack_1_from_test_dataset.csv')
    xset = [col for col in att_data.columns if col not in [
        'Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
    feature_dims = 43
    hide_layers = 128
    constrained_variables = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]

if dataset == 'WADI':
    attack_ids = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    att_data = pd.read_csv(data_folder+'/attack_1_from_test_dataset.csv')
    xset = [col for col in att_data.columns if col not in [
        'Row', 'DATETIME', 'ATT_FLAG', '2_MV_001_STATUS', '2_LT_001_PV', '2_MV_002_STATUS']]
    feature_dims = 82
    hide_layers = 256
    constrained_variables = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80]
    
yset = ['ATT_FLAG']

if __name__ == '__main__':
    advAE = Adversarial_AE(feature_dims, hide_layers)
    ben_data = pd.read_csv(data_folder+'/train_dataset.csv')
    advAE.attacker_scaler = advAE.attacker_scaler.fit(ben_data[xset])
    if pretrain_generator:
        advAE.train_advAE(ben_data, xset)
        advAE.generator.save('adversarial_models/' +
                             dataset+'/generator_100_percent.h5')

    advAE.generator = load_model(
        'adversarial_models/'+dataset+'/generator_100_percent.h5')

    for i in attack_ids:
        print('\nAttack: ' +str(i))
        try:
            f = open("./constraints/"+dataset+"/constraint_variables_attack_" +str(i)+"_AE.txt", 'r').read()
            variables = eval(f)
        except:
            variables = {}
        
        if fixed_constraints:
            f = open("./constraints/"+dataset+"/constraint_variables_attack_" +
                    str(i)+"_AE.txt", 'r').read()
            variables = eval(f)
        for n in constrained_variables:
            if fixed_constraints:
                constraints = variables[n]
                print('Imposed constraint:' + str(constraints))
            debug = False
            att_number = i
            att_data = pd.read_csv(data_folder+'/attack_' +
                                str(i)+'_from_test_dataset.csv')
            y_att = att_data[yset]
            X = pd.DataFrame(index=att_data.index,
                            columns=xset, data=att_data[xset])
      
            binary_dataframe = pd.DataFrame(columns=xset)
            gen_examples = advAE.generator.predict(
                advAE.attacker_scaler.transform(X))
            gen_examples = advAE.fix_sample(pd.DataFrame(
                columns=xset, data=advAE.attacker_scaler.inverse_transform(gen_examples)),dataset)
            if conceal_up_to_n and not(fixed_constraints):
                gen_examples, binary_dataframe = advAE.decide_concealment(n, binary_dataframe, pd.DataFrame(columns=xset, data=advAE.attacker_scaler.transform(gen_examples)),
                                                                        pd.DataFrame(columns=xset, data=advAE.attacker_scaler.transform(X)), xset)
                gen_examples = advAE.attacker_scaler.inverse_transform(gen_examples)
            else:
                gen_examples = advAE.conceal_fixed(constraints, pd.DataFrame(columns=xset, data=advAE.attacker_scaler.transform(gen_examples)),
                                                pd.DataFrame(columns=xset, data=advAE.attacker_scaler.transform(X)))
                gen_examples = advAE.attacker_scaler.inverse_transform(gen_examples)

            if fixed_constraints:
                if not os.path.exists('results/'+dataset+'/AE_max_concealable_var_'+str(n)+'/'):
                    os.makedirs(
                         'results/'+dataset+'/AE_max_concealable_var_'+str(n)+'/')
                pd.DataFrame(columns=xset, data=gen_examples).to_csv(
                    'results/'+dataset+'/AE_max_concealable_var_'+str(n)+'/new_advAE_attack_'+str(i)+'_from_test_dataset_max'+str(n)+'.csv')
            if conceal_up_to_n and not(fixed_constraints):
                lst = binary_dataframe.sum().sort_values(
                    ascending=False).head(n).keys().tolist()
                variables[n] = lst
                print('Extracted Constraint for attack '+str(att_number)+' n = '+ str(n) + ':' + str(lst))
        if conceal_up_to_n and not(fixed_constraints):
            print('Saving constraints for attack ' + str(att_number)+'\n')
            with open("./constraints/"+dataset+"/constraint_variables_attack_"+str(att_number)+"_AE.txt", 'w') as f:
                f.write(str(variables))
