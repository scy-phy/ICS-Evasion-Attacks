import sys
sys.path.append('../../../')
import keras
from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

# import required to load the attacked model
from autoencoder_BATADAL import load_AEED
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def sort_temp_and_drop(row_index, temp):
    """
    sort descending and drop values below theta.

    given the current recontructions error we sort them descending and drop the values lower then theta treshold

    Parameters
    ----------
    row_index : int
    temp : Pandas DataFrame
        Autoencoder output vector

    Returns
    --------
    recontrution vector sorted descending and dropped
    """
    # print(temp)
    temp = temp.sort_values(by=row_index, axis=1, ascending=False)
    i = 0
    for col in temp.columns:
        if temp.loc[row_index, col] < theta:
            break
        i = i + 1
    temp = temp.drop(columns=temp.columns[i:43])
    return(temp.copy())


def scale_input_and_detect_single(index, X):
    """
    given a row of the dataset we transform it with the scaler and we see if it is detected by the model.

    Parameters
    ----------
    index : int
        row index
    X : pandas DataFrame
        Dataframe Containing one row of sensor readings
    Returns
    --------
    bool
        detection outcome
    float
        average recontrution error
    pandas dataframe
        reconstruction error vector for the considered sensor readings
    """
    X_transformed = pd.DataFrame(
        index=[index], columns=xset, data=scaler.transform(X))
    Yhat, error, temp, _ = autoencoder.detect(
        X_transformed, theta=theta, window=1, average=True)
    return Yhat, error, temp


def scale_input_and_detect(index, X):
    """
    given a row of the dataset we transform it with the scaler and we see if it is detected by the model.

    Parameters
    ----------
    index : int
        row index
    X : pandas DataFrame
        Dataframe Containing one row of sensor readings
    Returns
    --------
    bool
        detection outcome
    float
        average recontrution error
    pandas dataframe
        reconstruction error vector for the considered sensor readings
    """
    X_transformed = pd.DataFrame(
        columns=xset, data=scaler.transform(X), index=X.index)
    _, error, _, _ = autoencoder.detect(
        X_transformed, theta=theta, window=1, average=True)
    error_df = pd.DataFrame({'error': error})

    X = pd.concat([X, error_df], axis=1)
    X = X.iloc[X['error'].idxmin()]
    error = X['error']
    X = X.drop('error')
    X = pd.DataFrame([X])
    return X, error


def compute_mutation_factor(att_data, newBest):
    """
    compute how many columns values have been changed at the end of the transformation.

    Parameters
    ----------
    att_data : pandas DataFrame
        original sensor readings
    newBest : pandas DataFrame
        concealed sensor readings
    """
    X2 = pd.DataFrame(index=att_data.index,
                      columns=xset, data=att_data[xset])
    frames = [X2, newBest]
    merge = pd.concat(frames)
    merge.loc['Diff'] = merge.iloc[0] - merge.iloc[1]
    changed_columns[row_index] = merge.loc['Diff'].astype(bool).sum()
    print('changed tuples: ' + str(len(changed_columns)))


def choose_column(row_index, prev_col_name, changed_variables, max_concealable_variables): 
    """
    select the sensor value to be manipulated depending on the constrints

    Parameters
    ----------
    row_index : int

    prev_col_name : string

    changed_variables : list
        variables that can be manipulated

    max_concealable_variables : int
        number of variables that can manipulated

    Returns
    ----------
    string
        the column that will be manipulated in the current iteration
    """
    if(prev_col_name == None):
            return changed_variables[row_index][0]
    return changed_variables[row_index][(changed_variables[row_index].index(prev_col_name)+1) % max_concealable_variables]

def change_vector_label(row_index, att_data, solutions_found, changed_variables, variables):
    """
    this is the main algorithm, it actually transforms the input row trying to change its predicted label.
    updates after 5 changes on the same variable the ranking and optimizes the new ranked 1 variable

    Parameters
    ----------
    row_index : int

    att_data : pandas DataFrame
        original data to be concealed

    solutions_found : int
        counter of found solution
    Returns
    --------
    pandas dataframe
        solution found
    int
        updated counter of solutions
    """

    original_vector = att_data.copy()
    changes = 0
    found_solution = 0
    _, error, temp = scale_input_and_detect_single(row_index, att_data)
    previous_best_error = error[row_index]
    temp = sort_temp_and_drop(row_index, temp)
    prev_col_name = None
    num_changes_without_optimizations = 0
    last_optimization = 0
    newBest = att_data.copy()
    optimized = False
    changed_variables[row_index] = variables[max_concealable_variables]
    while changes < budget and (changes - last_optimization) < patience and not(found_solution):
        col_name = choose_column(row_index, prev_col_name, changed_variables, max_concealable_variables)
        prev_col_name = col_name
        if debug:
            print('______________________________')
            print(col_name)
            print('______________________________')

        values = np.arange(
            normal_op_ranges[col_name]['min'], normal_op_ranges[col_name]['max']+0.1, normal_op_ranges[col_name]['step'])
        # print(values)
        att_data = att_data.append(
            [att_data] * (len(values)), ignore_index=True)
        att_data = att_data[:-1]  # delete eccessive lenght
        # substitute column values usign normal operations
        att_data[col_name] = values
        att_data, error = scale_input_and_detect(row_index, att_data)
        if error < previous_best_error:
            if debug:
                print(error, previous_best_error)
            previous_best_error = error
            newBest = att_data.copy()
            last_optimization = changes
            num_changes_without_optimizations = 0
            optimized = True
            try:
                if not(col_name) in changed_variables[row_index]:
                    changed_variables[row_index].append(col_name)
            except:
                changed_variables[row_index] = [col_name]
        else:
            optimized = False

        if error < theta:
            solutions_found = solutions_found + 1
            found_solution = 1
            print('Found solution number: ' + str(solutions_found))

        if optimized == False:
            num_changes_without_optimizations = num_changes_without_optimizations + 1

        att_data = newBest.copy()
        _, error, temp = scale_input_and_detect_single(
            row_index, att_data)
        temp = sort_temp_and_drop(row_index, temp)
        changes = changes + 1
        if debug:
            print(temp)
            print('--__--__--')
            print(changes)
            print('--__--__--')
    compute_mutation_factor(original_vector, att_data.copy())

    return newBest.copy(), solutions_found

"""
Select wich dataset are you considering
(we are not allowed to publish WADI data, please request them itrust Singapore website)
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', type=str, default=['BATADAL'])
args = parser.parse_args()
print(args.data)

dataset = args.data[0]
data_folder = '../../Data/'+dataset

if dataset == 'BATADAL':
    attack_ids = range(1,15)
    att_data = pd.read_csv(data_folder+'/attack_1_from_test_dataset.csv')
    xset = [col for col in att_data.columns if col not in [
        'Unnamed: 0', 'DATETIME', 'ATT_FLAG']]
    budget = 200
    patience = 15
    constrained_variables = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
    
if dataset == 'WADI':
    attack_ids = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    att_data = pd.read_csv(data_folder+'/attack_1_from_test_dataset.csv')
    xset = [col for col in att_data.columns if col not in [
        'Row', 'DATETIME','ATT_FLAG', '2_MV_001_STATUS', '2_LT_001_PV', '2_MV_002_STATUS']]
    budget = 300
    patience = 40
    constrained_variables = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80]

yset = ['ATT_FLAG']
autoencoder = load_AEED("../../Attacked_Model/"+dataset+"/autoencoder.json", "../../Attacked_Model/"+dataset+"/autoencoder.h5")
scaler = pickle.load(open("../../Attacked_Model/"+dataset+"/scaler.p", "rb"))
with open("../../Attacked_Model/"+dataset+"/theta") as f:
        theta = float(f.read())
normal_op_ranges = pickle.load(open('dict_'+dataset+'.p', 'rb'))

for att_number in attack_ids:
    variables = {}
    f = open("./constraints/"+dataset+"/constraint_variables_attack_"+str(att_number)+".txt", 'r').read()
    variables = eval(f)
    for max_concealable_variables in constrained_variables:
        debug = False
        changed_columns = {}
        changed_variables = {}

        print('ATT NUMBER: '+str(att_number))
        att_data = pd.read_csv(
            data_folder+'/attack_'+str(att_number)+'_from_test_dataset.csv')
        
        y_att = att_data[yset]

        X = pd.DataFrame(index=att_data.index,
                         columns=xset, data=att_data[xset])
        new_tuples = pd.DataFrame(columns=xset)

        # main loop that iterates over every row in the dataset
        changed_rows = 0
        solutions_found = 0
        max_spent_time = 0
        sum_spent = 0
        times = []
        import time
        for row_index, row in X.iterrows():
            prov = pd.DataFrame(index=[row_index],
                                columns=xset, data=att_data[xset])
            Yhat, original_error, temp = scale_input_and_detect_single(
                row_index, prov)
            if Yhat[row_index]:
                start_time = time.time()
                modified_row, solutions_found = change_vector_label(
                    row_index, prov, solutions_found, changed_variables, variables)
                spent_time = time.time() - start_time
                print("--- %s seconds ---" % spent_time)
                sum_spent = sum_spent + spent_time
                if max_spent_time < spent_time:
                    max_spent_time = spent_time
                new_tuples = new_tuples.append(modified_row, ignore_index=True)
                changed_rows = changed_rows + 1
                times.append(spent_time)

            else:
                new_tuples = new_tuples.append(prov)

        new_tuples['DATETIME'] = att_data['DATETIME']
        new_tuples['ATT_FLAG'] = att_data['ATT_FLAG']
        new_tuples.to_csv('./results/'+dataset+'/max_constraints_fixed/whitebox_attack_' +
                          str(att_number)+'_from_test_dataset_max_'+str(max_concealable_variables)+'.csv')
        print('mean spent time: ' + str(sum_spent/changed_rows))
        
        with open('./results/'+dataset+'/max_constraints_fixed/time_spent_new_sequential_v2.0_fixed_budget.txt', 'a') as f:
            f.write('______attack: '+str(att_number)+'max: '+str(max_concealable_variables)+'______\n')
            f.write('Mean: ' + str(np.mean(times)))
            f.write('\n')
            f.write('STD: ' + str(np.std(times)))
            f.write('\n')