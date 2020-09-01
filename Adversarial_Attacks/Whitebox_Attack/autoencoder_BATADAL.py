# numpy stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import *

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score
from sklearn.preprocessing import MinMaxScaler

# os and time utils
import os
import time
import glob

import json


# ### Autoencoder classes

# classes
class AutoEncoder(object):
    """ Keras-based AutoEncoder (AE) class used for event detection.

        Attributes:
        params: dictionary with parameters defining the AE structure,
    """
    def __init__(self, **kwargs):
        """ Class constructor, stores parameters and initialize AE Keras model. """

        # Default parameters values. If nI is not given, the code will crash later.
        params = {
            'nI': None,
            'nH': 3,
            'cf': 1,
            'activation' : 'tanh',
            'optimizer' : None,
            'verbose' : 0
            }

        for key,item in kwargs.items():
            params[key] = item

        self.params = params

    def create_model(self):
        """ Creates Keras AE model.

            The model has nI inputs, nH hidden layers in the encoder (and decoder)
            and cf compression factor. The compression factor is the ratio between
            the number of inputs and the innermost hidden layer which stands between
            the encoder and the decoder. The size of the hidden layers between the
            input (output) layer and the innermost layer decreases (increase) linearly
            according to the cg.
        """

        # retrieve params
        nI = self.params['nI'] # number of inputs
        nH = self.params['nH'] # number of hidden layers in encoder (decoder)
        cf = self.params['cf'] # compression factor
        activation = self.params['activation'] # autoencoder activation function
        optimizer = self.params['optimizer'] # Keras optimizer
        verbose = self.params['verbose'] # echo on screen

        # get number/size of hidden layers for encoder and decoder
        temp = np.linspace(nI,nI/cf,nH + 1).astype(int)
        nH_enc = temp[1:]
        nH_dec = temp[:-1][::-1]

        # input layer placeholder
        input_layer = Input(shape=(nI,))

        # build encoder
        for i, layer_size in enumerate(nH_enc):
            if i == 0:
                # first hidden layer
                encoder = Dense(layer_size, activation=activation)(input_layer)
            else:
                # other hidden layers
                encoder = Dense(layer_size, activation=activation)(encoder)

        # build decoder
        for i, layer_size in enumerate(nH_dec):
            if i == 0:
                # first hidden layer
                decoder = Dense(layer_size, activation=activation)(encoder)
            else:
                # other hidden layers
                decoder = Dense(layer_size, activation=activation)(decoder)

        # create autoencoder
        autoencoder = Model(input_layer, decoder)
        if optimizer == None:
            optimizer = optimizers.Adam(lr = 0.001)

        # print autoencoder specs
        if verbose > 0:
            print('Created autoencoder with structure:');
            print(', '.join('layer_{}: {}'.format(v, i) for v, i in enumerate(np.hstack([nI,nH_enc,nH_dec]))))

        # compile and return model
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
        return autoencoder

    def train(self, x, **train_params):
        """ Train autoencoder,

            x: inputs (inputs == targets, AE are self-supervised ANN).
        """
        if self.params['verbose']:
            if self.ann == None:
                print('Creating model.')
                self.create_model()
        self.ann.fit(x, x, **train_params)


    def predict(self, x, test_params={}):
        """ Yields reconstruction error for all inputs,

            x: inputs.
        """
        return self.ann.predict(x, **test_params)

class AEED(AutoEncoder):
    """ This class extends the AutoEncoder class to include event detection
        functionalities.
    """
    def initialize(self):
        """ Create the underlying Keras model. """
        self.ann = self.create_model()

    def predict(self, x, **keras_params):
        """ Predict with autoencoder. """
        preds = pd.DataFrame(index=x.index,columns=x.columns,
                                            data=super(AEED, self).predict(x.values,keras_params))
        errors = (x-preds)**2
        return preds, errors

    def detect(self, x, theta, window = 1, average=False, sys_theta = 0, **keras_params):
        """ Detection performed based on (smoothed) reconstruction errors.

            x = inputs,
            theta = threshold, attack flagged if reconstruction error > threshold,
            window = length of the smoothing window (default = 1 timestep, i.e. no smoothing),
            average = boolean (default = False), if True the detection is performed
                on the average reconstruction error across all outputs,
            keras_params = parameters for the Keras-based AE prediction.
        """
        #        preds = super(AEED, self).predict(x,keras_params)
        preds, temp = self.predict(x, **keras_params)
        temp = (x-preds)**2
        if average:
            errors = temp.mean(axis=1).rolling(window=window).mean()
            detection = errors > theta
        else:
            errors = temp.rolling(window=window).mean()
            detection = errors.apply(lambda x: x>np.max(theta, sys_theta))

        return detection, errors, temp, preds

    def save(self, filename):
        """ Save AEED modelself.

            AEED parameters saved in a .json, while Keras model is stored in .h5 .
        """
        # parameters
        with open(filename+'.json', 'w') as fp:
            json.dump(self.params, fp)
        # keras model
        self.ann.save(filename+'.h5')
        # echo
        print('Saved AEED parameters to {0}.\nKeras model saved to {1}'.format(filename+'.json', filename+'.h5'))


# functions
def load_AEED(params_filename, model_filename):
    """ Load stored AEED. """
    # load params and create AEED
    with open(params_filename) as fd:
        params = json.load(fd)
    aeed = AEED(**params)

    # load keras model
    aeed.ann = load_model(model_filename)
    return aeed

if __name__ == "__main__":
    # ### Load and preprocess training data

    # In[16]:


    # load training dataset
    data_path = "./data/"
    df_train_orig = pd.read_csv(data_path + "dataset03.csv", parse_dates = ['DATETIME'], dayfirst=True)


    # In[17]:


    # get dates and columns with sensor readings
    dates_train = df_train_orig['DATETIME']
    sensor_cols = [col for col in df_train_orig.columns if col not in ['DATETIME','ATT_FLAG']]

    # scale sensor data
    scaler = MinMaxScaler()
    X = pd.DataFrame(index = df_train_orig.index, columns = sensor_cols, data = scaler.fit_transform(df_train_orig[sensor_cols]))

    # split into training and validation
    X1, X2, _, _  = train_test_split(X, X, test_size=0.33, random_state=42)


    # ### Train autoencoder

    # In[18]:


    # define model parameters
    params = {
        'nI' : X.shape[1],
        'nH' : 6,
        'cf' : 2.5,
        'activation' : 'tanh',
        'verbose' : 1,
    }

    # create AutoEncoder for Event Detection (AEED)
    autoencoder = AEED(**params)
    autoencoder.initialize()


    # In[19]:


    # train models with early stopping and reduction of learning rate on plateau
    earlyStopping= EarlyStopping(monitor='val_loss', patience=3, verbose=0,  min_delta=1e-4, mode='auto')
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, epsilon=1e-4, mode='min')

    # initialize time
    start_time = time.time()

    # train autoencoder
    print(X1.values.shape)
    autoencoder.train(X1.values,
                epochs=500,
                batch_size=32,
                shuffle=False,
                callbacks = [earlyStopping, lr_reduced],
                verbose = 2,
                validation_data=(X2.values, X2.values))


    # ### Test autoencoder

    # In[20]:


    # assess detection
    def compute_scores(Y,Yhat):
        fpr, recall, _ = roc_curve(Y, Yhat)
        return [accuracy_score(Y,Yhat),f1_score(Y,Yhat),precision_score(Y,Yhat),recall[1], fpr[1]]


    # In[21]:


    # Load dataset with attacks
    df_test_01 = pd.read_csv(data_path + "dataset04.csv", parse_dates = ['DATETIME'], dayfirst=True)
    df_test_02 = pd.read_csv(data_path + "test_dataset.csv", parse_dates = ['DATETIME'], dayfirst=True)

    # scale datasets
    X3 = pd.DataFrame(index = df_test_01.index, columns = sensor_cols,
                      data = scaler.transform(df_test_01[sensor_cols]))
    X4 = pd.DataFrame(index = df_test_02.index, columns = sensor_cols,
                      data = scaler.transform(df_test_02[sensor_cols]))

    # get targets
    Y3 = df_test_01['ATT_FLAG']
    Y4 = df_test_02['ATT_FLAG']


    # In[22]:


    # perform detection

    # get validation reconstruction errors
    _, validation_errors = autoencoder.predict(X2)
    validation_errors.to_csv('validation_errors.csv')

    # plot distribution of average validation reconstruction errors
    f, ax = plt.subplots(1, figsize = (8,4))
    sns.boxplot(validation_errors.mean(axis=1), ax=ax)
    ax.set_xlim([0,0.005])
    ax.set_title('Boxplot of average validation reconstruction errors')

    # set treshold as quantile of average reconstruction error
    theta = validation_errors.mean(axis = 1).quantile(0.995)

    Yhat3, _ = autoencoder.detect(X3, theta = theta , window = 1, average=True)
    Yhat4, _ = autoencoder.detect(X4, theta = theta, window = 1, average=True)


    # In[23]:


    results = pd.DataFrame(index = ['test dataset 01','test dataset 02'],
                           columns = ['accuracy','f1_score','precision','recall','fpr'])
    results.loc['test dataset 01'] = compute_scores(Y3,Yhat3)
    results.loc['test dataset 02'] = compute_scores(Y4,Yhat4)

    print('Results:\n')
    print(results)


    # In[24]:


    # plot figure
    shade_of_gray = '0.75'
    f, axes = plt.subplots(2,figsize = (20,8))
    axes[0].plot(Yhat3, color = shade_of_gray, label = 'predicted state')
    axes[0].fill_between(Yhat3.index, Yhat3.values, where=Yhat3.values <=1, interpolate=True, color=shade_of_gray)
    axes[0].plot(Y3, color = 'r', alpha = 0.85, lw = 5, label = 'real state')
    axes[0].set_title('Detection trajectory on test dataset 01', fontsize = 14)
    axes[0].set_yticks([0,1])
    axes[0].set_yticklabels(['NO ATTACK','ATTACK'])
    axes[0].legend(fontsize = 12, loc = 2)

    axes[1].plot(Yhat4, color = shade_of_gray, label = 'predicted state')
    axes[1].fill_between(Yhat4.index, Yhat4.values, where=Yhat4.values <=1, interpolate=True, color=shade_of_gray)
    axes[1].plot(Y4, color = 'r', alpha = 0.75, lw = 5, label = 'real state')
    axes[1].set_title('Detection trajectory on test dataset 02', fontsize = 14)
    axes[1].set_yticks([0,1])
    axes[1].set_yticklabels(['NO ATTACK','ATTACK'])


    # In[25]:


    # save autoencoder
    autoencoder.save('autoencoder')
    del autoencoder
