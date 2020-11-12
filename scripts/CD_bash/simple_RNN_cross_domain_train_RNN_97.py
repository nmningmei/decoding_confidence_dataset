#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:35:17 2019

@author: nmei

This script is to train multiple RNN models on the subset of the source data

To estimate the model perform will be in another script

"""
import re
import os
import gc
gc.collect() # clean garbage memory
from glob import glob

from shutil import copyfile
copyfile('../utils.py','utils.py')



from shutil import copyfile
copyfile('../utils.py','utils.py')

import tensorflow as tf
from tensorflow.keras       import layers, Model, optimizers, losses
from tensorflow.keras.utils import to_categorical

import numpy  as np
import pandas as pd

from utils import make_CallBackList

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils           import shuffle as util_shuffle
#from sklearn.metrics         import roc_auc_score

experiment          = 'cross_domain'
data_dir            = '../../data/'
model_dir           = '../../models/RNN_CD'
source_dir          = '../../data/4-point'
target_dir          = '../../data/targets/*/'
result_dir          = '../../results/RNN_CD'
source_data         = glob(os.path.join(source_dir, "*.csv"))
target_data         = glob(os.path.join(target_dir, "*.csv"))
source_df_name      = os.path.join(data_dir,f'{experiment}','source.csv')
target_df_name      = os.path.join(data_dir,f'{experiment}','target.csv')
batch_size          = 32
time_steps          = 7
confidence_range    = 4
n_splits            = 100
n_jobs              = -1
split               = False # split the data into high and low dprime-metadrpime
feature_properties  = 'hidden states' # or hidden states or feature importance

for d in [model_dir,result_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

df_source           = pd.read_csv(source_df_name)
df_target           = pd.read_csv(target_df_name)

df_target['domain'] = df_target['filename'].apply(lambda x:x.split('/')[3].split('-')[0])


features            = df_source[[f"feature{ii + 1}" for ii in range(time_steps)]].values
targets             = df_source["targets"].values.astype(int)
groups              = df_source["sub"].values
np.random.seed(12345)
features,targets,groups = util_shuffle(features,targets,groups)
cv                      = GroupShuffleSplit(n_splits        = n_splits,
                                            test_size       = 0.2,
                                            random_state    = 12345)

for fold,(train,valid) in enumerate(cv.split(features,targets,groups = groups)):
    X_train,y_train = features[train],targets[train]
    X_valid,y_valid = features[valid],targets[valid]
    if fold >= 96: # batch_change
        break
    
# reset the GPU memory
tf.keras.backend.clear_session()
try:
    tf.random.set_random_seed(12345) # tf 1.0
except:
    tf.random.set_seed(12345) # tf 2.0

# build a 3-layer RNN model
inputs                  = layers.Input(shape     = (time_steps,4),# time steps by features 
                                       name      = 'inputs')
# the recurrent layer
lstm,state_h,state_c    = layers.LSTM(units             = 1,
                                      return_sequences  = True,
                                      return_state      = True,
                                      name              = "lstm")(inputs)
# from the LSTM layer, we will have an output with time steps by features, but 
dimension_squeeze       = layers.Lambda(lambda x:tf.keras.backend.squeeze(x,2))(lstm)
outputs                 = layers.Dense(4,
                                       name             = "output",
                                       activation       = "softmax")(dimension_squeeze)
model                   = Model(inputs,
                                outputs)

X_train = to_categorical(X_train - 1, num_classes = confidence_range)
X_valid = to_categorical(X_valid - 1, num_classes = confidence_range)

y_train = to_categorical(y_train - 1, num_classes = confidence_range)
y_valid = to_categorical(y_valid - 1, num_classes = confidence_range)

model.compile(optimizer     = optimizers.SGD(lr = 1e-3),
              loss          = losses.binary_crossentropy,
              metrics       = ['mse'])
# early stopping
model_name     = os.path.join(model_dir,f'{experiment}_{fold + 1}.h5')
print(model_name)
# early stopping
callbacks = make_CallBackList(model_name    = model_name,
                              monitor       = 'val_loss',
                              mode          = 'min',
                              verbose       = 0,
                              min_delta     = 1e-4,
                              patience      = 5,
                              frequency     = 1,)

print('fitting')
model.fit(X_train,
          y_train,
          batch_size        = batch_size,
          epochs            = 1000,
          validation_data   = (X_valid,y_valid),
          shuffle           = True,
          callbacks         = callbacks,
          verbose           = 1,
          )














































