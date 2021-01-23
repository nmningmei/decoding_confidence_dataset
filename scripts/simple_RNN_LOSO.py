#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:35:17 2019

@author: nmei
"""
import os
import gc
gc.collect() # clean garbage memory
from glob import glob

import tensorflow as tf
from tensorflow.keras       import layers, Model, optimizers, losses
from tensorflow.keras.utils import to_categorical

import numpy  as np
import pandas as pd

from utils import make_CallBackList,check_column_type,scoring_func

from sklearn.model_selection import LeaveOneGroupOut,StratifiedShuffleSplit
from sklearn.utils           import shuffle as util_shuffle

experiment          = ['confidence','LOO','RNN']
data_dir            = '../data'
model_dir           = f'../models/{experiment[0]}/{experiment[1]}_{experiment[2]}'
working_dir         = '../data/4-point'
working_data        = glob(os.path.join(working_dir, "*.csv"))
working_df_name     = os.path.join(data_dir,experiment[0],experiment[1],'all_data.csv')
saving_dir          = f'../results/{experiment[0]}/{experiment[1]}'
batch_size          = 32
time_steps          = 7
confidence_range    = 4
n_jobs              = -1
verbose             = 1
debug               = True

df_def          = pd.read_csv(working_df_name,)


# pick one of the csv files
filename = '../data/4-point/data_Bang_2019_Exp1.csv'
df_sub = df_def[df_def['filename'] == filename]
df_sub = check_column_type(df_sub)

#for (filename),df_sub in df_def.groupby(["filename"]):
features    = df_sub[[f"feature{ii + 1}" for ii in range(time_steps)]].values
targets     = df_sub["targets"].values.astype(int)
groups      = df_sub["sub"].values
accuraies   = df_sub['accuracy'].values
kk          = filename.split('/')[-1].split(".csv")[0]
cv          = LeaveOneGroupOut()
csv_name    = os.path.join(saving_dir,f'{experiment[2]} cross validation results LOO ({kk}).csv')
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
print(csv_name)

if not os.path.exists(csv_name) or debug:
    results             = dict(
                               fold         = [],
                               score        = [],
                               n_sample     = [],
                               source       = [],
                               sub_name     = [],
                               accuracy     = [],
                               )
    for ii in range(confidence_range):
        results[f'score{ii + 1}'] = []
    for ii in range(time_steps):
        results[f'hidden state T-{time_steps - ii}'] = []
else:
    results = pd.read_csv(csv_name)
    results = {col_name:list(results[col_name].values) for col_name in results.columns}

for fold,(train_,test) in enumerate(cv.split(features,targets,groups=groups)):
    model_name  = os.path.join(model_dir,f'LOO_{kk}_fold{fold + 1}.h5')
    print(model_name)
    
    # leave out test data
    X_,y_           = features[train_],targets[train_]
    X_test, y_test  = features[test]  ,targets[test]
    acc_test        = accuraies[test]
    acc_train_      = accuraies[train_]
    
    # split into train and validation data
    np.random.seed(12345)
    X_,y_ = util_shuffle(X_,y_)
    
    # the for-loop does not mean any thing, we only take the last step/output of the for-loop
    for train,valid in StratifiedShuffleSplit(test_size = 0.2,
                                              random_state = 12345).split(features[train_],targets[train_],groups=groups[train_]):
        X_train,y_train = X_[train],y_[train]
        X_valid,y_valid = X_[valid],y_[valid]
        acc_valid       = acc_train_[valid]
    
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
    dimension_squeeze       = layers.Lambda(lambda x:tf.keras.backend.squeeze(x,2),
                                            name            = 'squeeze')(lstm)
    outputs                 = layers.Dense(4,
                                           name             = "output",
                                           activation       = "softmax")(dimension_squeeze)
    model                   = Model(inputs,
                                    outputs)
    
    X_train = to_categorical(X_train - 1, num_classes = confidence_range)
    X_valid = to_categorical(X_valid - 1, num_classes = confidence_range)
    X_test  = to_categorical(X_test  - 1, num_classes = confidence_range)
    
    y_train = to_categorical(y_train - 1, num_classes = confidence_range)
    y_valid = to_categorical(y_valid - 1, num_classes = confidence_range)
    y_test  = to_categorical(y_test  - 1, num_classes = confidence_range)
    
    model.compile(optimizer     = optimizers.SGD(lr = 1e-2),
                  loss          = losses.binary_crossentropy,
                  metrics       = ['mse'])
    # early stopping
    callbacks = make_CallBackList(model_name    = model_name,
                                  monitor       = 'val_loss',
                                  mode          = 'min',
                                  verbose       = 0,
                                  min_delta     = 1e-4,
                                  patience      = 5,
                                  frequency     = 1,)
    
    if not os.path.exists(os.path.join(*model_name.split('/')[:-1])):
        os.makedirs(os.path.join(*model_name.split('/')[:-1]))
    if not os.path.exists(model_name) or debug:
        print('trained model not found, start training ...')
        model.fit(X_train,
                  y_train,
                  batch_size        = batch_size,
                  epochs            = 1000,
                  validation_data   = (X_valid,y_valid),
                  shuffle           = True,
                  callbacks         = callbacks,
                  verbose           = verbose,
                  )
    
    del model
    model = tf.keras.models.load_model(model_name)
    # freeze the model
    for layer in model.layers:
        layers.trainable = False
    print('build hidden state model')
    hidden_model = Model(model.input,model.layers[1].output)
    
    preds_valid = model.predict(X_valid.astype('float32'),batch_size=batch_size)
    preds_test  = model.predict(X_test.astype('float32'), batch_size=batch_size)
    print('get hidden states')
    hidden_state_valid,h_state_valid,c_state_valid = hidden_model.predict(X_valid,
                                                                  batch_size = batch_size,
                                                                  verbose = 1)
    
    print('on train')
    for acc_ in [0,1]:
        _idx, = np.where(acc_valid == acc_)
        if len(_idx) > 1:
            score_valid = scoring_func(y_valid[_idx],preds_valid[_idx],confidence_range = confidence_range)
            results['fold'].append(fold)
            results['score'].append(np.mean(score_valid))
            [results[f'score{ii + 1}'].append(score_valid[ii]) for ii in range(confidence_range)]
            results['n_sample'].append(X_valid[_idx].shape[0])
            results['source'].append('train')
            results['sub_name'].append('train')
            [results[f'hidden state T-{time_steps - ii}'].append(hidden_state_valid.mean(0)[ii,0]) for ii in range(time_steps)]
            results['accuracy'].append(acc_)
    
    print('on test')
    for acc_ in [0,1]:
        _idx, = np.where(acc_test == acc_)
        if len(_idx) > 1:
            score_test = scoring_func(y_test[_idx],preds_test[_idx],confidence_range = confidence_range)
            hidden_state_test,h_state_test,c_state_test = hidden_model.predict(X_test[_idx],
                                                                               batch_size = batch_size,
                                                                               verbose = 1)
            print('{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_'.format(*list(hidden_state_test.mean(0).reshape(7,))))
            
            results['fold'].append(fold)
            results['score'].append(np.mean(score_test))
            [results[f'score{ii + 1}'].append(score_test[ii]) for ii in range(confidence_range)]
            results['n_sample'].append(X_test[_idx].shape[0])
            results['source'].append('same')
            results['sub_name'].append(np.unique(groups[test])[0])
            [results[f'hidden state T-{time_steps - ii}'].append(hidden_state_test.mean(0)[ii,0]) for ii in range(time_steps)]
            
    gc.collect()
    
    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(csv_name,index = False)

