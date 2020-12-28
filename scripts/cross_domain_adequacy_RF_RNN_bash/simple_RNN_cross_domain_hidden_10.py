#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:17:58 2019

@author: nmei
"""

import os
import gc
gc.collect() # clean garbage memory


import tensorflow as tf
from tensorflow.keras       import layers, Model
from tensorflow.keras.utils import to_categorical

import numpy   as np
import pandas  as pd
import seaborn as sns
from shutil import copyfile
copyfile('../utils.py','utils.py')

from utils import scoring_func

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils           import shuffle as util_shuffle

sns.set_style('white')
sns.set_context('talk')

experiment          = ['cross_domain','adequacy','RNN']
feature_properties  = 'hidden states' # or hidden states or feature importance
data_dir            = '../../data/'
model_dir           = f'../../models/{experiment[1]}/{experiment[2]}_CD'
source_dir          = '../../data/4-point'
target_dir          = '../../data/targets/*/'
result_dir          = f'../../results/{experiment[1]}/{experiment[2]}_CD'
hidden_dir          = f'../../results/{experiment[1]}/{experiment[2]}_CD_{"".join(feature_properties.split(" "))}'
source_df_name      = os.path.join(data_dir,experiment[1],experiment[0],'source.csv')
target_df_name      = os.path.join(data_dir,experiment[1],experiment[0],'target.csv')
batch_size          = 32
time_steps          = 7
confidence_range    = 4
n_splits            = 100
n_jobs              = -1
split               = False # split the data into high and low dprime-metadrpime

for d in [model_dir,result_dir,hidden_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

df_source           = pd.read_csv(source_df_name)
df_target           = pd.read_csv(target_df_name)


df_target['domain'] = df_target['filename'].apply(lambda x:x.split('/')[3].split('-')[0])


results             = dict(
                           fold         = [],
                           score        = [],
                           n_sample     = [],
                           source       = [],
                           sub_name     = [],
                           )
for ii in range(confidence_range):
    results[f'score{ii + 1}'] = []
for ii in range(time_steps):
    results[f'{feature_properties} T-{time_steps - ii}'] = []

features    = df_source[[f"feature{ii + 1}" for ii in range(time_steps)]].values
targets     = df_source["targets"].values.astype(int)
groups      = df_source["sub"].values
np.random.seed(12345)
features,targets,groups = util_shuffle(features,targets,groups)
cv                      = GroupShuffleSplit(n_splits        = n_splits,
                                            test_size       = 0.2,
                                            random_state    = 12345)

for fold,(train,valid) in enumerate(cv.split(features,targets,groups = groups)):
    X_train,y_train = features[train],targets[train]
    X_valid,y_valid = features[valid],targets[valid]
    if fold >= 9: # batch_change
        break



csv_saving_name     = f'RNN cross validation results (fold {fold + 1}).csv'
model_name     = os.path.join(model_dir,f'{"_".join(experiment)}_{fold + 1}.h5')
csv_saving_name = os.path.join(result_dir,csv_saving_name)
print(model_name)
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
try:
    model.load_weights(model_name)
except:
    del model
    model = tf.keras.models.load_model(model_name)

hidden_model = Model(model.input,model.layers[1].output)

X_train = to_categorical(X_train - 1, num_classes = confidence_range)
X_valid = to_categorical(X_valid - 1, num_classes = confidence_range)

y_train = to_categorical(y_train - 1, num_classes = confidence_range)
y_valid = to_categorical(y_valid - 1, num_classes = confidence_range)

preds_valid = model.predict(X_valid,batch_size = batch_size,verbose = 1)
score_train = scoring_func(y_valid,preds_valid,confidence_range = confidence_range)
hidden_state_valid,h_state_valid,c_state_valid = hidden_model.predict(X_valid,
                                                                      batch_size = batch_size,
                                                                      verbose = 1)
df_valid = pd.DataFrame(hidden_state_valid[:,:,0],columns = [f'T{ii - time_steps}' for ii in range(time_steps)])
df_valid['sub'] = groups[valid]
df_valid_ave = df_valid.groupby(['sub']).mean().reset_index()
df_valid_ave['data'] = 'Train'
df_valid_ave['score'] = np.mean(score_train)

results['fold'].append(fold)
results['score'].append(np.mean(score_train))
[results[f'score{ii + 1}'].append(score_train[ii]) for ii in range(confidence_range)]
results['n_sample'].append(X_valid.shape[0])
results['source'].append('train')
results['sub_name'].append('train')
[results[f'{feature_properties} T-{time_steps - ii}'].append(hidden_state_valid.mean(0)[ii,0]) for ii in range(time_steps)]

# 
for (sub_name,target_domain),df_sub in df_target.groupby(['sub','domain']):
    df_sub
    features_        = df_sub[[f"feature{ii + 1}" for ii in range(7)]].values
    targets_         = df_sub["targets"].values.astype(int)
    groups_          = df_sub["sub"].values
    X_test,y_test    = features_.copy(),targets_.copy()
    X_test           = to_categorical(X_test - 1, num_classes = confidence_range)
    y_test           = to_categorical(y_test - 1, num_classes = confidence_range)
    
    
    preds_test  = model.predict(X_test,batch_size=batch_size,verbose = 1)
    score_test  = scoring_func(y_test,preds_test,confidence_range = confidence_range)
    
    print(f'training score = {np.mean(score_train):.4f} with {len(train)} instances, testing score = {np.mean(score_test):.4f} with {len(y_test)} instances')
    
    print('get hidden states')
    hidden_state_test,h_state_test,c_state_test = hidden_model.predict(X_test,
                                                                       batch_size = batch_size,
                                                                       verbose = 1)
    print('{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}'.format(*list(hidden_state_test.mean(0).reshape(7,))))
    
    results['fold'].append(fold)
    results['score'].append(np.mean(score_test))
    [results[f'score{ii + 1}'].append(score_test[ii]) for ii in range(confidence_range)]
    results['n_sample'].append(X_test.shape[0])
    results['source'].append(target_domain)
    results['sub_name'].append(sub_name)
    [results[f'{feature_properties} T-{time_steps - ii}'].append(np.abs(hidden_state_test.mean(0)[ii,0])) for ii in range(time_steps)]
    
    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(csv_saving_name,index = False)











