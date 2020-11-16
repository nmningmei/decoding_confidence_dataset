#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:10:54 2019

@author: nmei
"""

import os
import gc
import multiprocessing
gc.collect() # clean garbage memory
print(f'{multiprocessing.cpu_count()} cpus available')

import numpy  as np
import pandas as pd

from shutil import copyfile
copyfile('../utils.py','utils.py')

from utils import build_RF,scoring_func,get_RF_feature_importance

from tensorflow.keras.utils  import to_categorical

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils           import shuffle as util_shuffle

experiment          = 'cross_domain_confidence'
experiment_folder   = 'confidence'
data_dir            = '../../data/'
model_dir           = f'../../models/{experiment_folder}/RNN_CD'
source_dir          = '../../data/4-point'
target_dir          = '../../data/targets/*/'
result_dir          = f'../../results/{experiment_folder}/RF_CD'
hidden_dir          = f'../../results/{experiment_folder}/RF_CD_FI'
source_df_name      = os.path.join(data_dir,f'{experiment}','source.csv')
target_df_name      = os.path.join(data_dir,f'{experiment}','target.csv')
batch_size          = 32
time_steps          = 7
confidence_range    = 4
n_splits            = 100
n_jobs              = -1
split               = False # split the data into high and low dprime-metadrpime
feature_properties  = 'feature importance' # or hidden states or feature importance

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
    if fold >= 66: # batch_change
        break

csv_saving_name     = f'RF cross validation results (fold {fold + 1}).csv'

randomforestclassifier = build_RF(n_jobs = n_jobs,
                                  n_estimators = 500,
                                  sklearnlib = False,
                                  )
print('fitting...')
randomforestclassifier.fit(X_train,y_train)
preds_valid = randomforestclassifier.predict_proba(X_valid)
print('done fitting')
y_valid = to_categorical(y_valid - 1, num_classes = confidence_range)
score_train = scoring_func(y_valid,preds_valid,confidence_range = confidence_range)

feature_importance,results,_ = get_RF_feature_importance(randomforestclassifier,
                                                         features,
                                                         targets,
                                                         valid,
                                                         results,
                                                         feature_properties,
                                                         time_steps,)

results['fold'].append(fold)
results['score'].append(np.mean(score_train))
[results[f'score{ii + 1}'].append(score_train[ii]) for ii in range(confidence_range)]
results['n_sample'].append(X_valid.shape[0])
results['source'].append('train')
results['sub_name'].append('train')


# cross domain testing
for (sub_name,target_domain),df_sub in df_target.groupby(['sub','domain']):
    features_       = df_sub[[f"feature{ii + 1}" for ii in range(7)]].values
    targets_    = df_sub["targets"].values.astype(int)
    groups_         = df_sub["sub"].values
    X_test,y_test   = features_.copy(),targets_.copy()
    y_test          = to_categorical(y_test - 1, num_classes = confidence_range,)
    
    preds_test  = randomforestclassifier.predict_proba(X_test)
    score_test  = scoring_func(y_test,preds_test,confidence_range = confidence_range)
    
    print(f'training score = {np.mean(score_train):.4f} with {len(train)} instances, testing score = {np.mean(score_test):.4f} with {len(y_test)} instances')
    print('get feature importance')
    feature_importance,results,c = get_RF_feature_importance(randomforestclassifier,
                                                             features_,
                                                             targets_,
                                                             np.arange(features_.shape[0]),
                                                             results,
                                                             feature_properties,
                                                             time_steps,)
    print('{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_'.format(*c))
    results['fold'].append(fold)
    results['score'].append(np.mean(score_test))
    [results[f'score{ii + 1}'].append(score_test[ii]) for ii in range(confidence_range)]
    results['n_sample'].append(X_test.shape[0])
    results['source'].append(target_domain)
    results['sub_name'].append(sub_name)
    gc.collect()
    
    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(os.path.join(result_dir,f'{csv_saving_name}'),
                           index = False)































