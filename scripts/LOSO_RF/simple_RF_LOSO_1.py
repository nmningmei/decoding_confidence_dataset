#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:35:17 2019

@author: nmei
"""
import os
import re
import gc
gc.collect() # clean garbage memory
from glob import glob

from shutil import copyfile
copyfile('../utils.py','utils.py')


from tensorflow.keras.utils  import to_categorical

import numpy  as np
import pandas as pd

from utils import preprocess,build_RF,get_RF_feature_importance

import sklearn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils           import shuffle as util_shuffle
from sklearn.metrics         import roc_auc_score

experiment          = 'LOO_confidence'
data_dir            = '../../data'
model_dir           = f'../../models/{experiment}_RF'
working_dir         = '../../data/4-point'
working_data        = glob(os.path.join(working_dir, "*.csv"))
working_df_name     = os.path.join(data_dir,f'{experiment}','all_data.csv')
saving_dir          = f'../../results/{experiment}'
batch_size          = 32
time_steps          = 7
confidence_range    = 4
n_jobs              = -1
verbose             = 0

df_def          = pd.read_csv(working_df_name,)


# pick one of the csv files
filename = '../data/4-point/data_Bang_2019_Exp1.csv'
df_sub = df_def[df_def['filename'] == filename]
for name in df_sub.columns:
    if name == 'filename':
        pass
    else:
        try:
            df_sub[name] = df_sub[name].apply(lambda x:int(re.findall('\d+',x)[0]))
        except:
            print(f'column {name} contains strings')

#for (filename),df_sub in df_def.groupby(["filename"]):
features    = df_sub[[f"feature{ii + 1}" for ii in range(7)]].values
targets     = df_sub["targets"].values.astype(int)
groups      = df_sub["sub"].values
kk          = filename.split('/')[-1].split(".csv")[0]
cv          = LeaveOneGroupOut()
csv_name    = os.path.join(saving_dir,f'RF cross validation results LOO ({kk}).csv')
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
print(csv_name)

if not os.path.exists(csv_name):
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
        results[f'feature importance T-{time_steps - ii}'] = []
else:
    results = pd.read_csv(csv_name)
    results = {col_name:list(results[col_name].values) for col_name in results.columns}

for fold,(train_,test) in enumerate(cv.split(features,targets,groups=groups)):
    model_name  = os.path.join(model_dir,f'RF_{kk}_fold{fold + 1}.h5')
    print(model_name)
    if not os.path.exists(model_name):
        # leave out test data
        X_,y_           = features[train_],targets[train_]
        X_test, y_test  = features[test]  ,targets[test]
        
        # split into train and validation data
        np.random.seed(12345)
        X_,y_ = util_shuffle(X_,y_)
        
        # the for-loop does not mean any thing, we only take the last step/output of the for-loop
        for train,valid in cv.split(features[train_],targets[train_],groups=groups[train_]):
            X_train,y_train = X_[train],y_[train]
            X_valid,y_valid = X_[valid],y_[valid]
        
        
        
        randomforestclassifier = build_RF(n_jobs = n_jobs,
                                          n_estimators = 100,
                                          )
        
        print('fitting...')
        randomforestclassifier.fit(X_train,y_train)
        preds_valid = randomforestclassifier.predict_proba(X_valid)
        print('done fitting')
        y_valid = to_categorical(y_valid - 1, num_classes = confidence_range)
        
        score_train = []
        for ii in range(4):
            try:
                score_train.append(roc_auc_score(y_valid[:,ii],preds_valid[:,ii]))
            except:
                score_train.append(roc_auc_score(np.concatenate([y_valid[:,ii],[0,1]]),
                                                 np.concatenate([preds_valid[:,ii],[0.5,0.5]])
                                                 ))
        print('getting validation feature importance')
        feature_importance,results,c = get_RF_feature_importance(randomforestclassifier,
                                                       features,
                                                       targets,
                                                       valid,
                                                       results,)
        
        results['fold'].append(fold)
        results['score'].append(np.mean(score_train))
        [results[f'score{ii + 1}'].append(score_train[ii]) for ii in range(confidence_range)]
        results['n_sample'].append(X_valid.shape[0])
        results['source'].append('train')
        results['sub_name'].append('train')
        
        y_test = to_categorical(y_test - 1, num_classes = confidence_range)
        preds_test = randomforestclassifier.predict_proba(X_test)
        score_test = []
        for ii in range(4):
            try:
                score_test.append(roc_auc_score(y_test[:,ii],preds_test[:,ii]))
            except:
                score_test.append(roc_auc_score(np.concatenate([y_test[:,ii],[0,1]]),
                                                np.concatenate([preds_test[:,ii],[0.5,0.5]])
                                                ))
        feature_importance,results,c = get_RF_feature_importance(randomforestclassifier,
                                                       features,
                                                       targets,
                                                       test,
                                                       results,)
        
        print('{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_'.format(*list(c.reshape(7,))))
        
        results['fold'].append(fold)
        results['score'].append(np.mean(score_test))
        [results[f'score{ii + 1}'].append(score_test[ii]) for ii in range(confidence_range)]
        results['n_sample'].append(X_test.shape[0])
        results['source'].append('same')
        results['sub_name'].append(np.unique(groups[test])[0])
        
        gc.collect()
        
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(csv_name,index = False)











































