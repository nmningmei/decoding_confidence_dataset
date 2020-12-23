#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:36:16 2019

@author: dsb
"""
import re
import gc

import numpy as np
import pandas as pd

try:
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
except:
    pass

from tqdm import tqdm

from joblib import Parallel,delayed

from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

gc.collect()

def preprocess(working_data,
               time_steps = 7,
               target_columns = ['Confidence'],
               n_jobs = 8,
               verbose = 1
               ):
    """
    Inputs
    -----------------------
    working_data: list of csv names
    time_steps: int, trials looking back
    target_columns: list of columns names that we want to parse to become the RNN features, i.e. Confidence, accuracy, RTs
    n_jobs: number of CPUs we want to parallize the for-loop job
    verbose: if > 0, we print out the parallized for-loop processes
    
    Outputs
    -----------------------
    df_def: concatenated pandas dataframe that contains features at each time point and targets
    """
    df_for_concat = []
    for f in tqdm(working_data,desc = 'concat'):
        df_temp = pd.read_csv(f,header = 0)
        df_temp['accuracy'] = np.array(df_temp['Stimulus'] == df_temp['Response']).astype(int)
        df_temp['filename'] = f
        df_temp = df_temp[np.concatenate([['Subj_idx','filename'],target_columns])]
        df_for_concat.append(df_temp)
    df_concat = pd.concat(df_for_concat)
    
    df = dict(sub = [],
              filename = [],
              targets = [])
    for ii in range(7):
        df[f'feature{ii + 1}'] = []
    
    for (sub,filename), df_sub in tqdm(df_concat.groupby(['Subj_idx','filename']),desc = 'feature generating'):
    #    print(sub,filename)
        values = df_sub[target_columns].values
        data_gen = TimeseriesGenerator(values,values,
                                       length = time_steps,
                                       sampling_rate = 1,
                                       batch_size = 1)
        
        for features_,targets_ in list(data_gen):
            df['sub'].append(sub)
            df['filename'].append(filename)
            [df[f"feature{ii + 1}"].append(f) for ii,f in enumerate(features_.flatten())]
            df["targets"].append(targets_.flatten()[0])
    df = pd.DataFrame(df)
    df = df[['filename', 'sub','feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6','feature7', 'targets']]
    """
    REMOVING FEATURES AND TARGETS DIFFERENT FROM 1-4
    """
    df_temp = df.dropna()
    ###################### parallelize the for-loop to multiple CPUs ############################
    def detect(row):
        values = np.array([item for item in row[2:]])
        return np.logical_and(values < 5, values > 0)
    
    idx_within_range = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(detect)(**{'row':row})for ii,row in df_temp.iterrows())
    #############################################################################################
    idx = np.sum(idx_within_range,axis = 1) == (time_steps + 1) # ALL df_pepe columns must be true(1) & sum 8
    df_def = df_temp.loc[idx,:]
    return df_def

def meta_adequacy(x):
    if x['accuracy'] == 1:
        return x['Confidence']
    else:
        return 5 - x['Confidence']

def check_column_type(df_sub):
    for name in df_sub.columns:
        if name == 'filename':
            pass
        else:
            df_sub[name] = df_sub[name].astype(int)
    return df_sub

def add_adequacy(working_data,
                 time_steps = 7,
                 n_jobs = 8,
                 verbose = 1
                 ):
    """
    Inputs
    -----------------------
    working_data: list of csv names
    time_steps: int, trials looking back
    n_jobs: number of CPUs we want to parallize the for-loop job
    verbose: if > 0, we print out the parallized for-loop processes
    
    Outputs
    -----------------------
    df_def: concatenated pandas dataframe that contains features at each time point and targets
    """
    df_for_concat = []
    for f in tqdm(working_data,desc = 'concat'):
        df_temp = pd.read_csv(f,header = 0)
        df_temp['accuracy'] = np.array(df_temp['Stimulus'] == df_temp['Response']).astype(int)
        df_temp['filename'] = f
        df_temp['metaAdequacy'] = df_temp.apply(meta_adequacy,axis = 1)
        df_temp = df_temp[np.concatenate([['Subj_idx','filename'],['metaAdequacy']])]
        df_for_concat.append(df_temp)
    df_concat = pd.concat(df_for_concat)
    
    df = dict(sub = [],
              filename = [],
              targets = [])
    for ii in range(7):
        df[f'feature{ii + 1}'] = []
    
    for (sub,filename), df_sub in tqdm(df_concat.groupby(['Subj_idx','filename']),desc = 'feature generating'):
    #    print(sub,filename)
        values = df_sub['metaAdequacy'].values
        data_gen = TimeseriesGenerator(values,values,
                                       length = time_steps,
                                       sampling_rate = 1,
                                       batch_size = 1)
        
        for features_,targets_ in list(data_gen):
            df['sub'].append(sub)
            df['filename'].append(filename)
            [df[f"feature{ii + 1}"].append(f) for ii,f in enumerate(features_.flatten())]
            df["targets"].append(targets_.flatten()[0])
    df = pd.DataFrame(df)
    df = df[['filename', 'sub','feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6','feature7', 'targets']]
    """
    REMOVING FEATURES AND TARGETS DIFFERENT FROM 1-4
    """
    df_temp = df.dropna()
    ###################### parallelize the for-loop to multiple CPUs ############################
    def detect(row):
        values = row[2:].values
        return np.logical_and(values < 5, values > 0)
    
    idx_within_range = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(detect)(**{'row':row})for ii,row in df_temp.iterrows())
    #############################################################################################
    idx = np.sum(idx_within_range,axis = 1) == (time_steps + 1) # ALL df_pepe columns must be true(1) & sum 8
    df_def = df_temp.loc[idx,:]
    return df_def

# the most important helper function: early stopping and model saving
def make_CallBackList(model_name,monitor='val_loss',mode='min',verbose=0,min_delta=1e-4,patience=50,frequency = 1):
    from tensorflow.keras.callbacks             import ModelCheckpoint,EarlyStopping
    """
    Make call back function lists for the keras models
    
    Inputs
    -------------------------
    model_name: directory of where we want to save the model and its name
    monitor:    the criterion we used for saving or stopping the model
    mode:       min --> lower the better, max --> higher the better
    verboser:   printout the monitoring messages
    min_delta:  minimum change for early stopping
    patience:   temporal windows of the minimum change monitoring
    frequency:  temporal window steps of the minimum change monitoring
    
    Return
    --------------------------
    CheckPoint:     saving the best model
    EarlyStopping:  early stoppi
    """
    checkPoint = ModelCheckpoint(model_name,# saving path
                                 monitor          = monitor,# saving criterion
                                 save_best_only   = True,# save only the best model
                                 mode             = mode,# saving criterion
#                                 save_freq        = 'epoch',# frequency of check the update 
                                 verbose          = verbose,# print out (>1) or not (0)
#                                 load_weights_on_restart = True,
                                 )
    earlyStop = EarlyStopping(   monitor          = monitor,
                                 min_delta        = min_delta,
                                 patience         = patience,
                                 verbose          = verbose, 
                                 mode             = mode,
#                                 restore_best_weights = True,
                                 )
    return [checkPoint,earlyStop]

def make_hidden_state_dataframe(states,sign = -1,time_steps = 7,):
    df = pd.DataFrame(sign * states[:,:,0],columns = [f'T{ii - time_steps}' for ii in range(time_steps)])
    df = pd.melt(df,value_vars = df.columns,var_name = ['Time'],value_name = 'Hidden Activation')
    return df

def convert_object_to_float(df):
    for name in df.columns:
        if name == 'filename':
            pass
        else:
            try:
                df[name] = df[name].apply(lambda x:int(re.findall('\d+',x)[0]))
            except:
                print(f'column {name} contains strings')
    return df

def build_RF(n_jobs             = 1,
             learning_rate      = 1e-1,
             max_depth          = 3,
             n_estimators       = 100,
             objective          = 'binary:logistic',
             subsample          = 0.9,
             colsample_bytree   = 0.9,
             reg_alpha          = 0,
             reg_lambda         = 1,
             importance_type    = 'gain',
             sklearnlib         = True,
             ):
    if sklearnlib:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators    = n_estimators,
                                    criterion       = 'entropy',
                                    n_jobs          = n_jobs,
                                    class_weight    = 'balanced',
                                    random_state    = 12345,
                                    )
        return rf
    else:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
                            learning_rate                           = learning_rate,
                            max_depth                               = max_depth,
                            n_estimators                            = n_estimators,
                            objective                               = objective,
                            booster                                 = 'gbtree', # default
                            subsample                               = subsample,
                            colsample_bytree                        = colsample_bytree,
                            reg_alpha                               = reg_alpha,
                            reg_lambda                              = reg_lambda,
                            random_state                            = 12345, # not default
                            importance_type                         = importance_type,
                            n_jobs                                  = n_jobs,# default to be 1
                                                  )
        return xgb

def get_RF_feature_importance(randomforestclassifier,
                              features,
                              targets,
                              idx,
                              results,
                              feature_properties = 'feature importance',
                              time_steps = 7,):
    
    print('permutation feature importance...')
    from sklearn.inspection import permutation_importance
    feature_importance = permutation_importance(randomforestclassifier,
                                                features[idx],
                                                targets[idx],
                                                n_repeats       = 10,
                                                n_jobs          = -1,
                                                random_state    = 12345,
                                                )
    c = feature_importance['importances_mean']
    
    [results[f'{feature_properties} T-{time_steps - ii}'].append(c[ii]) for ii in range(time_steps)]
    return feature_importance,results,c

def append_dprime_metadprime(df,df_metadprime):
    temp = []
    for (filename,sub_name),df_sub in tqdm(df.groupby(['filename','sub']),desc='dprime'):
        df_sub
        idx_ = np.logical_and(df_metadprime['sub_names' ] == sub_name,
                              df_metadprime['file'      ] == filename.split('/')[-1],)
        row = df_metadprime[idx_]
        if len(row) > 0:
            df_sub['metadprime'] = row['metadprime'].values[0]
            df_sub['dprime'    ] = row['dprime'    ].values[0]
            temp.append(df_sub)
    df = pd.concat(temp)
    return df

def label_high_low(df,n_jobs = 1):
    """
    to determine the high and low metacognition, the M-ratio should be used
    M-ratio = frac{meta-d'}{d'}
    """
    df['m-ratio'] = df['metadprime'] / (df['dprime']  + 1e-12)
    df_temp = df.groupby(['filename','sub']).mean().reset_index()
    m_ratio = df_temp['metadprime'].values / (df_temp['dprime'].values + 1e-12)
    criterion = np.median(m_ratio)
    df['level']  = df['m-ratio'].apply(lambda x: 'high' if x >= criterion else 'low')

    return df

def scoring_func(y_true,y_pred,confidence_range = 4):
    score  = []
    for ii in range(confidence_range):
        try:
            score.append(roc_auc_score(y_true[:,ii],y_pred[:,ii]))
        except:
            score.append(roc_auc_score(np.concatenate([y_true[:,ii],[0,1]]),
                                       np.concatenate([y_pred[:,ii],[0.5,0.5]])
                                       ))
    return score