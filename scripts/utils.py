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
    t = tqdm(working_data,)
    for f in t:
        df_temp = pd.read_csv(f,header = 0)
        df_temp['accuracy'] = np.array(df_temp['Stimulus'] == df_temp['Response']).astype(int)
        df_temp['filename'] = f
        if target_columns[0] == 'metaAdequacy':
            t.set_description(f'add metaAdequacy,concatinating')
            df_temp['metaAdequacy'] = df_temp.apply(meta_adequacy,axis = 1)
        else:
            t.set_description(f'add confidence,concatinating')
        df_temp = df_temp[np.concatenate([['Subj_idx','filename','accuracy'],target_columns])]
        df_for_concat.append(df_temp)
    df_concat = pd.concat(df_for_concat)
    
    df = dict(sub = [],
              filename = [],
              targets = [],
              accuracy = [],)
    for ii in range(time_steps):
        df[f'feature{ii + 1}'] = []
    
    for (sub,filename), df_sub in tqdm(df_concat.groupby(['Subj_idx','filename']),desc = 'feature generating'):
    #    print(sub,filename)
        values = df_sub[target_columns].values
        accuracy = df_sub['accuracy'].values
        data_gen = TimeseriesGenerator(values,values,
                                       length = time_steps,
                                       sampling_rate = 1,
                                       batch_size = 1)
        
        for (features_,targets_),accuracy_ in zip(list(data_gen),accuracy[time_steps:]):
            df['sub'].append(sub)
            df['filename'].append(filename)
            [df[f"feature{ii + 1}"].append(f) for ii,f in enumerate(features_.flatten())]
            df["targets"].append(targets_.flatten()[0])
            df["accuracy"].append(accuracy_)
    df = pd.DataFrame(df)
    # re-order the columns
    df = df[np.concatenate([
             ['filename', 'sub','accuracy'],
             [f'feature{ii + 1}' for ii in range(time_steps)],
             ['targets']
             ])]
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
    try:
        from sklearn.inspection import permutation_importance
    except:
        print('why, IT?')
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

def resample_ttest(x,
                   baseline         = 0.5,
                   n_ps             = 100,
                   n_permutation    = 10000,
                   one_tail         = False,
                   n_jobs           = 12, 
                   verbose          = 0,
                   full_size        = True
                   ):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    one_tail: whether to perform one-tailed comparison
    """
    import numpy as np
    import gc
    from joblib import Parallel,delayed
    # statistics with the original data distribution
    t_experiment    = np.mean(x)
    null            = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    
    if null.shape[0] > int(1e4): # catch for big data
        full_size   = False
    if not full_size:
        size        = int(1e3)
    else:
        size = null.shape[0]
    
    
    gc.collect()
    def t_statistics(null,size,):
        """
        null: shifted data distribution
        size: tuple of 2 integers (n_for_averaging,n_permutation)
        """
        null_dist   = np.random.choice(null,size = size,replace = True)
        t_null      = np.mean(null_dist,0)
        if one_tail:
            return ((np.sum(t_null >= t_experiment)) + 1) / (size[1] + 1)
        else:
            return ((np.sum(np.abs(t_null) >= np.abs(t_experiment))) + 1) / (size[1] + 1) /2
    if n_ps == 1:
        ps = t_statistics(null, (size,int(n_permutation)))
    else:
        ps = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                        'null':null,
                        'size':(size,int(n_permutation)),}) for i in range(n_ps))
        ps = np.array(ps)
    return ps
def resample_ttest_2sample(a,b,
                           n_ps                 = 100,
                           n_permutation        = 10000,
                           one_tail             = False,
                           match_sample_size    = True,
                           n_jobs               = 6,
                           verbose              = 0):
    from joblib import Parallel,delayed
    import gc
    # when the samples are dependent just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,
                                     baseline       = 0,
                                     n_ps           = n_ps,
                                     n_permutation  = n_permutation,
                                     one_tail       = one_tail,
                                     n_jobs         = n_jobs,
                                     verbose        = verbose,)
        return ps
    else: # when the samples are independent
        t_experiment        = np.mean(a) - np.mean(b)
        if not one_tail:
            t_experiment    = np.abs(t_experiment)
            
        def t_statistics(a,b):
            group           = np.concatenate([a,b])
            np.random.shuffle(group)
            new_a           = group[:a.shape[0]]
            new_b           = group[a.shape[0]:]
            t_null          = np.mean(new_a) - np.mean(new_b)
            if not one_tail:
                t_null      = np.abs(t_null)
            return t_null
        
        gc.collect()
        ps = np.zeros(n_ps)
        for ii in range(n_ps):
            t_null_null = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                            'a':a,
                            'b':b}) for i in range(n_permutation))
            if one_tail:
                ps[ii] = ((np.sum(t_null_null >= t_experiment)) + 1) / (n_permutation + 1)
            else:
                ps[ii] = ((np.sum(np.abs(t_null_null) >= np.abs(t_experiment))) + 1) / (n_permutation + 1) / 2
        return ps

class MCPConverter(object):
    import statsmodels as sms
    """
    https://gist.github.com/naturale0/3915e2def589553e91dce99e69d138cc
    https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    input: array of p-values.
    * convert p-value into adjusted p-value (or q-value)
    """
    def __init__(self, pvals, zscores = None):
        self.pvals                    = pvals
        self.zscores                  = zscores
        self.len                      = len(pvals)
        if zscores is not None:
            srted                     = np.array(sorted(zip(pvals.copy(), zscores.copy())))
            self.sorted_pvals         = srted[:, 0]
            self.sorted_zscores       = srted[:, 1]
        else:
            self.sorted_pvals         = np.array(sorted(pvals.copy()))
        self.order                    = sorted(range(len(pvals)), key=lambda x: pvals[x])
    
    def adjust(self, method           = "holm"):
        import statsmodels as sms
        """
        methods = ["bonferroni", "holm", "bh", "lfdr"]
         (local FDR method needs 'statsmodels' package)
        """
        if method == "bonferroni":
            return [np.min([1, i]) for i in self.sorted_pvals * self.len]
        elif method == "holm":
            return [np.min([1, i]) for i in (self.sorted_pvals * (self.len - np.arange(1, self.len+1) + 1))]
        elif method == "bh":
            p_times_m_i = self.sorted_pvals * self.len / np.arange(1, self.len+1)
            return [np.min([p, p_times_m_i[i+1]]) if i < self.len-1 else p for i, p in enumerate(p_times_m_i)]
        elif method == "lfdr":
            if self.zscores is None:
                raise ValueError("Z-scores were not provided.")
            return sms.stats.multitest.local_fdr(abs(self.sorted_zscores))
        else:
            raise ValueError("invalid method entered: '{}'".format(method))
            
    def adjust_many(self, methods = ["bonferroni", "holm", "bh", "lfdr"]):
        if self.zscores is not None:
            df = pd.DataFrame(np.c_[self.sorted_pvals, self.sorted_zscores], columns=["p_values", "z_scores"])
            for method in methods:
                df[method] = self.adjust(method)
        else:
            df = pd.DataFrame(self.sorted_pvals, columns=["p_values"])
            for method in methods:
                if method != "lfdr":
                    df[method] = self.adjust(method)
        return df

def stars(x):
    if x < 0.001:
        return '***'
    elif x < 0.01:
        return '**'
    elif x < 0.05:
        return '*'
    else:
        return 'n.s.'