#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 05:30:49 2021

@author: nmei
"""

import os
import gc
import utils

import numpy as np
import pandas as pd

from glob import glob
from sklearn.preprocessing import MinMaxScaler as scaler
from sklearn.preprocessing import RobustScaler
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (ShuffleSplit,
                                     permutation_test_score,
                                     cross_validate)


experiment = 'adequacy' # confidence or adequacy
working_dir = f'../results/{experiment}/R*CD/'
stats_dir = f'../stats/{experiment}/CD/'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
working_data = np.sort(glob(os.path.join(working_dir,"*.csv")))

df = []
for f in working_data:
    temp = pd.read_csv(f)
    decoder = f.split('/')[-1].split(' ')[0]
    experiment = f.split('/')[-1].split('(')[-1].split(')')[0]
    temp['model'] = decoder
    temp['experiment'] = experiment
    col_to_rename = [item for item in temp.columns if ('T-' in item)]
    rename_mapper = {item:f'{item.split(" ")[-1]}' for item in col_to_rename}
    temp = temp.rename(columns = rename_mapper)
    # normalize with each decoding
    temp_array = temp[[item for item in temp.columns if ('T-' in item)]].values
    if decoder == 'RNN':
        temp_array = np.abs(temp_array)
    temp_array = scaler().fit_transform(temp_array.T)
    temp[[item for item in temp.columns if ('T-' in item)]] = temp_array.T
    df.append(temp)
df = pd.concat(df)

df_plot = df[df['source'] != 'train']

res_scores = dict(source = [],
                  model = [],
                  score_mean = [],
                  score_std = [],
                  pval = [],
                  )
res_features = dict(source = [],
                    model = [],
                    slope_mean = [],
                    slope_std = [],
                    intercept_mean = [],
                    intercept_std = [],
                    cv_score = [],
                    pval = [],
                    y_mean = [],
                    y_std = [],
                    )
res_slopes = dict(source = [],
                  model = [],
                  slope = [],)
for (source,model),df_sub in df_plot.groupby(['source','model']):
    # on the scores: compare against to theorectial chance level
    scores = df_sub['score'].values
    gc.collect()
    ps = utils.resample_ttest(scores,
                              0.5,
                              one_tail = True,
                              n_ps = 1,
                              n_permutation = int(1e5),
                              n_jobs = -1,
                              verbose = 0,)
    gc.collect()
    res_scores['source'].append(source)
    res_scores['model'].append(model)
    res_scores['score_mean'].append(np.mean(scores))
    res_scores['score_std'].append(np.std(scores))
    res_scores['pval'].append(np.mean(ps))
    
    # on the feature contributions
    features = df_sub[[f'T-{7 - ii}' for ii in range(7)]].values
    features = np.abs(features)
    xx = np.vstack([np.arange(7) for _ in range(features.shape[0])])
    cv = ShuffleSplit(n_splits = 100, test_size = 0.2,random_state = 12345)
    # a regularized linear regression
#    pipeline = linear_model.RidgeCV(alphas = np.logspace(-9,9,19),
#                                    scoring = 'neg_mean_squared_error',
#                                    cv = None,# set to None for efficient LOO algorithm
#                                    )
    pipeline = linear_model.BayesianRidge(fit_intercept = True)
    # permutation test to get p values
    _score,_,pval = permutation_test_score(pipeline,xx.reshape(-1,1),features.reshape(-1,1).ravel(),
                                           cv = cv,
                                           n_jobs = -1,
                                           random_state = 12345,
                                           n_permutations = int(1e3),
                                           scoring = 'neg_mean_squared_error',
                                           verbose = 1,
                                           )
    # cross validation to get the slopes and intercepts
    gc.collect()
    _res = cross_validate(pipeline,xx.reshape(-1,1),features.reshape(-1,1).ravel(),
                          cv = cv,
                          n_jobs = -1,
                          verbose = 1,
                          scoring = 'neg_mean_squared_error',
                          return_estimator = True,
                          )
    gc.collect()
    coefficients = np.array([est.coef_[0] for est in _res['estimator']])
    # save for 2 sample t tests
    for item in coefficients:
        res_slopes['source'].append(source)
        res_slopes['model'].append(model)
        res_slopes['slope'].append(item)
    intercepts = np.array([est.intercept_ for est in _res['estimator']])
    xxx = np.linspace(0,6,1000)
    temp = np.array([est.predict(xxx.reshape(-1,1),return_std = True) for est in _res['estimator']])
    y_mean = temp[:,0,:]
    y_std = temp[:,0,:]
    res_features['source'].append(source)
    res_features['model'].append(model)
    res_features['slope_mean'].append(np.mean(coefficients))
    res_features['slope_std'].append(np.std(coefficients))
    res_features['intercept_mean'].append(np.mean(intercepts))
    res_features['intercept_std'].append(np.std(intercepts))
    res_features['pval'].append(pval)
    res_features['cv_score'].append(_score)
    res_features['y_mean'].append(y_mean.mean(0))
    res_features['y_std'].append(y_std.mean(0))
res_scores = pd.DataFrame(res_scores)
res_features = pd.DataFrame(res_features)
res_slopes = pd.DataFrame(res_slopes)

temp = []
for model,df_sub in res_scores.groupby(['model']):
    df_sub = df_sub.sort_values(['pval'])
    pvals = df_sub['pval'].values
    converter = utils.MCPConverter(pvals = pvals)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
res_scores = pd.concat(temp)

temp = []
for model,df_sub in res_features.groupby(['model']):
    df_sub = df_sub.sort_values(['pval'])
    pvals = df_sub['pval'].values
    converter = utils.MCPConverter(pvals = pvals)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
res_features = pd.concat(temp)

feature_comparison = dict(source = [],
                          pval = [],
                          slope_rf_mean = [],
                          slope_rf_std = [],
                          slope_rnn_mean = [],
                          slope_rnn_std = [],
                          )
for (source),df_sub in res_slopes.groupby(['source']):
    rf = df_sub[df_sub['model'] == 'RF']['slope'].values
    rn = df_sub[df_sub['model'] == 'RNN']['slope'].values
    ps = utils.resample_ttest_2sample(rf,rn,
                                      one_tail = False,
                                      match_sample_size = False,
                                      n_jobs = -1,
                                      n_ps = 2,
                                      n_permutation = int(1e5),
                                      verbose = 1)
    feature_comparison['source'].append(source)
    feature_comparison['pval'].append(np.mean(ps))
    feature_comparison['slope_rf_mean'].append(rf.mean())
    feature_comparison['slope_rf_std'].append(rf.std())
    feature_comparison['slope_rnn_mean'].append(rn.mean())
    feature_comparison['slope_rnn_std'].append(rn.std())
feature_comparison = pd.DataFrame(feature_comparison)

res_scores['stars'] = res_scores['p_corrected'].apply(utils.stars)
res_scores.to_csv(os.path.join(stats_dir,'scores.csv'),index = False)

res_features['stars'] = res_features['p_corrected'].apply(utils.stars)
res_features.to_csv(os.path.join(stats_dir,'features.csv'),index = False)

feature_comparison.to_csv(os.path.join(stats_dir,'slope_comparison.csv'),index = False)