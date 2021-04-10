#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:49:24 2021

@author: ning
"""

import os
import gc
import utils

from glob import glob

import numpy   as np
import pandas  as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from sklearn import linear_model
from sklearn.model_selection import (LeaveOneGroupOut,
                                     permutation_test_score,
                                     cross_validate)

sns.set_style('whitegrid')
sns.set_context('poster')

experiment  = 'confidence' # confidence or adequacy
cv_type     = 'cross_domain' # LOO or cross_domain
decoder     = 'regression' #
working_dir = f'../results/{experiment}/{cv_type}/'
stats_dir   = f'../stats/{experiment}/{cv_type}'
figures_dir = f'../figures/{experiment}/{cv_type}'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

df = []
for f in glob(os.path.join(working_dir,'*csv')):
    temp            = pd.read_csv(f)
    temp['decoder'] = f.split('/')[-1].split('_')[0]
    df.append(temp)
df = pd.concat(df)

# common settings
ylim    = (0.1,1.1)
confidence_range= 4
time_steps      = np.arange(7)
dict_rename     = {0:'incorrect trials',1:'correct trials'}
xargs           = dict(hue          = 'accuracy_test',
                       hue_order    = ['correct trials','incorrect trials',],
                       split        = True,
                       inner        = 'quartile',
                       cut          = 0,
                       scale        = 'width',
                       palette      = ['deepskyblue','tomato'],
                       )

for col_name in ['accuracy_train','accuracy_test']:
    df[col_name] = df[col_name].map(dict_rename)

# significance of scores
np.random.seed(12345)
results = dict(accuracy_train   = [],
               accuracy_test    = [],
               score_mean       = [],
               score_std        = [],
               ps               = [],
               decoder          = [],
               target_data      = [],
               )
for (_decoder,acc_train,acc_test,target_data),df_sub in df.groupby(
        ['decoder','accuracy_train','accuracy_test','source']):
    scores = df_sub['score'].values
    
    gc.collect()
    ps = utils.resample_ttest(scores,
                              baseline      = .5,
                              n_ps          = 2,
                              n_permutation = int(1e4),
                              n_jobs        = -1,
                              verbose       = 1,
                              )
    gc.collect()
    
    results['accuracy_train'].append(acc_train)
    results['accuracy_test' ].append(acc_test)
    results['score_mean'    ].append(np.mean(scores))
    results['score_std'     ].append(np.std(scores))
    results['ps'            ].append(np.mean(ps))
    results['decoder'       ].append(_decoder)
    results['target_data'   ].append(target_data)
results = pd.DataFrame(results)

temp = []
for (_decoder),df_sub in results.groupby(['decoder']):
    df_sub                  = df_sub.sort_values(['ps'])
    pvals                   = df_sub['ps'].values
    converter               = utils.MCPConverter(pvals = pvals)
    d                       = converter.adjust_many()
    df_sub['ps_corrected']  = d['bonferroni'].values
    temp.append(df_sub)
results             = pd.concat(temp)
results['stars']    = results['ps_corrected'].apply(utils.stars)
results.to_csv(os.path.join(stats_dir,'scores.csv'),index = False)
# plot scores
g = sns.catplot(x           = 'accuracy_train',
                y           = 'score',
                order       = ['correct trials','incorrect trials'],
                col         = 'decoder',
                col_order   = ['regression','RNN'],
                row         = 'source',
                row_order   = ['cognitive','mem_4','mixed_4'],
                data        = df,
                kind        = 'violin',
                aspect      = 1.5,
                **xargs)
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
## add stars
for ax_row,target_data in zip(g.axes,['cognitive','mem_4','mixed_4']):
    for _ax,_decoder in zip(ax_row,['regression','RNN']):
        df_sub = results[np.logical_and(results['decoder'] == _decoder,
                                        results['target_data'] == target_data)]
        df_sub = df_sub.sort_values(['accuracy_train','accuracy_test'])
        
        for ii,text_obj in enumerate(xtick_order):
            # print(target_data,_decoder)
            position        = text_obj.get_position()
            xtick_label     = text_obj.get_text()
            df_sub_stats    = df_sub[df_sub['accuracy_train'] == xtick_label].sort_values(['accuracy_test'])
            for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.125,0.125]):
                # print(temp_row['stars'])
                if '*' in temp_row['stars']:
                    _ax.annotate(temp_row['stars'],
                                xy          = (ii + adjustment,1.05),
                                ha          = 'center',
                                fontsize    = 14)
(g.set_axis_labels("Training data","ROC AUC")
  .set(ylim = ylim)
  .set_titles(''))
# [ax.set_title(title) for ax,title in zip(g.axes.flatten(),['Lnear support vector machine','Recurrent neural network'])]
[ax.axhline(0.5,linestyle = '--',alpha = .7,color = 'black') for ax in g.axes.flatten()]
g._legend.set_title("Testing data")

g.savefig(os.path.join(figures_dir,'scores.jpg'),
          dpi = 300,
          bbox_inches = 'tight')

##############################################################################
# get the weights of the regression model
df_reg = df[df['decoder'] == decoder]
df_ave = df_reg.groupby(['fold','accuracy_train','accuracy_test','source','decoder']).mean().reset_index()

results = dict(accuracy_train = [],
               coefficient = [],
               ps = [],
               )
results[experiment] = []
slopes = dict(accuracy_train = [],
              coefficient = [],
              intercept = [],
              )
slopes[experiment] = []
# fit a regression to show the linear trend of the weights
for (acc_train),df_sub in df_ave.groupby(['accuracy_train']):
    for idx_confidence in range(confidence_range):
        weight_for_fit = df_sub[[f'weight T-{7-ii} C-{idx_confidence}' for ii in range(len(time_steps))]].values
        xx = np.vstack([np.arange(7) for _ in range(weight_for_fit.shape[0])])
        groups = df_sub['fold'].values
        groups = np.vstack([groups for _ in range(len(time_steps))]).T
        cv = LeaveOneGroupOut()
        pipeline = linear_model.BayesianRidge(fit_intercept = True)
        # permutation test to get p values
        _score,_,pval_slope = permutation_test_score(pipeline,
                                                     xx.reshape(-1,1),
                                                     weight_for_fit.reshape(-1,1).ravel(),
                                                     cv = cv,
                                                     groups = groups.reshape(-1,1).ravel(),
                                                     n_jobs = -1,
                                                     random_state = 12345,
                                                     n_permutations = int(1e4),
                                                     scoring = 'neg_mean_squared_error',
                                                     verbose = 1,
                                                     )
        # cross validation to get the slopes and intercepts
        gc.collect()
        _res = cross_validate(pipeline,
                              xx.reshape(-1,1),
                              weight_for_fit.reshape(-1,1).ravel(),
                              groups = groups.reshape(-1,1).ravel(),
                              cv = cv,
                              n_jobs = -1,
                              verbose = 1,
                              scoring = 'neg_mean_squared_error',
                              return_estimator = True,
                              )
        gc.collect()
        coefficients = np.array([est.coef_[0] for est in _res['estimator']])
        intercepts = np.array([est.intercept_ for est in _res['estimator']])
        results['accuracy_train'].append(acc_train)
        results[experiment].append(idx_confidence)
        results['ps'].append(pval_slope)
        results['coefficient'].append(np.mean(coefficients))
        for coef_,inte_ in zip(coefficients,intercepts):
            slopes['accuracy_train'].append(acc_train)
            slopes[experiment].append(idx_confidence)
            slopes['coefficient'].append(coef_)
            slopes['intercept'].append(inte_)
results = pd.DataFrame(results)
slopes = pd.DataFrame(slopes)

results.to_csv(os.path.join(stats_dir,'features.csv'),index = False)
slopes.to_csv(os.path.join(stats_dir,'slopes.csv'),index = False)

df_weights = []
for acc_train in ['correct trials','incorrect trials']:
    df_sub = df_ave[df_ave['accuracy_train'] == acc_train]
    weights = df_sub[[col for col in df_sub.columns if ('weight' in col)]].values
    w = np.concatenate([[w.reshape(7,-1).T] for w in weights])
    for ii in range(confidence_range):
        weight_for_plot = w[:,ii,:]
        temp = pd.DataFrame(weight_for_plot,columns = [f'T-{len(time_steps)-ii}' for ii in range(len(time_steps))])
        temp['accuracy_train'] = acc_train
        temp[experiment] = ii
        df_weights.append(temp)
df_weights = pd.concat(df_weights)
df_weights_plot = pd.melt(df_weights,
                          id_vars = ['accuracy_train',experiment],
                          value_vars = [f'T-{len(time_steps)-ii}' for ii in range(len(time_steps))],
                          var_name = ['Time'],
                          value_name = 'Weights')

colors = ['blue','orange','green','red']
df_weights_plot['x'] = df_weights_plot['Time'].apply(lambda x:-int(x[-1]))

g = sns.lmplot(x = 'x',
               y = 'Weights',
               hue = experiment,
               col = 'accuracy_train',
               col_order = ['correct trials','incorrect trials'],
               data = df_weights_plot,
               palette = colors,
               markers = '.',
               robust = True,
               seed = 12345,
               aspect = 1.5,
               )
(g.set_titles("train on {col_name}")
  .set_axis_labels('Trial','Weights (A.U.)')
  .set(xlim = (-7.5,-0.5),
       xticks = np.arange(-7,0),
       xticklabels = [f'T-{len(time_steps)-ii}' for ii in range(len(time_steps))])
  )
g.savefig(os.path.join(figures_dir,'features.jpg'),
          dpi = 300,
          bbox_inches = 'tight')













