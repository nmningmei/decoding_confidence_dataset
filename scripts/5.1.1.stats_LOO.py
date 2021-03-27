#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:36:16 2021

@author: ning

statistics of the results

"""

import os
import re
import gc
import utils

from glob import glob

import numpy   as np
import pandas  as pd
import seaborn as sns

from matplotlib import pyplot as plt

sns.set_style('whitegrid')
sns.set_context('poster')

experiment  = 'confidence' # confidence or adequacy
cv_type     = 'LOO' # LOO or cross_domain
decoder     = 'regression' #
working_dir = f'../results/{experiment}/{cv_type}/'
stats_dir   = f'../stats/{experiment}/{cv_type}'
figures_dir = f'../stats/{experiment}/{cv_type}'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

df = []
for f in glob(os.path.join(working_dir,'*csv')):
    temp            = pd.read_csv(f)
    study_name      = re.findall('\(([^)]+)',f)[0]
    temp['study_name'] = study_name
    temp['decoder'] = f.split('/')[-1].split(' ')[0]
    df.append(temp)
df = pd.concat(df)

# common settings
ylim    = {'confidence':(0.45,0.9),
           'adequacy':(0.3,0.9)}[experiment]
confidence_range= 4
time_steps      = [f'T-{7-ii}' for ii in range(7)]
dict_rename     = {0:'Incorrect trials',1:'Correct trials'}
xargs           = dict(hue          = 'accuracy_test',
                       hue_order    = ['Correct trials','Incorrect trials',],
                       split        = True,
                       inner        = 'quartile',
                       cut          = 0,
                       scale        = 'width',
                       palette      = ['deepskyblue','tomato'],
                       )

for col_name in ['accuracy_train','accuracy_test']:
    df[col_name] = df[col_name].map(dict_rename)

# averge within each study
df_ave = df.groupby(['decoder',
                     'study_name',
                     'accuracy_train',
                     'accuracy_test']).mean().reset_index()
df_ave.to_csv(os.path.join(stats_dir,'scores.csv'),index = False)

# significance of scores
np.random.seed(12345)
results = dict(accuracy_train   = [],
               accuracy_test    = [],
               score_mean       = [],
               score_std        = [],
               ps               = [],
               decoder          = [],
               )
for (_decoder,acc_train,acc_test),df_sub in df_ave.groupby(['decoder','accuracy_train','accuracy_test']):
    scores = df_sub['score'].values
    
    gc.collect()
    ps = utils.resample_ttest(scores,
                              baseline      = .5,
                              n_ps          = 100,
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

# plot scores
g = sns.catplot(x           = 'accuracy_train',
                y           = 'score',
                col         = 'decoder',
                col_order   = ['regression','RNN'],
                data        = df_ave,
                kind        = 'violin',
                aspect      = 1.5,
                **xargs)
(g.set_axis_labels("Training data","ROC AUC")
  .set(ylim = ylim))
[ax.set_title(title) for ax,title in zip(g.axes.flatten(),['Ridge regression','Recurrent neural network'])]
[ax.axhline(0.5,linestyle = '--',alpha = .7,color = 'black') for ax in g.axes.flatten()]
g._legend.set_title("Testing data")
## add stars
for ax,_decoder in zip(g.axes.flatten(),['regression','RNN']):
    df_sub = results[results['decoder'] == _decoder]
    df_sub = df_sub.sort_values(['accuracy_train','accuracy_test'])
    xtick_order = list(ax.xaxis.get_majorticklabels())
    
    for ii,text_obj in enumerate(xtick_order):
        position        = text_obj.get_position()
        xtick_label     = text_obj.get_text()
        df_sub_stats    = df_sub[df_sub['accuracy_train'] == xtick_label].sort_values(['accuracy_test'])
        for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.125,0.125]):
            if '*' in temp_row['stars']:
                print(temp_row['stars'])
                ax.annotate(temp_row['stars'],
                            xy          = (ii + adjustment,.85),
                            ha          = 'center',
                            fontsize    = 14)
g.savefig(os.path.join(figures_dir,'scores.jpg'),
          dpi = 300,
          bbox_inches = 'tight')

# get the weights of the regression model
df_reg = df_ave[df_ave['decoder'] == decoder]
weights = df_reg[[col for col in df_reg.columns if ('weight' in col)]].values
w = np.concatenate([[w.reshape(7,-1).T] for w in weights])

# plot the weights
fig,ax = plt.subplots(figsize = (10,6))
colors = ['blue','orange','green','red']
for ii in range(confidence_range):
    weight_for_plot = w[:,ii,:]
    w_mean = weight_for_plot.mean(0)
    w_std = weight_for_plot.std(0)
    ax.plot(time_steps,
            w_mean,
            color = colors[ii],
            alpha = 1.,
            )
    ax.fill_between(time_steps,
                    w_mean + w_std,
                    w_mean - w_std,
                    color = colors[ii],
                    alpha = .5,
                    label = f'confidence {ii+1}')
ax.legend(loc = 'upper left')
ax.set(title = '',
       ylabel = 'Weight (A.U)',
       xlabel = 'Trial')
fig.savefig(os.path.join(figures_dir,'features.jpg'),
            dpi = 300,
            bbox_inches = 'tight')

# fit a regression to show the linear trend of the weights








