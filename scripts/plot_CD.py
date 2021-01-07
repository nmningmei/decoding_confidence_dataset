#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:16:06 2019

@author: nmei
"""

import os
from glob import glob
from sklearn.preprocessing import MinMaxScaler as scaler
import pandas as pd
import numpy as np
from scipy import stats
from functools import partial

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

experiment = 'adequacy' # confidence or adequacy
working_dir = f'../results/{experiment}/R*CD/'
stats_dir = f'../stats/{experiment}/CD'
figure_dir = f'../figures/{experiment}/CD'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
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
df_plot = df_plot.sort_values(['source','model'])
df_stat_score = pd.read_csv(os.path.join(stats_dir,'scores.csv'))
df_stat_features = pd.read_csv(os.path.join(stats_dir,'features.csv'))

xargs = dict(hue = 'model',
             hue_order = ['RF','RNN'],
             split = True,
             inner = 'quartile',
             cut = 0,
             scale = 'width',)


fig,ax = plt.subplots(figsize = (16,16))
ax = sns.violinplot(x = 'source',
                    y = 'score',
                    data = df_plot,
                    ax = ax,
                    **xargs,
                    )
xtick_order = list(ax.xaxis.get_majorticklabels())

for ii,text_obj in enumerate(xtick_order):
    position = text_obj.get_position()
    ytick_label = text_obj.get_text()
    df_sub_stats = df_stat_score[df_stat_score['source'] == ytick_label].sort_values(['model'])
    for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.1,0.1]):
        if '*' in temp_row['stars']:
            print(temp_row['stars'])
            ax.annotate(temp_row['stars'],
                        xy = (ii + adjustment,0.95),
                        ha = 'center',
                        fontsize = 16)
ax.set(ylim = (0.15,1.05),ylabel = 'ROC AUC',xlabel = 'Target study')
ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5,)
ax.legend(loc = 'lower right')

fig.savefig(os.path.join(figure_dir,'cross domain decoding scores.jpeg',
                         ),
            bbox_inches = 'tight',)

df_plot = pd.melt(df,id_vars = [item for item in df.columns if ('T' not in item)],
                  value_vars = [item for item in df.columns if ('T' in item)],
                  var_name = ['Time'],
                  value_name = 'hidden states/feature importance')
df_plot = df_plot[df_plot['source'] != 'train']

fig,axes = plt.subplots(figsize = (16,7 * 3),
                        nrows = 3,
                        )
for ax,(source,df_sub_plot) in zip(axes.flatten(),df_plot.groupby('source')):
    ax = sns.stripplot(x = 'Time',
                       y = 'hidden states/feature importance',
                       hue = 'model',
                       hue_order = ['RF','RNN',],
                       data = df_sub_plot,
                       ax = ax,
                       dodge = True,
                       alpha = 0.01,)
    temp_func = partial(stats.trim_mean,**dict(proportiontocut=0.05))
    ax = sns.pointplot(x = 'Time',
                       y = 'hidden states/feature importance',
                       data = df_sub_plot,
                       ax = ax,
                       hue = 'model',
                       hue_order = ['RF','RNN'],
                       dodge = 0.4,
                       palette = 'dark',
                       estimator = temp_func,
                       markers = 'd',
                       join = False,
                       ci = None,
                       scale = 0.75,
                       )
    ###########################################################################
    df_sub_plot['x'] = df_sub_plot['Time'].map({f'T-{7-ii}':ii for ii in range(7)})
    rf_x = df_sub_plot[df_sub_plot['model'] == 'RF']['x'].values
    rn_x = df_sub_plot[df_sub_plot['model'] == 'RNN']['x'].values
    df_stat_features_sub = df_stat_features[df_stat_features['source'] == source]
    rf = df_stat_features_sub[df_stat_features_sub['model'] == 'RF']
    rn = df_stat_features_sub[df_stat_features_sub['model'] == 'RNN']
    for temp,col in zip([rf,rn],['blue','orange']):
        
        y_mean = np.array([item for item in temp['y_mean'].values[0].replace('[','').replace(']','').replace('\n','').replace('  ',' ').split(' ') if len(item) > 0],
                           dtype = 'float32')
        y_std = np.array([item for item in temp['y_std'].values[0].replace('[','').replace(']','').replace('\n','').replace('  ',' ').split(' ') if len(item) > 0],
                          dtype = 'float32')
        xxx = np.linspace(0,6,y_mean.shape[0])
        ax.plot(xxx,y_mean,linestyle = '--',color = col,alpha = 0.7)
        ax.fill_between(xxx,
                        y_mean + y_std,
                        y_mean - y_std,
                        color = col,
                        alpha = 0.05)
    ###########################################################################
    ax.set(xlabel = '',
           ylabel = 'A.U.',
           title = f'{source}, n_sample={int(rf_x.shape[0])}',
           xticklabels = [],)
    handles,labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
ax.set(xlabel = 'Time Steps',xticklabels = [f'T-{7-ii}' for ii in range(7)])
fig.legend(handles[2:],labels[2:],loc = (0.88,0.475),title = '')
fig.savefig(os.path.join(figure_dir,'hidden states of time steps.jpeg'),
          bbox_inches = 'tight')









































