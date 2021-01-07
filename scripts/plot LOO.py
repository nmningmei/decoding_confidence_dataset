#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:39:57 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
import numpy as np
from scipy import stats
from functools import partial
from sklearn.preprocessing import MinMaxScaler as scaler
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('whitegrid')
sns.set_context('poster')

experiment = 'adequacy' # confidence or adequacy
working_dir = f'../results/{experiment}/LOO/'
stats_dir = f'../stats/{experiment}/LOO_compare_RNN_RF/'
figure_dir = f'../figures/{experiment}/LOO_compare_RNN_RF/'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
working_data = np.sort(glob(os.path.join(working_dir,'*.csv')))

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
df_plot = df_plot.sort_values(['experiment','model'])
df_stat_score = pd.read_csv(os.path.join(stats_dir,'scores.csv'))
df_stat_features = pd.read_csv(os.path.join(stats_dir,'features.csv'))

xargs = dict(hue = 'model',
             hue_order = ['RF','RNN'],
             split = True,
             inner = 'quartile',
             cut = 0,
             scale = 'width',)

fig,ax = plt.subplots(figsize = (16,16))
ax = sns.violinplot(y = 'experiment',
                    x = 'score',
                    data = df_plot,
                    ax = ax,
                    **xargs)
ytick_order = list(ax.yaxis.get_majorticklabels())

for ii,text_obj in enumerate(ytick_order):
    position = text_obj.get_position()
    ytick_label = text_obj.get_text()
    df_sub_stats = df_stat_score[df_stat_score['experiment'] == ytick_label].sort_values(['model'])
    for (jj,temp_row),adjustment in zip(df_sub_stats.iterrows(),[-0.225,0.225]):
        if '*' in temp_row['stars']:
            print(temp_row['stars'])
            ax.annotate(temp_row['stars'],
                        rotation = 90,
                        xy = (1.01,ii + adjustment,),
                        verticalalignment = 'center',
                        fontsize = 16)
ax.set(xlim = (0.3,1.05),xlabel = 'ROC AUC',ylabel = 'Study')
ax.axvline(0.5,linestyle = '--',color = 'black',alpha = 0.5,)
ax.legend(loc = 'upper left')
fig.savefig(os.path.join(figure_dir,
                         'RNN vs RF LOO.jpeg'),
#            dpi = 400,
            bbox_inches = 'tight')

unique_experiment = pd.unique(df_plot['experiment'])
fig,axes = plt.subplots(figsize = (28,36),
                        nrows = 4,
                        ncols = int(unique_experiment.shape[0] / 4),
                        sharey = True,)
for ii,(ax,experiment) in enumerate(zip(axes.flatten(),unique_experiment)):
    df_sub = df_plot[df_plot['experiment'] == experiment].sort_values(['model'])
    df_sub_plot = df_sub.melt(
                          id_vars = ['fold','sub_name','model','experiment',],
                          value_vars = ['T-7', 'T-6', 'T-5', 'T-4', 'T-3', 'T-2', 'T-1'],)
    df_sub_stats = df_stat_features[df_stat_features['experiment'] == experiment]
    ax = sns.stripplot(x = 'variable',
                       y = 'value',
                       data = df_sub_plot,
                       ax = ax,
                       hue = 'model',
                       hue_order = ['RF','RNN'],
                       dodge = True,
                       alpha = 0.1,)
    temp_func = partial(stats.trim_mean,**dict(proportiontocut=0.05))
    ax = sns.pointplot(x = 'variable',
                       y = 'value',
                       data = df_sub_plot,
                       ax = ax,
                       hue = 'model',
                       hue_order = ['RF','RNN'],
                       dodge = 0.5,
                       palette = 'dark',
                       estimator = temp_func,
                       markers = 'd',
                       join = False,
                       ci = None,
                       scale = 0.5,
                       )
    ########################################################################
    df_sub_plot['x'] = df_sub_plot['variable'].map({f'T-{7-ii}':ii for ii in range(7)})
    rf_x = df_sub_plot[df_sub_plot['model'] == 'RF']['x'].values
    rn_x = df_sub_plot[df_sub_plot['model'] == 'RNN']['x'].values
    df_stat_features_sub = df_stat_features[df_stat_features['experiment'] == experiment]
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
    ########################################################################
    
    ax.set(xlabel = '',
           ylabel = '',
           title = f'{experiment}\nn_sample={int(rf_x.shape[0])}',
           xticklabels = [],)
    if ii % 4 == 0:
        ax.set(ylabel = 'A.U.')
    if ii >=12:
        ax.set(xlabel = 'Time Steps',xticklabels = [f'T-{7-ii}' for ii in range(7)])
    handles,labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
#plt.subplots_adjust(top = 1.1)
fig.legend(handles[2:],labels[2:],loc = (0.91,0.475),title = '')
fig.savefig(os.path.join(figure_dir,
                         'RNN vs RF features.jpeg'),
#            dpi = 400,
#            bbox_inches = 'tight',
            )























