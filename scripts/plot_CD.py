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
ax.set(ylim = (0.15,0.85))
ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5,)
ax.set(xlabel = 'Target data',ylabel = 'ROC AUC',)
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
    ax.set(xlabel = '',
           ylabel = 'A.U.',
           title = source,)
    handles,labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
ax.set(xlabel = 'Time Steps')
fig.legend(handles[2:],labels[2:],loc = (0.88,0.475),title = '')
fig.savefig(os.path.join(figure_dir,'hidden states of time steps.jpeg'),
          bbox_inches = 'tight')









































