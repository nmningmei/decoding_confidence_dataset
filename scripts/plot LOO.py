#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:39:57 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as scaler
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('whitegrid')
sns.set_context('poster')

working_dir = '../results/LOO/'
figure_dir = '../figures/LOO_compare_RNN_RF/'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
working_data = glob(os.path.join(working_dir,'*.csv'))

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
        temp_array = - temp_array
    temp_array = scaler().fit_transform(temp_array.T)
    temp[[item for item in temp.columns if ('T-' in item)]] = temp_array.T
    df.append(temp)
df = pd.concat(df)

df_plot = df[df['source'] != 'train']

fig,ax = plt.subplots(figsize = (16,16))
ax = sns.barplot(y = 'experiment',
                 x = 'score',
                 hue = 'model',
                 data = df_plot,
                 ax = ax,
                 )
fig.savefig(os.path.join(figure_dir,
                         'RNN vs RF LOO.jpeg'),
#            dpi = 400,
            bbox_inches = 'tight')
unique_experiment = pd.unique(df_plot['experiment'])
fig,axes = plt.subplots(figsize = (28,28),
                        nrows = 4,
                        ncols = int(unique_experiment.shape[0] / 4),)
for ax,experiment in zip(axes.flatten(),unique_experiment):
    df_sub = df_plot[df_plot['experiment'] == experiment].sort_values(['model'])
    df_sub_plot = df_sub.melt(
                          id_vars = ['fold','sub_name','model','experiment',],
                          value_vars = ['T-7', 'T-6', 'T-5', 'T-4', 'T-3', 'T-2', 'T-1'],)
    ax = sns.barplot(x = 'variable',
                     y = 'value',
                     hue = 'model',
                     data = df_sub_plot,
                     ax = ax,
                     )
    ax.set(xlabel = '',
           ylabel = '',
           title = experiment,)
fig.savefig(os.path.join(figure_dir,
                         'RNN vs RF features.jpeg'),
#            dpi = 400,
            bbox_inches = 'tight')























