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

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

experiment = 'confidence' # confidence or adequacy
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

fig,ax = plt.subplots(figsize = (16,16))
ax = sns.barplot(x = 'source',
                 y = 'score',
                 hue = 'model',
                 hue_order = ['RF','RNN'],
                 data = df_plot,
                 ax = ax,
                 )
ax.set(xlabel = 'Target data',ylabel = 'ROC AUC',)
fig.savefig(os.path.join(figure_dir,'cross domain decoding scores.jpeg',
                         ),
            bbox_inches = 'tight',)

df_plot = pd.melt(df,id_vars = [item for item in df.columns if ('T' not in item)],
                  value_vars = [item for item in df.columns if ('T' in item)],
                  var_name = ['Time'],
                  value_name = 'hidden states/feature importance')
df_plot = df_plot[df_plot['source'] != 'train']

g = sns.catplot(x = 'Time',
                y = 'hidden states/feature importance',
                hue = 'model',
                row = 'source',
                hue_order = ['RF','RNN'],
                data = df_plot,
                kind = 'bar',
                aspect = 2,
                )
(g.set_axis_labels('Time Steps','Weights')
  .set_titles('{row_name}'))
g.savefig(os.path.join(figure_dir,'hidden states of time steps.jpeg'),
          bbox_inches = 'tight')









































