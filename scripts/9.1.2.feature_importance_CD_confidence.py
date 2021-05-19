#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:30:26 2021

@author: ning
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

from sklearn import linear_model
from sklearn.model_selection import (LeaveOneGroupOut,
                                     permutation_test_score,
                                     cross_validate)

sns.set_style('whitegrid')
sns.set_context('poster')

experiment      = 'confidence' # confidence or adequacy
cv_type         = 'cross_domain' # LOO or cross_domain
decoder         = 'RF' #
confidence_range= 4
time_steps      = 7
domains         = ['Cognitive','Memory','Mixed']
xargs           = dict(hue          = 'accuracy_test',
                       hue_order    = ['correct trials','incorrect trials',],
                       col          = 'accuracy_train',
                       col_order    = ['correct trials','incorrect trials',],
                       row_order    = ['cognitive','mem_4','mixed_4'],
                       # split        = True,
                       # inner        = 'quartile',
                       # cut          = 0,
                       # scale        = 'width',
                       palette      = ['deepskyblue','tomato'],
                       # col_order    = ['SVM','RF','RNN'],
                       )
working_dir = f'../results/{experiment}/{cv_type}/'
stats_dir   = f'../stats/{experiment}/{cv_type}'
figures_dir = f'../figures/{experiment}/{cv_type}'
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

working_data = glob(os.path.join(working_dir,
                                 '*.csv'))
df_ave = utils.load_results(
    data_type      = experiment, # confidence or adequacy
    within_cross   = cv_type, # LOO or cross_domain
    working_data   = working_data,
    dict_rename    = {0:'incorrect trials',1:'correct trials'},
    dict_condition = None,
    )

# get the weights of the regression model
df_rf = df_ave[df_ave['decoder'] == decoder]
feature_importance = df_rf[[f'feature importance T-{7-ii}' for ii in range(time_steps)]]
for col in ['fold', 'source', 'accuracy_train', 'accuracy_test']:
    feature_importance[col] = df_rf[col]

df_plot = pd.melt(feature_importance,
                  id_vars = ['fold', 'source', 'accuracy_train', 'accuracy_test'],
                  value_vars = [f'feature importance T-{7-ii}' for ii in range(time_steps)],
                  var_name = 'Time',
                  value_name = 'feature importance',
                  )
df_plot['x'] = df_plot['Time'].apply(lambda x: x.split(' ')[-1])

g = sns.catplot(x = 'x',
                y = 'feature importance',
                row = 'source',
                data = df_plot,
                kind = 'bar',
                aspect = 1.5,
                **xargs
                )
(g.set_titles('Trained on {col_name}')
 .set_axis_labels('Trial','Feature importance'))
g._legend.set_title("Testing data")

















