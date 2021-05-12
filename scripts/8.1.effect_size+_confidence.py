#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 06:39:02 2021

@author: nmei
"""

import os
import utils

from glob import glob

import pandas as pd
import numpy as np

data_type = 'confidence'
working_dir = f'../results/{data_type}'

cv_type = 'LOO'
working_data = glob(os.path.join(working_dir,cv_type,'*.csv'))
working_data = [item for item in working_data if ('past' not in item) and ('recent' not in item)]


df = dict(cv_type = [],
          minimum = [],
          maximum = [],
          target_data = [],
          )

df_ave = utils.load_results(data_type = data_type,
                            within_cross = cv_type,
                            working_data = working_data,
                            dict_condition = None,
                            )
for _,df_sub in df_ave.groupby(['decoder','accuracy_train', 'accuracy_test',]):
    scores = df_sub['score'].values
    df['cv_type'].append(cv_type)
    df['minimum'].append(scores.min())
    df['maximum'].append(scores.max())
    df['target_data'].append('perceptual')

cv_type = 'cross_domain'
working_data = glob(os.path.join(working_dir,cv_type,'*.csv'))
working_data = [item for item in working_data if ('past' not in item) and ('recent' not in item)]

df_ave = utils.load_results(data_type = data_type,
                            within_cross = cv_type,
                            working_data = working_data,
                            dict_condition = None,
                            )
for temp,df_sub in df_ave.groupby(['source','decoder','accuracy_train', 'accuracy_test',]):
    scores = df_sub['score'].values
    df['cv_type'].append(cv_type)
    df['minimum'].append(scores.min())
    df['maximum'].append(scores.max())
    df['target_data'].append(temp[0])

df = pd.DataFrame(df)
