#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 04:04:53 2020

@author: nmei

add metaAdequacy

"""

import os

from glob import glob
from utils import preprocess

experiment          = 'adequacy'
data_dir            = '../data'
model_dir           = '../models/{experiment}'
working_dir         = '../data/4-point'
working_data        = glob(os.path.join(working_dir, "*.csv"))
working_df_name     = os.path.join(data_dir,f'{experiment}','LOO','all_data_adequacy.csv')
saving_dir          = '../results/{experiment}'
batch_size          = 32
time_steps          = 7
confidence_range    = 4
target_columns      = ['metaAdequacy']
n_jobs              = -1
verbose             = 1

df_def          = preprocess(working_data,target_columns = target_columns,n_jobs = n_jobs)
if not os.path.exists(os.path.join(data_dir,f'{experiment}')):
    os.mkdir(os.path.join(data_dir,f'{experiment}'))
df_def.to_csv(working_df_name,index=False)
