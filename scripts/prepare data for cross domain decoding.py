# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:24:08 2020

@author: ning
"""

import os
from glob import glob
from utils import add_adequacy
import pandas as pd
import numpy as np

experiment          = 'cross_domain'
data_dir            = '../data/'
model_dir           = '../models'
source_dir          = '../data/4-point'
target_dir          = '../data/targets/*/'
source_data         = np.array(glob(os.path.join(source_dir, "*.csv")))
target_data         = np.array(glob(os.path.join(target_dir, "*.csv")))
source_df_name      = os.path.join(data_dir,f'{experiment}','source.csv')
target_df_name      = os.path.join(data_dir,f'{experiment}','target.csv')
target_columns      = ['Confidence']
time_steps          = 7
confidence_range    = 4
n_jobs              = -1

if not os.path.exists(os.path.join(data_dir,experiment)):
    os.mkdir(os.path.join(data_dir,experiment))

df_source       = add_adequacy(source_data,n_jobs = n_jobs)
df_target       = add_adequacy(target_data,n_jobs = n_jobs)

df_source.to_csv(os.path.join(data_dir,experiment,'source.csv'),index = False)
df_target.to_csv(os.path.join(data_dir,experiment,'target.csv'),index = False)