#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 04:04:53 2020

@author: nmei

add metaAdequacy

"""

import os
import pandas as pd

from glob import glob
from utils import add_adequacy

experiment          = 'LOO'
data_dir            = '../data'
working_dir         = '../data/4-point'
working_data        = glob(os.path.join(working_dir, "*.csv"))
working_df_name     = os.path.join(data_dir,f'{experiment}','all_data_adequacy.csv')
n_jobs              = -1

if not os.path.exists(working_df_name):
    df_def          = add_adequacy(working_data,n_jobs = n_jobs)
    if not os.path.exists(os.path.join(data_dir,f'{experiment}')):
        os.mkdir(os.path.join(data_dir,f'{experiment}'))
    df_def.to_csv(working_df_name,index=False)
else:
    df_def          = pd.read_csv(working_df_name,)