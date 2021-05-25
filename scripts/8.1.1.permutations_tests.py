#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 04:42:05 2021

@author: nmei
"""

import os
import utils
import pandas as pd
import numpy as np

from glob import glob

# paired comparison
data_type = 'confidence'
confidence_range = 4
time_steps = 7
dict_rename = {0:'incorrect trials',1:'correct trials'}
working_dir = f'../results/{data_type}'
for cv_type in ['LOO','cross_domain']:
    working_data = glob(os.path.join(working_dir,cv_type,'*.csv'))
    df_ave = utils.load_results(
        data_type      = data_type,
        within_cross   = cv_type,
        working_data   = working_data,
        dict_rename    = dict_rename,
        dict_condition = None,
        )
    asdf