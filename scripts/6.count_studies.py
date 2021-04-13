# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:47:15 2021

@author: ning
"""

import os

from glob import glob

import numpy as np
import pandas as pd

working_dir = '../data/confidence/cross_domain'
perception = os.path.join(working_dir,'source.csv')
others = os.path.join(working_dir,'target.csv')

df_perception = pd.read_csv(perception)
df_others = pd.read_csv(others)
df_others['domain'] = df_others['filename'].apply(lambda x:x.split('/')[3].split('-')[0])

temp = df_perception.groupby(['filename','sub']).count().reset_index()[['filename','sub']]
temp = temp.groupby(['filename']).count().reset_index()
temp['domain'] = 'Perceptual domain'
temp = [temp,(df_others.groupby(['domain','filename','sub'])
.count()
.reset_index()[['domain','filename','sub']]
.groupby(['domain','filename',])
.count()
.reset_index()
)]
temp = pd.concat(temp)

temp['study'] = temp['filename'].apply(lambda x:x.split('/')[-1].replace('.csv',''))

def _process(x):
    temp = x.split('_')
    if len(temp) == 3:
        return f'{temp[1]},{temp[2]}'
    elif len(temp) == 4:
        return f'{temp[1]}, {temp[2]}, {temp[3]}'
temp['study'] = temp['study'].apply(_process)

temp.to_csv('../results/study counts.csv',index = False)