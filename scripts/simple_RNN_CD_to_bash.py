#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:26:17 2019

@author: nmei
"""
import os
import re
import pandas as pd
import numpy as np
from glob import glob

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils           import shuffle as util_shuffle

templates           = ['simple_RNN_cross_domain_RF.py',
                       'simple_RNN_cross_domain_train_RNN.py',
                       'simple_RNN_cross_domain_hidden.py']
experiment          = ['cross_domain','confidence','RF','RNN']
data_dir            = '../data'
source_dir          = '../data/4-point'
target_dir          = '../data/targets/*/'
source_data         = glob(os.path.join(source_dir, "*.csv"))
target_data         = glob(os.path.join(target_dir, "*.csv"))
source_df_name      = os.path.join(data_dir,experiment[1],experiment[0],'source.csv')
target_df_name      = os.path.join(data_dir,experiment[1],experiment[0],'target.csv')
node                = 1
core                = 16
mem                 = 2 * core * node
cput                = 24 * core * node
n_splits            = 100
level               = 'high'

df_source       = pd.read_csv(source_df_name)
df_target       = pd.read_csv(target_df_name)

add = """from shutil import copyfile
copyfile('../utils.py','utils.py')

"""
bash_folder = f'{"_".join(experiment)}_bash'
if not os.path.exists(bash_folder):
    os.mkdir(bash_folder)
    os.mkdir(os.path.join(bash_folder,'outputs'))

add_on = """from shutil import copyfile
copyfile('../utils.py','utils.py')

"""

if not os.path.exists(f'{bash_folder}/outputs'):
    os.mkdir(f'{bash_folder}/outputs')

features    = df_source[[f"feature{ii + 1}" for ii in range(7)]].values
targets     = df_source["targets"].values.astype(int)
groups      = df_source["sub"].values
np.random.seed(12345)
features,targets,groups = util_shuffle(features,targets,groups)
cv                      = GroupShuffleSplit(n_splits = n_splits,
                                            test_size = 0.2,
                                            random_state = 12345)
collection = []
for template in templates:
    for ii,(tr,te) in enumerate(cv.split(features,targets,groups)):
        new_script_name = os.path.join(bash_folder,template.replace('.py',f'_{ii+1}.py'))
        with open(new_script_name,'w') as new_file:
            with open(template,'r') as old_file:
                for line in old_file:
                    if "batch_change" in line:
                        line = line.replace('0',f'{ii}')
                    elif '../' in line:
                        line = line.replace('../','../../')
                    elif 'add here' in line:
                        line = add_on
                    elif "from glob" in line:
                        line = line + '\n' + add + '\n'
                    elif "verbose             =" in line:
                        line = "verbose             = 0\n"
                    elif "n_splits            = 100" in line:
                        line = f"n_splits            = {n_splits}\n"
                    elif "dprime_level" in line:
                        line = line.replace('low',level)
                    new_file.write(line)
                old_file.close()
            new_file.close()
        collection.append(new_script_name)
    

collection = np.array(collection).reshape(3,-1).T

for ii,row in enumerate(collection):
    rf,train,hidden = row
    new_batch_script_name = os.path.join(bash_folder,f'{"_".join(experiment)}{ii+1}')
    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N {"_".join(experiment)}_{ii+1}
#PBS -o outputs/out_{ii+1}.txt
#PBS -e outputs/err_{ii+1}.txt

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo fold-{ii}

python "{rf.split('/')[-1]}"
python "{train.split('/')[-1]}"
python "{hidden.split('/')[-1]}"
    """
    with open(new_batch_script_name,'w') as f:
        f.write(content)
        f.close()
    

with open(f'{bash_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

with open(f'{bash_folder}/qsub_jobs.py','a') as f:
    for ii in range(n_splits):
        if ii == 0:
            f.write(f'\nos.system("qsub {"_".join(experiment)}{ii+1}")\n')
        else:
            f.write(f'time.sleep(1)\nos.system("qsub {"_".join(experiment)}{ii+1}")\n')
    f.close()
    

