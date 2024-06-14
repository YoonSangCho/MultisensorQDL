# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:52:00 2020

@author: yscho

# http://www.blackarbs.com/blog/advanced-time-series-plots-in-python/1/6/2017

import matplotlib.pyplot as plt
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

######################################## Path
path_proj = 'D:/RESEARCH/Multisensor-RCA-QDL/'
path_code = path_proj+'code/'
os.chdir(path_code)
path_data = path_proj+'data/'

from utils.scenarios import RootCauses_200912
yymmdd, dataset_shape_list, root_cause_list = RootCauses_200912()
path_synthetic = path_data+'synthetic/{}/'.format(yymmdd)
path_results_prediction = path_proj+'results/synthetic/{}/prediction/'.format(yymmdd)
path_results_model = path_proj+'results/synthetic/{}/model/'.format(yymmdd)
path_results_rcm = path_proj+'results/synthetic/{}/RCM/'.format(yymmdd)

############################################################################## Load synthetic data
data_list = os.listdir(path_synthetic)
n_dataset = int(len(data_list)/2)

for idx in range(n_dataset):
    #idx=0
    # load
    with open(path_synthetic+data_list[2*idx], 'rb') as f:
        dataset = pickle.load(f)
    with open(path_synthetic+data_list[2*idx+1], 'rb') as f:
        y = pickle.load(f)
    f.close()
    ############################## Visualization
    
    n_process = dataset_shape_list[int(idx/3)-1][1]
    root_causes = root_cause_list[idx]
    process_ID = []
    for p in range(n_process):
        process_ID.append('Process {}'.format(p+1))
    
    n_causal_proc = len(root_causes)
    n_causal_time = int(root_causes[0][2]-root_causes[0][1])
    
    i=0
    tmp_data = pd.DataFrame(dataset[i].T, columns=process_ID)
    tmp_y = y[i]
    defect = tmp_y
    fig, axes = plt.subplots(n_process,1, figsize=(30,15), sharex=True)
    axes[0].set_title("Defect: {}".format(defect))
    tmp_data[process_ID].plot(subplots=True, ax=axes, ylim=(-defect-2,defect+2), rot=0)
    for ax in axes:
        for rc in root_causes:
            ax.legend(loc="upper right")
            axes[rc[0]].axvspan(rc[1], rc[2], color=sns.xkcd_rgb['red'], alpha=0.01)
    plt.show()
    ############################################################
