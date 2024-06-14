# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:38:18 2020

@author: yscho
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, jaccard_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy.random import seed

######################################## Path
path_proj = 'D:/RESEARCH/Multisensor-RCA-QDL/'
path_code = path_proj+'code/'
path_data = path_proj+'data/'
os.chdir(path_code)

from utils.scenarios import RootCauses_200912
yymmdd, dataset_shape_list, root_cause_list = RootCauses_200912()
path_synthetic = path_data+'synthetic/{}/'.format(yymmdd)
path_results_prediction = path_proj+'results/synthetic/{}/prediction/'.format(yymmdd)
path_results_model = path_proj+'results/synthetic/{}/model/'.format(yymmdd)
path_results_rcm = path_proj+'results/synthetic/{}/RCM/'.format(yymmdd)
path_results_ram = path_proj+'results/synthetic/{}/RAM/'.format(yymmdd)
path_results_performance = path_proj+'results/synthetic/{}/performance/'.format(yymmdd)
data_list = os.listdir(path_synthetic)

################################################################################
# import tensorflow as tf
# from tensorflow.compat.v1 import ConfigProto, InteractiveSession
# from tensorflow.keras import backend as K
# # from utils.QDL import VisHistory, getRAMs, getRCM, getRCM_minmax, getRCM_standard, VisRAMs, VisRCM_2D, VisRCM_3D
# # from models.cae import reg_cae_upsampled, reg_cae_transposed
# # from models.deconvnet import reg_deconvnet
# # from models.segnet import reg_segnet
# # from models.fcn import reg_fcn_vgg, reg_fcn_densenet
# # from models.unet import reg_unet
# model_list = [reg_cae_upsampled, reg_cae_transposed,
#               reg_deconvnet,
#               reg_fcn_vgg, reg_fcn_densenet, 
#               reg_segnet,  reg_unet]

model_name_list = ['cae_upsampled', 'DeconvNet', 'SegNet', 'fcn_vgg', 'fcn_densenet', 'unet']
n_block_list = [1,2,3]
n_dataset = len(root_cause_list)

# for tmp_model in model_name_list:
#     for n_block in n_block_list:
tmp_model = 'SegNet'
n_block = 1 #n_block_list[0]
for idx in range(0, n_dataset):
    # idx = 6
    n_scenario = data_list[2*idx].split('_')[1]
    n_scenario_sub = data_list[2*idx].split('_')[2]
    dataset_shape = dataset_shape_list[int(idx/3)-1]

    with open(path_results_rcm+'{}/'.format(tmp_model)+'dataset_{}_{}_{}_RAMs.pkl'.format(idx+1, tmp_model, n_block), 'rb') as f:
        RAMs = pickle.load(f)
    # with open(path_results_rcm+'{}/'.format(tmp_model)+'dataset_{}_{}_{}_RCM.pkl'.format(idx+1, tmp_model, n_block), 'rb') as f:
    #     RCM_raw = pickle.load(f)
    # with open(path_results_rcm+'{}/'.format(tmp_model)+'dataset_{}_{}_{}_RCM_minmax.pkl'.format(idx+1, tmp_model, n_block), 'rb') as f:
    #     RCM_minmax = pickle.load(f)
    # with open(path_results_rcm+'{}/'.format(tmp_model)+'dataset_{}_{}_{}_RCM_standard.pkl'.format(idx+1, tmp_model, n_block), 'rb') as f:
    #     RCM_standard = pickle.load(f)
    f.close()
    
    #######################################  Datasets
    with open(path_synthetic+data_list[2*idx], 'rb') as f:
        dataset = pickle.load(f)
    with open(path_synthetic+data_list[2*idx+1], 'rb') as f:
        y = pickle.load(f)
    f.close()
    
    #%% Visualization of causal maps for extracting prediccted causal regions for each RAM
    # indexation of target product
    y_max = max(y)
    tmp_idx = np.where(y==y_max)[0][0]
    
    tmp_RAM = RAMs[tmp_idx]#.shape

    # num_best = 10
    # num_worst = 10

    # y_sorted = pd.DataFrame(y, columns = ['y']).sort_values(by = 'y')
    # idx_best = list(y_sorted.iloc[0:num_worst].index)
    # idx_best.sort()
    # idx_worst = list(y_sorted.iloc[-num_best:].index)
    # idx_worst.sort()

    # data index
    n_process = dataset_shape_list[int(idx/3)-1][1]
    process_ID = []
    for p in range(n_process):
        process_ID.append('Process {}'.format(p))#+1
    tmp_data = pd.DataFrame(dataset[tmp_idx].T, columns=process_ID)
    defect = y[tmp_idx]
    
    # root cause: TRUE
    root_causes = root_cause_list[idx]
    n_root_causes = len(root_causes)
    # n_causal_proc = len(root_causes)
    # n_causal_time = int(root_causes[0][2]-root_causes[0][1])
    
    # # plotting (1): heatmap
    # plt.figure(figsize=(30,15))
    # sns.heatmap(tmp_RAM, cbar_kws={"shrink": 0.5},cmap='coolwarm')#, ,cmap='YlGnBu', vmin=-1,vmax=1,linewidths=0.005 #imshow(tmp_ram, alpha=0.5, cmap='jet')
    # plt.show()

    # plotting (2): heatmap & sensor data
    fig, axes = plt.subplots(n_process,1, figsize=(50,20), sharex=True)
    axes[0].set_title(f"Defect: {defect} root causes: {root_causes}")
    tmp_data[process_ID].plot(subplots=True, ax=axes, ylim=(-defect-2,defect+2), rot=0)
    for ax in axes:
        for proc in range(tmp_RAM.shape[0]):
            ax.legend(loc="right")
            for time in range(tmp_RAM.shape[1]-1):
                if tmp_RAM[proc, time] >= 0.5:
                    axes[proc].axvspan(time, time+1, color=sns.xkcd_rgb['red'], alpha=0.05*tmp_RAM[proc, time])
                else:
                    axes[proc].axvspan(time, time+1, color=sns.xkcd_rgb['blue'], alpha=0.05*(1-tmp_RAM[proc, time]))
    plt.savefig(path_results_ram+f'RAM_{tmp_model}_block_{n_block}_{n_scenario}_{n_scenario_sub}_defect_{defect}.png', bbox_inches = 'tight', pad_inches = 0.05,dpi=1000)
    plt.show()

