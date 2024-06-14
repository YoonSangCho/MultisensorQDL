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

for tmp_model in model_name_list:
    # tmp_model = model_name_list[5]
    for n_block in n_block_list:
        # n_block = n_block_list[0]
        for idx in range(0, n_dataset):
            # idx = 0
            n_scenario = data_list[2*idx].split('_')[1]
            n_scenario_sub = data_list[2*idx].split('_')[2]
            dataset_shape = dataset_shape_list[int(idx/3)-1]
    
            with open(path_results_rcm+'{}/'.format(tmp_model)+'dataset_{}_{}_{}_RAMs.pkl'.format(idx+1, tmp_model, n_block), 'rb') as f:
                RAMs = pickle.load(f)
            with open(path_results_rcm+'{}/'.format(tmp_model)+'dataset_{}_{}_{}_RCM.pkl'.format(idx+1, tmp_model, n_block), 'rb') as f:
                RCM_raw = pickle.load(f)
            with open(path_results_rcm+'{}/'.format(tmp_model)+'dataset_{}_{}_{}_RCM_minmax.pkl'.format(idx+1, tmp_model, n_block), 'rb') as f:
                RCM_minmax = pickle.load(f)
            with open(path_results_rcm+'{}/'.format(tmp_model)+'dataset_{}_{}_{}_RCM_standard.pkl'.format(idx+1, tmp_model, n_block), 'rb') as f:
                RCM_standard = pickle.load(f)
            f.close()
            
            ####################################### Localization evaluation
            ##### (1) evaluation with ROC curve and AUC
            ##### y_actual
            root_causes = root_cause_list[idx]
            n_root_causes = len(root_causes)
    
            root_causes_true = np.zeros(dataset_shape[1:])
            for r in range(n_root_causes):
                root_causes_true[root_causes[r][0], root_causes[r][1]:root_causes[r][2]] = 1
            root_causes_true_1d = root_causes_true.flatten()
            
            ######################################################################
            root_causes_proc = []
            root_causes_time = []
            for r in range(n_root_causes):
                # r=0
                root_causes_proc.append(root_causes[r][0])
                root_causes_time.append(list(range(root_causes[r][1],root_causes[r][2])))
            
            RCM_list = [RCM_raw] #, RCM_standard
            for RCM_idx, RCM in enumerate(RCM_list):
                if RCM_idx==0:
                    RCM_name = "RCM_raw"
                else:
                    RCM_name = "RCM_standard"
                
                ######################### roc curve
                fpr, tpr, thresholds = roc_curve(root_causes_true_1d, RCM.flatten())
                auc = roc_auc_score(root_causes_true_1d, RCM.flatten())
                auc = np.round(auc, 3)
                len(root_causes_true_1d)
                len(RCM.flatten())
        
                if not os.path.exists(path_results_performance+'{}/'.format(tmp_model)):
                    os.makedirs(path_results_performance+'{}/'.format(tmp_model))
        
                ns_probs = [0 for _ in range(len(root_causes_true_1d))]
                fpr_random, tpr_random, thresholds = roc_curve(root_causes_true_1d, ns_probs)
                ns_auc = roc_auc_score(root_causes_true_1d, ns_probs)
                print('No Skill: ROC AUC=%.3f' % (ns_auc))
                print('{}: ROC AUC=%.3f'.format(tmp_model) % (auc))
                plt.figure(figsize=(8,8))
                plt.title(f"dataset_{idx+1}_{tmp_model}_{n_block}_{RCM_name}_auc: {auc}")
                plt.plot(fpr_random, tpr_random, linestyle='--', label='random')
                plt.plot(fpr, tpr, marker='.', label='{}'.format(tmp_model))
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.savefig(path_results_performance+'{}/'.format(tmp_model)+f'dataset_{idx+1}_{tmp_model}_{n_block}_{RCM_name}_auc_{auc}.png')
                plt.show()
                    
                ######################################################################
                ##### Threshold
                pctl_70 = np.percentile(RCM, 70)
                pctl_80 = np.percentile(RCM, 80)
                pctl_90 = np.percentile(RCM, 90)
                pctl_95 = np.percentile(RCM, 95)
                pctl_99 = np.percentile(RCM, 99)
                threshold_list = [pctl_70, pctl_80, pctl_90, pctl_95, pctl_99]
                performance_list = []
                
                for threshold in threshold_list:
                    # threshold = pctl_80
                    
                    ##### y_pred
                    root_causes_pred = RCM > threshold
                    root_causes_pred = root_causes_pred.astype(int)
                    root_causes_pred_1d = root_causes_pred.flatten()
                    
                    
                    len(root_causes_true_1d)
                    len(root_causes_pred_1d)
                    
                    ##### get F1 socres
                    tn, fp, fn, tp =  confusion_matrix(root_causes_true_1d, root_causes_pred_1d).ravel() #, sample_weight=None, normalize=None, labels=["normal","root causes"
                    # print(confusion_matrix(root_causes_true_1d, root_causes_pred_1d))
                    
                    acc = (tp+tn) / (tn+fp+fn+tp)
                    recall = tp / (tp+fn) # = sensitivity, TPR
                    precision = tp / (tp+fp)
                    f1 = (2*recall*precision) / (recall+precision)
                    IOU = tp / (tp+fp+fn) # IOU = jaccard_score(root_causes_true_1d, root_causes_pred_1d)
        
                    acc = np.round(acc, 3)
                    recall = np.round(recall, 3)
                    precision = np.round(precision, 3)
                    f1 = np.round(f1, 3)
                    IOU = np.round(IOU, 3)
        
                    performance = [acc, recall, precision, f1, IOU]
                    # print(performance)
                    
                    performance_list.append(performance)
                    print('dataset_{}_{}_{}, acc:{},recall:{},precision:{},f1:{},IOU:{}'.format(idx+1, tmp_model, n_block, acc, recall, precision, f1, IOU))
                performance_results= pd.DataFrame(performance_list, index = [70, 80, 90, 95, 99], columns=["acc", "recall", "precision", "f1", "IOU"])
                performance_results = performance_results.fillna('-')
                performance_results.to_csv(path_results_performance+'{}/'.format(tmp_model)+f'dataset_{idx+1}_{tmp_model}_{n_block}_{RCM_name}_auc_{auc}.csv')
