# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:52:00 2020

@author: yscho

# http://www.blackarbs.com/blog/advanced-time-series-plots-in-python/1/6/2017

"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from numpy.random import seed
import tensorflow as tf
tf.random.set_seed(2017010500)
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

######################################## Path
path_proj = 'D:/RESEARCH/Multisensor-RCA-QDL/'
path_code = path_proj+'code/'
path_data = path_proj+'data/'
os.chdir(path_code)
import datetime
GPU_fraction = 0.5
config = ConfigProto(device_count = {'GPU': 1})
config.gpu_options.per_process_gpu_memory_fraction = GPU_fraction
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#################################################################
from utils.scenarios import RootCauses_200912
yymmdd, dataset_shape_list, root_cause_list = RootCauses_200912()
#################################################################

path_synthetic = path_data+'synthetic/{}/'.format(yymmdd)
path_results_prediction = path_proj+'results/synthetic/{}/prediction/'.format(yymmdd)
path_results_model = path_proj+'results/synthetic/{}/model/'.format(yymmdd)
path_results_rcm = path_proj+'results/synthetic/{}/RCM/'.format(yymmdd)

data_list = os.listdir(path_synthetic)
n_dataset = int(len(data_list)/2)

########################################################################################################################
from utils.scenarios import RootCauses_200912
from utils.QDL import VisHistory, RAM, getRAMs, getRCM, getRCM_minmax, getRCM_standard, VisRAMs, VisRCM_2D, VisRCM_3D
from models.deconvnet import reg_cae_upsampled, reg_cae_transposed
from models.segnet import reg_segnet
from models.fcn import reg_fcn_vgg, reg_fcn_densenet
from models.unet import reg_unet

model_list = [reg_segnet]#reg_fcn_vgg,reg_cae_upsampled, reg_segnet, reg_fcn_vgg, reg_unet, reg_cae_transposed, reg_unet

n_dataset = len(root_cause_list)
normalization=True
for tmp_model in model_list:
    # tmp_model = reg_unet
    ########################################
    for idx in range(0, n_dataset):
        #idx=0
        
        # for idx in range(0, 3):
        # load
        with open(path_synthetic+data_list[2*idx], 'rb') as f:
            dataset = pickle.load(f)
        with open(path_synthetic+data_list[2*idx+1], 'rb') as f:
            y = pickle.load(f)
        f.close()
        
        ################################################################################ Modeling
        
        
        #################### Hyperparameter
        n_epochs = dataset.shape[2] #100
        n_batch = 8 
        
        #################### Training
        if normalization==True:
            dataset_n = dataset.transpose(0,2,1)            
            print("dataset_n", dataset_n.shape)
            
            # scailing X_train / X_test
            dataset_normshape = dataset_n.reshape(-1,dataset_n.shape[2])
            print("dataset_normshape.shape", dataset_normshape.shape)
            
            scaler_standard = StandardScaler()
            scaler_standard.fit(dataset_normshape)
            dataset_norm = scaler_standard.transform(dataset_normshape)
            print("dataset_norm.shape", dataset_norm.shape)

            X = dataset_norm.reshape(dataset_n.shape[0],dataset_n.shape[1],dataset_n.shape[2],1).astype(float)
            X = np.transpose(X, axes=(0, 2, 1, 3)) #(# observation, sensor ID, process time)
            
            print("X.shape: ", X.shape)
            input_shape = X.shape[1:]
            
        else:
            X = dataset.reshape(dataset.shape[0],dataset.shape[1],dataset.shape[2],1).astype(float)
            input_shape = X.shape[1:]

            print("X.shape", X.shape)
            
        K.set_learning_phase(1) #set learning phase 1: training, 0: testing
        # model = reg_cae_upsampled(input_shape)
        model = tmp_model(input_shape, 1)
        es = EarlyStopping(monitor='val_loss', mode='auto', patience=20, restore_best_weights=True)
        hist_reg = model.fit(x=X, y=y, 
                             epochs=n_epochs,
                             batch_size=n_batch,
                             validation_data=(X, y),callbacks=[es])
        
        n_scenario = data_list[2*idx].split('_')[1]
        n_scenario_sub = data_list[2*idx].split('_')[2]

        if not os.path.exists(path_results_model+'{}/'.format(model.name)):
            os.makedirs(path_results_model+'{}/'.format(model.name))
        model.save(path_results_model+'{}/'.format(model.name)+'dataset_{}_{}_{}/'.format(n_scenario, n_scenario_sub, model.name))
        hist_reg.model.summary()
        VisHistory(hist_reg)
        
        #################### Testing
        K.set_learning_phase(0) #set learning phase 1: training, 0: testing
        y_pred = model.predict(X)
        print("MAE: ", np.mean(np.abs(y-y_pred)))
        defect_rates = pd.DataFrame(y_pred, columns=['defect rates'])
        defect_rates_sorted = defect_rates.sort_values(by=['defect rates'])
        
        ################################################################################ QDL
        RAMs = getRAMs(X, defect_rates_sorted, hist_reg.model)
        RCM = getRCM(RAMs[:-1], defect_rates_sorted) 
        RCM_minmax = getRCM_minmax(RAMs, defect_rates_sorted)
        RCM_standard = getRCM_standard(RAMs, defect_rates_sorted)
        
        ##### 2-dimensions Visualization: RAM & RCM
        # save
        if not os.path.exists(path_results_rcm+'{}/'.format(model.name)):
            os.makedirs(path_results_rcm+'{}/'.format(model.name))
        VisRAMs(RAMs, defect_rates_sorted, "normal", 10)
        VisRAMs(RAMs, defect_rates_sorted, "abnormal", 10)
        VisRCM_2D(RCM, path_results_rcm+'{}/'.format(model.name)+'dataset_{}_{}_{}_RCM_2D.png'.format(n_scenario, n_scenario_sub, model.name))
        VisRCM_3D(RCM, path_results_rcm+'{}/'.format(model.name)+'dataset_{}_{}_{}_RCM_3D.png'.format(n_scenario, n_scenario_sub, model.name))
        VisRCM_3D(RCM_minmax, path_results_rcm+'{}/'.format(model.name)+'dataset_{}_{}_{}_RCM_minmax_3D.png'.format(n_scenario, n_scenario_sub, model.name))
        VisRCM_3D(RCM_standard, path_results_rcm+'{}/'.format(model.name)+'dataset_{}_{}_{}_RCM_standard_3D.png'.format(n_scenario, n_scenario_sub, model.name))
        
        ####################################### Save RCMs
        with open(path_results_rcm+'{}/'.format(model.name)+'dataset_{}_{}_{}_RAMs.pkl'.format(n_scenario, n_scenario_sub, model.name), 'wb') as f:
            pickle.dump(RAMs, f)
        with open(path_results_rcm+'{}/'.format(model.name)+'dataset_{}_{}_{}_RCM.pkl'.format(n_scenario, n_scenario_sub, model.name), 'wb') as f:
            pickle.dump(RCM, f)
        with open(path_results_rcm+'{}/'.format(model.name)+'dataset_{}_{}_{}_RCM_minmax.pkl'.format(n_scenario, n_scenario_sub, model.name), 'wb') as f:
            pickle.dump(RCM_minmax, f)
        with open(path_results_rcm+'{}/'.format(model.name)+'dataset_{}_{}_{}_RCM_standard.pkl'.format(n_scenario, n_scenario_sub, model.name), 'wb') as f:
            pickle.dump(RCM_standard, f)
        f.close()
        
        K.clear_session()

    
    
        ####################################### Localization evaluation
        # root_causes = root_cause_list[idx]
        
        # n_root_causes = len(root_causes)
        # root_causes_proc = []
        # root_causes_time = []
        # for r in range(n_root_causes):
        #     # r=0
        #     root_causes_proc.append(root_causes[r][0])
        #     root_causes_time.append(list(range(root_causes[0][1],root_causes[0][2])))
        
        # pctl_70 = np.percentile(RCM, 70)
        # pctl_80 = np.percentile(RCM, 80)
        # pctl_90 = np.percentile(RCM, 90)
        # pctl_95 = np.percentile(RCM, 95)
        # pctl_99 = np.percentile(RCM, 99)
        
        # threshold_list = [pctl_70, pctl_80, pctl_90, pctl_95, pctl_99]
        # threshold = pctl_95
        # TPR_list = []
        # for threshold in threshold_list:
        #     from utils.QDL import getRootCause, getGround, getRecall
        #     root_causes_proc_pred, root_causes_time_pred = getRootCause(RCM, threshold)
                
        #     ground_actual = getGround(dataset.shape, root_causes_proc, root_causes_time)
        #     ground_pred = getGround(dataset.shape, root_causes_proc_pred, root_causes_time_pred)
        #     ground_combined = ground_actual+ground_pred
        #     TPR = getRecall(ground_actual, ground_combined)
        #     TPR_list.append(TPR)
        #     print('TPR_{}_{}_{}: '.format(n_scenario, n_scenario_sub, model.name), TPR)
        # Recall = pd.DataFrame(TPR_list, index = [70, 80, 90, 95, 99], columns=["TPR"])
        # Recall.to_csv(path_results_rcm+'dataset_{}_{}_{}_recall.csv'.format(n_scenario, n_scenario_sub, model.name))

        # session.close()
    

