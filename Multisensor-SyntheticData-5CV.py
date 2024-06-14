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

from math import sqrt
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold, StratifiedKFold

from numpy.random import seed
import tensorflow as tf
tf.random.set_seed(2017010500)
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
GPU_fraction = 0.3
config = ConfigProto(device_count = {'GPU': 1})
config.gpu_options.per_process_gpu_memory_fraction = GPU_fraction
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# session.close()

######################################## Define Paths & Load synthetic data
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


data_list = os.listdir(path_synthetic)
n_dataset = int(len(data_list)/2)
############################################################################## 
from utils.QDL import VisHistory, RAM, getRAMs, getRCM, getRCM_minmax, getRCM_standard, VisRAMs, VisRCM_2D, VisRCM_3D
from models.deconvnet import reg_cae_upsampled, reg_cae_transposed
from models.segnet import reg_segnet
from models.fcn import reg_fcn_vgg, reg_fcn_densenet
from models.unet import reg_unet

model_list = [reg_segnet] #reg_fcn_densenet, reg_cae_transposed, reg_unet, reg_cae_upsampled
normalization = True

for tmp_model in model_list:
    # tmp_model = reg_cae_transposed
    results_cv = pd.DataFrame([],index=range(0,n_dataset), columns=['R2_mean','R2_std','MAE_mean','MAE_std','RMSE_mean','RMSE_std'])
    ########################################
    for idx in range(0, n_dataset):
        #idx=0
        # load
        with open(path_synthetic+data_list[2*idx], 'rb') as f:
            dataset = pickle.load(f)
        with open(path_synthetic+data_list[2*idx+1], 'rb') as f:
            y = pickle.load(f)
        f.close()
        
        ################################################################################ Modeling
        
        #################### Hyperparameter
        n_epochs = 100
        n_batch = 8 
        n_block = 1
        #################### Training
        # dataset_input = dataset.reshape(dataset.shape[0],dataset.shape[1],dataset.shape[2],1).astype(float)
        # input_shape = dataset_input.shape[1:]
        # kfold = StratifiedKFold(n_splits=5, shuffle=False)
        # kfold = KFold(n_splits=5, shuffle=True, random_state=2017010500)
        
        ts_r2 = []
        ts_mae = []
        ts_rmse = []
        
        dat_X = dataset.copy()

        n_bin = int(y.max()/10)

        range_bin = 50
        bins = np.array(range(int(y.min()),int(y.max()),range_bin))
        y_bins = np.digitize(y, bins)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017010500)
        # train, xtest, ytrain, ytest = train_test_split(dat_X, y, test_size=1/5,  random_state=85, stratify=y_bins)

        for idx_train, idx_test in kfold.split(dat_X,y_bins):
            
            if normalization==True:
                dat_X_train, dat_X_test, y_train, y_test = dat_X[idx_train], dat_X[idx_test], y[idx_train], y[idx_test]
                print("dat_X_train.shape", dat_X_train.shape, "dat_X_test.shape", dat_X_test.shape)

                dat_X_train_n = dat_X_train.transpose(0,2,1)
                dat_X_test_n = dat_X_test.transpose(0,2,1)
                print("dat_X_train_n", dat_X_train_n.shape, "dat_X_test_n", dat_X_test_n.shape)
                
                # scailing X_train / X_test
                dat_X_train_normshape = dat_X_train_n.reshape(-1,dat_X_train_n.shape[2])
                dat_X_test_normshape = dat_X_test_n.reshape(-1,dat_X_test_n.shape[2])
                
                print("dat_X_train_normshape.shape", dat_X_train_normshape.shape)
                print("dat_X_test_normshape.shape", dat_X_test_normshape.shape)
                
                scaler_standard = StandardScaler()
                scaler_standard.fit(dat_X_train_normshape)
                dat_X_train_norm = scaler_standard.transform(dat_X_train_normshape)
                dat_X_test_norm = scaler_standard.transform(dat_X_test_normshape)
        
                print("dat_X_train_norm.shape", dat_X_train_norm.shape)
                print("dat_X_test_norm.shape", dat_X_test_norm.shape)
                
    
                X_train = dat_X_train_norm.reshape(dat_X_train_n.shape[0],dat_X_train_n.shape[1],dat_X_train_n.shape[2],1).astype(float)
                X_train = np.transpose(X_train, axes=(0, 2, 1, 3)) #(# observation, sensor ID, process time)
                
                X_test = dat_X_test_norm.reshape(dat_X_test_n.shape[0],dat_X_test_n.shape[1],dat_X_test_n.shape[2],1).astype(float)
                X_test = np.transpose(X_test, axes=(0, 2, 1, 3)) #(# observation, sensor ID, process time)
        
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 2020)
    
                print("X_train.shape: ", X_train.shape)
                print("X_val.shape: ", X_val.shape)
                print("X_test.shape: ", X_test.shape)
                
            else:
                dat_X_train, dat_X_test, y_train, y_test = dat_X[idx_train], dat_X[idx_test], y[idx_train], y[idx_test]
                X_train = np.transpose(np.expand_dims(dat_X_train, axis=1), axes=(0, 2, 3, 1))
                X_test = np.transpose(np.expand_dims(dat_X_test, axis=1), axes=(0, 2, 3, 1))

                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 2020)

                print("X_train.shape", X_train.shape, "X_test.shape", X_test.shape)                
            
            # ############################## Visualization
            # n_process = 30
            # root_causes = root_cause_list[0]
            # process_ID = []
            # for p in range(30):
            #     process_ID.append('Process {}'.format(p+1))
            # n_causal_proc = len(root_causes)
            # n_causal_time = int(root_causes[0][2]-root_causes[0][1])
            # i=0
            # X_train[0,:,:,0].shape
            # tmp_data = pd.DataFrame(X_train[0,:,:,0].T, columns=process_ID)
            # tmp_y = y[i]
            # defect = tmp_y
            # fig, axes = plt.subplots(n_process,1, figsize=(15,30), sharex=True)
            # axes[0].set_title("Defect: {}".format(defect))
            # tmp_data[process_ID].plot(subplots=True, ax=axes, ylim=(-10,10), rot=0)
            # for ax in axes:
            #     for rc in root_causes:
            #         ax.legend(loc="upper right")
            #         axes[rc[0]].axvspan(rc[1], rc[2], color=sns.xkcd_rgb['red'], alpha=0.09)
            # plt.show()
            ############################################################
            
            input_shape = X_train.shape[1:]
            ##### Training
            K.set_learning_phase(1) #set learning phase 1: training, 0: testing
            es = EarlyStopping(monitor='val_loss', mode='auto', patience=20, restore_best_weights=True)
            
            model = tmp_model(input_shape, n_block)
            model.fit(x=X_train, y=y_train, 
                      epochs=n_epochs, 
                      batch_size=n_batch,
                      validation_data=(X_val, y_val),
                      callbacks=[es])
            
            ##### Testing
            K.set_learning_phase(0) #set learning phase 1: training, 0: testing
    
            y_pred_vl = model.predict(X_test)
    
            ts_tmp_r2 = np.round(metrics.r2_score(y_test, y_pred_vl),2)
            ts_r2.append(ts_tmp_r2)
            ts_tmp_mae = np.round(metrics.mean_absolute_error(y_test,y_pred_vl),2)
            ts_mae.append(ts_tmp_mae)
            ts_tmp_rmse = np.round(sqrt(metrics.mean_squared_error(y_test,y_pred_vl)),2)
            ts_rmse.append(ts_tmp_rmse)
            print("ts_tmp_r2, ts_tmp_mae,ts_tmp_rmse: ", ts_tmp_r2, ts_tmp_mae,ts_tmp_rmse)
            
            K.clear_session()

        
        r2_mean, r2_std = np.mean(ts_r2).round(2), np.std(ts_r2).round(2)
        mae_mean, mae_std = np.mean(ts_mae).round(2), np.std(ts_mae).round(2)
        rmse_mean, rmse_std = np.mean(ts_rmse).round(2), np.std(ts_rmse).round(2)
            
        # save the results of every cross-validation fold
        results_cv.iloc[idx] = np.array([r2_mean, r2_std, mae_mean, mae_std, rmse_mean, rmse_std])
    
    # 예측성능 엑셀파일 저장하기 
    n_scenario = data_list[2*idx].split('_')[1]
    n_scenario_sub = data_list[2*idx].split('_')[2]
    writer = pd.ExcelWriter(path_results_prediction+'dataset_{}.xlsx'.format(model.name))
    results_cv.to_excel(writer,'results')
    writer.save()

plt.figure(figsize=(6,6))
plt.scatter(y_test,y_pred_vl)
plt.xlim(0,100)
plt.ylim(0,100)