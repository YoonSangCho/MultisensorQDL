# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:16:48 2020

@author: yscho
"""
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt

# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.layers import Reshape, Input, Dense, Activation, Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, UpSampling2D, Conv2DTranspose
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import Model, model_from_json
# from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
# from tensorflow.keras import backend as K


def VisHistory(history):
    # summarize history for loss
    plt.figure(figsize=(6,6))
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


def cropping(decoder, input_shape):
    if decoder.shape[1:-1] != input_shape[:-1]:
        crop_height = decoder.shape[1] - input_shape[0] # sensor ID
        crop_width = decoder.shape[2] - input_shape[1] # process time
        crop = Cropping2D(cropping=((crop_height, 0), (crop_width, 0)))(decoder)
    else:
        crop = decoder
    # assert int(crop.shape[1:-1] == input_shape[:-1]) is 1, "INPUT SHAPE is NOT equal to OUTPUT SHAPE"
    return crop

def RAM(model_ram, image):

    #original_img = X[0]
    original_img = image
    process_time, process_sensor_ID, _ = original_img.shape
    
    img = original_img.reshape(1,original_img.shape[0],original_img.shape[1],original_img.shape[2])
    
    # get layer            
    input_layer = model_ram.get_layer('input')
    #final_conv_layer = model_ram.get_layer('decoder_3_acv_2') # final conv layer (featuremaps before GAP)
    final_conv_layer = model_ram.get_layer('final_conv_layer') # final conv layer (featuremaps before GAP)
    
    dense_layer = model_ram.get_layer('dense')
    
    response_weights = dense_layer.get_weights()[0] # dense layer's weight 
    
    get_output_conv = K.function([input_layer.input], [final_conv_layer.output]) #, dense_layer.output
    outputs_conv = get_output_conv([img])
    conv_outputs = outputs_conv[0]
    
    # Create the response activation map.
    ram = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
    assert int(conv_outputs.shape[3] == response_weights.shape[0]) == 1, "# of featuremaps is not equal to # of reponse weights"
    for w in range(response_weights.shape[0]): ##### index of featuremaps = index of weight for each featuremap 
        #w = 0
        tmp = conv_outputs[0,:,:,w] * np.abs(response_weights[w,0])
        ram += tmp
    # ram normalization
    ram_min = np.min(ram) 
    ram_max = np.max(ram)
    ram = (ram - ram_min)/(ram_max - ram_min)
    return ram

def getRAMs(dataset_input, y, model_ram):
    #################### RAM
    RAM_list = []
    for i in range(dataset_input.shape[0]):
        # i=0
        # get RAM
        # tmp_idx = defect_rates_sorted.iloc[i].name
        tmp_X = dataset_input[i].copy()
        
        # RAM(model_ram, tmp_X).shape
        tmp_ram = RAM(model_ram, tmp_X) # .T T is for transpose
        RAM_list.append(tmp_ram)
        print("{}th RAM is done".format(i) )
    RAMs = np.stack(RAM_list)
    print("RAMs.shape: ", RAMs.shape)
    return RAMs

def getRCM(RAMs, y):
    # Get RCM(root cause map) of each observation in ascending order of y
    RCM = np.zeros(dtype = np.float32, shape = RAMs.shape[1:])
    for c in range(len(RAMs)):
        #c=0
        tmp = RAMs[c,:,:]
        tmp_y_hat = y[c]
        """ We need to consider that normalize the 'tmp_y_hat' to 0~1 """
        RCM+=tmp*tmp_y_hat
    
    print("RCM.shape: ", RCM.shape)
    RCM_max = np.max(RCM)
    RCM_min = np.min(RCM)
    RCM_norm = (RCM - RCM_min)/(RCM_max-RCM_min)
    print("RCM_norm.shape: ", RCM_norm.shape)
    return RCM_norm

def getRCM_standard(RAMs, y):
    
    scaler = StandardScaler()
    y_norm = scaler.fit_transform(y.reshape(y.shape[0],1))
    
    # Get RCM(root cause map) of each observation in ascending order of y
    RCM = np.zeros(dtype = np.float32, shape = RAMs.shape[1:])
    for c in range(len(RAMs)):
        #c=0
        tmp = RAMs[c,:,:]
        tmp_y_hat = y_norm[c][0]
        """ We need to consider that normalize the 'tmp_y_hat' to 0~1 """
        RCM+=tmp*tmp_y_hat
    
    print("RCM.shape: ", RCM.shape)
    RCM_max = np.max(RCM)
    RCM_min = np.min(RCM)
    RCM_norm = (RCM - RCM_min)/(RCM_max-RCM_min)
    print("RCM_norm.shape: ", RCM_norm.shape)
    return RCM_norm

def getRCM_minmax(RAMs, y):
    scaler = MinMaxScaler()
    y_norm = scaler.fit_transform(y.reshape(y.shape[0],1))
    
    # Get RCM(root cause map) of each observation in ascending order of y
    RCM = np.zeros(dtype = np.float32, shape = RAMs.shape[1:])
    for c in range(len(RAMs)):
        #c=0
        tmp = RAMs[c,:,:]
        tmp_y_hat = y_norm[c][0]
        """ We need to consider that normalize the 'tmp_y_hat' to 0~1 """
        RCM+=tmp*tmp_y_hat
    
    print("RCM.shape: ", RCM.shape)
    RCM_max = np.max(RCM)
    RCM_min = np.min(RCM)
    RCM_norm = (RCM - RCM_min)/(RCM_max-RCM_min)
    print("RCM_norm.shape: ", RCM_norm.shape)
    return RCM_norm

def VisRAMs(RAMs, y, groub, nb_ram):
    y_sort = y.copy()
    y_sort.sort()
    range_groub = range(0, nb_ram)
    
    if groub == "normal":
        quality = np.unique(y_sort)[:nb_ram]
    else:
        quality = np.unique(y_sort)[-nb_ram:]

    for i in range_groub:
        # i=0
        s = np.where(y == quality[i])[0][0]
        print("{}: {}".format(groub, y[s]))
        plt.figure(figsize=(16,8))
        plt.title(y[s])
        sns.heatmap(RAMs[s], cmap='coolwarm')
        plt.show()

def VisRCM_2D(RCM, path):
    ##### 2-dimensions Visualization: RCM
    plt.figure(figsize=(16,8))
    sns.heatmap(RCM, cmap='coolwarm')
    plt.savefig(path, bbox_inches = 'tight', pad_inches = 0.05,dpi=1000)
    plt.show()

def VisRCM_3D(RCM, path):
    ##### 3-dimensions Visualization: RCM
    x_axis = np.repeat(np.array(list(range(0,RCM.shape[0]))), RCM.shape[1]).reshape(RCM.shape[0], RCM.shape[1])
    y_axis = np.repeat(np.array(list(range(0,RCM.shape[1]))), RCM.shape[0]).reshape(RCM.shape[1], RCM.shape[0])
    z_axis = RCM
    from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting     
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_axis, y_axis.T, z_axis, cmap='coolwarm', edgecolor='none')#'viridis'
    ax.set_title('RCM')
    ax.set_xlabel('process_ID')
    ax.set_ylabel('processing_time')
    ax.set_zlabel('score')
    plt.savefig(path, bbox_inches = 'tight', pad_inches = 0.05,dpi=1000)
    plt.show()


# def getRAMs(dataset_input, defect_rates_sorted, model_ram):
#     ######################## RAM
#     RAM_list = []
#     for i in range(dataset_input.shape[0]):
#         # i=0
#         # get RAM
#         tmp_idx = defect_rates_sorted.iloc[i].name
#         tmp_X = dataset_input[tmp_idx].copy()
        
#         # RAM(model_ram, tmp_X).shape
#         tmp_ram = RAM(model_ram, tmp_X) # .T T is for transpose
#         RAM_list.append(tmp_ram)
#         print("{}th RAM is done".format(i) )
#     RAMs = np.stack(RAM_list)
#     print("RAMs.shape: ", RAMs.shape)
#     return RAMs

# def getRCM(RAMs, defect_rates_sorted):
    
#    # Get RCM(root cause map) of each observation in ascending order of y
#     RCM = np.zeros(dtype = np.float32, shape = RAMs.shape[1:])
#     for c in range(len(RAMs)):
#         #c=0
#         tmp = RAMs[c,:,:]
#         tmp_y_hat = defect_rates_sorted.iloc[c].values[0]
#         """ We need to consider that normalize the 'tmp_y_hat' to 0~1 """
#         RCM+=tmp*tmp_y_hat
    
#     print("RCM.shape: ", RCM.shape)
#     RCM_max = np.max(RCM)
#     RCM_min = np.min(RCM)
#     RCM_norm = (RCM - RCM_min)/(RCM_max-RCM_min)
#     print("RCM_norm.shape: ", RCM_norm.shape)
#     return RCM_norm

# def getRCM_standard(RAMs, defect_rates_sorted):
    
#     scaler = StandardScaler()
#     y_norm_sort = scaler.fit_transform(defect_rates_sorted.values)
    
#     # Get RCM(root cause map) of each observation in ascending order of y
#     RCM = np.zeros(dtype = np.float32, shape = RAMs.shape[1:])
#     for c in range(len(RAMs)):
#         #c=0
#         tmp = RAMs[c,:,:]
#         #tmp_y_hat = defect_rates_sorted.iloc[c].values[0] #*defect_rates_sorted.iloc[c].values[0]
#         tmp_y_hat = y_norm_sort[c][0] #*defect_rates_sorted.iloc[c].values[0]
#         """ We need to consider that normalize the 'tmp_y_hat' to 0~1 """
#         RCM+=tmp*tmp_y_hat
    
#     print("RCM.shape: ", RCM.shape)
#     RCM_max = np.max(RCM)
#     RCM_min = np.min(RCM)
#     RCM_norm = (RCM - RCM_min)/(RCM_max-RCM_min)
#     print("RCM_norm.shape: ", RCM_norm.shape)
#     return RCM_norm

# def getRCM_minmax(RAMs, defect_rates_sorted):
    
    
#     scaler = MinMaxScaler()
#     y_norm_sort = scaler.fit_transform(defect_rates_sorted.values)
    
#     # Get RCM(root cause map) of each observation in ascending order of y
#     RCM = np.zeros(dtype = np.float32, shape = RAMs.shape[1:])
#     for c in range(len(RAMs)):
#         #c=0
#         tmp = RAMs[c,:,:]
#         #tmp_y_hat = defect_rates_sorted.iloc[c].values[0] #*defect_rates_sorted.iloc[c].values[0]
#         tmp_y_hat = y_norm_sort[c][0] #*defect_rates_sorted.iloc[c].values[0]
#         """ We need to consider that normalize the 'tmp_y_hat' to 0~1 """
#         RCM+=tmp*tmp_y_hat
    
#     print("RCM.shape: ", RCM.shape)
#     RCM_max = np.max(RCM)
#     RCM_min = np.min(RCM)
#     RCM_norm = (RCM - RCM_min)/(RCM_max-RCM_min)
#     print("RCM_norm.shape: ", RCM_norm.shape)
#     return RCM_norm

# def VisRAMs(RAMs, defect_rates_sorted, groub, nb_ram):
#     if groub == "normal":
#         range_groub = range(0, nb_ram)
#     else:
#         range_groub = range(len(defect_rates_sorted)-nb_ram, len(defect_rates_sorted))

#     for i in range_groub:
#         # i=0
#         print("{}: {}".format(groub, defect_rates_sorted.iloc[i].values))
#         plt.figure(figsize=(16,8))
#         plt.title(defect_rates_sorted.iloc[i].values)
#         sns.heatmap(RAMs[i], cmap='coolwarm')
#         plt.show()

# def getRootCause(RCM, threshold):
#     root_causes_proc_pred = []
#     root_causes_time_pred = []
#     causes_count = []
#     for p in range(RCM.shape[0]):
#         #p=0
#         tmp_proc_cause = RCM[p, :] > threshold
#         tmp_proc_count = sum(tmp_proc_cause)
#         causes_count.append(tmp_proc_count)
#         if sum(tmp_proc_cause)==0:
#             pass
#         else:
#             tmp_time_axis = np.where(tmp_proc_cause ==True)[0].tolist()
#             root_causes_proc_pred.append(p)
#             root_causes_time_pred.append(tmp_time_axis)
#     # print("root_causes_axis_proc:", root_causes_proc_pred)
#     # print("root_causes_axis_time:", root_causes_time_pred)
#     return root_causes_proc_pred, root_causes_time_pred


# def getGround(shape, root_causes_proc, root_causes_time):
#     gound = np.zeros((shape[1], shape[2]))
#     for p in range(len(root_causes_proc)):
#         for t in range(len(root_causes_time)):
#             gound[root_causes_proc[p], root_causes_time[t]] = 1
#     return gound
# def getRecall(ground_actual,ground_combined):
#     n_gound_actual = sum(sum(ground_actual))
#     n_gound_TP = len(np.where(ground_combined==2)[0])
#     TPR = np.round(n_gound_TP/n_gound_actual, 2)
#     return TPR

    





################################################################################################################################
# import cv2 # conda install -c conda-forge opencv

# def generate_grad_cam(img_tensor, model, class_index, activation_layer):
#     """
#     params:
#     -------
#     img_tensor: resnet50 모델의 이미지 전처리를 통한 image tensor
#     model: pretrained resnet50 모델 (include_top=True)
#     class_index: 이미지넷 정답 레이블
#     activation_layer: 시각화하려는 레이어 이름

#     return:
#     grad_cam: grad_cam 히트맵
#     """
#     inp = model.input
#     y_c = model.output.op.inputs[0][0, class_index]
#     A_k = model.get_layer(activation_layer).output
    
#     ## 이미지 텐서를 입력해서
#     ## 해당 액티베이션 레이어의 아웃풋(a_k)과
#     ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
#     get_output = K.function([inp], [A_k, K.gradients(y_c, A_k)[0], model.output])
#     [conv_output, grad_val, model_output] = get_output([img_tensor])

#     ## 배치 사이즈가 1이므로 배치 차원을 없앤다.
#     conv_output = conv_output[0]
#     grad_val = grad_val[0]
    
#     ## 구한 gradient를 픽셀 가로세로로 평균내서 a^c_k를 구한다.
#     weights = np.mean(grad_val, axis=(0, 1))
    
#     ## 추출한 conv_output에 weight를 곱하고 합하여 grad_cam을 얻는다.
#     grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
#     for k, w in enumerate(weights):
#         grad_cam += w * conv_output[:, :, k]
    
#     grad_cam = cv2.resize(grad_cam, (224, 224))

#     ## ReLU를 씌워 음수를 0으로 만든다.
#     grad_cam = np.maximum(grad_cam, 0)

#     grad_cam = grad_cam / grad_cam.max()
#     return grad_cam


# def generate_cam(img_tensor, model, class_index, activation_layer):
#     """
#     params:
#     -------
#     img_tensor: resnet50 모델의 이미지 전처리를 통한 image tensor
#     model: pretrained resnet50 모델 (include_top=True)
#     class_index: 이미지넷 정답 레이블
#     activation_layer: 시각화하려는 레이어 이름

#     return:
#     cam: cam 히트맵
#     """
#     inp = model.input
#     A_k = model.get_layer(activation_layer).output
#     outp = model.layers[-1].output

#     ## 이미지 텐서를 입력해서
#     ## 해당 액티베이션 레이어의 아웃풋(a_k)과
#     ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
#     get_output = K.function([inp], [A_k, outp])
#     [conv_output, predictions] = get_output([img_tensor])
    
#     ## 배치 사이즈가 1이므로 배치 차원을 없앤다.
#     conv_output = conv_output[0]
    
#     ## 마지막 소프트맥스 레이어의 웨이트 매트릭스에서
#     ## 지정한 레이블에 해당하는 횡벡터만 가져온다.
#     weights = model.layers[-1].get_weights()[0][:, class_index]
    
#     ## 추출한 conv_output에 weight를 곱하고 합하여 cam을 얻는다.
#     cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
#     for k, w in enumerate(weights):
#         cam += w * conv_output[:, :, k]
    
#     cam = cv2.resize(cam, (224, 224))
#     cam = cam / cam.max()
    
#     return cam
