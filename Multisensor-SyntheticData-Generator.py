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

 
#################### Defining funtions for generateSynthetic
def generateSynthetic(root_causes_list, n_observations, n_process, n_time, n_scenario):
    # n_observations = 1000
    # n_process = 30
    # n_time = 100
    # root_causes_list = root_cause_list[0]
    
    dataset_list = []
    y_list = []
    process_ID = []
    for p in range(n_process):
        process_ID.append('Process {}'.format(p+1))
    
    seed = 2017010500
    np.random.seed(seed)
    y = np.random.normal(60, 20, size=1000).astype(int)
    plt.figure(figsize=(6,6))
    plt.hist(y)

    for s, root_causes in enumerate(root_causes_list):
        # root_causes = root_causes_list
        n_causal_proc = len(root_causes)
        n_causal_time = int(root_causes[0][2]-root_causes[0][1])
        
        ######################################## Dataset
        dataset = np.random.normal(0.0, 0.1, size=(n_observations, n_process, n_time))
        # print("dataset.shape", dataset.shape)
        for i in range(n_observations):
            #i=0
            ############################## Visualization: normal data
            tmp_dat = dataset[i].copy()
            tmp_dat_df = pd.DataFrame(tmp_dat.T, columns=process_ID)
            fig, axes = plt.subplots(n_process, 1, figsize=(20,15), sharex=True)
            axes[0].set_title("Defect: {}".format(0))
            tmp_dat_df[process_ID].plot(subplots=True, ax=axes, ylim=(-1,1), rot=0)
            for ax in axes:
                for rc in root_causes:
                    ax.legend(loc="upper right")
                    axes[rc[0]].axvspan(rc[1], rc[2], color=sns.xkcd_rgb['red'], alpha=0.01)
            plt.show()
            ############################################################
            
            defect = y[i]
            
            ########## abnormal signals for root causes 
            idx_type = int(n_causal_proc/3)
            for idx, rc in enumerate(root_causes):
                # abnormal_type_2 = np.sin(range(0, n_causal_time))+defect
                # dataset[i][rc[0],rc[1]:rc[2]]  = abnormal_type_2
                
                ##### sin funtion for abnormal signals
                # print(idx, rc)
                # rc = root_causes[0]
                '''
                np.sin(range(0, 30))
                plt.plot(np.sin(range(0, 30)))                
                x = np.linspace(-np.pi, np.pi, 30)
                plt.plot(np.sin(x))
                '''
                
                if idx < idx_type*1:
                    abnormal_type_1 = np.sin(range(0, n_causal_time))*defect
                    dataset[i][rc[0],rc[1]:rc[2]] = abnormal_type_1
                    # plt.plot(np.arange(0, len(abnormal_type_1)), abnormal_type_1)
                    # plt.ylim(-defect-2,defect+2)
                    # plt.show()
                    # print("type1")
                elif (idx_type*1 <= idx) and (idx < idx_type*2):
                    abnormal_type_2 = np.sin(range(0, n_causal_time))+defect
                    dataset[i][rc[0],rc[1]:rc[2]]  = abnormal_type_2
                    # plt.plot(np.arange(0, len(abnormal_type_2)), abnormal_type_2)
                    # plt.ylim(0,defect+2)
                    # plt.show()
                    # print("type2")
                else:
                    x = np.linspace(-np.pi, np.pi, n_causal_time)
                    # x = np.linspace(-np.pi, np.pi, 30)
                    # np.sin(range(0,30))
                    abnormal_type_3 = np.sin(x)*defect
                    dataset[i][rc[0],rc[1]:rc[2]] = abnormal_type_3
                    # plt.plot(np.arange(0, len(abnormal_type_3)), abnormal_type_3)
                    # plt.ylim(-defect-2,defect+2)
                    # plt.show()
                    # print("type3")
            '''
            # ##### temporal causes: process
            # tmp_causal_proc_1 = np.random.randint(0, n_process, 1)[0]
            # tmp_causal_proc_2 = np.random.randint(0, n_process, 1)[0]
            # if tmp_causal_proc_2 == tmp_causal_proc_1:
            #     if tmp_causal_proc_2+1 <= n_process and tmp_causal_proc_2 > 0:
            #         tmp_causal_proc_2 = tmp_causal_proc_2-1
            #     elif tmp_causal_proc_2 == 0:
            #         tmp_causal_proc_2+1
            #     else:
            #         tmp_causal_proc_2+1
            # else:
            #     pass
        
            # ##### temporal causes: processing time
            # tmp_causal_time_1 = np.random.randint(0, n_time-n_causal_time, 1)[0]
            # tmp_causal_time_2 = np.random.randint(0, n_time-n_causal_time, 1)[0]
            # # print("tmp_causal_proc_1: ", tmp_causal_time_1, "tmp_causal_proc_2: ", tmp_causal_time_2)
            
            # causes = [[tmp_causal_proc_1, tmp_causal_proc_1, tmp_causal_proc_1+n_causal_time], 
            #           [tmp_causal_proc_2, tmp_causal_proc_2, tmp_causal_proc_2+n_causal_time]] ##### scenario 4

            ########## abnormal signals for temporal causes 
            # for c in causes:
            #     # c = causes[0]
            #     dataset[i][c[0],c[1]:c[2]] = np.sin(range(0, n_causal_time))*defect*0.1
            '''
            
            '''
            ######################################## previous abnormal signals
            ##### normal distribution for abnormal signals
            # for rc in root_causes:
            #     # rc = root_causes[0]
            #     mu_ab, sigma_ab = weight_rc_m*defect, weight_rc_s*defect
            #     abnormal_rand = np.random.normal(mu_ab, sigma_ab, size=n_causal_time)
            #     dataset[i][rc[0],rc[1]:rc[2]] = dataset[i][rc[0],rc[1]:rc[2]] + abnormal_rand
            
            # for c in causes:
            #     # c = causes[0]
            #     tmp_mu_ab, tmp_sigma_ab = weight_tc_m*defect, weight_tc_s*defect
            #     tmp_abnormal_rand = np.random.normal(tmp_mu_ab, tmp_sigma_ab, size=n_causal_time)
            #     dataset[i][c[0],c[1]:c[2]] = dataset[i][c[0],c[1]:c[2]] + tmp_abnormal_rand        
            '''
            
            ############################## Visualization: Abnormal
            tmp_dat_ab_df = pd.DataFrame(dataset[i].T, columns=process_ID)
            fig, axes = plt.subplots(n_process,1, figsize=(20,15), sharex=True)
            axes[0].set_title("Defect: {}".format(defect))
            tmp_dat_ab_df[process_ID].plot(subplots=True, ax=axes, ylim=(-defect*1.5,defect*1.5), rot=0)
            for ax in axes:
                for rc in root_causes:
                    ax.legend(loc="upper right")
                    axes[rc[0]].axvspan(rc[1], rc[2], color=sns.xkcd_rgb['red'], alpha=0.01)
            plt.show()
            ############################################################
        
        ####################################### Save synthetic data
        # save
        with open(path_synthetic+'dataset_{}_{}_X.pkl'.format(n_scenario, (s+1)), 'wb') as f:
            pickle.dump(dataset, f)
        with open(path_synthetic+'dataset_{}_{}_Y.pkl'.format(n_scenario, (s+1)), 'wb') as f:
            pickle.dump(y, f)
        f.close()

        dataset_list.append(dataset)
        y_list.append(y)

        print('dataset_{}_{}_Y.pkl'.format(n_scenario, (s+1)), "completed")
    return dataset_list, y_list

from utils.scenarios import RootCauses_200912
yymmdd, dataset_shape_list, root_cause_list = RootCauses_200912()
print("yymmdd", yymmdd)

######################################## Path
path_proj = 'D:/RESEARCH/Multisensor-RCA-QDL/'
path_code = path_proj+'code/'
os.chdir(path_code)
path_data = path_proj+'data/'

# import datetime
# today = datetime.datetime.today()
# yy = today.strftime('%y')
# mm = today.strftime('%m')
# dd = today.strftime('%d')
# yymmdd = yy+mm+dd

path_synthetic = path_data+'synthetic/{}/'.format(yymmdd)
if not os.path.exists(path_synthetic):
    os.mkdir(path_synthetic)
path_results = path_proj+'results/synthetic/{}/'.format(yymmdd)
if not os.path.exists(path_results):
    os.mkdir(path_results)
path_results_prediction = path_results+'prediction/'
path_results_model = path_results+'model/'
path_results_rcm = path_results+'RCM/'
if not os.path.exists(path_results_prediction):
    os.mkdir(path_results_prediction)
if not os.path.exists(path_results_model):
    os.mkdir(path_results_model)
if not os.path.exists(path_results_rcm):
    os.mkdir(path_results_rcm)

######################################## Synthetic Data Generateion
dataset_list_1, y_list_1 = generateSynthetic(root_cause_list[0:3],
                                             dataset_shape_list[0][0], 
                                             dataset_shape_list[0][1], 
                                             dataset_shape_list[0][2], 
                                             1)

dataset_list_2, y_list_2 = generateSynthetic(root_cause_list[3:6],
                                             dataset_shape_list[1][0], 
                                             dataset_shape_list[1][1], 
                                             dataset_shape_list[1][2], 
                                             2)
dataset_list_3, y_list_3 = generateSynthetic(root_cause_list[6:9],
                                             dataset_shape_list[2][0], 
                                             dataset_shape_list[2][1], 
                                             dataset_shape_list[2][2], 
                                             3)




################################################################################ Defining root cause regions
'''
* 1-1. root cause region + abnormal 정규분포(m = y* 0.1, s = y* 0.1)
* 1-2. temporal cause region + abnormal 정규분포(m = y * 0.05, s = y * 0.05)
    - basically consider: (Root causes process 붙어있는 2개씩),(randomly temporal causes: 서로 다른 공정에서 떨어진 경우)
    - considering 1: (Root causes process 2개씩 붙어 있는 경우)
    - considering 2: (Root causes process 서로 다른 공정에서 떨어져 발생된 경우)
    - considering 3: (Root causes process 서로 같은 공정에서 떨어져 발생된 경우)
'''

################################################################################ 200831
'''
#################### [1000, 30, 100] scenario 1,2,3
# n_observations = 1000
# n_process = 30
# n_time = 100

dataset_shape_1 = (1000, 30, 100)

root_causes_1_1 = [[13, 40, 70],[14, 40, 70],[15, 40, 70]] 
root_causes_1_2 = [[14, 10, 30],[15, 10, 30],[14, 70, 90],[15, 70, 90]] 
root_causes_1_3 = [[3, 10, 40],[4, 10, 40],[27, 60, 90],[28, 60, 90]] 

#################### [1000, 100, 500] scenario 1,2,3
# n_observations = 1000
# n_process = 100
# n_time = 500

dataset_shape_2 = (1000, 100, 500)

root_causes_2_1 = [[50, 100, 200],[51, 100, 200],[52, 100, 200],[53, 100, 200],[54, 100, 200],[55, 100, 200]] 
root_causes_2_2 = [[20, 50, 150],[21, 50, 150],[22, 50, 150],[20, 350, 450],[21, 350, 450],[22, 350, 450]]
root_causes_2_3 = [[20, 50, 150],[21, 50, 150],[22, 50, 150],[80, 350, 450],[81, 350, 450],[82, 350, 450]]

#################### [1000, 300, 1000] scenario 1,2,3
# n_observations = 1000
# n_process = 50
# n_time = 250

dataset_shape_3 = (1000, 50, 250)

root_causes_3_1 = [[10, 100, 200],[11, 100, 200],[12, 100, 200],[13, 100, 200],[14, 100, 200],[15, 100, 200]] 
root_causes_3_2 = [[31, 100, 200],[32, 100, 200],[33, 100, 200],[34, 100, 200],[35, 100, 200],
                   [36, 100, 200],[37, 100, 200],[38, 100, 200],[39, 100, 200],[40, 100, 200]]
root_causes_3_3 = [[5, 0, 50],[6, 0, 50],[7, 0, 50],[8, 200, 250],[9, 200, 250],[10, 200, 250],
                   [40, 0, 50],[41, 0, 50],[42, 0, 50],[43, 200, 250],[44, 200, 250],[45, 200, 250]]

root_cause_list = [root_causes_1_1, root_causes_1_2, root_causes_1_3,
                   root_causes_2_1, root_causes_2_2, root_causes_2_3,
                   root_causes_3_1, root_causes_3_2, root_causes_3_3]

'''
'''
################################################################################ 200906
weight_rc_m, weight_rc_s = 1, 0.01
weight_tc_m, weight_tc_s = 0.1, 0.001

#################### [1000, 30, 100] scenario 1,2,3
# n_observations = 1000
# n_process = 50
# n_time = 100

root_causes_list_1_config = (1000, 50, 100)
root_causes_1_1 = [[6, 40, 70],[7, 40, 70],[8, 40, 70]] 
root_causes_1_2 = [[24, 10, 30],[25, 10, 30],[26, 10, 30],
                   [24, 70, 90],[25, 70, 90],[26, 70, 90]]
root_causes_1_3 = [[1, 0, 50],[2, 0, 50],[3, 0, 50],[4, 0, 50],[5, 0, 50],
                   [45, 50, 100],[46, 50, 100],[47, 50, 100],[48, 50, 100],[49, 50, 100]] 
root_causes_list_1 = [root_causes_1_1, root_causes_1_2, root_causes_1_3]

dataset_list_1, y_list_1 = generateSynthetic(root_causes_list_1,
                                              root_causes_list_1_config[0], 
                                              root_causes_list_1_config[1], 
                                              root_causes_list_1_config[2], 
                                              1,
                                              weight_rc_m, weight_rc_s,
                                              weight_tc_m, weight_tc_s)


#################### [1000, 100, 500] scenario 1,2,3
# n_observations = 1000
# n_process = 100
# n_time = 200
root_causes_list_2_config = (1000, 100, 200)
root_causes_2_1 = [[21, 100, 200],[22, 100, 200],[23, 100, 200],[24, 100, 200],[25, 100, 200]] 
root_causes_2_2 = [[51, 0, 50],[52, 0, 50],[53, 0, 50],[54, 0, 50],[55, 0, 50],
                   [51, 150, 200],[52, 150, 200],[53, 150, 200],[54, 150, 200],[55, 150, 200]]
root_causes_2_3 = [[21, 50, 150],[22, 50, 150],[23, 50, 150],[24, 50, 150],[25, 50, 150],
                   [81, 100, 200],[82, 100, 200],[83, 100, 200],[84, 100, 200],[85, 100, 200]]
root_causes_list_2 = [root_causes_2_1, root_causes_2_2, root_causes_2_3]

dataset_list_2, y_list_2 = generateSynthetic(root_causes_list_2,
                                              root_causes_list_2_config[0], 
                                              root_causes_list_2_config[1], 
                                              root_causes_list_2_config[2], 
                                              2,
                                              weight_rc_m, weight_rc_s,
                                              weight_tc_m, weight_tc_s)

#################### [1000, 300, 1000] scenario 1,2,3

# n_observations = 1000
# n_process = 150
# n_time = 300
root_causes_list_3_config = (1000, 150, 300)
root_causes_3_1 = [[31, 100, 200],[32, 100, 200],[33, 100, 200],[34, 100, 200],[35, 100, 200],
                   [36, 100, 200],[37, 100, 200],[38, 100, 200],[39, 100, 200],[40, 100, 200]]
root_causes_3_2 = [[71, 0, 100],[72, 0, 100],[73, 0, 100],[74, 0, 100],[75, 0, 100],
                   [76, 200, 300],[77, 200, 300],[78, 200, 300],[79, 200, 300],[70, 200, 300]]
root_causes_3_3 = [[6, 50, 100],[7, 50, 100],[8, 50, 100],[9, 50, 100],[10, 50, 100],
                   [51, 200, 250],[52, 200, 250],[53, 200, 250],[54, 200, 250],[55, 200, 250],
                   [101, 50, 100],[102, 50, 100],[103, 50, 100],[104, 50, 100],[105, 50, 100],
                   [141, 200, 250],[142, 200, 250],[143, 200, 250],[144, 200, 250],[145, 200, 250]]
root_causes_list_3 = [root_causes_3_1, root_causes_3_2, root_causes_3_3]

root_cause_list = [root_causes_1_1, root_causes_1_2, root_causes_1_3,
                   root_causes_2_1, root_causes_2_2, root_causes_2_3,
                   root_causes_3_1, root_causes_3_2, root_causes_3_3]

dataset_list_3, y_list_3 = generateSynthetic(root_causes_list_3,
                                              root_causes_list_3_config[0], 
                                              root_causes_list_3_config[1], 
                                              root_causes_list_3_config[2], 
                                              3,
                                              weight_rc_m, weight_rc_s,
                                              weight_tc_m, weight_tc_s)


dataset_shape_list = [root_causes_list_1_config, root_causes_list_2_config, root_causes_list_3_config]

'''

'''

################################################################################ 200911
# type_abnormal  = ["noise", 'fluctuation', "sin"]
# weight_rc_m, weight_rc_s = 1, 0.01
# weight_tc_m, weight_tc_s = 0.1, 0.001

#################### [1000, 30, 100] scenario 1,2,3
# n_observations = 1000
# n_process = 50
# n_time = 100

root_causes_list_1_config = (1000, 50, 100)
root_causes_1_1 = [[6, 40, 70],[7, 40, 70],[8, 40, 70]] 
root_causes_1_2 = [[24, 10, 30],[25, 10, 30],[26, 10, 30],
                   [24, 70, 90],[25, 70, 90],[26, 70, 90]]
root_causes_1_3 = [[0, 0, 50],[1, 0, 50],[2, 0, 50],[3, 0, 50],[4, 0, 50],[5, 0, 50],
                   [44, 50, 100],[45, 50, 100],[46, 50, 100],[47, 50, 100],[48, 50, 100],[49, 50, 100]] 
root_causes_list_1 = [root_causes_1_1, root_causes_1_2, root_causes_1_3]

dataset_list_1, y_list_1 = generateSynthetic(root_causes_list_1,
                                              root_causes_list_1_config[0], 
                                              root_causes_list_1_config[1], 
                                              root_causes_list_1_config[2], 
                                              1)


#################### [1000, 100, 500] scenario 1,2,3
# n_observations = 1000
# n_process = 100
# n_time = 200
root_causes_list_2_config = (1000, 100, 200)
root_causes_2_1 = [[20, 100, 200],[21, 100, 200],[22, 100, 200],[23, 100, 200],[24, 100, 200],[25, 100, 200]] 
root_causes_2_2 = [[50, 0, 50],[51, 0, 50],[52, 0, 50],[53, 0, 50],[54, 0, 50],[55, 0, 50],
                   [50, 150, 200],[51, 150, 200],[51, 150, 200],[52, 150, 200],[53, 150, 200],[54, 150, 200],[55, 150, 200]]
root_causes_2_3 = [[20, 50, 150],[21, 50, 150],[22, 50, 150],[23, 50, 150],[24, 50, 150],[25, 50, 150],
                   [80, 100, 200],[81, 100, 200],[82, 100, 200],[83, 100, 200],[84, 100, 200],[85, 100, 200]]
root_causes_list_2 = [root_causes_2_1, root_causes_2_2, root_causes_2_3]

dataset_list_2, y_list_2 = generateSynthetic(root_causes_list_2,
                                              root_causes_list_2_config[0], 
                                              root_causes_list_2_config[1], 
                                              root_causes_list_2_config[2], 
                                              2)

#################### [1000, 300, 1000] scenario 1,2,3

# n_observations = 1000
# n_process = 150
# n_time = 300
root_causes_list_3_config = (1000, 150, 300)
root_causes_3_1 = [[31, 100, 200],[32, 100, 200],[33, 100, 200],[34, 100, 200],[35, 100, 200],
                   [36, 100, 200],[37, 100, 200],[38, 100, 200],[39, 100, 200]]
root_causes_3_2 = [[71, 0, 100],[72, 0, 100],[73, 0, 100],[74, 0, 100],[75, 0, 100],
                   [76, 200, 300],[77, 200, 300],[78, 200, 300],[79, 200, 300]]
root_causes_3_3 = [[6, 50, 100],[7, 50, 100],[8, 50, 100],[9, 50, 100],[10, 50, 100],[11, 50, 100],
                   [51, 200, 250],[52, 200, 250],[53, 200, 250],[54, 200, 250],[55, 200, 250],[56, 200, 250],
                   [101, 50, 100],[102, 50, 100],[103, 50, 100],[104, 50, 100],[105, 50, 100],[106, 50, 100],
                   [141, 200, 250],[142, 200, 250],[143, 200, 250],[144, 200, 250],[145, 200, 250],[146, 200, 250]]
root_causes_list_3 = [root_causes_3_1, root_causes_3_2, root_causes_3_3]

root_cause_list = [root_causes_1_1, root_causes_1_2, root_causes_1_3,
                   root_causes_2_1, root_causes_2_2, root_causes_2_3,
                   root_causes_3_1, root_causes_3_2, root_causes_3_3]

dataset_list_3, y_list_3 = generateSynthetic(root_causes_list_3,
                                              root_causes_list_3_config[0], 
                                              root_causes_list_3_config[1], 
                                              root_causes_list_3_config[2], 
                                              3)

dataset_shape_list = [root_causes_list_1_config, root_causes_list_2_config, root_causes_list_3_config]



# # load
# with open(path_synthetic+'dataset_scenario_1_X.pkl'.format(str(root_causes)), 'rb') as f:
#     dataset = pickle.load(f)
# with open(path_synthetic+'dataset_scenario_1_Y.pkl'.format(str(root_causes)), 'rb') as f:
#     y = pickle.load(f)
# f.close()


'''