import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torchvision
import cv2
import os
import _pickle as pickle



def save_tensor(filter_maps, MNIST_Scale, prj_name, category, base_path, stage):

    if 'clf' != stage:
        # filter_tensor = torch.cat(filter_maps, dim = 1)
        filter_tensor = torch.stack(filter_maps, dim = 1)
    else:
        filter_tensor = filter_maps

    if 'c1' == stage or 's2b' == stage:
        center_crop = torchvision.transforms.CenterCrop(140)
        filter_tensor = center_crop(filter_tensor)
        print('after crop filter_tensor : ',filter_tensor.shape)

    filter_numpy = filter_tensor.clone().cpu().numpy()
    # filter_numpy = filter_numpy - np.min(filter_numpy)
    # filter_numpy = filter_numpy/np.max(filter_numpy)

    # if len(filter_numpy.shape) == 2:
    #     filter_numpy = filter_numpy[:,:][None][None]

    # For rdm_corr
    # # 1
    # # filter_numpy = np.mean(filter_numpy, axis = (1,2,3))
    # # 2
    # # filter_numpy = np.amax(filter_numpy, axis = (1,2,3))
    # # 3
    # filter_numpy = np.amax(filter_numpy, axis = (2,3))
    # # if stack else comment this
    # # filter_numpy = np.amax(filter_numpy, axis = 2)
    # filter_numpy = np.mean(filter_numpy, axis = 1)

    # Option 4 --> FLatten
    filter_numpy = filter_numpy.reshape(filter_numpy.shape[0], -1)
    filter_numpy = np.mean(filter_numpy, axis = 0)
    print(stage + ' filter_numpy : ' , filter_numpy.shape)

    # rdm_thomas
    # job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_thomas", prj_name)
    # rdm_corr
    job_dir = os.path.join(base_path, prj_name)
    print('self.prj_name : ', prj_name)
    # job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/color_cnn_FFhGRU_center_real_hGRU_illusions_one/corr_plots"
    # os.makedirs(job_dir, exist_ok=True)
    file_name = os.path.join(job_dir, f"filters_data_{MNIST_Scale}.pkl")

    open_file = open(file_name, "rb")
    filters_data = pickle.load(open_file)
    print('filters_data : ',filters_data.keys())
    open_file.close()

    key_name = stage + '_scale_' + str(int(MNIST_Scale*1000)) + '_cat_' + str(category)
    print('key_name : ',key_name)
    

    if key_name in filters_data:
        filters_data[key_name] = np.concatenate([filters_data[key_name], filter_numpy], axis = 0)
    else:
        filters_data[key_name] = filter_numpy
    
    open_file = open(file_name, "wb")
    pickle.dump(filters_data, open_file)
    open_file.close()