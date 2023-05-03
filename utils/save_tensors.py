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
import random



def save_tensor(filter_maps, MNIST_Scale, prj_name, category, base_path, stage):

    print('\nstage : ',stage)
    print('filter_maps len : ', len(filter_maps))
    print('filter_maps shape : ', filter_maps[0].shape)

    if 'c1' == stage or 's2b' == stage or 'c2' == stage:
        for fm_i in range(len(filter_maps)):
            center_crop = torchvision.transforms.CenterCrop(140)
            filter_maps[fm_i] = center_crop(filter_maps[fm_i])

    if 'clf' != stage:
        # filter_tensor = torch.cat(filter_maps, dim = 1)
        filter_tensor = torch.stack(filter_maps, dim = 1)
    else:
        filter_tensor = filter_maps

    print('filter_tensor shape : ', filter_tensor.shape)

    # if 'c1' == stage or 's2b' == stage or 'c2' == stage:
    #     center_crop = torchvision.transforms.CenterCrop(40)
    #     filter_tensor = center_crop(filter_tensor)
    #     print('after crop filter_tensor : ',filter_tensor.shape)


    filter_numpy = filter_tensor.clone().cpu().numpy()    

    ############################################################################################
    ############################################################################################
    # Method 1 -  Add Noise
    # print('stage : ',stage)
    # print('filter_numpy max : ',np.max(filter_numpy))
    # print('filter_numpy min : ',np.min(filter_numpy))
    # print('filter_numpy abs mean : ',np.mean(np.abs(filter_numpy)))


    # filter_numpy = filter_numpy + np.random.randn(*filter_numpy.shape) * 1
    ############################################################################################

    # filter_numpy = np.amax(filter_numpy, axis = 1)
    # print('after max filter_tensor : ',filter_numpy.shape)

    # Option 4 --> FLatten
    filter_numpy = filter_numpy.reshape(filter_numpy.shape[0], filter_numpy.shape[1], -1)
    filter_numpy = np.mean(filter_numpy, axis = 0)
    # filter_numpy = filter_numpy[0]
    print(stage + ' filter_numpy : ' , filter_numpy.shape)

    ############################################################################################
    ############################################################################################
    # Method 2 - Randomly Sample 10000 elements
    num_samples = 78400

    if 'c1' == stage or 's2b' == stage:
        indices = list(range(len(filter_numpy[0])))
        # random.shuffle(indices)

        # Select a random sample of size 3 from the shuffled indices
        sample_indices = sorted(random.sample(indices, k=num_samples))

        print('sample_indices : ',sample_indices[:100])

        temp_filter_numpy = np.zeros((len(filter_numpy), num_samples))
        for f_i in range(len(filter_numpy)):
            # temp_filter_numpy[f_i] = np.random.choice(filter_numpy[f_i], size = num_samples)
            # temp_filter_numpy[f_i] = np.array(random.sample(list(filter_numpy[f_i]), num_samples))

            # indices = list(range(len(filter_numpy[f_i])))
            # random.shuffle(indices)

            # # Select a random sample of size 3 from the shuffled indices
            # sample_indices = sorted(random.sample(indices, k=num_samples))

            # Get the corresponding items from the original array based on the selected indices
            temp_filter_numpy[f_i] = np.array([filter_numpy[f_i][i] for i in sample_indices])

        filter_numpy = temp_filter_numpy
        print(stage + ' sampled filter_numpy : ' , filter_numpy.shape)

    ############################################################################################

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