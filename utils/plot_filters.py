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

def pad_to_size(a, size):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom))

    return a
    
def plt_filter_func(x_input, filter_maps, prj_name, MNIST_Scale, stage):

    if stage == 'S1':
        ori_size = (x_input[0][0].shape[-1], x_input[0][0].shape[-1])
    else:
        ori_size = (x_input[-int(np.ceil(len(x_input)/2))][0].shape[-1], x_input[-int(np.ceil(len(x_input)/2))][0].shape[-1])

    combined_vertical_image = x_input[-int(np.ceil(len(x_input)/2))][0].clone()

    combined_vertical_image = combined_vertical_image - torch.min(combined_vertical_image)
    combined_vertical_image = (combined_vertical_image/torch.max(combined_vertical_image))

    if stage == 'S1':
        combined_vertical_image = pad_to_size(combined_vertical_image, ori_size)

    combined_vertical_image = combined_vertical_image.cpu().numpy()
    # CxHxW --> HxWxC
    combined_vertical_image = combined_vertical_image.transpose(1,2,0)
    combined_vertical_image = combined_vertical_image[:,:]*255.0
    combined_vertical_image = combined_vertical_image.astype('uint8')
    combined_vertical_image = cv2.copyMakeBorder(combined_vertical_image,3,3,3,3,cv2.BORDER_CONSTANT,value=255)

    
    for s_i in range(1, len(filter_maps)+1):
        scale_maps = filter_maps[-s_i].clone()

        # scale_maps = scale_maps - torch.min(scale_maps)
        # scale_maps = (scale_maps/torch.max(scale_maps))

        scale_maps = pad_to_size(scale_maps, ori_size)
        # print('scale_maps : ',scale_maps.shape)
        scale_maps_clone = scale_maps.clone()
        scale_maps_clone = scale_maps_clone[0]


        if stage == 'S2b':
            filter_maps = torch.mean(scale_maps_clone[300:400], dim = 0)
        else:
            filter_maps = scale_maps_clone[0]

        filter_maps = filter_maps - torch.min(filter_maps)
        filter_maps = (filter_maps/torch.max(filter_maps))

        filter_maps_numpy = filter_maps.cpu().numpy()
        # CxHxW --> HxWxC
        filter_maps_numpy = filter_maps_numpy.reshape(1, *filter_maps_numpy.shape)
        filter_maps_numpy = filter_maps_numpy.transpose(1,2,0)
        # filter_maps_numpy = filter_maps_numpy - np.min(filter_maps_numpy)
        # filter_maps_numpy = (filter_maps_numpy/np.max(filter_maps_numpy))*255.0
        filter_maps_numpy = filter_maps_numpy*255.0
        filter_maps_numpy = filter_maps_numpy.astype('uint8')
        filter_maps_numpy = cv2.copyMakeBorder(filter_maps_numpy,3,3,3,3,cv2.BORDER_CONSTANT,value=255)

        combined_vertical_image = cv2.vconcat([combined_vertical_image, filter_maps_numpy])



    main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters/' + stage
    os.makedirs(main_dir, exist_ok=True)
    job_dir = os.path.join(main_dir, "prj_{}".format(prj_name))
    os.makedirs(job_dir, exist_ok=True)

    # out_path = os.path.join(job_dir, "{}_filters_temp.png".format(self.MNIST_Scale))
    # cv2.imwrite(out_path, combined_image)

    out_path = os.path.join(job_dir, "{}_filters_temp.npy".format(int(1000*MNIST_Scale)))
    np.save(out_path, combined_vertical_image)