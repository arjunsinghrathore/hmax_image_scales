import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torchvision
import cv2
import os
from pytorch_lightning import Trainer, seed_everything
from torchvision import datasets, models
import _pickle as pickle

from utils.save_tensors import save_tensor
from utils.plot_filters import plt_filter_func
from utils.foveation_and_cortical_magn import warp_image

from .dicarlo_back_ends import ResNetBackEnd, Bottleneck, AlexNetBackEnd, CORnetSBackEnd

import random


# seed_everything(42, workers=True)

# if False and l_size in [15]:
    #     s1_filt_list = []

    #     save_dir = os.path.join('/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/learned_gabor_viz/', prj_name)
    #     for filt_ind, s1_filt in enumerate(filt_c):
    #             os.makedirs(save_dir, exist_ok=True)
    #             image_name = str(filt_ind) + '_init.png'
    #             s1_filt_3d = np.concatenate([s1_filt.reshape(l_size,l_size,1), s1_filt.reshape(l_size,l_size,1), s1_filt.reshape(l_size,l_size,1)], axis = -1)

    #             s1_filt_list.append(s1_filt_3d)
    #             # cv2.imwrite(os.path.join(save_dir, image_name), s1_filt_3d*255.0)

    #     for v_i in range(n_ori):
    #         for h_i in range(n_phi):
    #             if h_i == 0:
    #                 s1_filt_list_pad = cv2.copyMakeBorder(s1_filt_list[h_i*n_ori + v_i],1,1,1,1,cv2.BORDER_CONSTANT,value=[1,1,1])
    #                 hori_img = s1_filt_list_pad
    #             else:
    #                 s1_filt_list_pad = cv2.copyMakeBorder(s1_filt_list[h_i*n_ori + v_i],1,1,1,1,cv2.BORDER_CONSTANT,value=[1,1,1])
    #                 hori_img = cv2.hconcat([hori_img, s1_filt_list_pad])

    #         if v_i == 0: 
    #             vertical_img = hori_img
    #         else:
    #             vertical_img = cv2.vconcat([vertical_img, hori_img])
                
    #     print('INit vertical_img shape : ',vertical_img.shape)
    #     vertical_img = cv2.resize(vertical_img, (vertical_img.shape[1]*5,vertical_img.shape[0]*5))
    #     vertical_img = (vertical_img - np.min(vertical_img)) / (np.max(vertical_img) - np.min(vertical_img)) # normalize (min-max)
    #     # vertical_img = vertical_img/np.max(vertical_img)
    #     image_name = f'Init_collage_{l_size}_normalized_plt_no_norm_gabor.png'
    #     cv2.imwrite(os.path.join(save_dir, image_name), vertical_img*255.0)

def pad_to_size(a, size, pad_mode = 'constant'):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    # Do 0 padding only if padding size becomes more than image size
    try:
        a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode = pad_mode)
    except:
        a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode = 'constant')

    # # Do 0 padding only if padding size becomes more than image size
    # if (total_pad_h < current_size[0] and total_pad_w < current_size[1]) or pad_mode == 'constant':
    #     pad_top = total_pad_h // 2
    #     pad_bottom = total_pad_h - pad_top

    #     pad_left = total_pad_w // 2
    #     pad_right = total_pad_w - pad_left
        
    #     a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode = pad_mode)
    # else:
    #     # First round of reflect pad with as much as possible
    #     total_pad_h = current_size[0]-1
    #     pad_top = total_pad_h // 2
    #     pad_bottom = total_pad_h - pad_top

    #     total_pad_w = current_size[1]-1
    #     pad_left = total_pad_w // 2
    #     pad_right = total_pad_w - pad_left

    #     a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode = pad_mode)

    #     # 2nd round of reflect pad with the left required padding
    #     current_size = (a.shape[-2], a.shape[-1])
        
    #     total_pad_h = size[0] - current_size[0]
    #     pad_top = total_pad_h // 2
    #     pad_bottom = total_pad_h - pad_top

    #     total_pad_w = size[1] - current_size[1]
    #     pad_left = total_pad_w // 2
    #     pad_right = total_pad_w - pad_left

    #     a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode = pad_mode)

    return a

def get_gabor(l_size, la, si, n_ori, aspect_ratio, n_phi, prj_name = None):
    """generate the gabor filters

    Args
    ----
        l_size: float
            gabor sizes
        la: float
            lambda
        si: float
            sigma
        n_ori: type integer
            number of orientations
        aspect_ratio: type float
            gabor aspect ratio

    Returns
    -------
        gabor: type nparray
            gabor filter

    """

    gs = l_size

    # TODO: inverse the axes in the begining so I don't need to do swap them back
    # thetas for all gabor orientations
    th = np.array(range(n_ori)) * (np.pi / n_ori) #+ np.pi / 2.
    th = th[sp.newaxis, sp.newaxis, :]

    ######################## Phi ########################
    phi_s = np.array(range(n_phi)) * (2*np.pi / n_phi) #+ np.pi / 2.
    # print('complex phi_s : ',phi_s.shape)
    # print('complex phi_s : ',phi_s)
    # phi_s = phi_s[sp.newaxis, sp.newaxis, :]
    ######################################################


    hgs = (gs - 1) / 2.
    yy, xx = sp.mgrid[-hgs: hgs + 1, -hgs: hgs + 1]
    xx = xx[:, :, sp.newaxis];
    yy = yy[:, :, sp.newaxis]

    x = xx * np.cos(th) + yy * np.sin(th)
    y = - xx * np.sin(th) + yy * np.cos(th)

    ######################## No Phi ######################
    # filt = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * si ** 2)) * np.cos(2 * np.pi * x / la)
    # filt[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0
    ######################################################

    filt = []
    for phi in phi_s:
        filt_temp = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * si ** 2)) * np.cos((2 * np.pi * x / la) + phi)
        filt_temp[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0

        filt.append(filt_temp)

    filt = np.concatenate(filt, axis = -1)

    # gabor normalization (following cns hmaxgray package)
    for ori in range(n_ori*n_phi):
        filt[:, :, ori] -= filt[:, :, ori].mean()
        filt_norm = fastnorm(filt[:, :, ori])
        if filt_norm != 0: filt[:, :, ori] /= filt_norm
    filt_c = np.array(filt, dtype='float32').swapaxes(0, 2).swapaxes(1, 2)

    filt_c = torch.Tensor(filt_c)
    filt_c = filt_c.view(n_ori*n_phi, 1, gs, gs)
    filt_c = filt_c.repeat((1, 3, 1, 1))

    return filt_c


def fastnorm(in_arr):
    arr_norm = np.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)
    return arr_norm

def fastnorm_tensor(in_arr):
    arr_norm = torch.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)
    return arr_norm

class S1(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, scale, n_ori, padding, trainable_filters, la, si, visualize_mode = False, prj_name = None, MNIST_Scale = None, n_phi = 1, scale_factor = 4):

        super(S1, self).__init__()

        self.scale = scale
        self.la = la
        self.si = si
        self.visualize_mode = visualize_mode
        self.prj_name = prj_name
        self.MNIST_Scale = MNIST_Scale
        self.padding = padding
        self.scale_factor = scale_factor

        setattr(self, f's_{scale}', nn.Conv2d(3, n_ori*n_phi, kernel_size = scale, stride = 4, padding=padding))
        s1_cell = getattr(self, f's_{scale}')
        self.gabor_filter = get_gabor(l_size=scale, la=la, si=si, n_ori=n_ori, aspect_ratio=0.3, n_phi = n_phi, prj_name = self.prj_name)  # ??? What is aspect ratio
        print('self.gabor_filter : ', self.gabor_filter.shape)
        # self.gabor_weights = nn.Parameter(self.gabor_filter, requires_grad=False)
        s1_cell.weight = nn.Parameter(self.gabor_filter, requires_grad=trainable_filters)
        # for param in s1_cell.parameters():
        #     param.requires_grad = False

        # # # For normalization
        # setattr(self, f's_uniform_{scale}', nn.Conv2d(1, n_ori, scale, bias=False))
        # s1_uniform = getattr(self, f's_uniform_{scale}')
        # nn.init.constant_(s1_uniform.weight, 1)
        # for param in s1_uniform.parameters():
        #     param.requires_grad = False

        # self.batchnorm = nn.BatchNorm2d(n_ori, 1e-3)
        self.batchnorm = nn.Sequential(nn.BatchNorm2d(n_ori*n_phi, 1e-3),
                                       nn.ReLU(True),
                                      )
        

    def forward(self, x_pyramid, MNIST_Scale = None, batch_idx = None, prj_name = None, category = None, save_rdms = None, plt_filters = None):
        self.MNIST_Scale = MNIST_Scale
        s1_maps = []
        middle_scale = len(x_pyramid)//2
        # Loop over scales, normalizing.
        for p_i in range(len(x_pyramid)):

            x = x_pyramid[p_i]

            s1_cell = getattr(self, f's_{self.scale}')
            #  # s1_map = torch.abs(s1_cell(x))  # adding absolute value
            s1_map = s1_cell(x)
            # s1_map = F.conv2d(x, self.gabor_filter.to(device='cuda'), None, 4, self.padding)

            # s1_map = s1_map/(2**((2*(middle_scale-p_i))/self.scale_factor))

            # Normalization
            # s1_unorm = getattr(self, f's_uniform_{self.scale}')
            # s1_unorm = torch.sqrt(s1_unorm(x**2))
            # # s1_unorm = torch.sqrt(s1_unorm(x))
            # s1_unorm.data[s1_unorm == 0] = 1  # To avoid divide by zero
            # s1_map /= s1_unorm
            s1_map = self.batchnorm(s1_map)

            s1_maps.append(s1_map)

            # Padding (to get s1_maps in same size) ---> But not necessary for us
            ori_size = (x.shape[-2], x.shape[-1])
            s1_maps[p_i] = pad_to_size(s1_maps[p_i], ori_size)

        ############################################################################
        # RDMs

        if 's1' in save_rdms:
            save_tensor(s1_maps, MNIST_Scale, prj_name, category, base_path = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", stage = 's1')


        ##################################################################
        # Plot filters

        if 's1' in plt_filters:
            plt_filter_func(x_pyramid, s1_maps, prj_name, MNIST_Scale, stage = 'S1')

        return s1_maps

class C(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, global_pool, sp_kernel_size=[10, 8], sp_stride_factor=None, n_in_sbands=None,
                 num_scales_pooled=2, scale_stride=1,image_subsample_factor=1, visualize_mode = False, \
                 c1_bool = False, prj_name = None, MNIST_Scale = None, c2_bool = False, c3_bool = False, \
                 c2b_bool = False):

        super(C, self).__init__()
        print('c1_bool : ',c1_bool)
        print('c2_bool : ',c2_bool)
        print('c3_bool : ',c3_bool)
        print('c2b_bool : ',c2b_bool)

        self.c1_bool = c1_bool
        self.c2_bool = c2_bool
        self.c3_bool = c3_bool
        self.c2b_bool = c2b_bool
        self.prj_name = prj_name
        self.MNIST_Scale = MNIST_Scale
        self.visualize_mode = visualize_mode
        self.global_pool = global_pool
        self.sp_kernel_size = sp_kernel_size
        self.num_scales_pooled = num_scales_pooled
        self.sp_stride_factor = sp_stride_factor
        self.scale_stride = scale_stride
        self.n_in_sbands = n_in_sbands
        self.n_out_sbands = int(((n_in_sbands - self.num_scales_pooled) / self.scale_stride) + 1)
        self.img_subsample = image_subsample_factor 

        #################
        # Option 2:
        # self.inh_linear = nn.Sequential(nn.Linear(400, 400), nn.ReLU())
        # self.one_conv = nn.Sequential(nn.Conv2d(16, 16, kernel_size = 1, stride = 1), nn.ReLU())
        # self.scale_lin = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

        if c2b_bool:
            # self.scale_attention = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))
            self.scale_attention = nn.Sequential(nn.Linear(512, 64), nn.Linear(64, 1))
            # self.scale_lin_2 = nn.Sequential(nn.Linear(64, 1))

            # self.gate_fc = nn.Sequential(nn.Linear(512, 64), nn.Linear(64, 1))

            # self.inhibition_strength = nn.Parameter(torch.tensor(1.), requires_grad=True)

            # self.sigma_fc = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))

        if not self.global_pool:
            if self.sp_stride_factor is None:
                self.sp_stride = [1] * len(self.sp_kernel_size)
            else:
                self.sp_stride = [int(np.ceil(self.sp_stride_factor * kernel_size)) for kernel_size in self.sp_kernel_size]
                # self.sp_stride = [int(np.ceil(0.5 + kernel_size/self.sp_kernel_size[len(self.sp_kernel_size)//2])) for kernel_size in self.sp_kernel_size]

    def forward(self, x_pyramid, x_input = None, MNIST_Scale = None, batch_idx = None, category = None, \
                prj_name = None, same_scale_viz = None, base_scale = None, c1_sp_kernel_sizes = None, \
                c2_sp_kernel_sizes = None, image_scales = None, overall_max_scale_index = False, save_rdms = None, \
                plt_filters = None, argmax_bool = False):
        # TODO - make this whole section more memory efficient

        # print('####################################################################################')
        # print('####################################################################################')

        max_scale_index = [0]
        correct_scale_loss = 0

        c_maps = []
        if same_scale_viz:
            ori_size = (base_scale, base_scale)
        else:
            ori_size = x_pyramid[0].shape[2:4]

        # print('len(x_pyramid) : ',len(x_pyramid))

        # Single scale band case --> While Training
        if len(x_pyramid) == 1:
            if not self.global_pool:
                x = F.max_pool2d(x_pyramid[0], self.sp_kernel_size[0], self.sp_stride[0])
                x = pad_to_size(x, ori_size)

                c_maps.append(x)
            else:
                s_m = F.max_pool2d(x_pyramid[0], x_pyramid[0].shape[-1], 1)
                c_maps.append(s_m)

        # Multi Scale band case ---> While Testing
        else:
            #####################################################
            if not self.global_pool:
                if same_scale_viz:
                    ori_size = (base_scale, base_scale)
                else:
                    if len(x_pyramid) == 2:
                        ori_size = x_pyramid[0].shape[2:4]
                    else:
                        # ori_size = x_pyramid[-5].shape[2:4]
                        # ori_size = x_pyramid[-9].shape[2:4]
                        ori_size = x_pyramid[-int(np.ceil(len(x_pyramid)/2))].shape[2:4]

                # ####################################################
                # # MaxPool for C1 with 2 scales being max pooled over at a time
                for p_i in range(len(x_pyramid)-1):
                    # print('############################')
                    x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[0], self.sp_stride[0])
                    x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[1], self.sp_stride[1])

                    # First interpolating such that feature points match spatially
                    if x_1.shape[-1] > x_2.shape[-1]:
                        # x_2 = pad_to_size(x_2, x_1.shape[-2:])
                        x_2 = F.interpolate(x_2, size = x_1.shape[-2:], mode = 'bilinear')
                    else:
                        # x_1 = pad_to_size(x_1, x_2.shape[-2:])
                        x_1 = F.interpolate(x_1, size = x_2.shape[-2:], mode = 'bilinear')

                    # Then padding
                    x_1 = pad_to_size(x_1, ori_size)
                    x_2 = pad_to_size(x_2, ori_size)

                    ##################################
                    # Maxpool over scale groups
                    x = torch.stack([x_1, x_2], dim=4)

                    to_append, _ = torch.max(x, dim=4)
                    c_maps.append(to_append)

                # ####################################################
                # # MultiScale Training: MaxPool for C1 with 1 scales being max pooled over at a time
                # for p_i in range(len(x_pyramid)):
                #     # print('############################')
                #     x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[0], self.sp_stride[0])

                #     # Then padding
                #     x_1 = pad_to_size(x_1, ori_size)

                #     c_maps.append(x_1)

                ###################################################
                # # ArgMax case for C1
                # for p_i in range(len(x_pyramid)-1):
                #     scale_max = []

                #     # x_1 = F.max_pool2d(x_pyramid[p_i], x_pyramid[p_i].shape[-1])

                #     x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[0], self.sp_stride[0])


                #     # x_2 = F.max_pool2d(x_pyramid[p_i+1], x_pyramid[p_i+1].shape[-1])

                #     x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[1], self.sp_stride[1])

                #     if x_1.shape[-1] > x_2.shape[-1]:
                #         # x_2 = pad_to_size(x_2, x_1.shape[-2:])
                #         x_2 = F.interpolate(x_2, size = x_1.shape[-2:], mode = 'bilinear')
                #     else:
                #         # x_1 = pad_to_size(x_1, x_2.shape[-2:])
                #         x_1 = F.interpolate(x_1, size = x_2.shape[-2:], mode = 'bilinear')

                    
                #     x_1 = torch.sum(x_1, dim = (2,3)) #/image_scales[p_i]
                #     x_1 = x_1.reshape(*x_pyramid[p_i].shape[:2])
                #     x_1 = torch.sort(x_1, dim = -1)[0]
                #     scale_max.append(x_1)

                #     x_2 = torch.sum(x_2, dim = (2,3)) #/image_scales[p_i]
                #     x_2 = x_2.reshape(*x_pyramid[p_i+1].shape[:2])
                #     x_2 = torch.sort(x_2, dim = -1)[0]
                #     scale_max.append(x_2)

                #     # scale_max shape --> Scale x B x C
                #     scale_max = torch.stack(scale_max, dim=0) # Shape --> Scale x B x C

                #     x_max = []
                #     for b_i in range(scale_max.shape[1]):

                #         scale_max_argsort = torch.argsort(torch.stack([scale_max[0][b_i], scale_max[1][b_i]], dim=0), dim = 0) # Shape --> 2 x C
                #         # print('x_pyramid_flatten_argsort : ',x_pyramid_flatten_argsort)
                #         # Sum across the (CxHxW) dimension
                #         sum_scale_batch = torch.sum(scale_max_argsort, dim = 1) # SHape --> 2 x 1]
                #         # sum_scale_batch = sum_scale_batch.cpu().numpy()
                #         # sum_scale_batch = sum_scale_batch.astype(np.float32)
                #         # sum_scale_batch[0] = sum_scale_batch[0]/image_scales[max_scale_index[b_i]]
                #         # sum_scale_batch[1] = sum_scale_batch[1]/image_scales[p_i]
                #         # print('max_scale_index[b_i] : ', max_scale_index[b_i], ':: p_i : ', p_i, ' :: sum_scale_batch : ',sum_scale_batch)

                #         if sum_scale_batch[0] < sum_scale_batch[1]:
                #             # print('Smaller Scale : ')
                #             x_max_batch = F.max_pool2d(x_pyramid[p_i+1][b_i], self.sp_kernel_size[1], self.sp_stride[1])
                #         else:
                #             # print('Larger Scale : ')
                #             x_max_batch = F.max_pool2d(x_pyramid[p_i][b_i], self.sp_kernel_size[0], self.sp_stride[0])

                #         x_max_batch = pad_to_size(x_max_batch, ori_size)

                #         x_max.append(x_max_batch)

                #         x_1 = None
                #         x_2 = None
                #         x_max_batch = None

                #     x_max = torch.stack(x_max, dim=0)

                #     c_maps.append(x_max)
                
                ####################################################

            # ####################################################
            else:
                # # # MaxPool over all positions first then scales (C2b)
                if not argmax_bool:
                    # scale_max = []
                    # # Global MaxPool over positions for each scale separately
                    # # x_pyramid Shape --> [Scales, Batch, Channels, Height, Width]
                    # for p_i in range(len(x_pyramid)):
                    #     s_m = F.max_pool2d(x_pyramid[p_i], x_pyramid[p_i].shape[-1], 1)
                    #     scale_max.append(s_m)

                    scale_max = x_pyramid

                    
                    ####################################################

                    # Option 1:: Global Max Pooling over Scale i.e, Maxpool over scale groups
                    # print('scale_max : ',scale_max[0].shape)

                    x = torch.stack(scale_max, dim=4) # scale_max Shape --> [Batch, Channels, 1, 1, Scales]
                            

                    ####################################################

                    # print('x : ',x.shape)
                    to_append, _ = torch.max(x, dim=4)
                    c_maps.append(to_append)

                    # # Option 2:: No Global Max Pooling over Scale
                    # for s_i in range(len(scale_max)-1):
                    #     # Maxpool over scale groups
                    #     x = torch.stack([scale_max[s_i], scale_max[s_i+1]], dim=4)

                    #     to_append, _ = torch.max(x, dim=4)
                    #     c_maps.append(to_append)

                    # # Option 3: Give scale_max as c_maps
                    # # # Option 3.1 : Without Scale Max Pooling
                    # # c_maps = scale_max
                    # # Option 3.2 : With Scale Max Pooling
                    # x = torch.stack(scale_max, dim=4)
                    # to_append, _ = torch.max(x, dim=4)
                    # c_maps.append(to_append)

                # ####################################################
                # # # Argmax with global maxpool/sum over H, W
                else:
                    #####################################################
                    scale_max = []
                    for p_i in range(len(x_pyramid)):

                        # print('p_i argmax : ', p_i)

                        s_m = x_pyramid[p_i]

                        # #####################################################
                        # # Rescale
                        # new_dimension = int(image_scales[-(p_i+1)]) #int(s_m.shape[-1]/(image_scales[p_i]/image_scales[4]))
                        # # if new_dimension >= s_m.shape[-1]:
                        # #     new_dimension = s_m.shape[-1]
                        # print('Old Shape : ',s_m.shape[-1], ' ::: New Dimension : ',new_dimension)
                        # s_m = F.interpolate(s_m, size = (new_dimension, new_dimension), mode = 'bilinear')
                        # print('s_m shape : ',s_m.shape)
                        # # s_m = s_m.reshape(*s_m.shape[:2])
                        # #####################################################
                        # # Crop
                        # new_dimension = int(image_scales[-(p_i+1)]) #int(s_m.shape[-1]/(image_scales[p_i]/image_scales[4]))
                        # if new_dimension >= s_m.shape[-1]:
                        #     new_dimension = s_m.shape[-1]
                        # print('Old Shape : ',s_m.shape[-1], ' ::: New Dimension : ',new_dimension)
                        # center_crop = torchvision.transforms.CenterCrop(new_dimension)
                        # s_m = center_crop(s_m)
                        # print('s_m shape : ',s_m.shape)
                        # # s_m = s_m.reshape(*s_m.shape[:2])
                        #####################################################
                        # MaxPool H,W
                        s_m = F.max_pool2d(s_m, s_m.shape[-1], 1)
                        # s_m = F.avg_pool2d(s_m, s_m.shape[-1], 1)
                        s_m = s_m.reshape(*x_pyramid[p_i].shape[:2])
                        # Sum H,W
                        # s_m = torch.sum(s_m, dim = (2,3)) #/image_scales[p_i]
                        # s_m = s_m.reshape(*x_pyramid[p_i].shape[:2])
                        #####################################################
                        s_m = torch.sort(s_m, dim = -1)[0]
                        scale_max.append(s_m)

                    # scale_max shape --> Scale x B x C
                    scale_max = torch.stack(scale_max, dim=0) # Shape --> Scale x B x C
                    # print('scale_max : ',scale_max.shape)


                    # # Method 4 --> Attention weights for scale channels
                    # x_prime = scale_max # [No. of scales, Batch, Channel]

                    # #####################################################
                    # # # x_prime_clone = x_prime.clone()
                    # # x_prime_attn = self.scale_lin_1(x_prime)  # [No. of scales, Batch, hidden_channels]
                    # # x_prime_attn = self.scale_lin_2(x_prime_attn)  # [No. of scales, Batch, 1]
                    # # attention_weights = F.softmax(x_prime_attn, dim=0)  # Normalize weights across scales
                    # #####################################################

                    # # Attention Method 2
                    # num_scales = x_prime.shape[0]
                    # init_attention_weights = (torch.ones((num_scales, 1, 1))).to(x_prime.device)
                    # # init_attention_weights = (torch.zeros((num_scales, 1, 1), requires_grad = False)).to(x_prime.device)
                    # attention_weights = init_attention_weights.repeat(1, x_prime.shape[1], 1)  # [num_scales, batch_size, 1]
        
                    # # compute dynamic lateral inhibition
                    # for _ in range(4):
                    #     # attention_weights = F.softmax(attention_weights, dim=0)
                    #     # attention_weights_raw = attention_weights * torch.sigmoid(self.scale_attention(x_prime))
                    #     # attention_weights_raw = attention_weights * F.softmax(self.scale_attention(x_prime), dim=0)

                    #     ###################
                    #     attention_weights_raw = attention_weights * self.scale_attention(x_prime)
                    #     ###################
                    #     # gate = torch.sigmoid(self.gate_fc(x_prime))  # new line
                    #     # attention_weights_raw = gate * attention_weights + (1 - gate) * self.scale_attention(x_prime)  # modified line
                    #     ###################

                    #     attention_weights = F.softmax(attention_weights_raw, dim=0)
                    #     # attention_weights = attention_weights_raw

                    #     # compute expected scale
                    #     temperature = 0.01
                    #     attention_weights_es = F.softmax(attention_weights_raw/temperature, dim=0)
                    #     scales = torch.arange(num_scales).to(x_prime.device)  # [num_scales]
                    #     expected_scale = (scales[:, None, None] * attention_weights_es).sum(dim=0)  # [batch_size, 1]

                    #     # print('expected_scale : ', expected_scale, ' ::: Real Scale : ', torch.argmax(attention_weights, dim = 0))

                    #     # compute adaptive sigma
                    #     sigma = 0.5 # 1.5 #self.strength_of_inhibition * torch.sigmoid(self.sigma_fc(x_prime))
                        
                    #     # sigma = torch.where((scales[:, None, None] < 1) | (scales[:, None, None] > num_scales - 2), 
                    #     #                     0.1, 
                    #     #                     0.5)

                    #     # print('sigma : ',sigma)


                    #     # compute Gaussian-like weights
                    #     gaussian_weights = torch.exp(-((scales[:, None, None] - expected_scale) ** 2) / (2 * sigma ** 2))  # [num_scales, batch_size, 1]

                    #     # apply lateral inhibition
                    #     attention_weights = attention_weights * gaussian_weights
                    #     # attention_weights = F.softmax(attention_weights, dim=0)

                    # attention_weights = F.softmax(attention_weights, dim=0)

                    # #####################################################

                    # # Multiply the input tensor by the attention_weights
                    # output = x_prime * attention_weights
                    # # print('output : ',output.shape)
                    # scale_max = output #.permute(1,2,0)[:,:,None,None,:]
                    # # print('x : ',x.shape)


                    # print('image_scales : ',image_scales)

                    max_scale_index = [0]*scale_max.shape[1]
                    # max_scale_index = torch.tensor(max_scale_index).cuda()
                    for p_i in range(1, len(x_pyramid)):
                        
                        for b_i in range(scale_max.shape[1]):

                            # print(f'b_i : {b_i}, p_i {p_i}')
                            # print('scale_max[max_scale_index[b_i]][b_i] : ',scale_max[max_scale_index[b_i]][b_i])
                            # print('scale_max[p_i][b_i] : ',scale_max[p_i][b_i])

                            scale_max_argsort = torch.argsort(torch.stack([scale_max[max_scale_index[b_i]][b_i], scale_max[p_i][b_i]], dim=0), dim = 0) # Shape --> 2 x C
                            # print('x_pyramid_flatten_argsort : ',x_pyramid_flatten_argsort)
                            # Sum across the (CxHxW) dimension
                            sum_scale_batch = torch.sum(scale_max_argsort, dim = 1) # SHape --> 2 x 1]

                            # sum_scale_batch = sum_scale_batch.cpu().numpy()
                            # sum_scale_batch = sum_scale_batch.astype(np.float32)
                            # sum_scale_batch[0] = sum_scale_batch[0]/((image_scales[max_scale_index[b_i]]/image_scales[-1])**1)
                            # sum_scale_batch[1] = sum_scale_batch[1]/((image_scales[p_i]/image_scales[-1])**1)
                            # print('max_scale_index[b_i] : ', max_scale_index[b_i], ':: p_i : ', p_i, ' :: sum_scale_batch : ',sum_scale_batch)

                            if sum_scale_batch[0] < sum_scale_batch[1]:
                                max_scale_index[b_i] = p_i


                    # max_scale_index = torch.argmax(sum_scale)


                    to_append = []
                    for b_i in range(scale_max.shape[1]):
                        # print('max_scale_index : ',max_scale_index[b_i])
                        to_append_batch = F.max_pool2d(x_pyramid[max_scale_index[b_i]][b_i][None], x_pyramid[max_scale_index[b_i]][b_i][None].shape[-1], 1) # Shape --> 1 x C x 1 x 1
                        # print('to_append_batch shape : ',to_append_batch.shape)
                        to_append.append(to_append_batch)

                    to_append = torch.cat(to_append, dim = 0)
                    # print('to_append shape : ',to_append.shape)


                    c_maps.append(to_append)

                ####################################################

                
        ############################################################################
        ############################################################################
        # RDMs

        if 'c1' in save_rdms or 'c2b' in save_rdms:
            if self.c1_bool:
                stage_name = 'c1'
            else:
                stage_name = 'c2b'
            save_tensor(c_maps, MNIST_Scale, prj_name, category, base_path = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", stage = stage_name)

        
        ############################################################################
        ############################################################################
        # Plot filters

        if 'c1' in plt_filters:
            plt_filter_func(x_input, c_maps, prj_name, MNIST_Scale, stage = 'C1')


           
        if not self.global_pool: 
            return c_maps #, overall_max_scale_index
        else:
            return c_maps, max_scale_index, correct_scale_loss

class S2(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride):
        super(S2, self).__init__()

        # AlexNet
        self.bottleneck = nn.Conv2d(channels_in, 64, kernel_size=1, stride=1, bias=False)

        self.model_back_end = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3])
        
        # # # Original
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(channels_in, 256, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 3, stride = 2))

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU())

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU())

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 3, stride = 2))

        # # DiCarlo
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(64, 192, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 3, stride = 2))

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU())

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 3, stride = 2))

        # # VGG-16 BN
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(), 
        #     nn.MaxPool2d(kernel_size = 2, stride = 2))
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU())
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 2, stride = 2))
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU())
        # self.layer7 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 2, stride = 2))
        # self.layer8 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        # self.layer9 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        # self.layer10 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 2, stride = 2))
        # self.layer11 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        # self.layer12 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())
        # self.layer13 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 2, stride = 2))




    def forward(self, x_pyramid, prj_name = None, category = None, MNIST_Scale = None, x_input = None, save_rdms = None, plt_filters = None):
        

        # Convolve each kernel with each scale band
        s_maps_per_i = []
        for i in range(len(x_pyramid)):  # assuming S is last dimension

            x = x_pyramid[i]

            # # AlexNet
            # out = self.bottleneck(x)
            # out = self.layer2(out)
            # out = self.layer3(out)
            # out = self.layer4(out)
            # out = self.layer5(out)
            
            # # VGG-16 BN
            # out = self.layer2(x)
            # out = self.layer3(out)
            # out = self.layer4(out)
            # out = self.layer5(out)
            # out = self.layer6(out)
            # out = self.layer7(out)
            # out = self.layer8(out)
            # out = self.layer9(out)
            # out = self.layer10(out)
            # out = self.layer11(out)
            # out = self.layer12(out)
            # out = self.layer13(out)

            # ResNet50
            out = self.bottleneck(x)
            out = self.model_back_end(out)

            s_m = F.max_pool2d(out, out.shape[-1], 1)
            # scale_max.append(s_m)


            # s_maps_per_i.append(out)
            s_maps_per_i.append(s_m)

        s_maps = s_maps_per_i

        ############################################################################
        ############################################################################
        # RDMs

        if 's2b' in save_rdms:
            save_tensor(s_maps, MNIST_Scale, prj_name, category, base_path = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", stage = 's2b')


        ###################################################################
        # Plot filters

        if 's2b' in plt_filters:
            plt_filter_func(x_input, s_maps, prj_name, MNIST_Scale, stage = 'S2b')
            

        return s_maps


#########################################################################################################
# class HMAXify_Models(nn.Module):
#     def __init__(self,
#                  num_classes=10,
#                  prj_name = None,
#                  ip_scales = 18,
#                  s1_scale=11, #25 #23 #21 #19 #17 #15 #13 #11 # 7, #5
#                  s1_la=5.6, #14.1 #12.7 #11.5 #10.3 #9.1 #7.9 #6.8 #5.6 # 3.5, # 2.5
#                  s1_si=4.5, #11.3 #10.2 #9.2 #8.2 #7.3 #6.3 #5.4 #4.5 # 2.8, # 2
#                  n_ori=16,
#                  n_phi = 8,
#                  s1_trainable_filters=False,
#                  visualize_mode = False,
#                  MNIST_Scale = None,
#                  category = None,
#                  single_scale_bool = True,
#                  ):
#         super(HMAXify_Models, self).__init__()
# #########################################################################################################
class HMAXify_Models(nn.Module):
    def __init__(self,
                 num_classes=10,
                 prj_name = None,
                 ip_scales = 18,
                 s1_scale=15, #25 #23 #21 #19 #17 #15 #13 #11 # 7, #5
                 s1_la=7.9, #14.1 #12.7 #11.5 #10.3 #9.1 #7.9 #6.8 #5.6 # 3.5, # 2.5
                 s1_si=6.3, #11.3 #10.2 #9.2 #8.2 #7.3 #6.3 #5.4 #4.5 # 2.8, # 2
                 n_ori=16,
                 n_phi = 8,
                 s1_trainable_filters=False,
                 visualize_mode = False,
                 MNIST_Scale = None,
                 category = None,
                 single_scale_bool = True,
                 ):
        super(HMAXify_Models, self).__init__()
#########################################################################################################
# class HMAXify_Models(nn.Module):
#     def __init__(self,
#                  num_classes=10,
#                  prj_name = None,
#                  ip_scales = 18,
#                  s1_scale=21, #25 #23 #21 #19 #17 #15 #13 #11 # 7, #5
#                  s1_la=11.5, #14.1 #12.7 #11.5 #10.3 #9.1 #7.9 #6.8 #5.6 # 3.5, # 2.5
#                  s1_si=9.2, #11.3 #10.2 #9.2 #8.2 #7.3 #6.3 #5.4 #4.5 # 2.8, # 2
#                  n_ori=16,
#                  n_phi = 1,
#                  s1_trainable_filters=False,
#                  visualize_mode = False,
#                  MNIST_Scale = None,
#                  category = None,
#                  single_scale_bool = True,
#                  ):
#         super(HMAXify_Models, self).__init__()
# #########################################################################################################

        self.num_classes = num_classes

        self.ip_scales = 9
        self.single_scale_bool = True
        self.make_ip_2_bool = False
        self.force_const_size_bool = True

        self.pad_mode = 'reflect'
        # self.pad_mode = 'constant'

        self.argmax_bool = False
        self.same_scale_viz = None
        self.base_scale = None
        self.orcale_bool = None
        self.save_rdms = []
        self.plt_filters = []

        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.n_phi = n_phi
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.single_scale_bool = single_scale_bool
        self.scale = 4

        ########################################################
        ########################################################

       
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 3, stride = 2))

        ########################################################
        ########################################################
        # For scale C1
        # self.c1_sp_kernel_sizes = [22,18]
        # self.c1_sp_kernel_sizes = [14,12]
        # self.c1_sp_kernel_sizes = [12, 10]
        self.c1_sp_kernel_sizes = [10, 8]
        # self.c1_sp_kernel_sizes = [6,5]

        ########################################################
        ########################################################

        # Setting the scale stride and number of scales pooled at a time
        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled
        
        # Global pooling (spatially)
        self.c2b_scale_stride = self.c_scale_stride
        self.c2b_num_scales_pooled = ip_scales-1 #len(self.c1_sp_kernel_sizes)  # all of them

        ########################################################
        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters = True, #s1_trainable_filters,
                     la=self.s1_la, si=self.s1_si, visualize_mode = visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale, \
                     n_phi = n_phi, scale_factor = self.scale)
        # self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.5, n_in_sbands=ip_scales,
        #             num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride, visualize_mode = visualize_mode, \
        #             c1_bool = True, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        
        self.s2b = S2(channels_in=n_ori*n_phi, channels_out=128, kernel_size=3, stride=1)

        self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)

        # # # Classifier
        # self.classifier = nn.Sequential(
        #                                 nn.Dropout(0.2),  # TODO: check if this will be auto disabled if eval
        #                                 nn.Linear(256, 128),  # fc1
        #                                 nn.ReLU(),
        #                                 nn.Linear(128, num_classes)  # fc3
        #                                 )

        # # ResNet50 Classifier
        self.classifier = nn.Sequential(
                                        nn.Dropout(0.2),  # TODO: check if this will be auto disabled if eval
                                        nn.Linear(2048, 256),  # fc1
                                        nn.ReLU(),
                                        # nn.Linear(512, 128),  # fc1
                                        # nn.ReLU(),
                                        nn.Linear(256, num_classes)  # fc3
                                        )


    def make_ip(self, x, same_scale_viz = None, base_scale = None, ip_scales = None, scale = None):

        # scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
        # # scale_factor_list = [0.841, 1, 1.189]
        # scale_factor = random.choice(scale_factor_list)
        # # print('scale_factor 1 : ', scale_factor)
        # img_hw = x.shape[-1]
        # new_hw = int(img_hw*scale_factor)
        # x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
        # # print('x_rescaled : ',x_rescaled.shape)
        # if new_hw <= img_hw:
        #     x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
        # elif new_hw > img_hw:
        #     center_crop = torchvision.transforms.CenterCrop(img_hw)
        #     x_rescaled = center_crop(x_rescaled)

        if ip_scales and scale:
            # print("In right condition")
            ip_scales = ip_scales
            scale = scale
            const_size_bool = True or self.force_const_size_bool
        else:
            ip_scales = self.ip_scales
            scale = self.scale #5
            const_size_bool = False or self.force_const_size_bool

        # if self.MNIST_Scale == 1000:
        #     center_crop = torchvision.transforms.CenterCrop(140)
        #     x = center_crop(x)

        base_image_size = int(x.shape[-1]) 
        # print('base_image_size : ',base_image_size)
        
        if ip_scales == 1:
            image_scales_down = [base_image_size]
            image_scales_up = []
        elif ip_scales == 2:
            image_scales_up = []
            image_scales_down = [np.ceil(base_image_size/(2**(1/scale))), base_image_size]
        else:
            image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(ip_scales/2)))]
            image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(ip_scales/2)))]
        
        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]

        if const_size_bool:
            base_image_size = 224
        else:
            base_image_size = int(x.shape[-1]) 

        # print('base_image_size : ',base_image_size)
        # print('self.image_scales : ',self.image_scales)

        if len(self.image_scales) > 1:
            # print('Right Hereeeeeee: ', self.image_scales)
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                # print('i_s : ',i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                if const_size_bool:
                    # Padding or Cropping
                    if i_s <= base_image_size:
                        interpolated_img = pad_to_size(interpolated_img, (base_image_size, base_image_size), pad_mode = self.pad_mode)
                    elif i_s > base_image_size:
                        center_crop = torchvision.transforms.CenterCrop(base_image_size)
                        interpolated_img = center_crop(interpolated_img)
                
                # print('interpolated_img : ',interpolated_img.shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)
                image_pyramid.append(interpolated_img)

                # # print('image_pyramid : ',image_pyramid[-1].shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)

            # print('image_pyramid len : ',len(image_pyramid))

            return image_pyramid
        else:
            # print('Hereeeeeee')
            ##############################################################################
            if self.orcale_bool:
                # FOr oracle:
                if x.shape[-1] > 224:
                    center_crop = torchvision.transforms.CenterCrop(224)
                    x = center_crop(x)
                elif x.shape[-1] < 224:
                    x = pad_to_size(x, (224, 224))
            ##############################################################################

            return [x]

    def forward(self, x, batch_idx = None):

        x_pyramid = self.make_ip(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Out 17 Scales x BxCxHxW --> C = 3

        ###############################################
        s2b_maps = []
        for p_i in range(len(x_pyramid)):
            x_p = [x_pyramid[p_i]]

            s1_maps = self.s1(x_p, self.MNIST_Scale, batch_idx, prj_name = self.prj_name, category = self.category, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 17 Scales x BxCxHxW --> C = 4
            # c1_maps = self.c1(s1_maps, x_p, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, image_scales = self.image_scales, save_rdms = self.save_rdms, plt_filters = self.plt_filters)  # Out 16 Scales x BxCxHxW --> C = 4

            ###############################################
            # ByPass Route
            s2b_maps_temp = self.s2b(s1_maps, prj_name = self.prj_name, category = self.category, MNIST_Scale = self.MNIST_Scale, x_input = x_p, save_rdms = self.save_rdms, plt_filters = self.plt_filters) # Out 15 Scales x BxCxHxW --> C = 2000

            s2b_maps.append(s2b_maps_temp[0])

        c2b_maps, max_scale_index, correct_scale_loss = self.c2b(s2b_maps, x_p, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, \
                                                                image_scales = self.image_scales, save_rdms = self.save_rdms, plt_filters = self.plt_filters, \
                                                                argmax_bool = self.argmax_bool) # Overall x BxCx1x1 --> C = 2000

        ###############################################
        c2b_maps = torch.flatten(c2b_maps[0], 1) # Shape --> 1 x B x 400 x 1 x 1 

        # # Classify
        output = self.classifier(c2b_maps)

        # ###############################################
        # RDM
        if 'clf' in self.save_rdms:
            save_tensor(output, self.MNIST_Scale, self.prj_name, self.category, base_path = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", stage = 'clf')
        ###############################################


        return output, c2b_maps, max_scale_index, correct_scale_loss


