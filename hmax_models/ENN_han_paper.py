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
import _pickle as pickle

from utils.save_tensors import save_tensor
from utils.plot_filters import plt_filter_func

import random

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

class ENN_YH(nn.Module):
    def __init__(self,
                 ip_scales = 18,
                 num_classes=1000,
                 prj_name = None,
                 MNIST_Scale = None,
                 category = None,
                 ):
###########################################################
        super(ENN_YH, self).__init__()

        self.ip_scales = 11
        self.single_scale_bool = True
        self.force_const_size_bool = True

        # self.argmax_bool = False

        self.num_classes = num_classes
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.scale = 4
        

        ########################################################
        ########################################################

        # Feature extractors (in the order of the table in Figure 1)
        self.layer_1 = nn.Sequential(
                                    nn.Conv2d(3, 128, kernel_size = 11, stride = 1),
                                    nn.MaxPool2d(kernel_size=5, stride=2)
        )
        self.layer_2 = nn.Sequential(
                                    nn.Conv2d(128, 256, kernel_size = 5, stride = 1),
                                    nn.MaxPool2d(kernel_size=5, stride=2)
        )
        self.layer_3 = nn.Sequential(
                                    nn.Conv2d(256, 512, kernel_size = 5, stride = 1),
                                    nn.MaxPool2d(kernel_size=5, stride=2)
        )
        self.layer_4 = nn.Sequential(
                                    nn.Conv2d(512, 512, kernel_size = 5, stride = 1),
                                    nn.MaxPool2d(kernel_size=5, stride=2)
        )
    
        ########################################################
        # # Classifier
        self.classifier = nn.Sequential(
                                        # nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        nn.Linear(512, 256),  # fc1
                                        nn.Dropout(0.2),  # TODO: check if this will be auto disabled if eval
                                        # nn.BatchNorm1d(4096, 1e-3),
                                        # nn.ReLU(True),
                                        # nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        # nn.Linear(4096, 4096),  # fc2
                                        # nn.BatchNorm1d(256, 1e-3),
                                        # nn.ReLU(True),
                                        nn.Linear(256, num_classes)  # fc3
                                        )

        # self.overall_max_scale_index = []



    def make_ip(self, x, ip_scales = None, scale = None):

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
                        interpolated_img = pad_to_size(interpolated_img, (base_image_size, base_image_size))
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

        x_pyramid = self.make_ip(x) # Out 17 Scales x BxCxHxW --> C = 3

        feature_out_list = []
        for i in range(len(x_pyramid)):  # assuming S is last dimension

                x = x_pyramid[i]

                feature_out = self.layer_1(x)
                feature_out = self.layer_2(feature_out)
                feature_out = self.layer_3(feature_out)
                feature_out = self.layer_4(feature_out)

                feature_out_list.append(feature_out)

        scale_max = []
        for i in range(len(feature_out_list)):
                s_m = F.max_pool2d(feature_out_list[i], feature_out_list[i].shape[-1], 1)
                scale_max.append(s_m)

        #############################################################################
        scale_max_index = torch.stack(scale_max, dim=0) # Shape --> Scale x B x C
        scale_max_index = scale_max_index.squeeze()
        # print('scale_max_index : ',scale_max_index.shape)

        max_scale_index = [0]*scale_max_index.shape[1]
        # max_scale_index = torch.tensor(max_scale_index).cuda()
        for p_i in range(1, len(x_pyramid)):
            
            for b_i in range(scale_max_index.shape[1]):

                # print(f'b_i : {b_i}, p_i {p_i}')
                # print('scale_max[max_scale_index[b_i]][b_i] : ',scale_max[max_scale_index[b_i]][b_i])
                # print('scale_max[p_i][b_i] : ',scale_max[p_i][b_i])

                scale_max_argsort = torch.argsort(torch.stack([scale_max_index[max_scale_index[b_i]][b_i], scale_max_index[p_i][b_i]], dim=0), dim = 0) # Shape --> 2 x C
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

        #############################################################################

        scale_max = torch.stack(scale_max, dim=4)


        scale_max, _ = torch.max(scale_max, dim=4)
        scale_max = scale_max.squeeze()

        output = self.classifier(scale_max)

        return output, max_scale_index

#############################################################################
#############################################################################

