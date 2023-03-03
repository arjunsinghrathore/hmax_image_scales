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

seed_everything(42, workers=True)

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


class conv_bn_relu_func(nn.Module):
    def __init__(self, channels_in, kernel_size, stride):
        super(conv_bn_relu_func, self).__init__()

        self.conv_bn_relu_seq = nn.Sequential(
                                        nn.Conv2d(channels_in, 16, kernel_size, 1),
                                        nn.BatchNorm2d(16, 1e-3),
                                        nn.ReLU(True),

                                        nn.Conv2d(16, 16, kernel_size, stride),
                                        nn.BatchNorm2d(16, 1e-3),
                                        nn.ReLU(True),

                                        nn.Conv2d(16, 32, kernel_size, 1),
                                        nn.BatchNorm2d(32, 1e-3),
                                        nn.ReLU(True),

                                        nn.Conv2d(32, 32, kernel_size, stride),
                                        nn.BatchNorm2d(32, 1e-3),
                                        nn.ReLU(True),
                                        )

    def forward(self, x_pyramid):
        c_bn_r_maps = []
        # Loop over scales applying convolution --> batch normalizing --> relu
        # print('len x_pyramid conv_bn_relu_func :', len(x_pyramid))
        for p_i in range(len(x_pyramid)):
            c_bn_r_map = self.conv_bn_relu_seq(x_pyramid[p_i])
            c_bn_r_maps.append(c_bn_r_map)

        c_bn_r_maps = torch.stack(c_bn_r_maps, dim=4)

        return c_bn_r_maps


# # MaxPooling Over Positions First Then Scales
# class max_poooling_func(nn.Module):
#     def __init__(self):
#         super(max_poooling_func, self).__init__()
        
#     def forward(self, feature_maps_pyramid):

#         # Global MaxPool over positions for each scale separately
#         scale_max = []
#         # print('len feature_maps_pyramid max_poooling_func :', len(feature_maps_pyramid))
#         for p_i in range(len(feature_maps_pyramid)):
#             s_m = F.max_pool2d(feature_maps_pyramid[p_i], feature_maps_pyramid[p_i].shape[-1], 1)
#             scale_max.append(s_m)

#         # Option 1:: Global Max Pooling over Scale i.e, Maxpool over scale groups
#         scale_max = torch.stack(scale_max, dim=4)
#         feature_out, _ = torch.max(scale_max, dim=4)

#         # Option 2:: Global Avg Pooling over Scale i.e, Avgpool over scale groups

#         return feature_out  

# # MaxPooling Over Scales Directly
# class max_poooling_func(nn.Module):
#     def __init__(self):
#         super(max_poooling_func, self).__init__()
        
#     def forward(self, feature_maps_pyramid):

#         # Option 1:: Global Max Pooling over Scale i.e, Maxpool over scale groups
#         feature_out, _ = torch.max(feature_maps_pyramid, dim=4)

#         # Option 2:: Global Avg Pooling over Scale i.e, Avgpool over scale groups

#         return feature_out  

class max_poooling_func(nn.Module):
    def __init__(self):
        super(max_poooling_func, self).__init__()
        
    def forward(self, flattened_feature_maps_pyramid):

        # Global MaxPool
        flattened_feature_maps_pyramid = torch.stack(flattened_feature_maps_pyramid, dim=2)
        feature_out, _ = torch.max(flattened_feature_maps_pyramid, dim=2)

        # Option 2:: Global Avg Pooling over Scale i.e, Avgpool over scale groups

        return feature_out  


#########################################################################################################
#########################################################################################################
class fov_max(nn.Module):
    def __init__(self,
                 ip_scales = 18,
                 num_classes=10,
                 ):
        super(fov_max, self).__init__()

        self.ip_scales = ip_scales # 18
        self.num_classes = num_classes
        self.scale = 4
        
        ########################################################
        # Feature extractors (in the order conv_bn_relu_func --> max_poooling_func)
        self.conv_bn_relu = conv_bn_relu_func(channels_in = 1, kernel_size = 3, stride = 2)
        self.max_poooling = max_poooling_func()
        ########################################################

        # # Classifier
        # self.classifier = nn.Sequential(
        #                                 nn.Linear(32, 100),  
        #                                 nn.Dropout(0.15),
        #                                 nn.Linear(100, 10)  
        #                                 )
        
        self.linear_layer = nn.Sequential(
                                        nn.Linear(32, 100),  
                                        nn.Dropout(0.15),  
                                        )
        self.classifier = nn.Linear(100, 10)

    def make_ip(self, x):

        base_image_size = 112
        center_crop = torchvision.transforms.CenterCrop(base_image_size)
        x = center_crop(x)

        # base_image_size = int(x.shape[-1]) 
        scale = self.scale #5
        
        if self.ip_scales == 1:
            image_scales_down = [base_image_size]
            image_scales_up = []
        elif self.ip_scales == 2:
            image_scales_up = [base_image_size, np.ceil(base_image_size*(2**(1/scale)))]
            image_scales_down = []
        else:
            image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(self.ip_scales/2)))]
            image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(self.ip_scales/2)))]
        
        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]

        # print('image_scales : ',self.image_scales)

        base_image_size = 112

        if len(self.image_scales) > 1:
            # print('Right Hereeeeeee: ', self.image_scales)
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                # print('i_s : ',i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)

                # Gauss Blur
                # kernel_size_gauss = int(3*(i_s/base_image_size))
                # if kernel_size_gauss%2 == 0:
                #     kernel_size_gauss = kernel_size_gauss + 1
                # interpolated_img = torchvision.transforms.functional.gaussian_blur(interpolated_img, kernel_size_gauss, sigma = (7/8)*(i_s/base_image_size)).clamp(min=0, max=1)

                # Padding or Cropping
                if i_s <= base_image_size:
                    pad_input = pad_to_size(interpolated_img, (base_image_size, base_image_size))
                    image_pyramid.append(pad_input) # ??? Wgats is the range?
                elif i_s > base_image_size:
                    center_crop = torchvision.transforms.CenterCrop(base_image_size)
                    image_pyramid.append(center_crop(interpolated_img))

                # # # No Padding or Cropping
                # image_pyramid.append(interpolated_img)

                # # print('image_pyramid : ',image_pyramid[-1].shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)

            return image_pyramid

        else:
            # print('Hereeeeeee')
            ##############################################################################
            if self.orcale_bool:
            # if True:
                # FOr oracle:
                if x.shape[-1] > base_image_size:
                    center_crop = torchvision.transforms.CenterCrop(base_image_size)
                    x = center_crop(x)
                elif x.shape[-1] < base_image_size:
                    x = pad_to_size(x, (base_image_size, base_image_size))
            ##############################################################################

            return [x]

    def forward(self, x, batch_idx = None):

        if x.shape[1] == 3:
            # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
            # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
            x = x[:,0:1]

        max_scale_index = []
        correct_scale_loss = 0

        ###############################################
        x_pyramid = self.make_ip(x)

        ###############################################
        conv_bn_relu_maps = self.conv_bn_relu(x_pyramid) # Batch, Channels, H, W, Scales
        # max_poooling_maps = self.max_poooling(conv_bn_relu_maps)
        # print('max_poooling_maps : ',max_poooling_maps.shape)

        ###############################################
        # max_poooling_maps_flattened = torch.flatten(max_poooling_maps, 1) 

        linear_out_stack = []
        # print('conv_bn_relu_maps : ',conv_bn_relu_maps.shape)
        for s_ii in range(conv_bn_relu_maps.shape[4]):
            # conv_bn_relu_maps_flattned = torch.flatten(conv_bn_relu_maps[:,:,:,:,s_ii], 1)
            conv_bn_relu_maps_flattned = torch.flatten(F.avg_pool2d(conv_bn_relu_maps[:,:,:,:,s_ii], conv_bn_relu_maps[:,:,:,:,s_ii].shape[-1], 1), 1)
            # print('conv_bn_relu_maps_flattned : ',conv_bn_relu_maps_flattned.shape)

            linear_out = self.linear_layer(conv_bn_relu_maps_flattned)
            final_linear_out = self.classifier(linear_out)

            linear_out_stack.append(final_linear_out)
        
        # linear_out_stack --> Scales, num_classes
        max_poooling_out = self.max_poooling(linear_out_stack)
        # print('max_poooling_out : ',max_poooling_out.shape)

        # Classify
        # output = self.classifier(max_poooling_maps_flattened)
        # output = self.classifier(max_poooling_maps)

        return max_poooling_out , max_scale_index, correct_scale_loss

#############################################################################
#############################################################################