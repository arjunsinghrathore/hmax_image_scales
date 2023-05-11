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
from utils.foveation_and_cortical_magn import warp_image

import random


# seed_everything(42, workers=True)

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


#########################################################################################################
class HMAX_2_streams(nn.Module):
    def __init__(self,
                 num_classes=10,
                 prj_name = None,
                 model_pre = None
                 ):
        super(HMAX_2_streams, self).__init__()
#########################################################################################################

        self.num_classes = num_classes

        self.model_pre = model_pre
        # No Image Scale Pyramid
        self.model_pre.ip_scales = 1
        self.model_pre.single_scale_bool = False

        self.model_pre.force_const_size_bool = True

        self.stream_1_big = False
        
        self.stream_2_bool = True
        if self.stream_2_bool:
            self.stream_2_ip_scales = 5
            self.stream_2_scale = 4



    def forward(self, x, batch_idx = None):

        if x.shape[1] == 3:
            # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
            # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
            x = x[:,0:1]

        correct_scale_loss = 0.

        if self.stream_1_big:
            scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
            # scale_factor_list = [0.841, 1, 1.189]
            scale_factor = random.choice(scale_factor_list)
            # print('scale_factor 1 : ', scale_factor)
            img_hw = x.shape[-1]
            new_hw = int(img_hw*scale_factor)
            x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
            # print('x_rescaled : ',x_rescaled.shape)
            if new_hw <= img_hw:
                x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
            elif new_hw > img_hw:
                center_crop = torchvision.transforms.CenterCrop(img_hw)
                x_rescaled = center_crop(x_rescaled)
            # print('x_rescaled : ',x_rescaled.shape)
            
            stream_1_output, stram_1_c2b_feats, max_scale_index, _ = self.model_pre(x_rescaled, batch_idx, ip_scales = self.stream_2_ip_scales, scale = self.stream_2_scale)
            
        else:
            # print('Hereeeeeeeee')
            stream_1_output, stram_1_c2b_feats, max_scale_index, _ = self.model_pre(x, batch_idx) #, ip_scales = 2, scale = 4)

        if self.stream_2_bool:
            # print('Wrongggggg')
            scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
            # scale_factor_list = [0.841, 1, 1.189]
            scale_factor = random.choice(scale_factor_list)
            # print('scale_factor 2 : ', scale_factor)
            img_hw = x.shape[-1]
            new_hw = int(img_hw*scale_factor)
            x_rescaled = F.interpolate(x, size = (new_hw, new_hw), mode = 'bilinear').clamp(min=0, max=1)
            # print('x_rescaled : ',x_rescaled.shape)
            if new_hw <= img_hw:
                x_rescaled = pad_to_size(x_rescaled, (img_hw, img_hw))
            elif new_hw > img_hw:
                center_crop = torchvision.transforms.CenterCrop(img_hw)
                x_rescaled = center_crop(x_rescaled)
            # print('x_rescaled : ',x_rescaled.shape)
            
            stream_2_output, stram_2_c2b_feats, _, _ = self.model_pre(x_rescaled, batch_idx, ip_scales = self.stream_2_ip_scales, scale = self.stream_2_scale)

            correct_scale_loss = torch.mean(torch.abs(stram_1_c2b_feats - stram_2_c2b_feats))

            return stream_2_output, stram_1_c2b_feats, max_scale_index, correct_scale_loss

        

        return stream_1_output, stram_1_c2b_feats, max_scale_index, correct_scale_loss
        # return stream_2_output, stram_1_c2b_feats, max_scale_index, correct_scale_loss
