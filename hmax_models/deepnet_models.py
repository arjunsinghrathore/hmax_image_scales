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
class DeepNet_Models(nn.Module):
    def __init__(self,
                 num_classes=10,
                 prj_name = None,
                 ):
        super(DeepNet_Models, self).__init__()
#########################################################################################################

        self.num_classes = num_classes

        # self.deep_model = models.resnet50(pretrained=False)
        self.deep_model = models.vgg16_bn(pretrained=False)
        
        self.classifier = nn.Sequential(
                                        nn.Dropout(0.5),
                                        nn.Linear(1000, 256),  # fc1
                                        nn.ReLU(),
                                        nn.Linear(256, num_classes)  # fc2
                                        )



    def forward(self, x, batch_idx = None):

        # if x.shape[1] == 3:
        #     # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
        #     # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
        #     x = x[:,0:1]

        deep_out = self.deep_model(x)
        class_out = self.classifier(deep_out)
        

        return deep_out
        # return stream_2_output, stram_1_c2b_feats, max_scale_index, correct_scale_loss