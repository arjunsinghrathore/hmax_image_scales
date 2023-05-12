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


# seed_everything(42, workers=True)


#########################################################################################################
class HMAX_finetune(nn.Module):
    def __init__(self,
                 num_classes=10,
                 prj_name = None,
                 model_pre = None
                 ):
        super(HMAX_finetune, self).__init__()
#########################################################################################################

        self.num_classes = num_classes

        self.model_pre = model_pre
        # # No Image Scale Pyramid
        self.model_pre.ip_scales = 1
        self.model_pre.single_scale_bool = True
        # # # Image Scale Pyramid
        # self.model_pre.ip_scales = 18
        # self.model_pre.single_scale_bool = False

        self.model_pre.force_const_size_bool = True

        # # Classifier
        self.classifier_class = nn.Sequential(
                                        # nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        nn.Linear(self.model_pre.get_s4_in_channels(), 256),  # fc1
                                        # nn.Linear(self.get_s4_in_channels(), 512),  # fc1
                                        nn.Dropout(0.2),
                                        # nn.Linear(512, 256),  # fc1
                                        # nn.BatchNorm1d(4096, 1e-3),
                                        # nn.ReLU(True),
                                        # nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        # nn.Linear(4096, 4096),  # fc2
                                        # nn.BatchNorm1d(256, 1e-3),
                                        # nn.ReLU(True),
                                        nn.Linear(256, num_classes)  # fc3
                                        )

        # self.overall_max_scale_index = []


    def forward(self, x, batch_idx = None):

        if x.shape[1] == 3:
            # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
            # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
            x = x[:,0:1]

        _, pre_feats, max_scale_index, correct_scale_loss = self.model_pre(x, batch_idx)

        ###############################################
        pre_feats_flatten = torch.flatten(pre_feats, 1) # Shape --> 1 x B x 400 x 1 x 1 

        # Classify
        output = self.classifier_class(pre_feats_flatten)

        return output, pre_feats, max_scale_index, correct_scale_loss