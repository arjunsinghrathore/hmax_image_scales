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

def pad_to_size(a, size, pad_mode = 'constant'):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom), mode = pad_mode)

    return a



# #########################################################################################################
# class DeepNet_Models(nn.Module):
#     def __init__(self,
#                  num_classes=10,
#                  prj_name = None,
#                  ):
#         super(DeepNet_Models, self).__init__()
# #########################################################################################################

#         self.num_classes = num_classes

#         # self.deep_model = models.resnet50(pretrained=False)
#         self.deep_model = models.vgg16_bn(pretrained=False)
        
#         self.classifier = nn.Sequential(
#                                         nn.Dropout(0.5),
#                                         nn.Linear(1000, 256),  # fc1
#                                         nn.ReLU(),
#                                         nn.Linear(256, num_classes)  # fc2
#                                         )



#     def forward(self, x, batch_idx = None):

#         # if x.shape[1] == 3:
#         #     # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
#         #     # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
#         #     x = x[:,0:1]

#         deep_out = self.deep_model(x)
#         class_out = self.classifier(deep_out)
        

#         return deep_out
#         # return stream_2_output, stram_1_c2b_feats, max_scale_index, correct_scale_loss

#########################################################################################################
#########################################################################################################

def get_gabor(l_size, la, si, n_ori, aspect_ratio, n_phi):
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
    th = np.array(range(n_ori)) * np.pi / n_ori + np.pi / 2.
    th = th[sp.newaxis, sp.newaxis, :]

    ######################## Phi ########################
    phi_s = np.array(range(n_phi)) * np.pi / 4 + np.pi / 2.
    # print('complex phi_s : ',phi_s.shape)
    # print('complex phi_s : ',phi_s)
    # phi_s = phi_s[sp.newaxis, sp.newaxis, :]
    ######################################################


    hgs = (gs - 1) / 2.
    yy, xx = sp.mgrid[-hgs: hgs + 1, -hgs: hgs + 1]
    xx = xx[:, :, sp.newaxis];
    yy = yy[:, :, sp.newaxis]

    x = xx * np.cos(th) - yy * np.sin(th)
    y = xx * np.sin(th) + yy * np.cos(th)

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
    def __init__(self, scale, n_ori, padding, trainable_filters, la, si, visualize_mode = False, prj_name = None, MNIST_Scale = None, n_phi = 1):

        super(S1, self).__init__()

        self.scale = scale
        self.la = la
        self.si = si
        self.visualize_mode = visualize_mode
        self.prj_name = prj_name
        self.MNIST_Scale = MNIST_Scale
        self.padding = padding

        # setattr(self, f's_{scale}', nn.Conv2d(1, n_ori, scale, padding=padding))
        # s1_cell = getattr(self, f's_{scale}')
        self.gabor_filter = get_gabor(l_size=scale, la=la, si=si, n_ori=n_ori, aspect_ratio=0.3, n_phi = n_phi)  # ??? What is aspect ratio
        print('self.gabor_filter : ', self.gabor_filter.shape)
        # self.gabor_weights = nn.Parameter(self.gabor_filter, requires_grad=False)
        # s1_cell.weight = nn.Parameter(gabor_filter, requires_grad=trainable_filters)
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
        

        x = x_pyramid

        # s1_cell = getattr(self, f's_{self.scale}')
        # s1_map = torch.abs(s1_cell(x))  # adding absolute value
        # s1_map = s1_cell(x)
        s1_map = F.conv2d(x, self.gabor_filter.to(device='cuda'), None, 4, self.padding)

        # Normalization
        # s1_unorm = getattr(self, f's_uniform_{self.scale}')
        # s1_unorm = torch.sqrt(s1_unorm(x**2))
        # # s1_unorm = torch.sqrt(s1_unorm(x))
        # s1_unorm.data[s1_unorm == 0] = 1  # To avoid divide by zero
        # s1_map /= s1_unorm
        s1_map = self.batchnorm(s1_map)


        # Padding (to get s1_maps in same size) ---> But not necessary for us
        ori_size = (x.shape[-2], x.shape[-1])
        s1_map = pad_to_size(s1_map, ori_size)

        return s1_map

#########################################################################################################
class DeepNet_Models(nn.Module):
    def __init__(self,
                 num_classes=10,
                 prj_name = None,
                 ):
        super(DeepNet_Models, self).__init__()
#########################################################################################################

        self.num_classes = num_classes

        self.base_image_size = 224


        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        # self.layer1 = S1(scale=21, n_ori=12, padding='valid', trainable_filters = False, #s1_trainable_filters,
        #              la=11.5, si=9.2, visualize_mode = False, prj_name = prj_name, MNIST_Scale = 2, \
        #              n_phi = 6)

        # self.bottleneck = nn.Conv2d(72, 64, kernel_size=1, stride=1, bias=False)

        # AlexNet
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

        # VGG
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(72, 64, kernel_size=3, stride=1, padding=1),
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
        #     nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
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


        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(7*7*512, 4096),
        #     nn.ReLU())
        # self.fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU())
        # self.fc2= nn.Sequential(
        #     nn.Linear(4096, num_classes))

        # self.classifier = nn.Sequential(
        #                                 nn.Dropout(0.2),
        #                                 nn.Linear(512, 256),  # fc1
        #                                 nn.ReLU(),
        #                                 nn.Linear(256, num_classes)  # fc2
        #                                 )



    def forward(self, x, batch_idx = None):

        if x.shape[-1] < self.base_image_size:
            x = pad_to_size(x, (self.base_image_size, self.base_image_size), pad_mode = 'reflect')
        elif x.shape[-1] > self.base_image_size:
            center_crop = torchvision.transforms.CenterCrop(self.base_image_size)
            x = center_crop(x)

        # if x.shape[1] == 3:
        #     # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
        #     # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
        #     x = x[:,0:1]

        # deep_out = self.deep_model(x)
        # class_out = self.classifier(deep_out)

        # out = self.layer1(x)
        # out = self.bottleneck(out)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        # print('out : ',out.shape)
        out = self.fc(out)
        out = self.fc1(out)
        deep_out = self.fc2(out)
        
        return deep_out
        # return stream_2_output, stram_1_c2b_feats, max_scale_index, correct_scale_loss

