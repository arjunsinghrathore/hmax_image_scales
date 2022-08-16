import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn import init
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
np.random.seed(1)
# import pandas as pd
import cv2
import _pickle as pickle
import math
# from PIL import Image
# from torchsummary import summary
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from torch.autograd import Variable

import torchvision.models as models

from collections import OrderedDict
import skimage.color as sic

from sklearn.decomposition import PCA

import pytorch_lightning as pl
# from sklearn.preprocessing import LabelEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb


import scipy as sp
import h5py
from scipy.ndimage.filters import correlate
# from hmax_models.hmax import HMAX_latest_slim
# from hmax_models.hmax_changes import * 
from hmax_models.hmax_ip import * 
# from hmax_models.hmax_ip_basic import * 
from hmax_models.hmax_ivan import HMAX_latest_slim, HMAX_latest

import argparse
import os
import random
import shutil
import time
import warnings

from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)
# Seeds

# import random
# random.seed(1)




class HMAX_trainer(pl.LightningModule):
    def __init__(self, prj_name, n_ori, n_classes, lr, weight_decay, ip_scales, IP_bool = False, visualize_mode = False, \
                 MNIST_Scale = None, first_scale_test = False):
        super().__init__()
        
        self.parameter_dict = {'prj_name':prj_name, 'n_ori':n_ori, 'n_classes':n_classes, \
                                'lr':lr, 'weight_decay':weight_decay, 'ip_scales':ip_scales, \
                                'first_scale_test':first_scale_test, 'visualize_mode':visualize_mode, 'MNIST_Scale':MNIST_Scale}

        print('self.parameter_dict : ',self.parameter_dict)

        self.prj_name = prj_name 
        self.n_ori = n_ori 
        self.n_classes = n_classes 
        self.lr = lr 
        self.weight_decay = weight_decay 
        self.ip_scales = ip_scales
        self.IP_bool = IP_bool
        self.first_scale_test = first_scale_test
        self.visualize_mode = visualize_mode
        self.MNIST_Scale = MNIST_Scale

        ########################## While Testing ##########################
        
        ###################################################################

        if not self.IP_bool:
            # self.HMAX = HMAX(n_ori=self.n_ori,num_classes=self.n_classes)
            self.HMAX = HMAX_latest_slim(n_ori=self.n_ori,num_classes=self.n_classes)
        else:
            self.HMAX = HMAX_IP(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
            # self.HMAX = HMAX_IP_basic(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes)


        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss(reduce = False)


        # Val Loss
        self.val_losses = []
        self.min_loss = 1000
        self.acc1_list = []
        self.acc1_max = 0
        self.acc5_list = []
        self.acc5_max = 0

        # log hyperparameters
        self.save_hyperparameters()


    def force_cudnn_initialization(self):
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


    def forward(self, x, batch_idx = None):

        
        # if self.conv_init:
        #     self.force_cudnn_initialization()
        #     self.conv_init = False
        
        hmx_out = self.HMAX(x, batch_idx)

        return hmx_out
    
    #pytorch lighning functions
    def configure_optimizers(self):
        print('lrrr  : ',self.lr)
        optimiser = torch.optim.Adam(self.parameters(), self.lr, weight_decay = self.weight_decay)
        return optimiser

    # # learning rate warm-up
    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     # update params
    #     optimizer.step(closure=optimizer_closure)

    #     # Manual Learning Rate Schedule // 30))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr


    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res

    def training_step(self, batch, batch_idx):

        images, target = batch

        # print('target valueeeees : ',target)
        # print('target : ',target.shape)
        # print('images : ',images.shape)
        # print('target : ',target.dtype)
        # print('images : ',images.dtype)

        h_w = images.shape[-1]
        if len(images.shape) == 4:
            images = images.reshape(-1, 3, h_w, h_w)
            target = target.reshape(-1)

        ########################
        output = self(images)

        ########################
        loss = self.criterion(output, target)

        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

        ########################
        loss = torch.mean(loss)

        ########################
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_performance", {"acc1": acc1, "acc5": acc5})
        self.log('train_acc1', acc1, on_step=True, on_epoch=True, prog_bar=True)

        return loss #{'output' : output, 'target' : target} 

    # def training_step_end(self, out_tar):

    #     loss = criterion(out_tar['output'], out_tar['target'])

    #     acc1, acc5 = accuracy(output, target, topk=(1, 5))

    #     self.log('train_loss', loss,on_step=False, on_epoch=True,prog_bar=True)
        
    #     return loss


    def validation_step(self, batch, batch_idx):
        
        images, target = batch

        h_w = images.shape[-1]
        if len(images.shape) == 4:
            images = images.reshape(-1, 3, h_w, h_w)
            target = target.reshape(-1)

        # print('target : ',target.shape)
        # print('images : ',images.shape)
        # print('target : ',target.dtype)
        # print('images : ',images.dtype)

        ########################
        output = self(images)

        ########################
        val_loss = self.criterion(output, target)

        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

        ########################
        time.sleep(1)
        val_loss_list = val_loss.cpu().tolist()
        # print('val_loss_list : ',val_loss_list) 
        self.val_losses += val_loss_list

        acc1_list = acc1.cpu().tolist()
        self.acc1_list += acc1_list
        acc5_list = acc5.cpu().tolist()
        self.acc5_list += acc5_list

        ###########################
        # val_loss = torch.mean(val_loss)

        # return val_loss
        return acc1

    def validation_epoch_end(self,losses):

        losses = self.val_losses
        losses = np.array(losses)
        print('losses : ',losses.shape)
        losses = np.mean(losses)

        acc1_list = self.acc1_list
        acc1_list = np.array(acc1_list)
        print('acc1_list : ',acc1_list.shape)
        acc1_list = np.mean(acc1_list)

        acc5_list = self.acc5_list
        acc5_list = np.array(acc5_list)
        acc5_list = np.mean(acc5_list)

        #################################
        if losses < self.min_loss:
            self.min_loss = losses

        if acc1_list > self.acc1_max:
            self.acc1_max = acc1_list

        if acc5_list > self.acc5_max:
            self.acc5_max = acc5_list

        #################################
        result_summary = OrderedDict()

        result_summary["error" + "_mean"] = losses
        result_summary["Performance" + "_mean_top1"] = acc1_list
        result_summary["Performance" + "_mean_top5"] = acc5_list

        ##################################
        print(result_summary)

        ##################################
        self.log('val_loss', losses, on_step=False, on_epoch=True, prog_bar=True) #,on_step=False, on_epoch=True,prog_bar=True)
        # self.log("val_performance", {"acc1": acc1, "acc5": acc5})
        self.log('val_acc1', acc1_list, on_step=False, on_epoch=True, prog_bar=True)

        ##################################
        self.val_losses = []
        self.acc1_list = []
        self.acc5_list = []


    def test_step(self, batch, batch_idx):
        
        images, target = batch

        h_w = images.shape[-1]
        if len(images.shape) == 4:
            images = images.reshape(-1, 3, h_w, h_w)
            target = target.reshape(-1)

        # print('target valueeeees : ',target)
        # print('target : ',target.shape)
        # print('images : ',images.shape)
        # print('target : ',target.dtype)
        # print('images : ',images.dtype)

        ########################
        output = self(images, batch_idx)

        # return 0

        ########################
        val_loss = self.criterion(output, target)

        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

        ########################
        time.sleep(1)
        val_loss_list = val_loss.cpu().tolist()
        # print('val_loss_list : ',val_loss_list) 
        self.val_losses += val_loss_list

        acc1_list = acc1.cpu().tolist()
        self.acc1_list += acc1_list
        acc5_list = acc5.cpu().tolist()
        self.acc5_list += acc5_list

        ###########################
        # val_loss = torch.mean(val_loss)

        # return val_loss
        return acc1

    def test_epoch_end(self,losses):

        losses = self.val_losses
        losses = np.array(losses)
        print('losses : ',losses.shape)
        losses = np.mean(losses)

        acc1_list = self.acc1_list
        acc1_list = np.array(acc1_list)
        print('acc1_list : ',acc1_list.shape)
        acc1_list = np.mean(acc1_list)

        acc5_list = self.acc5_list
        acc5_list = np.array(acc5_list)
        acc5_list = np.mean(acc5_list)

        #################################
        if losses < self.min_loss:
            self.min_loss = losses

        if acc1_list > self.acc1_max:
            self.acc1_max = acc1_list

        if acc5_list > self.acc5_max:
            self.acc5_max = acc5_list

        #################################
        result_summary = OrderedDict()

        result_summary["error" + "_mean"] = losses
        result_summary["Performance" + "_mean_top1"] = acc1_list
        result_summary["Performance" + "_mean_top5"] = acc5_list

        ##################################
        print(result_summary)

        ##################################
        self.log('test_loss', losses, on_step=False, on_epoch=True, prog_bar=True) #,on_step=False, on_epoch=True,prog_bar=True)
        # self.log("val_performance", {"acc1": acc1, "acc5": acc5})
        self.log('test_acc1', acc1_list, on_step=False, on_epoch=True, prog_bar=True)

        ##################################
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/save_pickles", self.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "scale_test_acc.pkl")

        if not self.first_scale_test:
            open_file = open(file_name, "rb")
            test_accs = pickle.load(open_file)
            open_file.close()
            test_accs = test_accs + [acc1_list]
            ########################
            open_file = open(file_name, "wb")
            pickle.dump(test_accs, open_file)
            open_file.close()
        else:
            test_accs = [acc1_list]
            open_file = open(file_name, "wb")
            pickle.dump(test_accs, open_file)
            open_file.close()

        ##################################
        self.val_losses = []
        self.acc1_list = []
        self.acc5_list = []

        

    