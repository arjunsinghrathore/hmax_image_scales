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
from hmax_models.hmax_ivan import HMAX_latest_slim

from hmax_models.hmax_ip_basic_single_band import HMAX_IP_basic_single_band

from hmax_models.hmax_ip_basic_single_band_caps import HMAX_IP_basic_single_band_caps

from hmax_models.CapsNet import CapsNet

import argparse
import os
import random
import shutil
import time
import warnings

from pytorch_lightning import Trainer, seed_everything

# torch.autograd.set_detect_anomaly(True)

seed_everything(42, workers=True)
# Seeds

# import random
# random.seed(1)




class HMAX_trainer(pl.LightningModule):
    def __init__(self, prj_name, n_ori, n_classes, lr, weight_decay, ip_scales, IP_bool = False, visualize_mode = False, \
                 MNIST_Scale = None, first_scale_test = False, capsnet_bool = False, IP_capsnet_bool = False):
        super().__init__()
        
        self.parameter_dict = {'prj_name':prj_name, 'n_ori':n_ori, 'n_classes':n_classes, \
                                'lr':lr, 'weight_decay':weight_decay, 'ip_scales':ip_scales, \
                                'first_scale_test':first_scale_test, 'visualize_mode':visualize_mode, \
                                'MNIST_Scale':MNIST_Scale, 'capsnet_bool':capsnet_bool, 'IP_capsnet_bool':IP_capsnet_bool}

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

        self.capsnet_bool = capsnet_bool
        self.IP_capsnet_bool = IP_capsnet_bool

        ########################## While Testing ##########################
        
        ###################################################################

        if self.capsnet_bool:
            self.CapsuleNet = CapsNet()
        elif self.IP_capsnet_bool:
            self.HMAX = HMAX_IP_basic_single_band_caps(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        elif not self.IP_bool:
            # self.HMAX = HMAX(n_ori=self.n_ori,num_classes=self.n_classes)
            self.HMAX = HMAX_latest_slim(n_ori=self.n_ori,num_classes=self.n_classes)
            # self.HMAX = HMAX_latest(n_ori=self.n_ori,num_classes=self.n_classes)
        else:
            # self.HMAX = HMAX_IP(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
            #                     visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)

            # self.HMAX = HMAX_IP_sep_ideal(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                # visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
            # self.HMAX = HMAX_IP_basic(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes)

            self.HMAX = HMAX_IP_basic_single_band(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)


        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss(reduce = False)

        self.overall_max_scale_index = []


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


    def forward(self, x, batch_idx = None, target = None):

        
        # if self.conv_init:
        #     self.force_cudnn_initialization()
        #     self.conv_init = False

        if self.capsnet_bool:
            caps_out = self.CapsuleNet(x, target)
            return caps_out
        else:
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
        n_c = images.shape[1]
        if len(images.shape) == 4:
            images = images.reshape(-1, n_c, h_w, h_w)
            target = target.reshape(-1)

        ########################
        if self.capsnet_bool:
            target_eye = torch.sparse.torch.eye(10).cuda()
            target_eye = target_eye.index_select(dim=0, index=target)

            images = images[:,0:1]

            output, reconstructions, masked = self(images, target = target_eye)

            loss = self.CapsuleNet.loss(images, output, target_eye, reconstructions)

            acc1 = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target_eye.data.cpu().numpy(), 1)) / float(images.shape[0])
            # acc1 = torch.tensor(acc1)

        elif self.IP_capsnet_bool:

            # Getting Target Eye
            target_eye = torch.sparse.torch.eye(10).cuda()
            target_eye = target_eye.index_select(dim=0, index=target)

            # With recon
            images = images[:,0:1]

            # With recon
            output, reconstructions = self(images, target = target_eye)
            # No recon
            # output = self(images)

            # Loss
            # With recon
            loss = self.HMAX.loss(images, output, target_eye, reconstructions)
            # No recon
            # loss = self.HMAX.margin_loss(output, target_eye)

            # Getting masked for calculating acc
            # FInding the length of the vector which is the probability
            classes = torch.sqrt((output ** 2).sum(2))
            classes = F.softmax(classes)
            
            _, max_length_indices = classes.max(dim=1)
            masked = Variable(torch.sparse.torch.eye(10))
            if True:
                masked = masked.cuda()
            masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)

            # Getting accuracy
            acc1 = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target_eye.data.cpu().numpy(), 1)) / float(images.shape[0])

        else:
            output = self(images)

            ########################
            loss = self.criterion(output, target)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

            ########################
            loss = torch.mean(loss)

        ########################
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_performance", {"acc1": acc1, "acc5": acc5})
        self.log('train_acc1', acc1, on_step=True, on_epoch=True, prog_bar=True)

        return loss #{'output' : output, 'target' : target} 

    def training_step_end(self, losses):
        
        # for name, param in self.named_parameters():
        #     if name == 'HMAX.s2b.s_1.weight':
        #         # print('name : ',name)
        #         param.data = param.data / ((8/4)**2)
        #     elif name == 'HMAX.s2b.s_2.weight':
        #         # print('name : ',name)
        #         param.data = param.data / ((12/4)**2)
        #     elif name == 'HMAX.s2b.s_3.weight':
        #         # print('name : ',name)
        #         param.data = param.data / ((16/4)**2)
        

        
        losses = torch.mean(losses)

        self.log('train_loss', losses,on_step=True, on_epoch=True,prog_bar=True)
        
        return losses


    def validation_step(self, batch, batch_idx):
        
        images, target = batch

        h_w = images.shape[-1]
        n_c = images.shape[1]
        if len(images.shape) == 4:
            images = images.reshape(-1, n_c, h_w, h_w)
            target = target.reshape(-1)

        # print('target : ',target.shape)
        # print('images : ',images.shape)
        # print('target : ',target.dtype)
        # print('images : ',images.dtype)

        ########################
        if self.capsnet_bool:
            target_eye = torch.sparse.torch.eye(10).cuda()
            target_eye = target_eye.index_select(dim=0, index=target)

            images = images[:,0:1]

            output, reconstructions, masked = self(images, target = None)

            # print('images : ',images.shape)
            # print('target : ',target.shape)
            # print('output : ',output.shape)
            # print('reconstructions : ',reconstructions.shape)
            # print('masked : ',masked.shape)

            val_loss = self.CapsuleNet.loss(images, output, target_eye, reconstructions)

            # print('val_loss : ',[val_loss.item()])

            acc1 = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target_eye.data.cpu().numpy(), 1)) / float(images.shape[0])

            acc1_list = [acc1]
            self.acc1_list += acc1_list
            acc5_list = [0]
            self.acc5_list += acc5_list

        elif self.IP_capsnet_bool:

            # Getting Target Eye
            target_eye = torch.sparse.torch.eye(10).cuda()
            target_eye = target_eye.index_select(dim=0, index=target)

            # With recon
            images = images[:,0:1]

            # With recon
            output, reconstructions = self(images, target = None)
            # No recon
            # output = self(images)

            # print('output : ',output.shape)
            # print('reconstructions : ',reconstructions.shape)

            # Loss
            # With recon
            val_loss = self.HMAX.loss(images, output, target_eye, reconstructions)
            # No recon
            # val_loss = self.HMAX.margin_loss(output, target_eye)

            # Getting masked for calculating acc
            # FInding the length of the vector which is the probability
            classes = torch.sqrt((output ** 2).sum(2))
            classes = F.softmax(classes)
            
            _, max_length_indices = classes.max(dim=1)
            masked = Variable(torch.sparse.torch.eye(10))
            if True:
                masked = masked.cuda()
            masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)

            # Getting accuracy
            acc1 = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target_eye.data.cpu().numpy(), 1)) / float(images.shape[0])
                                
            #
            acc1_list = [acc1]
            self.acc1_list += acc1_list
            acc5_list = [0]
            self.acc5_list += acc5_list

        else:
            output = self(images)

            ########################
            val_loss = self.criterion(output, target)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

            ########################
            val_loss = torch.mean(val_loss)

            acc1_list = acc1.cpu().tolist()
            self.acc1_list += acc1_list
            acc5_list = acc5.cpu().tolist()
            self.acc5_list += acc5_list

        ########################
        time.sleep(1)
        # val_loss_list = val_loss.cpu().tolist()
        val_loss_list = [val_loss.item()]

        # print('val_loss_list : ',val_loss_list) 
        self.val_losses += val_loss_list

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
        n_c = images.shape[1]
        if len(images.shape) == 4:
            images = images.reshape(-1, n_c, h_w, h_w)
            target = target.reshape(-1)

        # print('target valueeeees : ',target)
        # print('target : ',target.shape)
        # print('images : ',images.shape)
        # print('target : ',target.dtype)
        # print('images : ',images.dtype)

        ########################
        if self.capsnet_bool:
            target_eye = torch.sparse.torch.eye(10).cuda()
            target_eye = target_eye.index_select(dim=0, index=target)

            images = images[:,0:1]

            output, reconstructions, masked = self(images, target = None)

            # print('images : ',images.shape)
            # print('target : ',target.shape)
            # print('output : ',output.shape)
            # print('reconstructions : ',reconstructions.shape)
            # print('masked : ',masked.shape)

            val_loss = self.CapsuleNet.loss(images, output, target_eye, reconstructions)

            # print('val_loss : ',[val_loss.item()])

            acc1 = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target_eye.data.cpu().numpy(), 1)) / float(images.shape[0])

            acc1_list = [acc1]
            self.acc1_list += acc1_list
            acc5_list = [0]
            self.acc5_list += acc5_list

        elif self.IP_capsnet_bool:

            # Getting Target Eye
            target_eye = torch.sparse.torch.eye(10).cuda()
            target_eye = target_eye.index_select(dim=0, index=target)

            # With recon
            images = images[:,0:1]

            # With recon
            output, reconstructions, max_scale_index = self(images, target = None)
            # No recon
            # output = self(images)

            # print('output : ',output.shape)
            # print('reconstructions : ',reconstructions.shape)

            # Loss
            # With recon
            val_loss = self.HMAX.loss(images, output, target_eye, reconstructions)
            # No recon
            # val_loss = self.HMAX.margin_loss(output, target_eye)

            # Getting masked for calculating acc
            # FInding the length of the vector which is the probability
            classes = torch.sqrt((output ** 2).sum(2))
            classes = F.softmax(classes)
            
            _, max_length_indices = classes.max(dim=1)
            masked = Variable(torch.sparse.torch.eye(10))
            if True:
                masked = masked.cuda()
            masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)

            # Getting accuracy
            acc1 = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target_eye.data.cpu().numpy(), 1)) / float(images.shape[0])
                                
            ########################
            time.sleep(1)
            acc1_list = [acc1]
            self.acc1_list += acc1_list
            acc5_list = [0]
            self.acc5_list += acc5_list
            # val_loss_list = val_loss.cpu().tolist()
            val_loss_list = [val_loss.item()]

            # print('val_loss_list : ',val_loss_list) 
            self.val_losses += val_loss_list

            self.overall_max_scale_index += max_scale_index

        else:

            ########################
            output, max_scale_index = self(images, batch_idx)

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

            self.overall_max_scale_index += max_scale_index

        ###########################
        # val_loss = torch.mean(val_loss)

        # return val_loss
        return acc1

    def test_epoch_end(self,losses):

        #################################
        # Creating histogram
        fig, axs = plt.subplots(1, 1,
                                figsize =(10, 7),
                                tight_layout = True)

        # print('self.overall_max_scale_index : ',self.overall_max_scale_index)
        print('self.overall_max_scale_index len : ',len(self.overall_max_scale_index))
        
        axs.hist(self.overall_max_scale_index, bins = 20)

        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/save_argmax_hist", self.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, f'scale_{self.HMAX.MNIST_Scale}.png')
        fig.savefig(os.path.join(job_dir, file_name))
        #################################

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
        self.overall_max_scale_index = []

        

    
