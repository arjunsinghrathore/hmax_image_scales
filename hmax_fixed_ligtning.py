import torch
# torch.manual_seed(1)
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
# np.random.seed(1)
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

from hmax_models.hmax_ip_basic_single_band import HMAX_IP_basic_single_band, HMAX_IP_basic_single_band_deeper, HMAX_IP_basic_single_band_deeper_cifar10

from hmax_models.hmax_ip_full_single_band import HMAX_IP_full_single_band

from hmax_models.hmax_ip_basic_multi_band import HMAX_IP_basic_multi_band

from hmax_models.hmax_ip_mulit_band_recon import HMAX_IP_basic_multi_band_recon

from hmax_models.hmax_ip_basic_single_band_caps import HMAX_IP_basic_single_band_caps

from hmax_models.MOCO_v2_dimin_semantics import MoCo_v2

from hmax_models.SimClr_Contrastive import SimClr

from hmax_models.hmax_2_streams import HMAX_2_streams

from hmax_models.hmax_finetune import HMAX_finetune

from hmax_models.CapsNet import CapsNet

from hmax_models.lindeberg_fov_max import fov_max

from deepnet_models.vgg import vgg16_bn

import argparse
import os
import random
import shutil
import time
import warnings

from pytorch_lightning import Trainer, seed_everything

# torch.autograd.set_detect_anomaly(True)

# seed_everything(42, workers=True)
# Seeds

# import random
# random.seed(1)




class HMAX_trainer(pl.LightningModule):
    def __init__(self, prj_name, n_ori, n_classes, lr, weight_decay, ip_scales, IP_bool = False, visualize_mode = False, \
                 MNIST_Scale = None, first_scale_test = False, capsnet_bool = False, IP_capsnet_bool = False, \
                 IP_contrastive_bool = False, lindeberg_fov_max_bool = False, IP_full_bool = False, \
                 IP_bool_recon = False, IP_contrastive_finetune_bool = False, model_pre = None, \
                 contrastive_2_bool = False, sim_clr_bool = False, batch_size = None, IP_2_streams = False, cifar_data_bool = False):
        super().__init__()
        
        self.parameter_dict = {'prj_name':prj_name, 'n_ori':n_ori, 'n_classes':n_classes, \
                                'lr':lr, 'weight_decay':weight_decay, 'ip_scales':ip_scales, \
                                'first_scale_test':first_scale_test, 'visualize_mode':visualize_mode, \
                                'MNIST_Scale':MNIST_Scale, 'capsnet_bool':capsnet_bool, 'IP_capsnet_bool':IP_capsnet_bool, \
                                'IP_contrastive_bool':IP_contrastive_bool, 'lindeberg_fov_max_bool':lindeberg_fov_max_bool, \
                                'IP_full_bool':IP_full_bool, 'IP_bool_recon':IP_bool_recon, \
                                'IP_contrastive_finetune_bool':IP_contrastive_finetune_bool, 'contrastive_2_bool':contrastive_2_bool, \
                                'batch_size':batch_size, 'sim_clr_bool':sim_clr_bool, 'IP_2_streams':IP_2_streams, \
                                'cifar_data_bool':cifar_data_bool}

        print('self.parameter_dict : ',self.parameter_dict)

        self.prj_name = prj_name 
        self.n_ori = n_ori 
        self.n_classes = n_classes 
        self.lr = lr 
        self.weight_decay = weight_decay 
        self.ip_scales = ip_scales
        self.IP_bool = IP_bool
        self.IP_full_bool = IP_full_bool
        self.IP_bool_recon = IP_bool_recon
        self.first_scale_test = first_scale_test
        self.visualize_mode = visualize_mode
        self.MNIST_Scale = MNIST_Scale

        self.capsnet_bool = capsnet_bool
        self.IP_capsnet_bool = IP_capsnet_bool
        self.IP_contrastive_bool = IP_contrastive_bool
        self.lindeberg_fov_max_bool = lindeberg_fov_max_bool
        self.IP_contrastive_finetune_bool = IP_contrastive_finetune_bool

        self.IP_2_streams = IP_2_streams

        self.cifar_data_bool = cifar_data_bool

        self.model_pre = model_pre
        
        self.contrastive_2_bool = contrastive_2_bool
        self.sim_clr_bool = sim_clr_bool

        self.batch_size = batch_size

        ###################################################################

        if self.capsnet_bool:
            self.CapsuleNet = CapsNet()
        elif self.IP_capsnet_bool:
            self.HMAX = HMAX_IP_basic_single_band_caps(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        elif self.IP_contrastive_bool:
            if self.sim_clr_bool:
                base_encoder = HMAX_IP_basic_single_band_deeper(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                    visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
                # base_encoder = HMAX_IP_basic_single_band(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                #                 visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)


                # self.HMAX = SimClr(base_encoder, batch_size = batch_size, dim=128, T=0.1, mlp=not(contrastive_2_bool), \
                #                     contrastive_2_bool = contrastive_2_bool)
                self.HMAX = SimClr(base_encoder, batch_size = batch_size, dim=128, T=0.1, mlp=True, \
                                    contrastive_2_bool = contrastive_2_bool)
            else:
                # base_encoder_q = HMAX_IP_full_single_band(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                #                         visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
                # base_encoder_k = HMAX_IP_full_single_band(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                #                         visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
                base_encoder_q = HMAX_IP_basic_single_band_deeper(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                    visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
                base_encoder_k = HMAX_IP_basic_single_band_deeper(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                    visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)

                self.HMAX = MoCo_v2(base_encoder_q, base_encoder_k, dim=128, K=8192, m=0.999, T=0.2, mlp=not(contrastive_2_bool), \
                                    alpha=2.0, contrastive_2_bool = contrastive_2_bool)
                # self.HMAX = MoCo_v2(base_encoder_q, base_encoder_k, dim=128, K=16384, m=0.999, T=0.2, mlp=True, alpha=2.0)

        elif self.IP_contrastive_finetune_bool:
            for params_pre in self.model_pre.parameters():
                params_pre.requires_grad = False  # not update by gradient

            self.HMAX = HMAX_finetune(num_classes=self.n_classes, prj_name = self.prj_name, model_pre = self.model_pre)

        elif self.IP_2_streams:
            # base_encoder = HMAX_IP_basic_single_band(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
            #                     visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
            base_encoder = HMAX_IP_basic_single_band_deeper(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)

            self.HMAX = HMAX_2_streams(num_classes=self.n_classes, prj_name = self.prj_name, model_pre = base_encoder)

        elif self.lindeberg_fov_max_bool:
            # print('In hmax_fixed_lightning succes')
            self.HMAX = fov_max(ip_scales = self.ip_scales, num_classes=self.n_classes)
        
        elif self.IP_bool:

            if self.cifar_data_bool:
                # # Single Band Deeper - Cifar10
                self.HMAX = HMAX_IP_basic_single_band_deeper_cifar10(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                    visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
            else:
            
                # # # Single Band
                # self.HMAX = HMAX_IP_basic_single_band(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                #                     visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
                # # Single Band Deeper
                self.HMAX = HMAX_IP_basic_single_band_deeper(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                    visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)

                # # # Multi-Band
                # self.HMAX = HMAX_IP_basic_multi_band(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                #                     visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        
        elif self.IP_bool_recon:
            self.HMAX = HMAX_IP_basic_multi_band_recon(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)

        elif self.IP_full_bool:
            self.HMAX = HMAX_IP_full_single_band(ip_scales = self.ip_scales, n_ori=self.n_ori,num_classes=self.n_classes, \
                                    visualize_mode = self.visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)

            # self.HMAX = vgg16_bn()


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

        if self.IP_contrastive_bool and self.sim_clr_bool:
            self.l_pos_list = []
            self.l_neg2_list = []
            self.l_neg_list = []

        self.plot_moco_hists = False

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

        if self.IP_contrastive_bool:
            optimiser = torch.optim.SGD(self.parameters(), self.lr, weight_decay = self.weight_decay, momentum = 0.9)

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=200, eta_min=self.lr / 50
            )
            return [optimiser], [lr_scheduler]
        else:
            optimiser = torch.optim.Adam(self.parameters(), self.lr, weight_decay = self.weight_decay)

            if self.cifar_data_bool:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=120, eta_min=self.lr / 50
                )

                return [optimiser], [lr_scheduler]

            return optimiser



    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
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

        if self.IP_contrastive_bool:
            target = target.reshape(-1)
        elif len(images.shape) == 4:
            h_w = images.shape[-1]
            n_c = images.shape[1]
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

        elif self.IP_contrastive_bool:

            correct_scale_loss = 0
            
            if self.sim_clr_bool:
                # Without Scale Loss
                loss, sim_argsort = self.HMAX(images[:,0], images[:,2], scale_loss = False, target = None)
                # # With Scale Loss
                # loss, sim_argsort, correct_scale_loss = self.HMAX(images[:,0], images[:,2], scale_loss = True, target = target)

                acc1 = (sim_argsort == 0).float().mean()
                acc5 = (sim_argsort < 5).float().mean()
            else:
                # # dimin_seman_mode
                # logits, labels = self.HMAX(images[:,0], images[:,1], images[:,2], n_non_sematic=1, dimin_seman_mode = True, target = target)
                # # Non dimin_seman_mode
                # logits, labels = self.HMAX(images[:,0], images[:,1], images[:,2], n_non_sematic=1, dimin_seman_mode = False)
                # Non dimin_seman_mode scale aug
                logits, labels = self.HMAX(images[:,0], images[:,2], images[:,2], n_non_sematic=1, dimin_seman_mode = False)
                # # Non dimin_seman_mode scale aug + scale loss
                # logits, labels, correct_scale_loss = self.HMAX(images[:,0], images[:,2], images[:,2], n_non_sematic=1, dimin_seman_mode = False, scale_loss = True, target = target)
                # # # Dimin_seman_mode for scale loss only 
                # logits, labels, correct_scale_loss = self.HMAX(images[:,0], images[:,1], images[:,2], n_non_sematic=1, dimin_seman_mode = True, scale_loss = True, target = target)


                loss = self.criterion(logits, labels)

                acc1, acc5 = self.accuracy(logits, labels, topk=(1, 5))

            ########################
            loss = torch.mean(loss)

            # Adding scale loss
            loss = loss + (correct_scale_loss*0.5)

        elif self.IP_bool_recon:
            output, max_scale_index, recon_loss = self(images)

            ########################
            loss = self.criterion(output, target)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

            ########################
            loss = torch.mean(loss)

            # Adding scale loss
            # loss = loss + (correct_scale_loss*0.125)
            loss = loss + (recon_loss*100.0)

            self.log('recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)

        else:
            output, c2b_maps, max_scale_index, correct_scale_loss = self(images)

            ########################
            loss = self.criterion(output, target)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

            ########################
            loss = torch.mean(loss)

            # Adding scale loss
            loss = loss + (correct_scale_loss*0.5)
            # loss = loss + (correct_scale_loss*0.08)

        ########################
        # if not self.IP_contrastive_bool:
        if True:
            # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            # self.log("train_performance", {"acc1": acc1, "acc5": acc5})
            # if not self.sim_clr_bool:
            self.log('train_acc1', acc1.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('correct_scale_loss', correct_scale_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss #{'output' : output, 'target' : target} 

    def training_step_end(self, losses):
        
        losses = torch.mean(losses)

        self.log('train_loss', losses,on_step=True, on_epoch=True,prog_bar=True)
        
        return losses


    def validation_step(self, batch, batch_idx):
        
        images, target = batch


        if self.IP_contrastive_bool:
            target = target.reshape(-1)
        elif len(images.shape) == 4:
            h_w = images.shape[-1]
            n_c = images.shape[1]
            images = images.reshape(-1, n_c, h_w, h_w)
            target = target.reshape(-1)

        ########################
        if self.capsnet_bool:
            target_eye = torch.sparse.torch.eye(10).cuda()
            target_eye = target_eye.index_select(dim=0, index=target)

            images = images[:,0:1]

            output, reconstructions, masked = self(images, target = None)

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

        elif self.IP_contrastive_bool:

            if self.plot_moco_hists:
                if not self.sim_clr_bool:
                    # # dimin_seman_mode
                    # logits, labels, l_pos, l_neg2, l_neg = self.HMAX(images[:,0], images[:,1], images[:,2], n_non_sematic=1, val_mode = True)
                    # # Non dimin_seman_mode
                    # logits, labels, l_pos, l_neg2, l_neg = self.HMAX(images[:,0], images[:,1], images[:,2], n_non_sematic=1, dimin_seman_mode = False, val_mode = True)
                    # Non dimin_seman_mode scale aug
                    logits, labels, l_pos, l_neg2, l_neg = self.HMAX(images[:,0], images[:,2], images[:,2], n_non_sematic=1, dimin_seman_mode = False, val_mode = True)
                    # # Non dimin_seman_mode scale aug + scale loss
                    # logits, labels, l_pos, l_neg2, l_neg, scale_loss = self.HMAX(images[:,0], images[:,2], images[:,2], n_non_sematic=1, dimin_seman_mode = False, val_mode = True, scale_loss = True, target = target)

                    self.l_pos_list = self.l_pos_list + l_pos.cpu().tolist()
                    self.l_neg2_list = self.l_neg2_list + l_neg2.cpu().tolist()
                    self.l_neg_list = self.l_neg_list + l_neg.cpu().tolist()

            ########################
            acc1_list = [0]
            self.acc1_list += acc1_list
            acc5_list = [0]
            self.acc5_list += acc5_list

            ########################
            # val_loss = nll

            val_loss = torch.zeros(1, device = images.device)

        elif self.IP_bool_recon:
            output, max_scale_index, recon_loss = self(images)

            ########################
            val_loss = self.criterion(output, target)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

            ########################
            val_loss = torch.mean(val_loss)

            # Adding scale loss
            # loss = loss + (correct_scale_loss*0.125)
            val_loss = val_loss + (recon_loss*100.0)

            acc1_list = acc1.cpu().tolist()
            self.acc1_list += acc1_list
            acc5_list = acc5.cpu().tolist()
            self.acc5_list += acc5_list

        else:
            # print('Correct Place')
            output, c2b_maps, max_scale_index, correct_scale_loss = self(images)

            ########################
            val_loss = self.criterion(output, target)

            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

            ########################
            val_loss = torch.mean(val_loss)

            # Adding scale loss
            val_loss = val_loss + (correct_scale_loss*0.125)
            # val_loss = val_loss + (correct_scale_loss*0.08)

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
        return val_loss

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
        if self.IP_contrastive_bool and self.plot_moco_hists:
            job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/MOCO_stuff/MOCO_hists", self.prj_name)
            os.makedirs(job_dir, exist_ok=True)

            # Creating histogram
    
            # print('self.overall_max_scale_index : ',self.overall_max_scale_index)
            print('Saving the Histogramsssss')
            
            fig, axs = plt.subplots(1, 1,
                                    figsize =(10, 7),
                                    tight_layout = True)
            print('self.l_pos_list : ', len(self.l_pos_list))
            # axs.hist(self.l_pos_list, bins = list(np.linspace(-1, 1, 50)))
            counts, bins = np.histogram(self.l_pos_list, bins = list(np.linspace(-1, 1, 50)))
            axs.stairs(counts, bins, fill = True)
            file_name = os.path.join(job_dir, 'l_pos.png')
            fig.savefig(os.path.join(job_dir, file_name))
            plt.close()

            fig, axs = plt.subplots(1, 1,
                                    figsize =(10, 7),
                                    tight_layout = True)
            print('self.l_neg_list : ', len(self.l_neg_list))
            # axs.hist(self.l_neg_list, bins = list(np.linspace(-1, 1, 50)))
            counts, bins = np.histogram(self.l_neg_list, bins = list(np.linspace(-1, 1, 50)))
            axs.stairs(counts, bins, fill = True)
            file_name = os.path.join(job_dir, 'l_neg.png')
            fig.savefig(os.path.join(job_dir, file_name))
            plt.close()

            fig, axs = plt.subplots(1, 1,
                                    figsize =(10, 7),
                                    tight_layout = True)
            print('self.l_neg2_list : ', len(self.l_neg2_list))
            # axs.hist(self.l_neg2_list, bins = list(np.linspace(-1, 1, 50)))
            counts, bins = np.histogram(self.l_neg2_list, bins = list(np.linspace(-1, 1, 50)))
            axs.stairs(counts, bins, fill = True)
            file_name = os.path.join(job_dir, 'l_neg2.png')
            fig.savefig(os.path.join(job_dir, file_name))
            plt.close()

            self.l_pos_list = []
            self.l_neg2_list = []
            self.l_neg_list = []


        ##################################
        self.val_losses = []
        self.acc1_list = []
        self.acc5_list = []


    def test_step(self, batch, batch_idx):
        
        images, target = batch

        if len(images.shape) == 4:
            h_w = images.shape[-1]
            n_c = images.shape[1]
            images = images.reshape(-1, n_c, h_w, h_w)
            target = target.reshape(-1)
        elif len(images.shape) == 5:
            h_w = images.shape[-1]
            n_c = images.shape[2]
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

        elif self.IP_contrastive_bool:

            # images = images.reshape()

            output, max_scale_index = self(images)

            ########################
            val_loss, prob = self.HMAX.loss(output, target, True)

            acc1_list = [0]
            self.acc1_list += acc1_list
            acc5_list = [0]
            self.acc5_list += acc5_list

            ########################
            # val_loss = torch.mean(val_loss)

            time.sleep(1)
            # print('prob : ', prob)
            # val_loss_list = val_loss.cpu().tolist()
            # print('val_loss_list : ',val_loss_list) 
            val_loss_list = [val_loss.item()]
            self.val_losses += val_loss_list

        elif self.IP_bool_recon:
            output, max_scale_index, recon_loss = self(images)

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


        else:

            ########################
            output, c2b_maps, max_scale_index, correct_scale_loss = self(images, batch_idx)

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
        return val_loss

    def test_epoch_end(self,losses):

        #################################
        # Creating histogram
        fig, axs = plt.subplots(1, 1,
                                figsize =(10, 7),
                                tight_layout = True)

        # print('self.overall_max_scale_index : ',self.overall_max_scale_index)
        print('self.overall_max_scale_index len : ',len(self.overall_max_scale_index))
        
        # axs.hist(self.overall_max_scale_index, bins = 20)
        axs.hist(self.overall_max_scale_index, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

        # Change Path into own folder
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
        # Change Path into own folder
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

        

    
