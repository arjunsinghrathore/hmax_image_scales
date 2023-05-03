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

# seed_everything(42, workers=True)

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class SimClr(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, batch_size, dim=128, T=0.07, mlp=False, contrastive_2_bool = False):
        """
        dim: feature dimension (default: 128)
        T: softmax temperature (default: 0.07)
        """
        super(SimClr, self).__init__()

        self.temperature = T
        self.contrastive_2_bool = contrastive_2_bool
        self.batch_size = batch_size

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = base_encoder

        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder.get_s4_in_channels()
            self.encoder.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))

            # self.encoder.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim), nn.ReLU())
            
            # self.encoder.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim_mlp, dim_mlp//2), nn.Linear(dim_mlp//2, dim))
           
        if contrastive_2_bool:
            self.ip_scales = 5
            self.scale = 4
        else:
            self.ip_scales = None
            self.scale = None

    def device_as(self, t1, t2):
        """
        Moves t1 to the device of t2
        """
        return t1.to(t2.device)

    def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def contrastive_loss(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)

       denominator = self.device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)

       ###########################################################
       # Get ranking position of positive example
       self_mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
       similarity_matrix.masked_fill_(self_mask, -9e15)
       # Find positive example -> batch_size//2 away from the original example
       pos_mask = self_mask.roll(shifts=similarity_matrix.shape[0]//2, dims=0)

       similarity_matrix = similarity_matrix / self.temperature

       comb_sim = torch.cat([similarity_matrix[pos_mask][:,None],  # First position positive example
                                similarity_matrix.masked_fill(pos_mask, -9e15)],
                                dim=-1)
       sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

       return loss, sim_argsort

    def forward(self, im_q, im_k, scale_loss = False, target = None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, q_scale_maps, _, _ = self.encoder(im_q, contrastive_scale_loss = scale_loss, contrastive_2_bool = self.contrastive_2_bool, ip_scales = 2, scale = 4)  # queries: NxC


        # Compute key features
        # if contrastive_2_bool:
        #     k_unnorm, k_scale_maps, _, _ = self.encoder_k(im_k, contrastive_scale_loss = scale_loss, scale = target)  # keys: NxC
        #     k = nn.functional.normalize(k_unnorm, dim=1)
        # else:
        k, k_scale_maps, _, _ = self.encoder(im_k, contrastive_scale_loss = scale_loss, contrastive_2_bool = self.contrastive_2_bool, ip_scales = self.ip_scales, scale = self.scale)  # keys: NxC


        loss, sim_argsort = self.contrastive_loss(q, k)

        if scale_loss:
            # # OPtion 1 --> Direct Scale Loss among scale channels
            # k_scale_maps = k_scale_maps.squeeze().permute(0,2,1) # --> Batch, No. of scales, Channels
            # k_scale_maps_target = torch.gather(k_scale_maps, 1, target.long().view(k_scale_maps.shape[0], 1, 1).repeat(1, 1, k_scale_maps.shape[2]))
            # k_scale_maps_target = k_scale_maps_target.squeeze()
            
            # correct_scale_loss = torch.tensor([0.], device = k_scale_maps[0].device)

            # # for sm_i in range(k_scale_maps.shape[1]):
            # #     for k_sm_i in range(len(k_scale_maps)):
            # #         if k_sm_i not in [int(target[sm_i])]:
            # #             # print('int(target[sm_i]) : ',int(target[sm_i]))
            # #             correct_scale_loss = correct_scale_loss + F.relu(k_scale_maps[k_sm_i][sm_i] - k_scale_maps[int(target[sm_i])][sm_i])
            
            # # Alternative
            # print_list = []
            # for k_sm_i in range(k_scale_maps.shape[1]):
            #         print_list.append(torch.mean(F.relu(k_scale_maps[:, k_sm_i, :] - k_scale_maps_target)).item())
            #         correct_scale_loss = correct_scale_loss + F.relu(k_scale_maps[:, k_sm_i, :] - k_scale_maps_target)
            # print('print_list : ',print_list)

            # correct_scale_loss = torch.mean(correct_scale_loss)

            # # OPtion 2 --> Make representation equal along with contrastive learning
            # k_scale_maps = k_scale_maps.squeeze().permute(0,2,1) # --> Batch, No. of scales, Channels
            # k_scale_maps_target = torch.gather(k_scale_maps, 1, target.long().view(k_scale_maps.shape[0], 1, 1).repeat(1, 1, k_scale_maps.shape[2]))
            # k_scale_maps_target = k_scale_maps_target.squeeze()

            # correct_scale_loss = q - k_scale_maps_target

            correct_scale_loss = q - k
            correct_scale_loss = torch.mean(torch.abs(correct_scale_loss))
            

        if scale_loss:
            return loss, sim_argsort, correct_scale_loss
        else:
            return loss, sim_argsort
