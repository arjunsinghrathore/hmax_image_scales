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

seed_everything(42, workers=True)

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class MoCo_v2(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder_q, base_encoder_k, dim=128, K=65536, m=0.999, T=0.07, mlp=False, alpha=1.0, contrastive_2_bool = False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_v2, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.alpha = alpha
        self.contrastive_2_bool = contrastive_2_bool

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder_q
        self.encoder_k = base_encoder_k

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.get_s4_in_channels()
            self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))

            # self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim), nn.ReLU())
            # self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim), nn.ReLU())

            # self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim_mlp, dim_mlp//2), nn.Linear(dim_mlp//2, dim))
            # self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim_mlp, dim_mlp//2), nn.Linear(dim_mlp//2, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if contrastive_2_bool:
            # self.encoder_k.make_ip_2_bool = True
            # print('MOCO bool heere')
            self.encoder_k.single_scale_bool = False
            self.encoder_k.ip_scales = 5
            self.encoder_k.scale = 4

            self.m = 0.9

            dim = self.encoder_k.get_s4_in_channels()

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, im_t, n_non_sematic=10, val_mode = False, dimin_seman_mode = False, scale_loss = False, target = None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_unnorm, q_scale_maps, _, _ = self.encoder_q(im_q, contrastive_scale_loss = scale_loss, contrastive_2_bool = self.contrastive_2_bool)  # queries: NxC
        q = nn.functional.normalize(q_unnorm, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # if contrastive_2_bool:
            #     k_unnorm, k_scale_maps, _, _ = self.encoder_k(im_k, contrastive_scale_loss = scale_loss, scale = target)  # keys: NxC
            #     k = nn.functional.normalize(k_unnorm, dim=1)
            # else:
            k_unnorm, k_scale_maps, _, _ = self.encoder_k(im_k, contrastive_scale_loss = scale_loss, contrastive_2_bool = self.contrastive_2_bool)  # keys: NxC
            k = nn.functional.normalize(k_unnorm, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_unnorm = self._batch_unshuffle_ddp(k_unnorm, idx_unshuffle)
            if scale_loss:
                k_scale_maps = self._batch_unshuffle_ddp(k_scale_maps, idx_unshuffle)

            if dimin_seman_mode:
                ts = []
                # for m in range(n_non_sematic):
                    # im_t_shuffle, idx_unshuffle = self._batch_shuffle_ddp(im_t[:, m].clone())
                im_t_shuffle, idx_unshuffle = self._batch_shuffle_ddp(im_t)
                t, t_scale_maps, _, _ = self.encoder_k(im_t_shuffle, contrastive_scale_loss = scale_loss)
                t = nn.functional.normalize(t, dim=1)

                t = self._batch_unshuffle_ddp(t, idx_unshuffle)
                t_scale_maps = self._batch_unshuffle_ddp(t_scale_maps, idx_unshuffle)

                ts.append(t)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # if dimin_seman_mode and (not scale_loss):
        #     # non-semantic negative logits
        #     # l_neg2 = [torch.einsum('nc,nc->n', [q, t]).unsqueeze(-1)*self.alpha for t in ts]
        #     l_neg2 = [(torch.einsum('nc,nc->n', [q, t])*target).unsqueeze(-1)*self.alpha for t in ts]
        # else:
        l_neg2 = [torch.zeros_like(l_pos, device= im_q.device)]

        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # if dimin_seman_mode and (not scale_loss):
        #     logits = torch.cat([l_pos, l_neg] + l_neg2, dim=1)
        # else:
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        if scale_loss:
            # corr_scale_loss = F.relu(k_scale_maps.squeeze() - q_scale_maps.squeeze())
            corr_scale_loss = F.relu(t_scale_maps.squeeze() - q_scale_maps.squeeze())
            # corr_scale_loss = F.relu(k_unnorm.squeeze() - q_unnorm.squeeze())
            corr_scale_loss = torch.mean(corr_scale_loss, dim = 1)
            corr_scale_loss = corr_scale_loss * target
            corr_scale_loss = torch.sum(corr_scale_loss) / torch.sum(target)

        if val_mode:
            if scale_loss:
                return logits, labels, l_pos, l_neg2[0]/self.alpha, torch.mean(l_neg, dim = 1), corr_scale_loss
            else:
                return logits, labels, l_pos, l_neg2[0]/self.alpha, torch.mean(l_neg, dim = 1)

        if scale_loss:
            return logits, labels, corr_scale_loss
        else:
            return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor.contiguous())
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output