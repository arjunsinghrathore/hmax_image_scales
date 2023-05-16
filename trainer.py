import torch
# torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
# np.random.seed(1)
# import pandas as pd
import cv2
# from PIL import Image
# from torchsummary import summary
import time
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
# from sklearn.preprocessing import LabelEncoder

# from pytorch_lightning.loggers import NeptuneLogger

import _pickle as pickle

from torch.utils.data import random_split, DataLoader, Dataset

from pytorch_lightning.callbacks import ModelCheckpoint


import dataloader_lightning
# import models_hGRU_center_GAP
# from hmax_models.hmax_ivan import HMAX_latest_slim, HMAX_latest
# print('Importedddddd HMAX_latest_slim')
import hmax_fixed_ligtning

from utils.rdm_corr import rdm_corr_scales_func, rdm_corr_func, direct_corr_func

import os
import shutil
import pickle
import random

# Seeds
from pytorch_lightning import Trainer, seed_everything

# seed_everything(42, workers=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        
        print("GPUs detected: = {}".format( torch.cuda.device_count()))
        
        for i in range(n_gpus):
            print("_______")
            print( torch.cuda.get_device_name( i ) )
            print("_______")


    # pt_model = model_center_pt.SQUEEZENET_1_1()
    # pt_model = pt_model.load_from_checkpoint('/users/aarjun1/data/aarjun1/color_cnn/checkpoints/unet_vgg-epoch=56-val_loss=2.96.ckpt')

    # base = pt_model.slice

    # Hyper-Parameters
    IP_bool = False
    IP_bool_recon = False
    IP_full_bool = False
    IP_2_streams = False
    capsnet_bool = False
    IP_capsnet_bool = False
    IP_contrastive_bool = False
    IP_contrastive_finetune_bool = False
    lindeberg_fov_max_bool = False
    deepnet_models_bool = False
    ENN_bool = True
    if capsnet_bool:
        prj_name = "checkpoint_CapsNet_data_shuffle_org_MNIST_data_64bs_1by4_1e4"
    elif IP_capsnet_bool:
        prj_name = "checkpoint_HMAX_basic_single_band_CapsNet_4_to_12_dim_vector_recon_23S1_22_C1_02_stride_real_S2b_normalize_alpha_square_data_shuffle_linear_classifier_my_lindeberg_data_no_smooth_no_nolin_4scale_224_64bs_1by4_1e4"
    elif IP_bool:
        prj_name = "checkpoint_HMAX_deeper_multi_band_5_02_drop_scale_lin_attn_better_recurrent_attn_second_gauss_sigma05_bndry01_temp001_t4_mnist_13S1_12C1_05stride_S1_correct_gabor_abs_NT_F_real_S2b_BN_relu_2scale_112_48bs_1by4_1e4"
    elif IP_2_streams: 
        prj_name = "checkpoint_HMAX_2_streams_ScaleBand_large_one_classifier_single_deeper_band_C1_overlap_02_drop_Equal_loss05_mnist_13S1_12C1_05stride_S1_correct_gabor_abs_NT_F_real_S2b_BN_relu_data_shuffle_linear_classifier_2scale_112_48bs_1by4_1e4"
    elif IP_bool_recon:
        prj_name = "checkpoint_HMAX_basic_multi_band_02_drop_recon_100_smoothed_nl_mnist_7S1_6_C1_05_stride_C1_no_interpolate_S1_correct_gabor_BN_relu_real_S2b_BN_relu_data_shuffle_linear_classifier_2scale_112_16bs_1by4_1e4"
    elif IP_full_bool:
        prj_name = "checkpoint_HMAX_full_single_band_fat_mask_presc16_noise05_02_drop_mnist_15S1_14C1_05stride_6C2_01stride_S1_correct_gabor_abs_NT_F_real_S2b_BN_relu_data_shuffle_linear_classifier_2scale_224_24bs_1by4_1e4_continued"
        # prj_name = "checkpoint_HMAX_full_single_band_mask_noise05_vgg"
    elif IP_contrastive_bool:
        prj_name = "checkpoint_HMAX_SimClr_idea_2_mlp_scaleAug_const_size_single_band_deeper_02_drop_mnist_19S1_18C1_05stride_S1_correct_gabor_abs_BN_bias_real_S2b_BN_relu_2scale_112_64bs_1by4_1e4"
    elif IP_contrastive_finetune_bool:
        prj_name = "checkpoint_HMAX_finetune_SimClr_idea_2_mlp_scaleAug_const_size_single_band_deeper_02_drop_mnist_7S1_6C1_05stride_S1_correct_gabor_abs_BN_bias_real_S2b_BN_relu_2scale_112_64bs_1by4_1e4"
    elif lindeberg_fov_max_bool:
        prj_name = "checkpoint_lindeberg_avgpool_fov_max_after_final_linear_5e4_lr_no_weight_decay_2scale_112_before_64bs_1by4"
    elif deepnet_models_bool:
        prj_name = "checkpoint_VGG16_BN_Scratch_05to8_scales_224_before_64bs_1by4"
    elif ENN_bool:
        prj_name = "checkpoint_ENN_1Stride_11scales_224_before_24bs_1by2"
    else:
        prj_name = "checkpoint_HMAX_latest_slim_PNAS_IVAN_50_MNIST_18_17s_no_S1_norm_192i_10e5_lr" #_new_stride"
        # prj_name = "checkpoint_HMAX_PNAS_100_MNIST_18_IP_GAP_17s_up_down_linderberg_C_S2_alpha_norm_like_S1_no_RELU_1by5_192_20_down_mp_like_HMAX_drop_s4_data_shuffle_linear_classifier_1e4"
    
    n_ori = 4
    n_classes = 10

    if IP_bool or IP_full_bool or capsnet_bool or IP_capsnet_bool or IP_bool_recon or IP_contrastive_finetune_bool or IP_2_streams or deepnet_models_bool or ENN_bool:
        # lr = 0.000001 # For 7 scales
        # lr = 0.00000005 # For 14 scales
        # lr = 0.00001 # Our
        lr = 1e-4 #1e-4 # IP_caps
        # lr = 1e-3 # caps
        # lr = 0.000005 # Lindeberg?
        weight_decay = 1e-4 #1e-2
        batch_size_per_gpu = 32
        num_epochs = 1000 # 1000
        ip_scales = 11 #18 # 9 #14 #7
        image_size = 224 #224 #128 #192 #80
        warp_image_bool = False

        # multi_scale_training_bool = True

        linderberg_bool = False
        my_data = True
        all_scales_train_bool = False
        orginal_mnist_bool = False

        cifar_data_bool = False

        oracle_bool = False
        argmax_bool = False

        oracle_plot_overlap_bool = False
        argmax_plot_overlap_bool = False
        oracle_argmax_plot_overlap_bool = False

        sim_clr_bool = True
        contrastive_2_bool = True
    
    

    elif IP_contrastive_bool:
        sim_clr_bool = True

        if sim_clr_bool:
            lr = 1e-4 #1e-4 # IP_caps
            
            weight_decay = 1e-4 #1e-2
            batch_size_per_gpu = 48 #128
            num_epochs = 200 # 1000
            ip_scales = 18 #18 # 9 #14 #7
            image_size = 112 #224 #128 #192 #80
            warp_image_bool = False

            contrastive_2_bool = True

        else:
            lr = 1e-2 #1e-4 # IP_caps
            
            weight_decay = 1e-4 #1e-2
            batch_size_per_gpu = 64 #128
            num_epochs = 200 # 1000
            ip_scales = 18 #18 # 9 #14 #7
            image_size = 112 #224 #128 #192 #80
            warp_image_bool = False

            contrastive_2_bool = True

        linderberg_bool = False
        my_data = True
        all_scales_train_bool = False
        orginal_mnist_bool = False

        oracle_bool = False
        argmax_bool = False

        oracle_plot_overlap_bool = False
        argmax_plot_overlap_bool = False
        oracle_argmax_plot_overlap_bool = False

    elif lindeberg_fov_max_bool:
        lr = 5e-4 # IP_caps
        weight_decay = 0
        batch_size_per_gpu = 128
        num_epochs = 500 # 1000
        ip_scales = 18 #18 # 9 #14 #7
        image_size = 224 #224 #128 #192
        linderberg_bool = False
        my_data = True
        all_scales_train_bool = False
        orginal_mnist_bool = False
        oracle_bool = False
        argmax_bool = False
        oracle_plot_overlap_bool = False
        argmax_plot_overlap_bool = False
        oracle_argmax_plot_overlap_bool = False

    else:
        lr = 10e-5# --> HMAX
        weight_decay = 1e-4
        batch_size_per_gpu = 4
        num_epochs = 1000
        ip_scales = 17
        image_size = 192 #224 # For HMAx
        linderberg_bool = False
        my_data = True


    # Mode
    test_mode = True
    val_mode = False
    continue_tr = False
    visualize_mode = False
    rdm_corr = False
    rdm_thomas = False

    featur_viz = False
    same_scale_viz = False

    # Get RDM's and RDM Correlation
    if test_mode and rdm_corr:
        save_rdms = ['c1'] #, 's2b' ,'c2b', 'clf'] #, 'c2b'] #, 'c2', 'c2b', 'c3', 'clf'] #['s1', 'c1', 's2b', 'c2b'] # None
    else:
        save_rdms = []  # ['s1', 'c1', 's2b', 'c2b'] # None

    # PLot Filter
    if test_mode and featur_viz:
        plt_filters = ['s1'] # # ['s1', 'c1', 's2b', 'c2b'] # None
    else:
        plt_filters = []  # None

    
    # Set some variables
    scale_datasets = [18,36,8,24,30,12,4,20,16]
    train_dataset = 24
    rdm_datasets = [18, 24]
    rdm_thomas_datasets = scale_datasets

    MNIST_Scale = train_dataset


    if IP_contrastive_finetune_bool:
        # Initializing the model
        pre_prj_name = "checkpoint_HMAX_SimClr_idea_2_mlp_scaleAug_const_size_single_band_deeper_02_drop_mnist_19S1_18C1_05stride_S1_correct_gabor_abs_BN_bias_real_S2b_BN_relu_2scale_112_64bs_1by4_1e4"
        model_pre = hmax_fixed_ligtning.HMAX_trainer(pre_prj_name, n_ori, n_classes, lr, weight_decay, ip_scales, IP_bool, visualize_mode, \
                                                MNIST_Scale, capsnet_bool = capsnet_bool, IP_capsnet_bool = IP_capsnet_bool, \
                                                IP_contrastive_bool = IP_contrastive_bool, lindeberg_fov_max_bool = lindeberg_fov_max_bool, \
                                                IP_full_bool = IP_full_bool, IP_bool_recon = IP_bool_recon, IP_contrastive_finetune_bool = False, \
                                                contrastive_2_bool = contrastive_2_bool, sim_clr_bool = sim_clr_bool, batch_size = batch_size_per_gpu)
        model_pre = model_pre.load_from_checkpoint('/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/' + pre_prj_name + '/HMAX-epoch=20-val_acc1=98.67788461538461-val_loss=0.04193095513108766.ckpt')
        if sim_clr_bool:
            model_pre = model_pre.HMAX.encoder
        else:
            model_pre = model_pre.HMAX.encoder_q

        print('########################################################')
        print('\n Loaded the pretrained model')

        model = hmax_fixed_ligtning.HMAX_trainer(prj_name, n_ori, n_classes, lr, weight_decay, ip_scales, IP_bool, visualize_mode, \
                                                MNIST_Scale, capsnet_bool = capsnet_bool, IP_capsnet_bool = IP_capsnet_bool, \
                                                IP_contrastive_bool = IP_contrastive_bool, lindeberg_fov_max_bool = lindeberg_fov_max_bool, \
                                                IP_full_bool = IP_full_bool, IP_bool_recon = IP_bool_recon, IP_contrastive_finetune_bool = True, model_pre = model_pre)
    else:
        # Initializing the model
        model = hmax_fixed_ligtning.HMAX_trainer(prj_name, n_ori, n_classes, lr, weight_decay, ip_scales, IP_bool, visualize_mode, \
                                                MNIST_Scale, capsnet_bool = capsnet_bool, IP_capsnet_bool = IP_capsnet_bool, \
                                                IP_contrastive_bool = IP_contrastive_bool, lindeberg_fov_max_bool = lindeberg_fov_max_bool, \
                                                IP_full_bool = IP_full_bool, IP_bool_recon = IP_bool_recon, IP_contrastive_finetune_bool = False, \
                                                contrastive_2_bool = contrastive_2_bool, sim_clr_bool = sim_clr_bool, batch_size = batch_size_per_gpu, \
                                                IP_2_streams = IP_2_streams, cifar_data_bool = cifar_data_bool, deepnet_models_bool = deepnet_models_bool, \
                                                ENN_bool = ENN_bool) #, \
                                                # multi_scale_training_bool = multi_scale_training_bool)

    # Loading weights if required
    if test_mode or val_mode or continue_tr or rdm_corr or rdm_thomas:
        # Change Path into own folder
        model = model.load_from_checkpoint('/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/' + prj_name + '/HMAX-epoch=2-val_acc1=98.02684294871794-val_loss=0.06468446119106375.ckpt')
        ########################## While Testing ##########################
        ## Need to force change some variables after loading a checkpoint
        if rdm_corr or rdm_thomas:
            model.prj_name = prj_name + "_ref_unorm"
            model.HMAX.prj_name = prj_name + "_ref_unorm"
        else:
            model.prj_name = prj_name
            model.HMAX.prj_name = prj_name
        model.visualize_mode = visualize_mode
        model.MNIST_Scale = MNIST_Scale
        model.HMAX.MNIST_Scale = MNIST_Scale
        model.lr = lr
        model.HMAX.same_scale_viz = same_scale_viz

        model.HMAX.argmax_bool = argmax_bool

        #
        model.HMAX.base_scale = image_size
        model.ip_scales = ip_scales
        model.HMAX.ip_scales = ip_scales

        if IP_2_streams:
            model.HMAX.model_pre.ip_scales = ip_scales
            model.HMAX.stream_2_bool = False
            model.HMAX.stream_1_big = False
            model.HMAX.model_pre.argmax_bool = argmax_bool

        model.HMAX.save_rdms = save_rdms
        model.HMAX.plt_filters = plt_filters

        if val_mode:
            model.plot_moco_hists = True
    
        ###################################################################

    print(model)
    print('Number of Parameters', count_parameters(model))

    if continue_tr:
        prj_name = prj_name + "_continued"

    # Setting the paths
    dataset = 'mnist_scale'
    traindir = "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/scale" + str(train_dataset) + "/train"
    valdir = "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/scale" + str(train_dataset) + "/test"
    testdir = "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/scale"
    linderberg_dir = "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr2p000_scte2p000-1.h5"
    linderberg_test_dir = {0.5: "mnist_large_scale_te10000_outsize112-112_scte0p500.h5",
                           0.595: "mnist_large_scale_te10000_outsize112-112_scte0p595.h5",
                           0.707: "mnist_large_scale_te10000_outsize112-112_scte0p707.h5",
                           0.841: "mnist_large_scale_te10000_outsize112-112_scte0p841.h5",

                           1: "mnist_large_scale_te10000_outsize112-112_scte1p000.h5",
                           1.189: "mnist_large_scale_te10000_outsize112-112_scte1p189.h5",
                           1.414: "mnist_large_scale_te10000_outsize112-112_scte1p414.h5",
                           1.682: "mnist_large_scale_te10000_outsize112-112_scte1p682.h5",

                           2: "mnist_large_scale_te10000_outsize112-112_scte2p000.h5",
                           2.378: "mnist_large_scale_te10000_outsize112-112_scte2p378.h5",
                           2.828: "mnist_large_scale_te10000_outsize112-112_scte2p828.h5",
                           3.364: "mnist_large_scale_te10000_outsize112-112_scte3p364.h5",

                           4: "mnist_large_scale_te10000_outsize112-112_scte4p000.h5",
                           4.757: "mnist_large_scale_te10000_outsize112-112_scte4p757.h5",
                           5.657: "mnist_large_scale_te10000_outsize112-112_scte5p657.h5",
                           6.727: "mnist_large_scale_te10000_outsize112-112_scte6p727.h5",
                           8: "mnist_large_scale_te10000_outsize112-112_scte8p000.h5"}

    my_dataset_scales = list(linderberg_test_dir.keys())
    my_dataset_scales = [int(mds*1000) for mds in my_dataset_scales]

    # my_dataset_scales_temp = []
    # for mds in my_dataset_scales:
    #     if mds >=2000:
    #         my_dataset_scales_temp.append(mds)
    # my_dataset_scales = my_dataset_scales_temp

    if not(cifar_data_bool):
        # my_dataset_scales = [2000, 4757] 
        my_dataset_scales = [2000, 500, 8000, 1000, 4000, 6727] #[4000] #, [2000, 8000] #[2000, 4000, 8000]
        # my_dataset_scales = [2000, 1682, 2378, 1414, 2828]
    else:
        my_dataset_scales = [1, 0.841, 0.5, 1.414, 2] #, 0.25, 4] # Cifar10 Scales

    print('my_dataset_scales : ',my_dataset_scales)

    my_dataset_traindir = '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_but_no_smoothning_no_non_linear/scale2000/train'
    my_dataset_valdir = '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_but_no_smoothning_no_non_linear/scale2000/val'
    my_dataset_testdir = '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_but_no_smoothning_no_non_linear/scale'
    #
    # my_dataset_traindir = '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2000/train'
    # my_dataset_valdir = '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2000/val'
    # my_dataset_testdir = '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale'

    # my_dataset_testdir = '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/cifar10_20_no_pad/scale'
    # my_dataset_testdir = '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/visual_images_RDM/scale'

    if linderberg_bool:
        rdm_thomas_datasets = list(linderberg_test_dir.keys())
    elif my_data:
        rdm_thomas_datasets = my_dataset_scales

    # Calling the dataloader
    if cifar_data_bool and not IP_contrastive_bool:
        print('In cifar10 data and not IP_contrastive_bool Condition')
        data = dataloader_lightning.dataa_loader_cifar10(image_size, my_dataset_traindir, my_dataset_valdir, my_dataset_testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                                                my_dataset_scales = my_dataset_scales, test_mode = False, all_scales_train_bool = all_scales_train_bool, warp_image_bool = warp_image_bool, 
                                                IP_contrastive_finetune_bool = IP_contrastive_finetune_bool)
    elif my_data and not IP_contrastive_bool:
        print('In my data and not IP_contrastive_bool Condition')
        data = dataloader_lightning.dataa_loader_my(image_size, my_dataset_traindir, my_dataset_valdir, my_dataset_testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                                                my_dataset_scales = my_dataset_scales, test_mode = False, all_scales_train_bool = all_scales_train_bool, warp_image_bool = warp_image_bool, 
                                                IP_contrastive_finetune_bool = IP_contrastive_finetune_bool)
    elif my_data and IP_contrastive_bool:
        print('In IP_contrastive_bool Condition')
        data = dataloader_lightning.dataa_loader_simclr(image_size, my_dataset_traindir, my_dataset_valdir, my_dataset_testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                                                my_dataset_scales = my_dataset_scales, test_mode = False, all_scales_train_bool = all_scales_train_bool, \
                                                warp_image_bool = warp_image_bool, contrastive_2_bool = contrastive_2_bool)
        
    elif orginal_mnist_bool:
        data = dataloader_lightning.dataa_loader(image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                                                orginal_mnist_bool = orginal_mnist_bool)
    else:
        data = dataloader_lightning.dataa_loader(image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                                                linderberg_bool = linderberg_bool, linderberg_dir = linderberg_dir)


    # Callbacks and Trainer
    if IP_contrastive_bool:
        checkpoint_callback = ModelCheckpoint(
                                monitor="train_loss",
                                # Change Path into own folder
                                dirpath="/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/" + prj_name,
                                filename="HMAX-{epoch}-{train_acc1}-{train_loss}",
                                save_top_k=20,
                                mode="min",
                            )
    else:
        checkpoint_callback = ModelCheckpoint(
                                monitor="val_loss",
                                # Change Path into own folder
                                dirpath="/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/" + prj_name,
                                filename="HMAX-{epoch}-{val_acc1}-{val_loss}",
                                save_top_k=20,
                                mode="min",
                            )

    # # create NeptuneLogger
    # neptune_logger = NeptuneLogger(
    #     api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYThkNDA5ZC0yYWM2LTRhYzgtOGMwYi03Y2ZlMzg2MjhiYzEifQ==",  # replace with your own
    #     project="Serre-Lab/monkey-ai",  # "<WORKSPACE/PROJECT>"
    #     tags=["training_HMAX_pyramid" if IP_bool else "training_HMAX"],  # optional
    #     source_files=['*.py'],
    # )
    
    if IP_capsnet_bool: # or IP_contrastive_bool: # or IP_contrastive_finetune_bool:
        print('Clipping grad norm to 0.5')
        trainer = pl.Trainer(max_epochs = num_epochs, devices=n_gpus, accelerator = 'gpu', strategy = 'ddp', callbacks = [checkpoint_callback]) #, precision=16) #, logger = neptune_logger) #, gradient_clip_val= 0.) #, \
                                                    # gradient_clip_algorithm="value") #, logger = wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs = num_epochs, devices=n_gpus, accelerator = 'gpu', strategy = 'dp', callbacks = [checkpoint_callback], accumulate_grad_batches = 1) #, precision=16) #, logger = neptune_logger) #, gradient_clip_val= 0.5, \
                                                # gradient_clip_algorithm="value") #, logger = wandb_logger)
    # Train
    if not(test_mode or val_mode or rdm_corr or rdm_thomas):
        trainer.fit(model, data)
    # Val
    elif val_mode:
        trainer.validate(model, data) 
    # Calculate RDM Correlation
    elif rdm_corr:

        prj_name_save = prj_name

        # ########################
        # Change Path into own folder
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", model.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, f"filters_data_{my_dataset_scales[0]}.pkl")

        open_file = open(file_name, "wb")
        pickle.dump({'Empty1':0}, open_file)
        open_file.close()

        # ########################
        # Change Path into own folder
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", model.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, f"filters_data_{my_dataset_scales[1]}.pkl")

        open_file = open(file_name, "wb")
        pickle.dump({'Empty2':0}, open_file)
        open_file.close()


        if not(linderberg_bool or my_data): 
            for s_i, s_data in enumerate(rdm_datasets):

                print('###################################################')
                print('###################################################')
                print('This is scale : ',s_data)
                print('model.prj_name : ', model.prj_name)
                print('model.HMAX.prj_name : ', model.HMAX.prj_name)

                model.HMAX.MNIST_Scale = s_data
                model.MNIST_Scale = s_data

                for c_i in range(10):

                    print('###################################################')
                    print('This is category : ',c_i)

                    model.HMAX.category = c_i

                    testdir_scale_cat = testdir + str(s_data) + "/test/" + str(c_i)
                    # Calling the dataloader
                    data = dataloader_lightning.dataa_loader(image_size, traindir, valdir, testdir_scale_cat, batch_size_per_gpu, n_gpus, False, True, featur_viz = featur_viz)

                    if s_i == 0:
                        model.first_scale_test = True
                    else:
                        model.first_scale_test = False

                    #
                    trainer.test(model, data)
        
        elif my_data:
            for s_i, s_data in enumerate(my_dataset_scales):

                    print('###################################################')
                    print('###################################################')
                    print('This is scale : ',s_data)

                    model.HMAX.MNIST_Scale = s_data
                    model.MNIST_Scale = s_data

                    if same_scale_viz:
                        my_dataset_testdir_scale = testdir + str(2000) + "/test_viz"
                    else:
                        my_dataset_testdir_scale = my_dataset_testdir + str(s_data) + "/test"
                    

                    if oracle_bool:
                        ########################################################################
                        ########################################################################
                        print('\n###################################################')
                        print('Oracle Version')
                        # Calling the dataloader oracle version
                        model.prj_name = prj_name_save + "_oracle"
                        model.HMAX.prj_name = prj_name_save + "_oracle"
                        prj_name = prj_name_save + "_oracle"

                        data = dataloader_lightning.dataa_loader_my(int(image_size/(s_data/4000)), my_dataset_traindir, my_dataset_valdir, my_dataset_testdir_scale, batch_size_per_gpu, n_gpus, featur_viz, \
                                                                    my_dataset_scales = my_dataset_scales, test_mode = True, warp_image_bool = warp_image_bool, 
                                                                    IP_contrastive_finetune_bool = IP_contrastive_finetune_bool)
                        model.orcale_bool = True

                        # + str(s_data)
                        # prj_name = prj_name + "_ivan_test"
                        if s_i == 0:
                            model.first_scale_test = True
                        else:
                            model.first_scale_test = False
                        #
                        trainer.test(model, data)
                    else:
                        print('\n###################################################')
                        print('Non Oracle Version')
                        # Calling the dataloader

                        if argmax_bool:
                            model.prj_name = prj_name_save + "_argmax"
                            model.HMAX.prj_name = prj_name_save + "_argmax"
                            prj_name = prj_name_save + "_argmax"

                        for c_i in range(n_classes):

                            print('###################################################')
                            print('This is category : ',c_i)

                            model.HMAX.category = c_i

                            testdir_scale_cat = my_dataset_testdir + str(s_data) + "/test/" + str(c_i)

                            if my_data and IP_contrastive_bool:
                                print('In IP_contrastive_bool Condition')
                                data = dataloader_lightning.dataa_loader_contrastive(image_size, my_dataset_traindir, my_dataset_valdir, my_dataset_testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                                                                        my_dataset_scales = my_dataset_scales, test_mode = True)
                            else:
                                data = dataloader_lightning.dataa_loader_my(image_size, my_dataset_traindir, my_dataset_valdir, testdir_scale_cat, batch_size_per_gpu, n_gpus, featur_viz, \
                                                                        my_dataset_scales = my_dataset_scales, test_mode = True, rdm_corr_mode = True, warp_image_bool = warp_image_bool, 
                                                                    IP_contrastive_finetune_bool = IP_contrastive_finetune_bool)

                            model.orcale_bool = False

                            #
                            # + str(s_data)
                            # prj_name = prj_name + "_ivan_test"
                            if s_i == 0:
                                model.first_scale_test = True
                            else:
                                model.first_scale_test = False
                            #
                            trainer.test(model, data)


        print('###################################################')
        print('###################################################')
        print('Now Loading the Data for sending to RDM Corr')

        # Change Path into own folder
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", model.prj_name)
        # job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/color_cnn_FFhGRU_center_real_hGRU_illusions_one/corr_plots"
        # os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, f"filters_data_{my_dataset_scales[0]}.pkl")

        open_file = open(file_name, "rb")
        filters_data_1 = pickle.load(open_file)
        # print('filters_data : ',filters_data.keys())
        open_file.close()

        # Change Path into own folder
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", model.prj_name)
        # job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/color_cnn_FFhGRU_center_real_hGRU_illusions_one/corr_plots"
        # os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, f"filters_data_{my_dataset_scales[1]}.pkl")

        open_file = open(file_name, "rb")
        filters_data_2 = pickle.load(open_file)
        # print('filters_data : ',filters_data.keys())
        open_file.close()

        filters_data = {**filters_data_1, **filters_data_2}
        print('filters_data : ',filters_data.keys())

        # stage_list = ['c1', 'c2', 'c2b', 'c3']
        # stage_list = ['c1', 's2b', 'c2b', 'clf']
        stage_list = save_rdms #['s1'] #, 'c1']

        spearman_corr_list = []
        for stage in stage_list:
            small_scale = []
            large_scale = []
            for s_i, s_data in enumerate(my_dataset_scales):
                for c_i in range(n_classes):
                    if not(linderberg_bool or my_data):
                        key_name = stage + '_scale_' + str(s_data) + '_cat_' + str(c_i)
                    else:
                        key_name = stage + '_scale_' + str(int(s_data*1000)) + '_cat_' + str(c_i)
                    temp_data = filters_data[key_name][:]

                    print('###################################################')
                    print('Kye_name : ', key_name, ' : Shape : ', temp_data.shape)

                    if s_i == 0:
                        small_scale.append(temp_data)
                    else:
                        large_scale.append(temp_data)

            small_scale = np.stack(small_scale, axis=0)
            large_scale = np.stack(large_scale, axis=0)

            print('###################################################')
            print('Going to spearman_corr : ',stage)

            spearman_corr = rdm_corr_func(small_scale, large_scale, job_dir, stage)
            # spearman_corr = direct_corr_func(small_scale, large_scale, job_dir, stage)

            spearman_corr_list.append(spearman_corr)

        # Plot
        fig, ax = plt.subplots(1,1) 
        ax.scatter(range(1,len(spearman_corr_list)+1),spearman_corr_list)

        # Set number of ticks for x-axis
        ax.set_xticks(range(1,len(spearman_corr_list)+1))
        # Set ticks labels for x-axis
        ax.set_xticklabels(stage_list, rotation='vertical', fontsize=18)

        ax.set_xlabel("Pooling Stages", fontweight="bold")
        ax.set_ylabel("RDM Correlation", fontweight="bold")

        ax.set_ylim(0, 1)

        ax.grid()

        # Change Path into own folder
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", model.prj_name)
        fig.savefig(os.path.join(job_dir, "rdm_correlation_plot.png"), dpi=199)

    elif rdm_thomas:



        print('rdm_thomas_datasets : ',rdm_thomas_datasets)

        ########################
        # Change Path into own folder
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_thomas", model.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "filters_data.pkl")

        open_file = open(file_name, "wb")
        pickle.dump({'Empty':0}, open_file)
        open_file.close()


        if not(linderberg_bool or my_data):
            for s_i, s_data in enumerate(rdm_thomas_datasets):

                print('###################################################')
                print('###################################################')
                print('This is scale : ',s_data)
                print('model.prj_name : ', model.prj_name)
                print('model.HMAX.prj_name : ', model.HMAX.prj_name)

                model.HMAX.MNIST_Scale = s_data
                model.MNIST_Scale = s_data

                for c_i in range(1):

                    print('###################################################')
                    print('This is category : ',c_i)

                    model.HMAX.category = c_i

                    testdir_scale_cat = testdir + str(s_data) + "/test/" + str(c_i)
                    if same_scale_viz:
                        testdir_scale_cat = testdir + str(train_dataset) + "/test_viz/" + str(c_i)
                    else:
                        testdir_scale_cat = testdir + str(s_data) + "/test/" + str(c_i)
                    # Calling the dataloader
                    data = dataloader_lightning.dataa_loader(image_size if not(same_scale_viz) else s_data, traindir, valdir, testdir_scale_cat, batch_size_per_gpu, \
                            n_gpus, False, True, featur_viz = featur_viz, same_scale_viz = same_scale_viz, linderberg_bool = linderberg_bool, linderberg_dir = linderberg_dir, linderberg_test = linderberg_bool)

                    if s_i == 0:
                        model.first_scale_test = True
                    else:
                        model.first_scale_test = False

                    #
                    trainer.test(model, data)
        
        elif my_data:
            for s_i, s_data in enumerate(my_dataset_scales):

                print('###################################################')
                print('###################################################')
                print('This is scale : ',s_data)

                model.HMAX.MNIST_Scale = s_data
                model.MNIST_Scale = s_data

                my_dataset_testdir_scale = my_dataset_testdir + str(s_data) + "/test"
                # Calling the dataloader
                data = dataloader_lightning.dataa_loader_my(image_size, my_dataset_traindir, my_dataset_valdir, my_dataset_testdir_scale, batch_size_per_gpu, n_gpus, featur_viz, \
                                                            my_dataset_scales = my_dataset_scales, test_mode = True, warp_image_bool = warp_image_bool, 
                                                                    IP_contrastive_finetune_bool = IP_contrastive_finetune_bool)

                if s_i == 0:
                    model.first_scale_test = True
                else:
                    model.first_scale_test = False
                #
                trainer.test(model, data)
        
        else:
            l_i = 0
            for l_data in linderberg_test_dir:

                print('###################################################')
                print('###################################################')
                print('This is l_data : ',l_data)

                model.HMAX.MNIST_Scale = l_data
                model.MNIST_Scale = l_data

                l_h5_file = linderberg_test_dir[l_data]

                linderberg_dir = "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/" + l_h5_file
                # Calling the dataloader
                data = dataloader_lightning.dataa_loader(image_size if not(same_scale_viz) else s_data, traindir, valdir, linderberg_dir, batch_size_per_gpu, \
                                                        n_gpus, True, featur_viz = featur_viz, same_scale_viz = same_scale_viz, \
                                                        linderberg_bool = linderberg_bool, linderberg_dir = linderberg_dir, linderberg_test = True)

                #
                # + str(s_data)
                # prj_name = prj_name + "_ivan_test"
                if l_i == 0:
                    model.first_scale_test = True
                else:
                    model.first_scale_test = False
                
                l_i += 1
                #
                trainer.test(model, data)



        print('###################################################')
        print('###################################################')
        print('Now Loading the Data for sending to RDM thomas')

        # Change Path into own folder
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_thomas", model.prj_name)
        # job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/color_cnn_FFhGRU_center_real_hGRU_illusions_one/corr_plots"
        # os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "filters_data.pkl")

        open_file = open(file_name, "rb")
        filters_data = pickle.load(open_file)
        print('filters_data : ',filters_data.keys())
        open_file.close()

        # stage_list = ['c1', 'c2', 'c2b', 'c3']
        stage_list = ['c1', 'c2b']

        for stage in stage_list:
            for s_i, s_data in enumerate(rdm_thomas_datasets):
                # if s_i == len(rdm_thomas_datasets)-1:
                #     continue
                for c_i in range(1):
                    if not(linderberg_bool or my_data):
                        key_name = stage + '_scale_' + str(s_data) + '_cat_' + str(c_i)
                    else:
                        # s_data = int(s_data*1000)
                        key_name = stage + '_scale_' + str(int(s_data*1000))
                    temp_data = filters_data[key_name][:8]

                    if not(linderberg_bool or my_data):
                        base_key_name = stage + '_scale_' + str(image_size) + '_cat_' + str(c_i)
                    else:
                        base_key_name = stage + '_scale_' + str(int(rdm_thomas_datasets[int(len(rdm_thomas_datasets)//2)]*1000)) 
                    # base_key_name = stage + '_scale_' + str(rdm_thomas_datasets[s_i+1]) + '_cat_' + str(c_i)
                    base_temp_data = filters_data[base_key_name][:8]

                    print('###################################################')
                    print('key_name : ', key_name, ' : Shape : ', temp_data.shape)
                    print('base_key_name : ', base_key_name, ' : Shape : ', base_temp_data.shape)

                    # rdm_corr_scales_func(scale_base_state_features, scale_state_features, scale, n_scales, save_dir, c_stage):
                    if my_data:
                        s_data = s_data/1000.0

                    rdm_corr_scales_func(base_temp_data, temp_data, s_data, len(rdm_thomas_datasets), job_dir, stage, (linderberg_bool or my_data))

                    

            print('###################################################')

    # Test
    else:
        prj_name_save = prj_name

        if not(oracle_plot_overlap_bool or argmax_plot_overlap_bool or oracle_argmax_plot_overlap_bool):
            if not(linderberg_bool or my_data or cifar_data_bool):
                for s_i, s_data in enumerate(scale_datasets):

                    print('###################################################')
                    print('###################################################')
                    print('This is scale : ',s_data)

                    model.HMAX.MNIST_Scale = s_data
                    model.MNIST_Scale = s_data

                    if same_scale_viz:
                        testdir_scale = testdir + str(train_dataset) + "/test_viz"
                    else:
                        testdir_scale = testdir + str(s_data) + "/test"
                    # Calling the dataloader
                    data = dataloader_lightning.dataa_loader(image_size if not(same_scale_viz) else s_data, traindir, valdir, testdir_scale, batch_size_per_gpu, n_gpus, True, featur_viz = featur_viz, same_scale_viz = same_scale_viz)

                    #
                    # + str(s_data)
                    # prj_name = prj_name + "_ivan_test"
                    if s_i == 0:
                        model.first_scale_test = True
                    else:
                        model.first_scale_test = False
                    #
                    trainer.test(model, data)
            elif my_data or cifar_data_bool:
                for s_i, s_data in enumerate(my_dataset_scales):

                    print('###################################################')
                    print('###################################################')
                    print('This is scale : ',s_data)

                    model.HMAX.MNIST_Scale = s_data
                    model.MNIST_Scale = s_data

                    if same_scale_viz:
                        my_dataset_testdir_scale = testdir + str(2000) + "/test_viz"
                    else:
                        my_dataset_testdir_scale = my_dataset_testdir + str(s_data) + "/test"
                    

                    if oracle_bool:
                        ########################################################################
                        ########################################################################
                        print('\n###################################################')
                        print('Oracle Version')
                        # Calling the dataloader oracle version
                        model.prj_name = prj_name_save + "_oracle"
                        model.HMAX.prj_name = prj_name_save + "_oracle"
                        prj_name = prj_name_save + "_oracle"

                        data = dataloader_lightning.dataa_loader_my(int(image_size/(s_data/4000)), my_dataset_traindir, my_dataset_valdir, my_dataset_testdir_scale, batch_size_per_gpu, n_gpus, featur_viz, \
                                                                    my_dataset_scales = my_dataset_scales, test_mode = True, warp_image_bool = warp_image_bool, 
                                                                    IP_contrastive_finetune_bool = IP_contrastive_finetune_bool)
                        model.orcale_bool = True

                        # + str(s_data)
                        # prj_name = prj_name + "_ivan_test"
                        if s_i == 0:
                            model.first_scale_test = True
                        else:
                            model.first_scale_test = False
                        #
                        trainer.test(model, data)
                    else:
                        print('\n###################################################')
                        print('Non Oracle Version')
                        # Calling the dataloader

                        if argmax_bool:
                            model.prj_name = prj_name_save + "_argmax"
                            model.HMAX.prj_name = prj_name_save + "_argmax"
                            prj_name = prj_name_save + "_argmax"


                        if cifar_data_bool and not IP_contrastive_bool:
                            rescaled_image_size = int(32*s_data)
                            if rescaled_image_size%2 != 0:
                                rescaled_image_size += 1
                            print('rescaled_image_size : ',rescaled_image_size)
                            data = dataloader_lightning.dataa_loader_cifar10(rescaled_image_size, my_dataset_traindir, my_dataset_valdir, my_dataset_testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                                                my_dataset_scales = my_dataset_scales, test_mode = False, all_scales_train_bool = all_scales_train_bool, warp_image_bool = warp_image_bool, 
                                                IP_contrastive_finetune_bool = IP_contrastive_finetune_bool)
                        elif my_data and IP_contrastive_bool:
                            print('In IP_contrastive_bool Condition')
                            data = dataloader_lightning.dataa_loader_contrastive(image_size, my_dataset_traindir, my_dataset_valdir, my_dataset_testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                                                                    my_dataset_scales = my_dataset_scales, test_mode = True)
                        else:
                            data = dataloader_lightning.dataa_loader_my(image_size, my_dataset_traindir, my_dataset_valdir, my_dataset_testdir_scale, batch_size_per_gpu, n_gpus, featur_viz, \
                                                                    my_dataset_scales = my_dataset_scales, test_mode = True, warp_image_bool = warp_image_bool, 
                                                                    IP_contrastive_finetune_bool = IP_contrastive_finetune_bool)

                        model.orcale_bool = False

                        #
                        # + str(s_data)
                        # prj_name = prj_name + "_ivan_test"
                        if s_i == 0:
                            model.first_scale_test = True
                        else:
                            model.first_scale_test = False
                        #
                        trainer.test(model, data)
            else:
                l_i = 0
                for l_data in linderberg_test_dir:

                    print('###################################################')
                    print('###################################################')
                    print('This is l_data : ',l_data)

                    model.HMAX.MNIST_Scale = l_data
                    model.MNIST_Scale = l_data

                    l_h5_file = linderberg_test_dir[l_data]

                    linderberg_dir = "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/" + l_h5_file
                    # Calling the dataloader
                    data = dataloader_lightning.dataa_loader(image_size if not(same_scale_viz) else s_data, traindir, valdir, linderberg_dir, batch_size_per_gpu, \
                                                            n_gpus, True, featur_viz = featur_viz, same_scale_viz = same_scale_viz, \
                                                            linderberg_bool = linderberg_bool, linderberg_dir = linderberg_dir, linderberg_test = True)

                    #
                    # + str(s_data)
                    # prj_name = prj_name + "_ivan_test"
                    if l_i == 0:
                        model.first_scale_test = True
                    else:
                        model.first_scale_test = False
                    
                    l_i += 1
                    #
                    trainer.test(model, data)

        
        if not(same_scale_viz or oracle_bool or argmax_bool or featur_viz or visualize_mode):
            # Change Path into own folder
            job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/save_pickles", prj_name_save)
            os.makedirs(job_dir, exist_ok=True)
            file_name = os.path.join(job_dir, "scale_test_acc.pkl")
            open_file = open(file_name, "rb")
            test_accs = pickle.load(open_file)
            open_file.close()

            if oracle_plot_overlap_bool:
                # Change Path into own folder
                job_dir_orc = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/save_pickles", prj_name_save + "_oracle")
                os.makedirs(job_dir_orc, exist_ok=True)
                file_name_orc = os.path.join(job_dir_orc, "scale_test_acc.pkl")
                open_file_orc = open(file_name_orc, "rb")
                test_accs_orc = pickle.load(open_file_orc)
                open_file_orc.close()
            elif argmax_plot_overlap_bool:
                # Change Path into own folder
                job_dir_arg = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/save_pickles", prj_name_save + "_argmax")
                os.makedirs(job_dir_arg, exist_ok=True)
                file_name_arg = os.path.join(job_dir_arg, "scale_test_acc.pkl")
                open_file_arg = open(file_name_arg, "rb")
                test_accs_arg = pickle.load(open_file_arg)
                open_file_arg.close()
            elif oracle_argmax_plot_overlap_bool:
                # Oracle
                # Change Path into own folder
                job_dir_orc = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/save_pickles", prj_name_save + "_oracle")
                os.makedirs(job_dir_orc, exist_ok=True)
                file_name_orc = os.path.join(job_dir_orc, "scale_test_acc.pkl")
                open_file_orc = open(file_name_orc, "rb")
                test_accs_orc = pickle.load(open_file_orc)
                open_file_orc.close()
                # Argmax
                # Change Path into own folder
                job_dir_arg = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/save_pickles", prj_name_save + "_argmax")
                os.makedirs(job_dir_arg, exist_ok=True)
                file_name_arg = os.path.join(job_dir_arg, "scale_test_acc.pkl")
                open_file_arg = open(file_name_arg, "rb")
                test_accs_arg = pickle.load(open_file_arg)
                open_file_arg.close()



            if linderberg_bool or my_data:
                test_accs = np.array(test_accs)

                fig = plt.figure(figsize=(45,30), dpi=250)
                ax = fig.add_subplot(111)
                if my_data:
                    scales_list = [mds/1000 for mds in my_dataset_scales]
                else:
                    scales_list = linderberg_test_dir.keys()

                # ax.scatter(list(scales_list), test_accs, s=100, c='b', label='Max Over Scales')
                ax.plot(list(scales_list), test_accs, c='b', label='Max Over Scales')

                # # if not(oracle_plot_overlap_bool):
                # for xy in zip(list(scales_list), test_accs):                                       # <--
                #     ax.annotate('(%.6s, %.6s)' % xy, xy=xy, textcoords='data', fontsize=25) # <--

                if oracle_plot_overlap_bool:
                    test_accs_orc = np.array(test_accs_orc)

                    ax.scatter(list(scales_list), test_accs_orc, s=100, c='r', label='Oracle Case -  Single Scale Test')
                    for xy in zip(list(scales_list), test_accs_orc):                                       # <--
                        ax.annotate('(%.6s, %.6s)' % xy, xy=xy, textcoords='data', fontsize=25) # <--

                elif argmax_plot_overlap_bool:
                    test_accs_arg = np.array(test_accs_arg)

                    # ax.scatter(list(scales_list), test_accs_arg, s=100, c='g', label='Argmax Case')
                    ax.plot(list(scales_list), test_accs_arg, c='g', label='Argmax Case')

                elif oracle_argmax_plot_overlap_bool:
                    # Oracle
                    test_accs_orc = np.array(test_accs_orc)

                    # ax.scatter(list(scales_list), test_accs_orc, s=100, c='r', label='Oracle Case -  Single Scale Test')
                    ax.plot(list(scales_list), test_accs_orc, c='r', label='Oracle Case -  Single Scale Test')

                    # for xy in zip(list(scales_list), test_accs_orc):                                       # <--
                    #     ax.annotate('(%.6s, %.6s)' % xy, xy=xy, textcoords='data', fontsize=25) # <--

                    # Argmax
                    test_accs_arg = np.array(test_accs_arg)

                    # ax.scatter(list(scales_list), test_accs_arg, s=100, c='g', label='Argmax Case')
                    ax.plot(list(scales_list), test_accs_arg, c='g', label='Argmax Case')

                    # for xy in zip(list(scales_list), test_accs_arg):                                       # <--
                    #     ax.annotate('(%.6s, %.6s)' % xy, xy=xy, textcoords='data', fontsize=25) # <--

                
                ax.legend(loc='lower right', fontsize=30)

                ax.set_xscale('log')
                ax.set_xticks(list(scales_list))
                ax.set_xticklabels(list(scales_list))

                ax.set_xlabel(
                "Scales",
                fontweight="bold",
                fontsize=40.0)

                ax.set_ylabel(
                "Accuracy",
                fontweight="bold",
                fontsize=40.0,)

                ax.tick_params(axis='both', labelsize=25)


            else:
                index_sort = np.argsort(scale_datasets)
                scale_datasets = np.array(scale_datasets)[index_sort]
                test_accs = np.array(test_accs)[index_sort]

                fig = plt.figure(figsize=(18,10), dpi=250)
                ax = fig.add_subplot(111)
                ax.scatter(scale_datasets, test_accs)
                ax.set_xlabel(
                "Scales",
                fontweight="bold",
                fontsize=15.0)
                ax.set_ylabel(
                "Accuracy",
                fontweight="bold",
                fontsize=15.0,)
                for xy in zip(scale_datasets, test_accs):                                       # <--
                    ax.annotate('(%.6s, %.6s)' % xy, xy=xy, textcoords='data') # <--

            ax.grid()

            # Change Path into own folder
            job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/scale_invar_plots"
            if oracle_plot_overlap_bool:
                fig.savefig(os.path.join(job_dir, prj_name_save + '_oracle_overlap'))
            elif argmax_plot_overlap_bool:
                fig.savefig(os.path.join(job_dir, prj_name_save + '_argmax_overlap'))
            elif oracle_argmax_plot_overlap_bool:
                fig.savefig(os.path.join(job_dir, prj_name_save + '_oracle_argmax_overlap'))
            else:
                fig.savefig(os.path.join(job_dir, prj_name_save))

        
        # #####################################################################################################

        if 's1' in plt_filters:
            main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters'
            # S1
            S1_dir = os.path.join(main_dir, 'S1')
            S1_dir = os.path.join(S1_dir, 'prj_' + prj_name_save)
            images_list = os.listdir(S1_dir)
            
            filtered_images_list = []
            for il in images_list:
                if il.split('.')[-1] == 'npy':
                    filtered_images_list.append(il)

            scales_list = np.array([int(fil.split('_')[0]) for fil in filtered_images_list])
            print('scales_list : ',scales_list)
            index_sort = np.argsort(scales_list)
            sorted_images_list = [filtered_images_list[i_s] for i_s in index_sort]

            sorted_images_list = [os.path.join(S1_dir, sil) for sil in sorted_images_list]

            print('sorted_images_list : ',sorted_images_list)

            combined_image = np.empty((1))
            for index, im_path in enumerate(sorted_images_list):
                # combined_vertical_image = cv2.imread(im_path)
                combined_vertical_image = np.load(im_path)

                if len(combined_image.shape) == 1:
                    combined_image = combined_vertical_image
                else:
                    # print('combined_image : ',combined_image.shape)
                    # print('combined_vertical_image : ',combined_vertical_image.shape)
                    combined_image = cv2.hconcat([combined_image, combined_vertical_image])

            out_path = os.path.join(S1_dir, "filters_all.png")
            cv2.imwrite(out_path, combined_image)

            plt.figure(figsize = (50, 100))
            plt.imshow(combined_image)
            plt.savefig(out_path.split('.')[0] + '_plt.png')


        # ######################################################################################################


        if 'c1' in plt_filters:
            main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters'
            # C1
            C1_dir = os.path.join(main_dir, 'C1')
            C1_dir = os.path.join(C1_dir, 'prj_' + prj_name_save)
            images_list = os.listdir(C1_dir)
            
            filtered_images_list = []
            for il in images_list:
                if il.split('.')[-1] == 'npy':
                    filtered_images_list.append(il)

            scales_list = np.array([int(fil.split('_')[0]) for fil in filtered_images_list])
            print('scales_list : ',scales_list)
            index_sort = np.argsort(scales_list)
            sorted_images_list = [filtered_images_list[i_s] for i_s in index_sort]

            sorted_images_list = [os.path.join(C1_dir, sil) for sil in sorted_images_list]

            print('sorted_images_list : ',sorted_images_list)

            combined_image = np.empty((1))
            for index, im_path in enumerate(sorted_images_list):
                # combined_vertical_image = cv2.imread(im_path)
                combined_vertical_image = np.load(im_path)

                print('combined_image : ',combined_image.shape)
                print('combined_vertical_image : ',combined_vertical_image.shape)

                if len(combined_image.shape) == 1:
                    combined_image = combined_vertical_image
                else:
                    combined_image = cv2.hconcat([combined_image, combined_vertical_image])

            out_path = os.path.join(C1_dir, "filters_all.png")
            cv2.imwrite(out_path, combined_image)

            plt.figure(figsize = (50, 100))
            plt.imshow(combined_image)
            plt.savefig(out_path.split('.')[0] + '_plt.png')

        # ######################################################################################################

        if 'c2b' in plt_filters:
            main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters'
            # C2
            C2_dir = os.path.join(main_dir, 'C2')
            C2_dir = os.path.join(C2_dir, 'prj_' + prj_name_save)
            images_list = os.listdir(C2_dir)
            
            filtered_images_list = []
            for il in images_list:
                if il.split('.')[-1] == 'npy':
                    filtered_images_list.append(il)

            scales_list = np.array([int(fil.split('_')[0]) for fil in filtered_images_list])
            print('scales_list : ',scales_list)
            index_sort = np.argsort(scales_list)
            sorted_images_list = [filtered_images_list[i_s] for i_s in index_sort]

            sorted_images_list = [os.path.join(C2_dir, sil) for sil in sorted_images_list]

            print('sorted_images_list : ',sorted_images_list)

            combined_image = np.empty((1))
            for index, im_path in enumerate(sorted_images_list):
                # combined_vertical_image = cv2.imread(im_path)
                combined_vertical_image = np.load(im_path)

                print('combined_image : ',combined_image.shape)
                print('combined_vertical_image : ',combined_vertical_image.shape)

                if len(combined_image.shape) == 1:
                    combined_image = combined_vertical_image
                else:
                    combined_image = cv2.hconcat([combined_image, combined_vertical_image])

            out_path = os.path.join(C2_dir, "filters_all.png")
            cv2.imwrite(out_path, combined_image)

            plt.figure(figsize = (50, 100))
            plt.imshow(combined_image)
            plt.savefig(out_path.split('.')[0] + '_plt.png')

        # ######################################################################################################


        if 's2b' in plt_filters:
            main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters'
            # S2b
            S2b_dir = os.path.join(main_dir, 'S2b')
            S2b_dir = os.path.join(S2b_dir, 'prj_' + prj_name_save)
            images_list = os.listdir(S2b_dir)
            
            filtered_images_list = []
            for il in images_list:
                if il.split('.')[-1] == 'npy':
                    filtered_images_list.append(il)

            scales_list = np.array([int(fil.split('_')[0]) for fil in filtered_images_list])
            print('scales_list : ',scales_list)
            index_sort = np.argsort(scales_list)
            sorted_images_list = [filtered_images_list[i_s] for i_s in index_sort]

            sorted_images_list = [os.path.join(S2b_dir, sil) for sil in sorted_images_list]

            print('sorted_images_list : ',sorted_images_list)

            combined_image = np.empty((1))
            for index, im_path in enumerate(sorted_images_list):
                # combined_vertical_image = cv2.imread(im_path)
                combined_vertical_image = np.load(im_path)

                if len(combined_image.shape) == 1:
                    combined_image = combined_vertical_image
                else:
                    # print('combined_image : ',combined_image.shape)
                    # print('combined_vertical_image : ',combined_vertical_image.shape)
                    combined_image = cv2.hconcat([combined_image, combined_vertical_image])

            out_path = os.path.join(S2b_dir, "filters_all.png")
            cv2.imwrite(out_path, combined_image)

            plt.figure(figsize = (50, 100))
            plt.imshow(combined_image)
            plt.savefig(out_path.split('.')[0] + '_plt.png')
