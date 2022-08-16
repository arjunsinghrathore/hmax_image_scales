import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
np.random.seed(1)
# import pandas as pd
import cv2
# from PIL import Image
# from torchsummary import summary
import time
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
# from sklearn.preprocessing import LabelEncoder

from pytorch_lightning.loggers import WandbLogger
import _pickle as pickle


from torch.utils.data import random_split, DataLoader, Dataset

from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

import dataloader_lightning
# import models_hGRU_center_GAP
# from hmax_models.hmax_ivan import HMAX_latest_slim, HMAX_latest
# print('Importedddddd HMAX_latest_slim')
import hmax_fixed_ligtning

from rdm_corr import rdm_corr_func

# api_key = 'b508002bdc18b80b784941855ce5a0e722ef50d8'
# os.environ["WANDB_API_KEY"] = api_key
# wandb.init()


# Seeds
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

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
    # prj_name = "checkpoint_HMAX_PNAS_IVAN_100_MNIST_18_17s_256i" #_new_stride"
    prj_name = "checkpoint_HMAX_PNAS_100_MNIST_18_IP_GAP_7s_up_down_linderberg_C_first_pos_1by3_192_13_down_mp_like_HMAX_continued"
    # prj_name = "checkpoint_HMAX_PNAS_100_MNIST_18_IP_GAP_7s_up_down_linderberg_C_first_pos_s_normalize_fastnorm_across_patches_1by3_192_13_down_mp_like_HMAX_continued"
    n_ori = 4
    n_classes = 10
    lr = 0.000001
    # lr = 0.0000005
    # lr = 0.00005

    # lr = 10e-6 # --> HMAX
    weight_decay = 1e-4
    batch_size_per_gpu = 12 #8
    num_epochs = 1000
    ip_scales = 7
    image_size = 192
    IP_bool = True

    # Mode
    test_mode = False
    val_mode = False
    continue_tr = False
    visualize_mode = False
    rdm_corr = True

    # Dataset Setting
    scale_datasets = [24,30,36,12,8,4,18,20,16]
    train_dataset = 18
    rdm_datasets = [18, 24]

    MNIST_Scale = train_dataset

    # Initializing the model
    model = hmax_fixed_ligtning.HMAX_trainer(prj_name, n_ori, n_classes, lr, weight_decay, ip_scales, IP_bool, visualize_mode, MNIST_Scale)

    if test_mode or val_mode or continue_tr or rdm_corr:
        model = model.load_from_checkpoint('/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/' + prj_name + '/HMAX-epoch=256-val_acc1=80.33853912353516-val_loss=0.5373715758323669.ckpt')

        ########################## While Testing ##########################
        ## Need to force change some variables after loading a checkpoint
        if rdm_corr:
            model.prj_name = prj_name + "_closer"
            model.HMAX.prj_name = prj_name + "_closer"
        model.visualize_mode = visualize_mode
        model.MNIST_Scale = MNIST_Scale
        model.HMAX.MNIST_Scale = MNIST_Scale
        model.lr = lr
    
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


    # Calling the dataloader
    data = dataloader_lightning.dataa_loader(image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus)


    # Callbacks and Trainer
    checkpoint_callback = ModelCheckpoint(
                            monitor="val_acc1",
                            dirpath="/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/" + prj_name,
                            filename="HMAX-{epoch}-{val_acc1}-{val_loss}",
                            save_top_k=8,
                            mode="max",
                        )

    # wandb_logger = WandbLogger(project=prj_name)
    # # log gradients, parameter histogram and model topology
    # wandb_logger.watch(model, log="all")
    
    trainer = pl.Trainer(max_epochs = num_epochs, gpus=-1, accelerator = 'dp', callbacks = [checkpoint_callback]) #, gradient_clip_val= 0.5, \
                                                # gradient_clip_algorithm="value") #, logger = wandb_logger)
    # Train
    if not(test_mode or val_mode or rdm_corr):
        trainer.fit(model, data)
    # Val
    elif val_mode:
        trainer.validate(model, data) 
    # Calculate RDM Correlation
    elif rdm_corr:

        ########################
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", model.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "filters_data.pkl")

        open_file = open(file_name, "wb")
        pickle.dump({'Empty':0}, open_file)
        open_file.close()


        for s_i, s_data in enumerate(rdm_datasets):

            print('###################################################')
            print('###################################################')
            print('This is scale : ',s_data)

            model.HMAX.MNIST_Scale = s_data
            model.MNIST_Scale = s_data

            for c_i in range(10):

                print('###################################################')
                print('This is category : ',c_i)

                model.HMAX.category = c_i

                testdir_scale_cat = testdir + str(s_data) + "/test/" + str(c_i)
                # Calling the dataloader
                data = dataloader_lightning.dataa_loader(image_size, traindir, valdir, testdir_scale_cat, batch_size_per_gpu, n_gpus, False, True)

                if s_i == 0:
                    model.first_scale_test = True
                else:
                    model.first_scale_test = False

                #
                trainer.test(model, data)

        print('###################################################')
        print('###################################################')
        print('Now Loading the Data for sending to RDM Corr')

        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", model.prj_name)
        # job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/color_cnn_FFhGRU_center_real_hGRU_illusions_one/corr_plots"
        # os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "filters_data.pkl")

        open_file = open(file_name, "rb")
        filters_data = pickle.load(open_file)
        print('filters_data : ',filters_data.keys())
        open_file.close()

        stage_list = ['c1', 'c2', 'c2b', 'c3']

        spearman_corr_list = []
        for stage in stage_list:
            small_scale = []
            large_scale = []
            for s_i, s_data in enumerate(rdm_datasets):
                for c_i in range(10):
                    key_name = stage + '_scale_' + str(s_data) + '_cat_' + str(c_i)
                    temp_data = filters_data[key_name][:36]

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

            spearman_corr_list.append(spearman_corr)

        # Plot
        fig, ax = plt.subplots(1,1) 
        ax.scatter(range(1,5),spearman_corr_list)

        # Set number of ticks for x-axis
        ax.set_xticks(range(1,5))
        # Set ticks labels for x-axis
        ax.set_xticklabels(stage_list, rotation='vertical', fontsize=18)

        ax.set_xlabel("Pooling Stages", fontweight="bold")
        ax.set_ylabel("RDM Correlation", fontweight="bold")

        ax.grid()

        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_corr", model.prj_name)
        fig.savefig(os.path.join(job_dir, "rdm_correlation_plot.png"))

    # Test
    else:
        # model.prj_name = model.prj_name + "_ivan_test"
        # model.HMAX.prj_name = model.HMAX.prj_name + "_ivan_test"
        # prj_name = prj_name + "_ivan_test"
        for s_i, s_data in enumerate(scale_datasets):

            print('###################################################')
            print('###################################################')
            print('This is scale : ',s_data)

            model.HMAX.MNIST_Scale = s_data
            model.MNIST_Scale = s_data

            testdir_scale = testdir + str(s_data) + "/test"
            # Calling the dataloader
            data = dataloader_lightning.dataa_loader(image_size, traindir, valdir, testdir_scale, batch_size_per_gpu, n_gpus, True)

            #
            # + str(s_data)
            # prj_name = prj_name + "_ivan_test"
            if s_i == 0:
                model.first_scale_test = True
            else:
                model.first_scale_test = False

            #
            trainer.test(model, data)

        # #
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/save_pickles", model.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "scale_test_acc.pkl")
        open_file = open(file_name, "rb")
        test_accs = pickle.load(open_file)
        open_file.close()

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

        job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/scale_invar_plots"
        fig.savefig(os.path.join(job_dir, prj_name))

        
        # #####################################################################################################

        # main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters'
        # # S1
        # S1_dir = os.path.join(main_dir, 'S1')
        # S1_dir = os.path.join(S1_dir, 'prj_' + prj_name)
        # images_list = os.listdir(S1_dir)
        
        # filtered_images_list = []
        # for il in images_list:
        #     if il.split('.')[-1] == 'npy':
        #         filtered_images_list.append(il)

        # scales_list = np.array([int(fil.split('_')[0]) for fil in filtered_images_list])
        # print('scales_list : ',scales_list)
        # index_sort = np.argsort(scales_list)
        # sorted_images_list = [filtered_images_list[i_s] for i_s in index_sort]

        # sorted_images_list = [os.path.join(S1_dir, sil) for sil in sorted_images_list]

        # print('sorted_images_list : ',sorted_images_list)

        # combined_image = np.empty((1))
        # for index, im_path in enumerate(sorted_images_list):
        #     # combined_vertical_image = cv2.imread(im_path)
        #     combined_vertical_image = np.load(im_path)

        #     if len(combined_image.shape) == 1:
        #         combined_image = combined_vertical_image
        #     else:
        #         # print('combined_image : ',combined_image.shape)
        #         # print('combined_vertical_image : ',combined_vertical_image.shape)
        #         combined_image = cv2.hconcat([combined_image, combined_vertical_image])

        # out_path = os.path.join(S1_dir, "filters_all.png")
        # cv2.imwrite(out_path, combined_image)

        # plt.figure(figsize = (50, 100))
        # plt.imshow(combined_image)
        # plt.savefig(out_path.split('.')[0] + '_plt.png')


        # ######################################################################################################

        # main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters'
        # # S1
        # C1_dir = os.path.join(main_dir, 'C1')
        # C1_dir = os.path.join(C1_dir, 'prj_' + prj_name)
        # images_list = os.listdir(C1_dir)
        
        # filtered_images_list = []
        # for il in images_list:
        #     if il.split('.')[-1] == 'npy':
        #         filtered_images_list.append(il)

        # scales_list = np.array([int(fil.split('_')[0]) for fil in filtered_images_list])
        # print('scales_list : ',scales_list)
        # index_sort = np.argsort(scales_list)
        # sorted_images_list = [filtered_images_list[i_s] for i_s in index_sort]

        # sorted_images_list = [os.path.join(C1_dir, sil) for sil in sorted_images_list]

        # print('sorted_images_list : ',sorted_images_list)

        # combined_image = np.empty((1))
        # for index, im_path in enumerate(sorted_images_list):
        #     # combined_vertical_image = cv2.imread(im_path)
        #     combined_vertical_image = np.load(im_path)

        #     if len(combined_image.shape) == 1:
        #         combined_image = combined_vertical_image
        #     else:
        #         # print('combined_image : ',combined_image.shape)
        #         # print('combined_vertical_image : ',combined_vertical_image.shape)
        #         combined_image = cv2.hconcat([combined_image, combined_vertical_image])

        # out_path = os.path.join(C1_dir, "filters_all.png")
        # cv2.imwrite(out_path, combined_image)

        # plt.figure(figsize = (50, 100))
        # plt.imshow(combined_image)
        # plt.savefig(out_path.split('.')[0] + '_plt.png')