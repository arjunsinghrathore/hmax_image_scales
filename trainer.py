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

from pytorch_lightning.loggers import NeptuneLogger

import _pickle as pickle


from torch.utils.data import random_split, DataLoader, Dataset

from pytorch_lightning.callbacks import ModelCheckpoint


import dataloader_lightning
# import models_hGRU_center_GAP
# from hmax_models.hmax_ivan import HMAX_latest_slim, HMAX_latest
# print('Importedddddd HMAX_latest_slim')
import hmax_fixed_ligtning

from rdm_corr import rdm_corr_func, rdm_corr_scales_func

import os
import shutil
import pickle
import random

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
    prj_name = "checkpoint_HMAX_PNAS_IVAN_100_GAP_MNIST_18_17s_224i_10e6_s4_drop_lr" #_new_stride"
    # prj_name = "checkpoint_HMAX_PNAS_100_MNIST_18_IP_GAP_7s_up_down_linderberg_C_first_pos_s_normalize_filt_norm_alpha_1by4_192_13_down_mp_like_HMAX_continued"
    n_ori = 4
    n_classes = 10
    IP_bool = False

    if IP_bool:
        # lr = 0.000001 # For 7 scales
        lr = 0.00000005 # For 14 scales
        # lr = 0.00005
        weight_decay = 1e-4
        batch_size_per_gpu = 8
        num_epochs = 1000
        ip_scales = 10 #14 #7
        image_size = 160 #128 #192
    else:
        lr = 10e-6# --> HMAX
        weight_decay = 1e-4
        batch_size_per_gpu = 4
        num_epochs = 1000
        ip_scales = 17
        image_size = 192 #224 # For HMAx


    # Mode
    test_mode = False
    val_mode = False
    continue_tr = False
    visualize_mode = False
    rdm_corr = False
    rdm_thomas = False

    featur_viz = False
    same_scale_viz = False

    # Dataset Setting
    if same_scale_viz:
        base_image_size = image_size
        scale = 4
        image_scales_down = [int(np.ceil(base_image_size/(2**(i/scale)))) if np.ceil(base_image_size/(2**(i/scale)))%2 == 0 else int(np.floor(base_image_size/(2**(i/scale)))) for i in range(int(np.ceil(ip_scales/2)))]
        image_scales_up = [int(np.ceil(base_image_size*(2**(i/scale)))) if np.ceil(base_image_size*(2**(i/scale)))%2 == 0 else int(np.floor(base_image_size*(2**(i/scale)))) for i in range(1, int(np.ceil(ip_scales/2)))]
        # image_scales = [np.ceil(base_image_size) if np.ceil(base_image_size)%2 == 0 else np.floor(base_image_size) for i in range(self.ip_scales)]

        image_scales = image_scales_down + image_scales_up
        # scale_datasets = sorted(image_scales)

        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        scale_datasets = [image_scales[i_s] for i_s in index_sort]
        
    else:
        scale_datasets = [18,24,30,36,12,8,4,20,16]

    train_dataset = 18
    rdm_datasets = [18, 24]
    rdm_thomas_datasets = scale_datasets

    MNIST_Scale = train_dataset

    # Initializing the model
    model = hmax_fixed_ligtning.HMAX_trainer(prj_name, n_ori, n_classes, lr, weight_decay, ip_scales, IP_bool, visualize_mode, MNIST_Scale)

    if test_mode or val_mode or continue_tr or rdm_corr or rdm_thomas:
        model = model.load_from_checkpoint('/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/' + prj_name + '/HMAX-epoch=443-val_acc1=86.18790064102564-val_loss=0.4148487138531445.ckpt')

        ########################## While Testing ##########################
        ## Need to force change some variables after loading a checkpoint
        if rdm_corr or rdm_thomas:
            model.prj_name = prj_name + "_ref"
            model.HMAX.prj_name = prj_name + "_ref"
        else:
            model.prj_name = prj_name
            model.HMAX.prj_name = prj_name
        model.visualize_mode = visualize_mode
        model.MNIST_Scale = MNIST_Scale
        model.HMAX.MNIST_Scale = MNIST_Scale
        model.lr = lr
        model.HMAX.same_scale_viz = same_scale_viz
        model.HMAX.base_scale = image_size
    
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
    data = dataloader_lightning.dataa_loader(image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, featur_viz)


    # Callbacks and Trainer
    checkpoint_callback = ModelCheckpoint(
                            monitor="val_acc1",
                            dirpath="/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/" + prj_name,
                            filename="HMAX-{epoch}-{val_acc1}-{val_loss}",
                            save_top_k=8,
                            mode="max",
                        )

    # create NeptuneLogger
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYThkNDA5ZC0yYWM2LTRhYzgtOGMwYi03Y2ZlMzg2MjhiYzEifQ==",  # replace with your own
        project="Serre-Lab/monkey-ai",  # "<WORKSPACE/PROJECT>"
        tags=["training_HMAX"],  # optional
        source_files=['*.py'],
    )
    
    trainer = pl.Trainer(max_epochs = num_epochs, devices=n_gpus, accelerator = 'gpu', strategy = 'dp', callbacks = [checkpoint_callback], logger = neptune_logger) #, gradient_clip_val= 0.5, \
                                                # gradient_clip_algorithm="value") #, logger = wandb_logger)
    # Train
    if not(test_mode or val_mode or rdm_corr or rdm_thomas):
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

    elif rdm_thomas:

        print('rdm_thomas_datasets : ',rdm_thomas_datasets)

        ########################
        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_thomas", model.prj_name)
        os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "filters_data.pkl")

        open_file = open(file_name, "wb")
        pickle.dump({'Empty':0}, open_file)
        open_file.close()


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
                data = dataloader_lightning.dataa_loader(image_size if not(same_scale_viz) else s_data, traindir, valdir, testdir_scale_cat, batch_size_per_gpu, n_gpus, False, True, featur_viz = featur_viz, same_scale_viz = same_scale_viz)

                if s_i == 0:
                    model.first_scale_test = True
                else:
                    model.first_scale_test = False

                #
                trainer.test(model, data)

        print('###################################################')
        print('###################################################')
        print('Now Loading the Data for sending to RDM Corr')

        job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_thomas", model.prj_name)
        # job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/color_cnn_FFhGRU_center_real_hGRU_illusions_one/corr_plots"
        # os.makedirs(job_dir, exist_ok=True)
        file_name = os.path.join(job_dir, "filters_data.pkl")

        open_file = open(file_name, "rb")
        filters_data = pickle.load(open_file)
        print('filters_data : ',filters_data.keys())
        open_file.close()

        stage_list = ['c1', 'c2', 'c2b', 'c3']

        for stage in stage_list:
            for s_i, s_data in enumerate(rdm_thomas_datasets):
                # if s_i == len(rdm_thomas_datasets)-1:
                #     continue
                for c_i in range(1):
                    key_name = stage + '_scale_' + str(s_data) + '_cat_' + str(c_i)
                    temp_data = filters_data[key_name][:8]

                    base_key_name = stage + '_scale_' + str(image_size) + '_cat_' + str(c_i)
                    # base_key_name = stage + '_scale_' + str(rdm_thomas_datasets[s_i+1]) + '_cat_' + str(c_i)
                    base_temp_data = filters_data[base_key_name][:8]

                    print('###################################################')
                    print('key_name : ', key_name, ' : Shape : ', temp_data.shape)
                    print('base_key_name : ', base_key_name, ' : Shape : ', base_temp_data.shape)

                    # rdm_corr_scales_func(scale_base_state_features, scale_state_features, scale, n_scales, scale_datasets, save_dir, c_stage):
                    rdm_corr_scales_func(base_temp_data, temp_data, s_data, len(rdm_thomas_datasets), job_dir, stage)

                    

            print('###################################################')

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

        '''
        Get feature vectors for some k image sizes and s image scales within it after the C stages.
        Get dissimilarity matrices between the k image scales in a pairwise fashion i.e., let's say after C1 stage, the dissimilarity matrix is 
        calculated between the s image scales feature vectors between the 1st and 2nd image size.
        '''

        main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters'
        # C1
        C1_dir = os.path.join(main_dir, 'C1')
        C1_dir = os.path.join(C1_dir, 'prj_' + prj_name)
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

        main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters'
        # C2
        C2_dir = os.path.join(main_dir, 'C2')
        C2_dir = os.path.join(C2_dir, 'prj_' + prj_name)
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
