import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
torch.manual_seed(1)
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
import os
import cv2
import numpy as np
np.random.seed(1)
import random
import scipy as sp
import matplotlib.pyplot as plt
import skimage.color as sic
import pickle
from pytorch_lightning import Trainer, seed_everything
from PIL import Image

from sklearn.model_selection import train_test_split

import rsatoolbox

seed_everything(42, workers=True)

####################################################################
def calc_rdms_thomas_torch(measurements_1, measurements_2):

    sum_sq_measurements_1 = torch.sum(measurements_1**2, dim=1, keepdim=True)
    sum_sq_measurements_2 = torch.sum(measurements_2**2, dim=1, keepdim=True)
    # Doing it the way ||x2 - x1||^2 = ||x2||^2 + ||x1||^2 - 2 <x2, x1> ----> Same thing goes for the y coordinates
    rdm = sum_sq_measurements_1 + sum_sq_measurements_2.t() - 2 * torch.matmul(measurements_1, measurements_2.t())

    return rdm

'''
Expecting Input to be of the shape --> Bx(C*S)xHxW 
Arguments:
scale_base_state_features --> The features of the reference scale. Shape --> Bx(C*S)xHxW 
scale_state_features --> The features of the other scale. Shape --> Bx(C*S)xHxW 
scale --> scale of the scale_state_features
n_scales --> Total number of scales
scale_datasets
'''
def rdm_corr_scales_func(scale_base_state_features, scale_state_features, scale, n_scales, save_dir, c_stage):

    print('####################################################################')
    print('scale : ',scale)
    print('scale_base_state_features shape : ',scale_base_state_features.shape)
    print('scale_state_features shape : ',scale_state_features.shape)

    # Step 1
    # Rearrange Bx(C*S)xHxW --> But then convert to ScalexBxCxHxW 
    B, C_S, H, W = scale_base_state_features.shape
    if c_stage == 'c1':
        n_scales_base = n_scales-1
    elif c_stage == 'c2' or c_stage == 'c2b':
        n_scales_base = n_scales-2
    elif c_stage == 'c3':
        n_scales_base = n_scales-3

    scale_base_state_features = scale_base_state_features.reshape(B, n_scales_base, int(C_S/n_scales_base), H, W)
    scale_base_state_features = scale_base_state_features.transpose(1,0,2,3,4)

    #
    B, C_S, H, W = scale_state_features.shape
    if c_stage == 'c1':
        n_scales_i = n_scales-1
    elif c_stage == 'c2' or c_stage == 'c2b':
        n_scales_i = n_scales-2
    elif c_stage == 'c3':
        n_scales_i = n_scales-3

    scale_state_features = scale_state_features.reshape(B, n_scales_i, int(C_S/n_scales_i), H, W)
    scale_state_features = scale_state_features.transpose(1,0,2,3,4)

    # Step 2
    # Take mean across CxHxW 
    scale_base_state_features_mean = np.mean(scale_base_state_features, axis = (2,3,4))
    scale_state_features_mean = np.mean(scale_state_features, axis = (2,3,4))

    print('scale_base_state_features_mean shape : ',scale_base_state_features_mean.shape)
    print('scale_state_features_mean shape : ',scale_state_features_mean.shape)

    # Step 3
    # Z Normalize seperately for each category
    scale_base_state_features_z_norm = (scale_base_state_features_mean - np.mean(scale_base_state_features_mean, axis = 1, keepdims = True)) / np.std(scale_base_state_features_mean, axis = 1, keepdims = True)
    scale_state_features_z_norm = (scale_state_features_mean - np.mean(scale_state_features_mean, axis = 1, keepdims = True)) / np.std(scale_state_features_mean, axis = 1, keepdims = True)

    # Preparing data for calculating RDM by converting to torch
    scale_base_state_features_z_norm_torch = torch.tensor(scale_base_state_features_z_norm)
    scale_state_features_z_norm_torch = torch.tensor(scale_state_features_z_norm)
    
    # Step 4
    # Build the RDM Matrices by taking pairwise category euclidean distance for the 2 states
    rdm = calc_rdms_thomas_torch(scale_base_state_features_z_norm_torch, scale_state_features_z_norm_torch)
    rdm_numpy = rdm.numpy()

    # Plotting and Saving
    ############################################
    # Getting the scale channel names
    # For base
    scales_ch_x = [str(int(scale/(160/18))) + '_is' for _ in range(n_scales_i)]
    scales_ch_y = ['18_is' for _ in range(n_scales_base)]

    ############################################
    
    figure = plt.figure()
    axes = figure.add_subplot(111)
    
    # using the matshow() function
    caxes = axes.matshow(rdm_numpy, interpolation ='nearest')
    figure.colorbar(caxes)
    
    axes.set_xticklabels(['']+scales_ch_x)
    axes.set_yticklabels(['']+scales_ch_y)

    # fig, ax, ret_val = rsatoolbox.vis.show_rdm(model_rdms, show_colorbar='figure')

    img_name = 'scale_state_features_rdm_stage_' + c_stage + '_scale_' + str(scale) + '.png'
    save_path = os.path.join(save_dir, img_name)
    figure.savefig(save_path, bbox_inches='tight', dpi=300)




####################################################################
'''
Expecting Input to be of the shape --> CatxBx(C*S)xHxW
And expecting that the input comes out to be for a single category already segregated
'''
def rdm_corr_func(small_state_features, large_state_features, save_dir, c_stage):

    print('small_state_features shape : ',small_state_features.shape)
    print('large_state_features shape : ',large_state_features.shape)

    # Step 2
    # Take mean across CxHxW 
    small_state_features_mean = np.mean(small_state_features, axis = (2,3,4))
    large_state_features_mean = np.mean(large_state_features, axis = (2,3,4))

    print('small_state_features_mean shape : ',small_state_features_mean.shape)
    print('large_state_features_mean shape : ',large_state_features_mean.shape)

    # Step 3
    # Z Normalize seperately for each category
    small_state_features_z_norm = (small_state_features_mean - np.mean(small_state_features_mean, axis = 1, keepdims = True)) / np.std(small_state_features_mean, axis = 1, keepdims = True)
    large_state_features_z_norm = (large_state_features_mean - np.mean(large_state_features_mean, axis = 1, keepdims = True)) / np.std(large_state_features_mean, axis = 1, keepdims = True)

    # Preparing data for calculating RDM
    small_state_features_data = rsatoolbox.data.Dataset(small_state_features_z_norm)
    large_state_features_data = rsatoolbox.data.Dataset(large_state_features_z_norm)

    # Step 4
    # Build the 2 RDM Matrices by taking pairwise category euclidean distance for the 2 states
    # calc_rmd returns a RDMs object
    small_state_features_rdm = rsatoolbox.rdm.calc_rdm(small_state_features_data, method='euclidean', descriptor=None, noise=None)
    large_state_features_rdm = rsatoolbox.rdm.calc_rdm(large_state_features_data, method='euclidean', descriptor=None, noise=None)

    # Plotting and Saving
    # Need to write code for saving

    fig, ax, ret_val = rsatoolbox.vis.show_rdm(small_state_features_rdm, show_colorbar='figure')
    img_name = 'small_state_features_rdm_' + c_stage + '.png'
    save_path = os.path.join(save_dir, img_name)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)

    fig, ax, ret_val = rsatoolbox.vis.show_rdm(large_state_features_rdm, show_colorbar='figure')
    img_name = 'large_state_features_rdm_' + c_stage + '.png'
    save_path = os.path.join(save_dir, img_name)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Step 5
    # Getting the SPearman Rank Correlation for the 2 RDMs
    spearman_corr = rsatoolbox.rdm.compare_spearman(small_state_features_rdm, large_state_features_rdm)

    print('spearman_corr : ', spearman_corr)

    return spearman_corr


# ####################################################################


# '''
# Calculating RDM Correlation for a HMAX Layer:

# State 1 --> Small Scale Images
# State 2 --> Large Scale Images

# 1. Take output of the images of a category from the CNN layer.
# 2. Average the output of the images of a category from the CNN layer such that we are left 
#    with a single value for each image in a category.
# 3. Do z-normalization for the category
# 4. Split the above resulting data into half for the category (namely odd and even).
# 5. Average across images in a category for the two halves, such that we are left with a 
#    single value for each category in the 2 halves
# 6. For each half find the pairwise euclidean distances separately......(this leads to having 
#    a 0 value across the diagonals?) --> Through this we got 2 RDM matrices from the 2 halves, 
#    so a total of 4 RDM Matrices
# 7. We take the off-diagonal values and take it as the category dissimilarity vector

# 8. For getting the raw RDM Correlation (correlated the category dissimilarity vectors across 
#    the two states of a given transformation across the two halves of the data) --> correlating 
#    odd run upper half off-diagonal elements of state 1 with even run lower half off-diagonal 
#    elements of state 2 and vice versa (correlating even run lower half off-diagonal elements 
#    of state 1 with odd run upper half off-diagonal elements of state 2), and then taking 
#    the average of these two correlations.

# 9. For getting the reliability of RDM correlation (by correlating the category dissimilarity 
#    vectors within the same state of a given transformation across the two halves of the 
#    data) --> correlating odd run upper half off-diagonal elements of state 1 with even 
#    run upper half off-diagonal elements of state 1 and correlating odd run lower half 
#    off-diagonal elements of state 2 with even run lower half off-diagonal elements of 
#    state 2, and then taking the average of these two correlations

# 10. The final corrected RDM correlation = Raw RDM correlation / reliability measure
# '''

# '''
# Expecting Input to be of the shape --> CatxBx(C*S)xHxW
# And expecting that the input comes out to be for a single category already segregated
# '''
# # def wrap_rdm(small_state_features, large_state_features):

# #     # Step 2
# #     # Take mean across CxHxW 
# #     small_state_features_mean = np.mean(small_state_features, axis = (1,2,3))
# #     large_state_features_mean = np.mean(large_state_features, axis = (1,2,3))

# #     # Step 3
# #     # Z Normalize seperately for each category
# #     small_state_features_z_norm = (small_state_features_mean - np.mean(small_state_features_mean, axis = 1, keepdims = True)) / torch.std(small_state_features_mean, axis = 1, keepdims = True)
# #     large_state_features_z_norm = (large_state_features_mean - np.mean(large_state_features_mean, axis = 1, keepdims = True)) / torch.std(large_state_features_mean, axis = 1, keepdims = True)

# #     # Step 4
# #     # Spliitng into odd and even halves
# #     len_small = len(small_state_features_z_norm)
# #     if len_small%2 != 0:
# #         small_state_features_z_norm = small_state_features_z_norm[:-1]
# #         len_small = len(small_state_features_z_norm)

# #     len_large = len(large_state_features_z_norm)
# #     if len_large%2 != 0:
# #         large_state_features_z_norm = large_state_features_z_norm[:-1]
# #         len_large = len(large_state_features_z_norm)

# #     small_state_features_half_odd, small_state_features_half_even, large_state_features_half_odd, large_state_features_half_even = train_test_split(
# #                                                                                                                                    small_state_features_z_norm,
# #                                                                                                                                    large_state_features_z_norm,
# #                                                                                                                                    test_size=0.5)

# #     # Step 5
# #     # Take avaerage across categories such that we get one value per category
# #     small_state_features_half_odd_mean = np.mean(small_state_features_half_odd, axis = 1)
# #     small_state_features_half_even_mean = np.mean(small_state_features_half_even, axis = 1)

# #     large_state_features_half_odd_mean = np.mean(large_state_features_half_odd, axis = 1)
# #     large_state_features_half_even_mean = np.mean(large_state_features_half_even, axis = 1)

# #     # Step 6
# #     # Build the 4 RDM Matrices by taking pairwise category euclidean distance for the 2 halves of the 2 states

    
