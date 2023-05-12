import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import torch
# torch.manual_seed(1)
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
import os
import cv2
import numpy as np
# np.random.seed(1)
import random
import scipy as sp
import matplotlib.pyplot as plt
import skimage.color as sic
import pickle
from pytorch_lightning import Trainer, seed_everything
from PIL import Image

from random import sample

from utils.foveation_and_cortical_magn import warp_image

import time


import h5py

# seed_everything(42, workers=True)
# Seeds

# import random
# random.seed(1)

def loader_func(image_path):


    image_path_mod = "/".join(image_path.split('/')[:-2]) + '/0/3.jpg'
    img = cv2.imread(image_path_mod)   
    img = img[:,:,::-1]
    
    return Image.fromarray(img)

class load_mnist_manually(Dataset):
    def __init__(self, image_dir, image_size, warp_image_bool = False):
      
      self.image_dir = image_dir
      self.image_size = image_size
      self.warp_image_bool = warp_image_bool

      if type(self.image_dir) == list:
        self.image_paths = []
        for id_i in range(len(self.image_dir)):
            image_paths_class = os.listdir(self.image_dir[id_i])
            for img_pth in image_paths_class:
                image_paths_ind = os.listdir(self.image_dir[id_i] + '/' + img_pth)
                image_paths_ind = [self.image_dir[id_i] + '/' + img_pth + '/' + img_pth_i for img_pth_i in image_paths_ind]
                self.image_paths = self.image_paths + image_paths_ind
      else:
        self.image_paths = os.listdir(self.image_dir)
        self.image_paths = [self.image_dir + '/' + img_pth for img_pth in self.image_paths]

      print('Len self.image_paths : ', len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # image_path_mod = os.path.join(self.image_dir, self.image_paths[idx])
        image_path_mod = self.image_paths[idx]

        img = cv2.imread(image_path_mod)
        img = img[:,:,::-1]

        if self.warp_image_bool:
            print('log polar img : ',img.shape)
            img = img/255.0

            # img = warp_image(img, output_size=img.shape[0], input_size=None, fill_value=0.)
            def to_log_polar(lin_img, max_radius=32.0):    
                # assert isinstance(img, Image.Image)
                
                dsize = lin_img.shape[:2]
                center = (lin_img.shape[0]//2, lin_img.shape[1]//2)
                flags = cv2.WARP_POLAR_LOG
                out = cv2.warpPolar(
                    lin_img, dsize=dsize, center=center, maxRadius=max_radius, flags=flags + cv2.WARP_FILL_OUTLIERS)

                return out

            
            img = to_log_polar(img, max_radius = img.shape[0])

            # img = img + np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * 0.1 + 0.


            img = img - np.min(img)
            img = (img/np.max(img))*255.0
            img = img.astype(np.uint8)
        
        # zero_mask = img.copy()
        # zero_mask[zero_mask != 0] = -1
        # zero_mask += 1
        # # img = img + np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * 1.5 * zero_mask
        # img = img + np.random.randint(0, 255, (img.shape[0], img.shape[1], img.shape[2])) * 2 * zero_mask
        # # img = np.random.randint(0, 255, (img.shape[0], img.shape[1], img.shape[2]))

        # img = img/255.0
        img = img - np.min(img)
        img = (img/np.max(img))

        # If Image size = 512
        # img = cv2.resize(img, (self.image_size, self.image_size))

        img = img.transpose(2,0,1)

        # zero_mask = img.copy()
        # zero_mask[zero_mask != 0] = -1
        # zero_mask += 1
        # img = img + np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * 1 * zero_mask
        # # img = img + np.random.randint(0, 255, img.shape[0], img.shape[1], img.shape[2]) * 1 * zero_mask

        category = int(image_path_mod.split('/')[-2]) #int(self.image_dir.split('/')[-1])

        return torch.tensor(img.copy(), dtype = torch.float32), torch.tensor(np.array([category]), dtype = torch.int64)

class linderberg_data(Dataset):
    def __init__(self, image_dir, image_size, train_bool = False, test_bool = False):
      
      self.image_dir = image_dir
      self.image_size = image_size

      with h5py.File(self.image_dir, 'r') as f:    
            if train_bool:
                self.x = np.array( f["/x_train"], dtype=np.float32)
                self.y = np.array( f["/y_train"], dtype=np.int32)
            elif not(test_bool):
                self.x = np.array( f["/x_val"], dtype=np.float32)
                self.y = np.array( f["/y_val"], dtype=np.int32)
            else:
                self.x = np.array( f["/x_test"], dtype=np.float32)
                self.y = np.array( f["/y_test"], dtype=np.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        img = self.x[idx]
        # print('img img img : ',img)
        # img = (img/255)
        img = img + np.max(img)
        img = img/np.max(img)

        # If Image size = 512
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.reshape(self.image_size, self.image_size, 1)

        img = img.transpose(2,0,1)

        category = self.y[idx]
        # print('category category category : ',category)

        return torch.tensor(img.copy(), dtype = torch.float32), torch.tensor(np.array([category]), dtype = torch.int64)

      

class dataa_loader(pl.LightningDataModule):
    def __init__(self, image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, test_mode = False, \
                 rdm_corr_mode = False, featur_viz = False, same_scale_viz = False, linderberg_bool = False, \
                 linderberg_dir = None, linderberg_test = False, orginal_mnist_bool = False):
        super().__init__()
          
        # Directory to load Data
        self.traindir = traindir
        self.valdir = valdir
        self.testdir = testdir
        self.test_mode = test_mode
        self.rdm_corr_mode = rdm_corr_mode
        self.featur_viz = featur_viz
        self.same_scale_viz = same_scale_viz

        self.image_size = int(image_size)

        self.linderberg_bool = linderberg_bool
        self.linderberg_dir = linderberg_dir
        self.linderberg_test = linderberg_test

        self.orginal_mnist_bool = orginal_mnist_bool

        # Batch Size
        self.batch_size = n_gpus*batch_size_per_gpu
          
  
    def setup(self, stage=None):

        if self.orginal_mnist_bool:
            trans = transforms.Compose([transforms.ToTensor()])
            self.train_data = datasets.MNIST(root='/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/orignal_mnist/', train=True, download=True, transform=trans)
            len_train_data = len(self.train_data)
            self.train_data, self.val_data = random_split(self.train_data, [int(len_train_data*0.8), int(len_train_data*0.2)], generator=torch.Generator().manual_seed(42))
            self.test_data = datasets.MNIST(root='/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/orignal_mnist/', train=False, download=True, transform=trans)

            print('Orignial MNIST Successfully Loaded')

        elif self.linderberg_bool:
            if self.linderberg_test:
                self.test_data = linderberg_data(self.linderberg_dir, self.image_size, test_bool = True)

                if self.featur_viz:
                    len_data = len(self.test_data)
                    self.test_data, self.throw_data = random_split(self.test_data, [8, len_data-8], generator=torch.Generator().manual_seed(42))
            else:
                self.train_data = linderberg_data(self.linderberg_dir, self.image_size, train_bool = True)
                self.val_data = linderberg_data(self.linderberg_dir, self.image_size, train_bool = False)
        else:
            self.train_data = datasets.ImageFolder(root=
                self.traindir,
                transform=
                transforms.Compose([
                    transforms.Resize(self.image_size),

                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))

            self.val_data = datasets.ImageFolder(root=
                self.valdir,
                transform=
                transforms.Compose([
                    transforms.Resize(self.image_size),

                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))

            if self.test_mode:
                self.test_data = datasets.ImageFolder(root=
                    self.testdir,
                    transform =
                    transforms.Compose([
                        transforms.Resize(self.image_size),

                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]), 
                    # loader = loader_func 
                    )
                
                if self.featur_viz:
                    len_data = len(self.test_data)
                    self.test_data, self.throw_data = random_split(self.test_data, [8, len_data-8], generator=torch.Generator().manual_seed(42))
            
            if self.rdm_corr_mode:
                self.test_data = load_mnist_manually(self.testdir, self.image_size)

                len_data = len(self.test_data)
                self.test_data, self.throw_data = random_split(self.test_data, [8, len_data-8], generator=torch.Generator().manual_seed(42))
  
        


    def train_dataloader(self):
        
        # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size, drop_last = True, num_workers = 8, pin_memory=False, shuffle = True)
  
    def val_dataloader(self):
        
        # Generating val_dataloader
        return DataLoader(self.val_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 8, pin_memory=False, shuffle = True)
  
    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 4, shuffle = False)



############################################################

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def to_log_polar(lin_img, max_radius=32.0):    
    # assert isinstance(img, Image.Image)
    
    dsize = lin_img.shape[:2]
    center = (lin_img.shape[0]//2, lin_img.shape[1]//2)
    flags = cv2.WARP_POLAR_LOG
    out = cv2.warpPolar(
        lin_img, dsize=dsize, center=center, maxRadius=max_radius, flags=flags + cv2.WARP_FILL_OUTLIERS)

    return out


def loader_func_warp(image_path):

    img = cv2.imread(image_path)   
    img = img[:,:,::-1]
    img = img/255.0

    # print('img shape : ', img.shape)
    # img = warp_image(img, output_size=img.shape[0], input_size=None, fill_value=0.)
    img = to_log_polar(img, max_radius = img.shape[0])

    img = img - np.min(img)
    img = (img/np.max(img))*255.0
    img = img.astype(np.uint8)
    
    return Image.fromarray(img)

class dataa_loader_my(pl.LightningDataModule):
    def __init__(self, image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                my_dataset_scales, test_mode, rdm_corr_mode = False, all_scales_train_bool = False, \
                warp_image_bool = False, IP_contrastive_finetune_bool = False):
        super().__init__()
          
        # Directory to load Data
        self.traindir = traindir
        self.valdir = valdir
        self.testdir = testdir
        self.featur_viz = featur_viz
        self.rdm_corr_mode = rdm_corr_mode
        self.all_scales_train_bool = all_scales_train_bool
        self.warp_image_bool = warp_image_bool
        self.IP_contrastive_finetune_bool = IP_contrastive_finetune_bool

        # self.IP_contrastive_bool = IP_contrastive_bool

        self.test_mode = test_mode

        self.image_size = int(image_size)

        # Batch Size
        self.batch_size = n_gpus*batch_size_per_gpu
          
  
    def setup(self, stage=None):

        if self.all_scales_train_bool:
            self.traindir = [
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale500/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale595/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale707/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale841/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1000/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1189/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1414/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1682/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2000/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2378/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2828/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale3364/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale4000/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale4757/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale5657/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale6727/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale8000/train',
                            ]
                            
            self.train_data = load_mnist_manually(self.traindir, self.image_size)

            self.valdir = [
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale500/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale595/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale707/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale841/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1000/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1189/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1414/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1682/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2000/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2378/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2828/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale3364/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale4000/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale4757/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale5657/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale6727/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale8000/val',
                            ]
            self.val_data = load_mnist_manually(self.valdir, self.image_size)
            
        else:
            if self.IP_contrastive_finetune_bool:
                mean = 0.5
                std = 0.5
            else:
                mean = 0.
                std = 1.
            if self.warp_image_bool:
                self.train_data = datasets.ImageFolder(root=
                    self.traindir,
                    transform=
                    transforms.Compose([
                        # transforms.Resize(self.image_size),
                        transforms.CenterCrop(self.image_size),

                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((mean,), (std,)),
                    ]),
                    loader = loader_func_warp)

                self.val_data = datasets.ImageFolder(root=
                self.valdir,
                transform=
                transforms.Compose([
                    # transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),

                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((mean,), (std,)),
                ]),
                loader = loader_func_warp)
            else:
                self.train_data = datasets.ImageFolder(root=
                self.traindir,
                transform=
                transforms.Compose([
                    # transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),

                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((mean,), (std,)),
                ]))

                self.val_data = datasets.ImageFolder(root=
                    self.valdir,
                    transform=
                    transforms.Compose([
                        # transforms.Resize(self.image_size),
                        transforms.CenterCrop(self.image_size),

                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((mean,), (std,)),
                    ]))

        if self.test_mode and not (self.rdm_corr_mode or self.featur_viz):
            if self.warp_image_bool:
                self.test_data = datasets.ImageFolder(root=
                    self.testdir,
                    transform =
                    transforms.Compose([
                        # transforms.Resize(self.image_size),
                        transforms.CenterCrop(self.image_size),

                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((mean,), (std,)),
                        # AddGaussianNoise(0., 1.),
                    ]), 
                    loader = loader_func_warp
                    # loader = loader_func 
                    )
            else:
                self.test_data = datasets.ImageFolder(root=
                    self.testdir,
                    transform =
                    transforms.Compose([
                        # transforms.Resize(self.image_size),
                        transforms.CenterCrop(self.image_size),

                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((mean,), (std,)),
                        # AddGaussianNoise(0., 1.),
                    ]), 
                    # loader = loader_func 
                    )
            
            # if self.featur_viz:
            #     len_data = len(self.test_data)
            #     self.test_data, self.throw_data = random_split(self.test_data, [10, len_data-10], generator=torch.Generator().manual_seed(42))

        if self.rdm_corr_mode or self.featur_viz:
            if self.featur_viz:
                self.test_data = load_mnist_manually([self.testdir], self.image_size, warp_image_bool = self.warp_image_bool)
            else:
                self.test_data = load_mnist_manually(self.testdir, self.image_size, warp_image_bool = self.warp_image_bool)

            len_data = len(self.test_data)
            self.test_data, self.throw_data = random_split(self.test_data, [6, len_data-6], generator=torch.Generator().manual_seed(42))
        
            
  
        print('traindir : ',self.traindir)
        print('valdir : ',self.valdir)
        print('testdir : ',self.testdir)


    def train_dataloader(self):
        
        # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size, drop_last = True, num_workers = 4, pin_memory=False, shuffle = True)
  
    def val_dataloader(self):
        
        # Generating val_dataloader
        return DataLoader(self.val_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 4, pin_memory=False, shuffle = True)
  
    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 1, shuffle = False)



############################################################

def pad_to_size(a, size):

    current_size = (a.shape[0], a.shape[1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = np.pad(a, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
       mode='constant', constant_values=3) 

    return a

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 4% of the image height/width
        self.kernel_size = kernel_size
        if self.kernel_size%2 == 0:
            self.kernel_size += 1 
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)
        # print('sample : ',sample.shape)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

class ContrastiveTransformations:
    def __init__(self, base_transforms, scale_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.scale_transforms = scale_transforms
        self.n_views = n_views

    def __call__(self, x):
        base_transformed = [self.base_transforms(x) for i in range(self.n_views)]
        scale_transformed = [self.scale_transforms(x)]
        return base_transformed + scale_transformed

def warp_func_torch(img):

    img = img.numpy()
    print('img : ',img.shape) 
    img = img[:,:,::-1]
    img = img/255.0

    # print('img shape : ', img.shape)
    # img = warp_image(img, output_size=img.shape[0], input_size=None, fill_value=0.)
    img = to_log_polar(img, max_radius = img.shape[0])

    img = img - np.min(img)
    img = (img/np.max(img))*255.0
    img = img.astype(np.uint8)
    
    return Image.fromarray(img)

class load_mnist_simclr(Dataset):
    def __init__(self, image_dir, image_size, warp_image_bool = False, val_mode = False, contrastive_2_bool = False):
      
      self.image_dir = image_dir
      self.image_size = image_size
      self.warp_image_bool = warp_image_bool
      self.contrastive_2_bool = contrastive_2_bool

      if type(self.image_dir) != list:
        self.image_dir = [self.image_dir]
      
      self.image_paths = []
      for id_i in range(len(self.image_dir)):
        image_paths_class = os.listdir(self.image_dir[id_i])
        for img_pth in image_paths_class:
            image_paths_ind = os.listdir(self.image_dir[id_i] + '/' + img_pth)
            image_paths_ind = [self.image_dir[id_i] + '/' + img_pth + '/' + img_pth_i for img_pth_i in image_paths_ind]
            self.image_paths = self.image_paths + image_paths_ind
    
      print('Len self.image_paths : ', len(self.image_paths))

      if not val_mode:
        self.base_transforms = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.RandomAffine(20, translate=None, scale=None, shear=None),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.8),
                GaussianBlur(kernel_size=int(0.04 * self.image_size), p=0.5),

                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                lambda x: x + 0.05 * torch.randn_like(x[0:1])
            ])

        self.scale_transforms = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.RandomAffine(20, translate=None, scale=None, shear=None),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
                GaussianBlur(kernel_size=int(0.04 * self.image_size), p=0.5),

                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                lambda x: x + 0.05 * torch.randn_like(x[0:1])
            ])

      else:
        # self.base_transforms = transforms.Compose([
        #             transforms.CenterCrop(self.image_size),
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.5,), (0.5,)),
        #         ])
        self.base_transforms = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.RandomAffine(20, translate=None, scale=None, shear=None),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.8),
                GaussianBlur(kernel_size=int(0.04 * self.image_size), p=0.5),

                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                lambda x: x + 0.05 * torch.randn_like(x[0:1])
            ])
            
        # self.scale_transforms = transforms.Compose([
        #             transforms.CenterCrop(self.image_size),
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.5,), (0.5,)),
        #         ])
        self.scale_transforms = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.RandomAffine(20, translate=None, scale=None, shear=None),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
                GaussianBlur(kernel_size=int(0.04 * self.image_size), p=0.5),

                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                lambda x: x + 0.05 * torch.randn_like(x[0:1])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # image_path_mod = os.path.join(self.image_dir, self.image_paths[idx])
        image_path_mod = self.image_paths[idx]

        img = cv2.imread(image_path_mod)
        img = img[:,:,::-1]

        if self.warp_image_bool:
            # print('log polar img : ',img.shape)
            img = img/255.0

            img = warp_image(img, output_size=img.shape[0], input_size=None, fill_value=0.)
            img = img - np.min(img)
            img = (img/np.max(img))*255.0
            img = img.astype(np.uint8)
        

        img = img - np.min(img)
        img = (img/np.max(img))*255.0
        img = img.astype(np.uint8)

        # img = img.transpose(2,0,1)
        pil_img = Image.fromarray(img)

        # img_tensor = torch.tensor(img.copy(), dtype = torch.float32)

        # Anchor
        anchor_tensor = self.base_transforms(pil_img)

        # Positive
        pos_tensor = self.base_transforms(pil_img)

        #############################################################
        # Resize Images
        if self.contrastive_2_bool:
            scale_factor_list = [0.707, 0.841, 1, 1.189, 1.414]
            scale_factor = random.choice(scale_factor_list)
            scale_index = scale_factor_list.index(scale_factor)
            # print('scale_factor : ',scale_factor)
        else:
            # scale_factor = random.uniform(0.6, 1.5)
            scale_factor = random.uniform(0.25, 3)

        rescaled_image_size = int(img.shape[0]*scale_factor)
        resized_img = cv2.resize(img, (rescaled_image_size, rescaled_image_size))
        # print('resized_img : ', resized_img.shape)


        # resized_img = resized_img - np.min(resized_img)
        # resized_img = (resized_img/np.max(resized_img))*255.0
        # resized_img = resized_img.astype(np.uint8)

        pil_resized_img = Image.fromarray(resized_img)

        # Scale Negative
        scale_neg_tensor = self.scale_transforms(pil_resized_img)
        # scale_neg_tensor = self.scale_transforms(pil_img)
        #############################################################

        concat_tensor = torch.stack([anchor_tensor, pos_tensor, scale_neg_tensor], dim = 0)
        # print('concat_tensor : ', concat_tensor.shape)

        # if idx < 10:
        #     job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/MOCO_stuff/MOCO_imgs/"
        #     os.makedirs(job_dir, exist_ok=True)

        #     cv2.imwrite(job_dir + f'anchor_tensor_{idx}.png',anchor_tensor.clone().numpy().transpose(1,2,0)[:,:,::-1]*255.0)
        #     # cv2.imwrite(job_dir + f'pos_tensor_{idx}.png',pos_tensor.clone().numpy().transpose(1,2,0)[:,:,::-1]*255.0)
        #     cv2.imwrite(job_dir + f'scale_neg_tensor_{idx}.png',scale_neg_tensor.clone().numpy().transpose(1,2,0)[:,:,::-1]*255.0)
        #     # cv2.imwrite(job_dir + f'img_{idx}.png',img[:,:,::-1])
        #     cv2.imwrite(job_dir + f'resized_img_{idx}.png',resized_img[:,:,::-1])

        if not self.contrastive_2_bool:
            # if scale_factor >= 1.15 or scale_factor <= 0.85:
            if scale_factor >= 1.5 or scale_factor <= 0.65:
                target = 1.
            else:
                target = 0.
        else:
            target = scale_index
            # print('target : ',target)

        return concat_tensor, torch.tensor(np.array([target]), dtype = torch.float32)

class dataa_loader_simclr(pl.LightningDataModule):
    def __init__(self, image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                my_dataset_scales, test_mode, rdm_corr_mode = False, all_scales_train_bool = False, \
                warp_image_bool = False, contrastive_2_bool = False):
        super().__init__()
          
        # Directory to load Data
        self.traindir = traindir
        self.valdir = valdir
        self.testdir = testdir
        self.featur_viz = featur_viz
        self.rdm_corr_mode = rdm_corr_mode
        self.all_scales_train_bool = all_scales_train_bool
        self.warp_image_bool = warp_image_bool

        self.contrastive_2_bool = contrastive_2_bool

        self.test_mode = test_mode

        self.image_size = int(image_size)

        # Batch Size
        self.batch_size = n_gpus*batch_size_per_gpu
          
  
    def setup(self, stage=None):

        if self.all_scales_train_bool:
            self.traindir = [
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale500/train',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale595/train',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale707/train',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale841/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1000/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1189/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1414/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1682/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2000/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2378/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2828/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale3364/train',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale4000/train',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale4757/train',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale5657/train',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale6727/train',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale8000/train',
                            ]
                            
            self.train_data = load_mnist_manually(self.traindir, self.image_size)

            self.valdir = [
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale500/val',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale595/val',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale707/val',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale841/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1000/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1189/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1414/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale1682/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2000/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2378/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale2828/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale3364/val',
                            '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale4000/val',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale4757/val',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale5657/val',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale6727/val',
                            # '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale8000/val',
                            ]
            self.val_data = load_mnist_manually(self.valdir, self.image_size)
            
        else:
            if self.warp_image_bool:
                self.train_data = datasets.ImageFolder(root=
                    self.traindir,
                    transform=
                    transforms.Compose([
                        transforms.Resize(self.image_size),

                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        lambda x: warp_func_torch(x)
                    ]),
                    loader = loader_func_warp)

                self.val_data = datasets.ImageFolder(root=
                self.valdir,
                transform=
                transforms.Compose([
                    transforms.Resize(self.image_size),

                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                loader = loader_func_warp)
            else:
                # self.train_data = datasets.ImageFolder(root=
                # self.traindir,
                # transform=
                # ContrastiveTransformations(
                # base_transforms = transforms.Compose([
                #     transforms.CenterCrop(self.image_size),
                #     transforms.RandomAffine(30, translate=None, scale=None, shear=0.3),
                #     transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.8),
                #     GaussianBlur(kernel_size=int(0.04 * self.image_size), p=0.5),

                #     transforms.ToTensor(),
                #     transforms.Normalize((0.5,), (0.5,)),
                #     lambda x: x + 0.05 * torch.randn_like(x)
                # ]),
                # scale_transforms = transforms.Compose([
                #     transforms.CenterCrop(self.image_size),
                #     transforms.RandomAffine(30, translate=None, scale=(0.25, 4), shear=0.3),
                #     transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.8),
                #     GaussianBlur(kernel_size=int(0.04 * self.image_size), p=0.5),

                #     transforms.ToTensor(),
                #     transforms.Normalize((0.5,), (0.5,)),
                #     lambda x: x + 0.05 * torch.randn_like(x)
                # ]),
                # n_views=2)
                # )

                # self.val_data = datasets.ImageFolder(root=
                # self.valdir,
                # transform=
                # ContrastiveTransformations(
                # base_transforms = transforms.Compose([
                #     transforms.CenterCrop(self.image_size),
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.5,), (0.5,)),
                # ]),
                # scale_transforms = transforms.Compose([
                #     transforms.CenterCrop(self.image_size),
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.5,), (0.5,)),
                # ]),
                # n_views=2)
                # )

                self.train_data = load_mnist_simclr(self.traindir, self.image_size, warp_image_bool = self.warp_image_bool, contrastive_2_bool = self.contrastive_2_bool)
                self.val_data = load_mnist_simclr(self.valdir, self.image_size, warp_image_bool = self.warp_image_bool, val_mode = True)

        if self.test_mode and not self.rdm_corr_mode:
            if self.warp_image_bool:
                self.test_data = datasets.ImageFolder(root=
                    self.testdir,
                    transform =
                    transforms.Compose([
                        transforms.Resize(self.image_size),

                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # AddGaussianNoise(0., 1.),
                    ]), 
                    loader = loader_func_warp
                    # loader = loader_func 
                    )
            else:
                self.test_data = datasets.ImageFolder(root=
                    self.testdir,
                    transform =
                    transforms.Compose([
                        transforms.Resize(self.image_size),

                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # AddGaussianNoise(0., 1.),
                    ]), 
                    # loader = loader_func 
                    )
            
            if self.featur_viz:
                len_data = len(self.test_data)
                self.test_data, self.throw_data = random_split(self.test_data, [10, len_data-10], generator=torch.Generator().manual_seed(42))

        if self.rdm_corr_mode:
            self.test_data = load_mnist_manually(self.testdir, self.image_size, warp_image_bool = self.warp_image_bool)

            len_data = len(self.test_data)
            self.test_data, self.throw_data = random_split(self.test_data, [8, len_data-8], generator=torch.Generator().manual_seed(42))
        
            
  
        print('traindir : ',self.traindir)
        print('valdir : ',self.valdir)
        print('testdir : ',self.testdir)


    def train_dataloader(self):
        
        # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size, drop_last = True, num_workers = 4, pin_memory=False, shuffle = True)
  
    def val_dataloader(self):
        
        # Generating val_dataloader
        return DataLoader(self.val_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 4, pin_memory=False, shuffle = True)
  
    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 1, shuffle = False)







####################################################################
# CIFAR-10 DataLoader

class dataa_loader_cifar10(pl.LightningDataModule):
    def __init__(self, image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                my_dataset_scales, test_mode, rdm_corr_mode = False, all_scales_train_bool = False, \
                warp_image_bool = False, IP_contrastive_finetune_bool = False):
        super().__init__()
          
        # Directory to load Data
        self.traindir = traindir
        self.valdir = valdir
        self.testdir = testdir
        self.featur_viz = featur_viz
        self.rdm_corr_mode = rdm_corr_mode
        self.all_scales_train_bool = all_scales_train_bool
        self.warp_image_bool = warp_image_bool
        self.IP_contrastive_finetune_bool = IP_contrastive_finetune_bool

        # self.IP_contrastive_bool = IP_contrastive_bool

        self.test_mode = test_mode

        self.image_size = int(image_size)

        # Batch Size
        self.batch_size = n_gpus*batch_size_per_gpu
          
  
    def setup(self, stage=None):

        data_root = "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_dige/database/cifar10"

        transform_train = transforms.Compose(
            [transforms.Pad(8, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine((-15, +15), translate=(0, 0.1), shear=(-15, 15)),
                transforms.RandomCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
                lambda x: x + 0.05 * torch.randn_like(x)
                ]
        )

        transform_test = transforms.Compose(
            [transforms.Resize(self.image_size),
            transforms.Pad((32-self.image_size)//2, padding_mode="reflect"),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ]
        )

        print('Right Place')

        self.train_data = datasets.CIFAR10(root=data_root, transform=transform_train, download=True, train=True)

        self.val_data = datasets.CIFAR10(root=data_root, transform=transform_test, download=True, train=False)

        self.test_data = self.val_data


    def train_dataloader(self):
        
        # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size, drop_last = True, num_workers = 4, pin_memory=False, shuffle = True)
  
    def val_dataloader(self):
        
        # Generating val_dataloader
        return DataLoader(self.val_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 4, pin_memory=False, shuffle = True)
  
    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 1, shuffle = False)

