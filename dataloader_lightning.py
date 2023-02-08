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

from random import sample

import time


import h5py

seed_everything(42, workers=True)
# Seeds

# import random
# random.seed(1)

def loader_func(image_path):


    image_path_mod = "/".join(image_path.split('/')[:-2]) + '/0/3.jpg'
    img = cv2.imread(image_path_mod)   
    img = img[:,:,::-1]
    
    return Image.fromarray(img)

class rdm_corr_data(Dataset):
    def __init__(self, image_dir, image_size):
      
      self.image_dir = image_dir
      self.image_size = image_size

      self.image_paths = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path_mod = os.path.join(self.image_dir, self.image_paths[idx])

        img = cv2.imread(image_path_mod)
        img = img[:,:,::-1]

        img = (img/255)

        # If Image size = 512
        img = cv2.resize(img, (self.image_size, self.image_size))

        img = img.transpose(2,0,1)

        category = int(self.image_dir.split('/')[-1])

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
                self.test_data = rdm_corr_data(self.testdir, self.image_size)

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

class dataa_loader_my(pl.LightningDataModule):
    def __init__(self, image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                my_dataset_scales, test_mode, rdm_corr_mode = False):
        super().__init__()
          
        # Directory to load Data
        self.traindir = traindir
        self.valdir = valdir
        self.testdir = testdir
        self.featur_viz = featur_viz
        self.rdm_corr_mode = rdm_corr_mode

        # self.IP_contrastive_bool = IP_contrastive_bool

        print('traindir : ',traindir)
        print('valdir : ',valdir)
        print('testdir : ',testdir)

        self.test_mode = test_mode

        self.image_size = int(image_size)

        # Batch Size
        self.batch_size = n_gpus*batch_size_per_gpu
          
  
    def setup(self, stage=None):

        
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

        if self.test_mode and not self.rdm_corr_mode:
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
                self.test_data, self.throw_data = random_split(self.test_data, [24, len_data-24], generator=torch.Generator().manual_seed(42))

        if self.rdm_corr_mode:
            self.test_data = rdm_corr_data(self.testdir, self.image_size)

            len_data = len(self.test_data)
            self.test_data, self.throw_data = random_split(self.test_data, [24, len_data-24], generator=torch.Generator().manual_seed(42))
            
  
        


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

class contrastive_data(Dataset):
    def __init__(self, image_dir, image_size, test_mode = False):
      
      self.image_dir = image_dir
      self.image_size = image_size
      self.test_mode = test_mode

      label_paths = os.listdir(self.image_dir)

      image_paths = []
      for l_p in label_paths:
        image_path_temp = os.listdir(os.path.join(self.image_dir, l_p))
        image_path_temp = [os.path.join(self.image_dir + '/' + l_p, ipt) for ipt in image_path_temp]

        image_paths = image_paths + image_path_temp

      self.image_paths = image_paths

      print('self.image_paths : ',len(self.image_paths))

      self.transforms = transforms.Compose([
            # transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # normalize,
        ])

      self.down_up_sizing_num = 2
      self.same_sizing = 2



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        time.sleep(1)

        image_path = self.image_paths[idx]

        # print('image_path : ',image_path)

        img = cv2.imread(image_path)
        img = img[:,:,::-1]

        if not self.test_mode:
            # Random Resizing..... 
            img_aug_list = []
            category_aug_list = []
            for rsz_i in range(self.same_sizing + 2*self.down_up_sizing_num):
                
                # Down-Size Aug
                if rsz_i in [0, self.down_up_sizing_num-1]:
                    rand_scales = [1.618, 2, 2.378, 2.828, 3.364]
                    rand_index = sample(range(5),1)
                    rand_index = rand_index[0]
                    image_rand_scale = rand_scales[rand_index]

                    # # image_rand_scale = np.random.uniform(1.5, 3.75)

                    image_sz = int(28*image_rand_scale)
                    # print('Down-Sizing : ',image_rand_scale,' ::: image_sz : ',image_sz)
                    img_aug_temp = cv2.resize(img, (image_sz, image_sz))
                    # print('img_aug_temp : ',img_aug_temp.shape)
                    img_aug_temp = pad_to_size(img_aug_temp, (self.image_size, self.image_size))
                    # print('img_aug_temp : ',img_aug_temp.shape)

                # Up Sizing
                elif rsz_i in [self.down_up_sizing_num, 2*self.down_up_sizing_num-1]:
                    rand_scales = [4.757, 5.657, 6.727, 8, 9.51]
                    rand_index = sample(range(5), 1)
                    rand_index = rand_index[0]
                    image_rand_scale = rand_scales[rand_index]
                    rand_index = rand_index + 5 # Because this up-sizing....else will overlap with down-sizing
                    
                    # image_rand_scale = np.random.uniform(4.25, 9)

                    image_sz = int(28*image_rand_scale)
                    # print('Up-Sizing : ',image_rand_scale,' ::: image_sz : ',image_sz)
                    img_aug_temp = cv2.resize(img, (image_sz, image_sz))
                    # print('img_aug_temp : ',img_aug_temp.shape)
                    if image_sz <= self.image_size:
                        img_aug_temp = pad_to_size(img_aug_temp, (self.image_size, self.image_size))
                    else:
                        center = img_aug_temp.shape
                        x = center[1]/2 - self.image_size/2
                        y = center[0]/2 - self.image_size/2

                        img_aug_temp = img_aug_temp[int(y):int(y+self.image_size), int(x):int(x+self.image_size)]

                    # print('img_aug_temp : ',img_aug_temp.shape)

                else:
                    img_aug_temp = cv2.resize(img, (self.image_size, self.image_size))

                category_aug_temp = int(image_path.split('/')[-2])

                # Giving Random Label for Negative integers
                if rsz_i in [0, 2*self.down_up_sizing_num-1]:
                    # category_aug_temp = np.random.randint(low = 10, high = 9223372036854775807, dtype=np.int64)
                    category_aug_temp = category_aug_temp + rand_index

                # img_aug_temp = torch.tensor(img_aug_temp.copy(), dtype = torch.float32)

                # print('Before')
                # print('img_aug_temp : ',img_aug_temp.shape)

                img_aug_temp = img_aug_temp[:,:,0]

                # print('img_aug_temp : ',img_aug_temp.shape)
                # print('img_aug_temp max : ',np.max(img_aug_temp))
                # print('img_aug_temp min : ',np.min(img_aug_temp))

                img_aug_temp = self.transforms(Image.fromarray(img_aug_temp, mode="L"))

                # img_aug_temp = img_aug_temp.transpose(2,0,1)

                # print('After')
                # print('img_aug_temp : ',img_aug_temp.shape)
                # print('img_aug_temp max : ',torch.max(img_aug_temp))
                # print('img_aug_temp min : ',torch.min(img_aug_temp))

                img_aug_temp = (img_aug_temp/255)

                img_aug_list.append(img_aug_temp)
                category_aug_list.append(torch.tensor(np.array([category_aug_temp]), dtype = torch.int64))

            img = torch.stack(img_aug_list, dim = 0)
            category = torch.stack(category_aug_list, dim = 0)

        else:

            img_aug_list = []
            category_aug_list = []

            # Image
            image_rand_scale = 2 #np.random.uniform(4.25, 9)
            image_sz = int(28*image_rand_scale)
            # print('Up-Sizing : ',image_rand_scale,' ::: image_sz : ',image_sz)
            img_aug_temp = cv2.resize(img, (image_sz, image_sz))
            # print('img_aug_temp : ',img_aug_temp.shape)
            if image_sz <= self.image_size:
                img_aug_temp = pad_to_size(img_aug_temp, (self.image_size, self.image_size))
            else:
                center = img_aug_temp.shape
                x = center[1]/2 - self.image_size/2
                y = center[0]/2 - self.image_size/2

                img_aug_temp = img_aug_temp[int(y):int(y+self.image_size), int(x):int(x+self.image_size)]

            # Category
            category_aug_temp = int(image_path.split('/')[-2])

            # Appending Stuff
            # img = torch.tensor((img/255).copy(), dtype = torch.float32)
            # img_aug_temp = torch.tensor((img_aug_temp/255).copy(), dtype = torch.float32)

            img_aug_temp = img_aug_temp[:,:,0]
            img = img[:,:,0]
            # img_aug_temp = img_aug_temp.transpose(2,0,1)
            img = self.transforms(Image.fromarray(img, mode="L"))
            img_aug_temp = self.transforms(Image.fromarray(img_aug_temp, mode="L"))
            img_aug_temp = (img_aug_temp/255)
            img = (img/255)

            img_aug_list.append(img)
            img_aug_list.append(img_aug_temp)

            category_aug_temp = torch.tensor(np.array([category_aug_temp]), dtype = torch.int64)
            category_aug_list.append(category_aug_temp)
            category_aug_list.append(category_aug_temp)

            img = torch.stack(img_aug_list, dim = 0)
            category = torch.stack(category_aug_list, dim = 0)

        # print('img : ', img.shape)
        # print('category : ', category.shape)

        return img, category

class dataa_loader_contrastive(pl.LightningDataModule):
    def __init__(self, image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, featur_viz, \
                my_dataset_scales, test_mode):
        super().__init__()
          
        # Directory to load Data
        self.traindir = traindir
        self.valdir = valdir
        self.testdir = testdir
        self.featur_viz = featur_viz

        # self.IP_contrastive_bool = IP_contrastive_bool

        print('traindir : ',traindir)
        print('valdir : ',valdir)
        print('testdir : ',testdir)

        self.test_mode = test_mode

        self.image_size = int(image_size)

        # Batch Size
        self.batch_size = n_gpus*batch_size_per_gpu
          
  
    def setup(self, stage=None):

        self.train_data = contrastive_data(self.traindir, self.image_size)

        self.val_data = contrastive_data(self.valdir, self.image_size)

        if self.test_mode:
            self.test_data = contrastive_data(self.valdir, self.image_size, self.test_mode)
        


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
                          batch_size = self.batch_size, drop_last = True, num_workers = 1, shuffle = False)
