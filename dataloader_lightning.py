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
                my_dataset_scales, test_mode):
        super().__init__()
          
        # Directory to load Data
        self.traindir = traindir
        self.valdir = valdir
        self.testdir = testdir
        self.featur_viz = featur_viz

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

