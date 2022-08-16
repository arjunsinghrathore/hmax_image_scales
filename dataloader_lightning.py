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

seed_everything(42, workers=True)
# Seeds

# import random
# random.seed(1)

# def loader_func(image_path):
#     image_path_mod = "/".join(image_path.split('/')[:-2]) + '/0/3.jpg'

#     img = cv2.imread(image_path_mod)

#     return Image.fromarray(img)

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

      

class dataa_loader(pl.LightningDataModule):
    def __init__(self, image_size, traindir, valdir, testdir, batch_size_per_gpu, n_gpus, test_mode = False, rdm_corr_mode = False):
        super().__init__()
          
        # Directory to load Data
        self.traindir = traindir
        self.valdir = valdir
        self.testdir = testdir
        self.test_mode = test_mode
        self.rdm_corr_mode = rdm_corr_mode

        self.image_size = image_size

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
                transform=
                transforms.Compose([
                    transforms.Resize(self.image_size),

                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]), 
                # loader = loader_func,
                )

            # len_data = len(self.test_data)
            
            # self.test_data, self.throw_data = random_split(self.test_data, [1, len_data-1], generator=torch.Generator().manual_seed(42))
        
        if self.rdm_corr_mode:
            self.test_data = rdm_corr_data(self.testdir, self.image_size)

            len_data = len(self.test_data)
            self.test_data, self.throw_data = random_split(self.test_data, [12, len_data-12], generator=torch.Generator().manual_seed(42))
  
        


    def train_dataloader(self):
        
        # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size, drop_last = True, num_workers = 16, pin_memory=False)
  
    def val_dataloader(self):
        
        # Generating val_dataloader
        return DataLoader(self.val_data,
                          batch_size = self.batch_size, drop_last = True, num_workers = 16, pin_memory=False) #, shuffle = False)
  
    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size = self.batch_size, drop_last = False, num_workers = 1, shuffle = False)

