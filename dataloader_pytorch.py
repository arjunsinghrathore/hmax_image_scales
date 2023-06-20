import torch.nn as nn
from .custom_transform import Binarize, Scale_0_1
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import torch
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
from random import sample

from utils.foveation_and_cortical_magn import warp_image

import time
import h5py

def get_data(args):

    # MNIST
    if args.dataset_name == 'MNIST' and args.my_data:
        args.n_classes = 10
        assert args.image_size == 224, "Value of args.image_size is not 224 for MY MNIST"
        args.base_image_size = args.image_size

        traindir = f'/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale{int(args.train_base_scale*1000)}/train'
        valdir = f'/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale{int(args.train_base_scale*1000)}/val'
        if args.category == None:
            testdir = f'/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale{int(args.train_base_scale*1000)}/test'
        else:
            testdir = f'/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/Like_Lindeberg_smoothning_and_non_linear/scale{int(args.train_base_scale*1000)}/test/' + str(args.category)

        mean = 0.
        std = 1.

        if args.warp_image_bool:
            loader_func = loader_func_warp
        else:
            loader_func = None

        if len(args.train_scale_aug_range) == 1:
            scale_range = 1
        elif len(args.train_scale_aug_range) == 2:
            scale_range = tuple(args.train_scale_aug_range)
        else:
            raise NotImplementedError()

        train_data = datasets.ImageFolder(root=
            traindir,
            transform=
            transforms.Compose([
                # transforms.Resize(self.image_size),
                transforms.RandomAffine(0, translate=None, scale=scale_range, shear=None),
                transforms.CenterCrop(args.image_size),

                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
        ]),
        loader = loader_func)

        val_data = datasets.ImageFolder(root=
            valdir,
            transform=
            transforms.Compose([
                # transforms.Resize(self.image_size),
                transforms.RandomAffine(0, translate=None, scale=scale_range, shear=None),
                transforms.CenterCrop(args.image_size),

                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]),
            loader = loader_func)

        if args.category == None:
            test_data = datasets.ImageFolder(root=
                testdir,
                transform=
                transforms.Compose([
                    # transforms.Resize(self.image_size),
                    transforms.RandomAffine(0, translate=None, scale=None, shear=None),
                    transforms.CenterCrop(args.image_size),

                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((mean,), (std,)),
                ]),
                loader = loader_func)
        else:
            if args.feature_viz:
                # testdir is wrapped in list as it is not single caytegory like rdm_corr
                test_data = load_mnist_manually([testdir], args.image_size, warp_image_bool = args.warp_image_bool)
            else:
                test_data = load_mnist_manually(testdir, args.image_size, warp_image_bool = args.warp_image_bool)

            len_data = len(test_data)
            test_data, throw_data = random_split(test_data, [6, len_data-6], generator=torch.Generator().manual_seed(42))

        print('My MNIST Successfully Loaded')

    elif args.dataset_name == 'MNIST' and args.linderberg_bool:
        args.n_classes = 10
        assert args.image_size == 112, "Value of args.image_size is not 112 for Lindeberg MNIST"
        args.base_image_size = args.image_size

        linderberg_dir = "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr2p000_scte2p000-1.h5"
        linderberg_test_dir = {0.5: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte0p500.h5",
                            0.595: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte0p595.h5",
                            0.707: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte0p707.h5",
                            0.841: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte0p841.h5",

                            1: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte1p000.h5",
                            1.189: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte1p189.h5",
                            1.414: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte1p414.h5",
                            1.682: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte1p682.h5",

                            2: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte2p000.h5",
                            2.378: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte2p378.h5",
                            2.828: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte2p828.h5",
                            3.364: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte3p364.h5",

                            4: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte4p000.h5",
                            4.757: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte4p757.h5",
                            5.657: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte5p657.h5",
                            6.727: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte6p727.h5",
                            8: "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/Linderberg_Data/mnist_large_scale_te10000_outsize112-112_scte8p000.h5"}

        train_data = linderberg_data(linderberg_dir, args.image_size, train_bool = True)
        val_data = linderberg_data(linderberg_dir, args.image_size, train_bool = False)
        test_data = linderberg_data(linderberg_test_dir[args.train_base_scale], args.image_size, test_bool = True)
        if args.feature_viz:
            len_data = len(test_data)
            test_data, throw_data = random_split(test_data, [6, len_data-6], generator=torch.Generator().manual_seed(42))

        print('Lindeberg MNIST Successfully Loaded')

    elif args.dataset_name == 'MNIST' and args.orginal_mnist_bool:
        args.n_classes = 10
        # assert args.image_size == 112, "Value of args.image_size is not 112 for Lindeberg MNIST"
        args.base_image_size = args.image_size

        trans = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(root='/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/orignal_mnist/', train=True, download=True, transform=trans)
        len_train_data = len(train_data)
        train_data, val_data = random_split(train_data, [int(len_train_data*0.8), int(len_train_data*0.2)], generator=torch.Generator().manual_seed(42))
        test_data = datasets.MNIST(root='/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_hmax/data/mnist_scale/arjun_data/orignal_mnist/', train=False, download=True, transform=trans)

        print('Orignial MNIST Successfully Loaded')

    elif args.dataset_name == 'Cifar10' and args.cifar10_data_bool:
        args.n_classes = 10
        assert args.image_size == 32, "Value of args.image_size is not 32 for Cifar10"
        args.base_image_size = args.image_size

        data_root = "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_dige/database/cifar10"

        transform_train = transforms.Compose(
            [transforms.Pad(8, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine((-15, +15), translate=(0, 0.1), shear=(-15, 15)),
                transforms.RandomCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
                lambda x: x + 0.05 * torch.randn_like(x)
                ]
        )

        rescaled_image_size = int(args.image_size * args.train_base_scale) // 2 * 2
        transform_test = transforms.Compose(
            [transforms.CenterCrop(args.image_size),
            transforms.Resize(rescaled_image_size), # To ensure that it is even sized for smooth padding
            # transforms.Pad(max(0, (args.image_size-rescaled_image_size)//2), padding_mode=args.pad_mode),
            # transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ]
        )

        train_data = datasets.CIFAR10(root=data_root, transform=transform_train, download=True, train=True)

        val_data = datasets.CIFAR10(root=data_root, transform=transform_test, download=True, train=False)

        test_data = val_data

        print('Cifar10 Successfully Loaded')

    elif args.dataset_name == 'Imagenette' and args.imagenette_data_bool:

        args.n_classes = 10
        assert args.image_size == 224, "Value of args.image_size is not 224 for Imagenette"
        args.base_image_size = args.image_size

        data_root = "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_shared/dataset/ImageNette"

        transform_train = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomVerticalFlip(),
                        # transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        transforms.CenterCrop(224),
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

        rescaled_image_size = int(args.image_size * args.train_base_scale) // 2 * 2
        transform_test = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Resize(rescaled_image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

        train_data = load_imagenette(data_root + '/train', transform_train = transform_train, transform_test = transform_test)
        val_data = load_imagenette(data_root + '/val', val_bool = True, , transform_train = transform_train, transform_test = transform_test)
        test_data = val_data

    else:
        raise NotImplementedError()


##################################################################################

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
    img = warp_image(img, output_size=img.shape[0], input_size=None, fill_value=0.)
    img = to_log_polar(img, max_radius = img.shape[0])

    img = img - np.min(img)
    img = (img/np.max(img))*255.0
    img = img.astype(np.uint8)
    
    return Image.fromarray(img)

def center_crop(img):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    if width >= height:
        dim = height
    else:
        dim = width
        
    # process crop width and height for max available dimension
    crop_width = dim
    crop_height = dim
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

# MY MNIST
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

        image_path_mod = self.image_paths[idx]

        img = cv2.imread(image_path_mod)
        img = img[:,:,::-1]

        img = cv2.resize(img, (self.image_size, self.image_size))

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

            img = img - np.min(img)
            img = (img/np.max(img))*255.0
            img = img.astype(np.uint8)

        # img = img/255.0
        img = img - np.min(img)
        img = (img/np.max(img))

        img = img.transpose(2,0,1)

        # To add noise
        # zero_mask = img.copy()
        # zero_mask[zero_mask != 0] = -1
        # zero_mask += 1
        # img = img + np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * 1 * zero_mask
        # # img = img + np.random.randint(0, 255, img.shape[0], img.shape[1], img.shape[2]) * 1 * zero_mask

        category = int(image_path_mod.split('/')[-2]) #int(self.image_dir.split('/')[-1])

        return torch.tensor(img.copy(), dtype = torch.float32), torch.tensor(np.array([category]), dtype = torch.int64)

# Lindeberg
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

# ImageNette
class load_imagenette(Dataset):
    def __init__(self, image_dir, val_bool = False, transform_train = None, transform_test = None):
      
        self.image_dir = image_dir

        self.image_dir_folders = os.listdir(self.image_dir)
        self.image_dir_folders = [self.image_dir + '/' + img_pth for img_pth in self.image_dir_folders]
        self.image_paths = []
        for id_i in range(len(self.image_dir_folders)):
            image_paths_class = os.listdir(self.image_dir_folders[id_i])
            image_paths_ind = [self.image_dir_folders[id_i] + '/' + img_pth_i for img_pth_i in image_paths_class]
            self.image_paths = self.image_paths + image_paths_ind

        # print('image paths : ', self.image_paths[:100])
        print('Len self.image_paths : ', len(self.image_paths))

        if not val_bool:
            self.transform = transform_train
        else:
            self.transform = transform_test

        self.cat_to_idx = {'cassetteplayer': 0, 'chainsaw': 1, 'church': 2, 'englishspringer': 3, 'frenchhorn': 4, \
                           'garbagetruck': 5, 'gaspump': 6, 'golfball': 7, 'parachute': 8, 'tench': 9}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # image_path_mod = os.path.join(self.image_dir, self.image_paths[idx])
        image_path_mod = self.image_paths[idx]

        img = cv2.imread(image_path_mod)
        img = img[:,:,::-1]

        img = img - np.min(img)
        img = (img/np.max(img))*255.0
        img = img.astype(np.uint8)

        img = center_crop(img)

        img = Image.fromarray(img)
        img = self.transform(img)

        category = self.cat_to_idx[image_path_mod.split('/')[-2]]

        return img, torch.tensor(np.array([category]), dtype = torch.int64)