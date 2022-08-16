import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np 
import torch.nn.functional as F
import scipy as sp
import h5py
import numpy as np
from scipy.ndimage.filters import correlate

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--local_rank", type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def get_gabors(l_sizes, l_divs, n_ori, aspect_ratio):
    """generate the gabor filters

    Args
    ----
        l_sizes: type list of floats
            list of gabor sizes
        l_divs: type list of floats
            list of normalization values to be used
        n_ori: type integer
            number of orientations
        aspect_ratio: type float
            gabor aspect ratio

    Returns
    -------
        gabors: type list of nparrays
            gabor filters

    Example
    -------
        aspect_ratio  = 0.3
        l_gabor_sizes = range(7, 39, 2)
        l_divs        = arange(4, 3.2, -0.05)
        n_ori         = 4
        get_gabors(l_gabor_sizes, l_divs, n_ori, aspect_ratio)

    """

    las = np.array(l_sizes)*2/np.array(l_divs)
    sis = las*0.8
    gabors = []

    # TODO: make the gabors as an array for better handling at the gpu level
    for i, scale in enumerate(l_sizes):
        la = las[i] ; si = sis[i]; gs = l_sizes[i]
        #TODO: inverse the axes in the begining so I don't need to do swap them back
        # thetas for all gabor orientations
        th = np.array(range(n_ori))*np.pi/n_ori + np.pi/2.
        th = th[sp.newaxis, sp.newaxis,:]
        hgs = (gs-1)/2.
        yy, xx = sp.mgrid[-hgs: hgs+1, -hgs: hgs+1]
        xx = xx[:,:,sp.newaxis] ; yy = yy[:,:,sp.newaxis]

        x = xx*np.cos(th) - yy*np.sin(th)
        y = xx*np.sin(th) + yy*np.cos(th)

        filt = np.exp(-(x**2 +(aspect_ratio*y)**2)/(2*si**2))*np.cos(2*np.pi*x/la)
        filt[np.sqrt(x**2+y**2) > gs/2.] = 0

        # gabor normalization (following cns hmaxgray package)
        for ori in range(n_ori):
            filt[:,:,ori] -= filt[:,:,ori].mean()
            filt_norm = fastnorm(filt[:,:,ori])
            if filt_norm !=0: filt[:,:,ori] /= filt_norm
        filt_c = np.array(filt, dtype = 'float32').swapaxes(0,2).swapaxes(1,2)
        gabors.append(filt_c)

    return gabors

def fastnorm(in_arr):
   arr_norm = np.dot(in_arr.ravel(), in_arr.ravel()).sum()**(1./2.)

   return arr_norm

aspect_ratio  = 0.3
l_gabor_sizes = range(7, 39, 2)
l_divs = np.arange(4, 3.2, -0.05)
n_ori = 4
gabors = get_gabors(l_gabor_sizes, l_divs, n_ori, aspect_ratio)
gabor7 = gabors[0:1]
gabor7 = torch.Tensor(gabor7*3)
gabor7 = gabor7.view(4, 3, 7, 7)
gabor9 = gabors[1:2]
gabor9 = torch.Tensor(gabor9*3)
gabor9 = gabor9.view(4, 3, 9, 9)
gabor11 = gabors[2:3]
gabor11 = torch.Tensor(gabor11*3)
gabor11 = gabor11.view(4, 3, 11, 11)
gabor13 = gabors[3:4]
gabor13 = torch.Tensor(gabor13*3)
gabor13 = gabor13.view(4, 3, 13, 13)
gabor15 = gabors[4:5]
gabor15 = torch.Tensor(gabor15*3)
gabor15 = gabor15.view(4, 3, 15, 15)
gabor17 = gabors[5:6]
gabor17 = torch.Tensor(gabor17*3)
gabor17 = gabor17.view(4, 3, 17, 17)
gabor19 = gabors[6:7]
gabor19 = torch.Tensor(gabor19*3)
gabor19 = gabor19.view(4, 3, 19, 19)
gabor21 = gabors[7:8]
gabor21 = torch.Tensor(gabor21*3)
gabor21 = gabor21.view(4, 3, 21, 21)
gabor23 = gabors[8:9]
gabor23 = torch.Tensor(gabor23*3)
gabor23 = gabor23.view(4, 3, 23, 23)
gabor25 = gabors[9:10]
gabor25 = torch.Tensor(gabor25*3)
gabor25 = gabor25.view(4, 3, 25, 25)
gabor27 = gabors[10:11]
gabor27 = torch.Tensor(gabor27*3)
gabor27 = gabor27.view(4, 3, 27, 27)
gabor29 = gabors[11:12]
gabor29 = torch.Tensor(gabor29*3)
gabor29 = gabor29.view(4, 3, 29, 29)
gabor31 = gabors[12:13]
gabor31 = torch.Tensor(gabor31*3)
gabor31 = gabor31.view(4, 3, 31, 31)
gabor33 = gabors[13:14]
gabor33 = torch.Tensor(gabor33*3)
gabor33 = gabor33.view(4, 3, 33, 33)
gabor35 = gabors[14:15]
gabor35 = torch.Tensor(gabor35*3)
gabor35 = gabor35.view(4, 3, 35, 35)
gabor37 = gabors[15:16]
gabor37 = torch.Tensor(gabor37*3)
gabor37 = gabor37.view(4, 3, 37, 37)

class DeepHMAX(nn.Module):
    def __init__(self):
        super(DeepHMAX, self).__init__()
        
        self.s1_7 = nn.Conv2d(3, 4, 7)
        self.s1_7.weight = nn.Parameter(gabor7, requires_grad=False)

        self.s1_9 = nn.Conv2d(3, 4, 7)
        self.s1_9.weight = nn.Parameter(gabor9, requires_grad=False)

        self.s1_11 = nn.Conv2d(3, 4, 7)
        self.s1_11.weight = nn.Parameter(gabor11, requires_grad=False)

        self.s1_13 = nn.Conv2d(3, 4, 7)
        self.s1_13.weight = nn.Parameter(gabor13, requires_grad=False)

        self.s1_15 = nn.Conv2d(3, 4, 7) 
        self.s1_15.weight = nn.Parameter(gabor15, requires_grad=False)

        self.s1_17 = nn.Conv2d(3, 4, 7) 
        self.s1_17.weight = nn.Parameter(gabor17, requires_grad=False)

        self.s1_19 = nn.Conv2d(3, 4, 7)
        self.s1_19.weight = nn.Parameter(gabor19, requires_grad=False)

        self.s1_21 = nn.Conv2d(3, 4, 7)
        self.s1_21.weight = nn.Parameter(gabor21, requires_grad=False)
            
        self.s1_23 = nn.Conv2d(3, 4, 7)
        self.s1_23.weight = nn.Parameter(gabor23, requires_grad=False)

        self.s1_25 = nn.Conv2d(3, 4, 7)
        self.s1_25.weight = nn.Parameter(gabor25, requires_grad=False)

        self.s1_27 = nn.Conv2d(3, 4, 7)
        self.s1_27.weight = nn.Parameter(gabor27, requires_grad=False)

        self.s1_29 = nn.Conv2d(3, 4, 7)
        self.s1_29.weight = nn.Parameter(gabor29, requires_grad=False)

        self.s1_31 = nn.Conv2d(3, 4, 7)
        self.s1_31.weight = nn.Parameter(gabor31, requires_grad=False)

        self.s1_33 = nn.Conv2d(3, 4, 7)
        self.s1_33.weight = nn.Parameter(gabor33, requires_grad=False)

        self.s1_35 = nn.Conv2d(3, 4, 7)
        self.s1_35.weight = nn.Parameter(gabor35, requires_grad=False)

        self.s1_37 = nn.Conv2d(3, 4, 7)
        self.s1_37.weight = nn.Parameter(gabor37, requires_grad=False)
        
        self.c1 = nn.Sequential(
            nn.MaxPool2d((3,3), stride=2),
            nn.Conv2d(8, 12, 3),
            nn.BatchNorm2d(12, 1e-3),
            nn.ReLU(True))
        
        self.s2 = nn.Sequential(
            nn.Conv2d(12, 16, 3), # RF = 17 pixels
            nn.BatchNorm2d(16, 1e-3),
            nn.ReLU(True))
        
        self.c2 = nn.Sequential(
	    nn.MaxPool2d((3,3), stride=2),
            nn.Conv2d(32, 48, 4), # RF = 33 pixels 
            nn.BatchNorm2d(48, 1e-3),
            nn.ReLU(True))
        
        self.s3 = nn.Sequential(
            nn.Conv2d(48, 64, 3), # RF = 41 pixels 
            nn.BatchNorm2d(64, 1e-3),
            nn.ReLU(True))
        
        self.s2b = nn.Sequential(
            nn.Conv2d(12, 16, 5, 2), # RF = 21 pixels
            nn.BatchNorm2d(16, 1e-3),
            nn.ReLU(True),
            nn.AvgPool2d(3,2),
            nn.Conv2d(16, 24, 5, 1), # RF = 53 pixels 
            nn.BatchNorm2d(24, 1e-3),
            nn.ReLU(True))
        
        self.c2b = nn.Sequential(
	    nn.MaxPool2d(2,1),
            nn.Conv2d(48, 64, 2, 1), # RF = 69 pixels
            nn.BatchNorm2d(64, 1e-3),
            nn.ReLU(True))
        
        self.c3 = nn.Sequential(
	    nn.MaxPool2d((3,3), stride=2),
            nn.Conv2d(128, 96, 5, 1), # RF = 81 pixels 
            nn.BatchNorm2d(96, 1e-3),
            nn.ReLU(True))
        
        self.s4 = nn.Sequential(
            nn.Conv2d(704, 512, 1, 1),
            nn.BatchNorm2d(512, 1e-3),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 1, 1),
            nn.BatchNorm2d(256, 1e-3),
            nn.ReLU(True),
            nn.MaxPool2d(3,2))
        
        self.norm = nn.Sequential(
            nn.BatchNorm2d(num_features=4, eps=1e-3))
        
        self.x_skip = nn.AdaptiveMaxPool2d(18)
        
        self.fc1 = nn.Sequential(
            nn.Linear(256*8*8, 4096),
            nn.BatchNorm1d(4096, 1e-3),
            nn.ReLU(True))
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096, 1e-3),
            nn.ReLU(True))

        self.fc3 = nn.Linear(4096, 1000)
        self.drop = nn.Dropout(0.5)
 
    def forward(self, x):
        # S1 to C1
        x7 = self.s1_7(x)
        x7 = torch.abs(self.norm(x7))
        x9 = self.s1_7(x)
        x9 = torch.abs(self.norm(x9))
        xa = torch.cat((x7, x9),1)
        xa = self.c1(xa)

        x11 = self.s1_7(x)
        x11 = torch.abs(self.norm(x11))
        x13 = self.s1_7(x)
        x13 = torch.abs(self.norm(x13))
        xb = torch.cat((x11, x13),1)
        xb = self.c1(xb)

        x15 = self.s1_7(x)
        x15 = torch.abs(self.norm(x15))
        x17 = self.s1_7(x)
        x17 = torch.abs(self.norm(x17))
        xc = torch.cat((x15, x17),1)
        xc = self.c1(xc)

        x19 = self.s1_7(x)
        x19 = torch.abs(self.norm(x19))
        x21 = self.s1_7(x)
        x21 = torch.abs(self.norm(x21))
        xd = torch.cat((x19, x21),1)
        xd = self.c1(xd)

        x23 = self.s1_7(x)
        x23 = torch.abs(self.norm(x23))
        x25 = self.s1_7(x)
        x25 = torch.abs(self.norm(x25))
        xe = torch.cat((x21, x23),1)
        xe = self.c1(xe)

        x27 = self.s1_7(x)
        x27 = torch.abs(self.norm(x27))
        x29 = self.s1_7(x)
        x29 = torch.abs(self.norm(x29))
        xf = torch.cat((x27, x29),1)
        xf = self.c1(xf)

        x31 = self.s1_7(x)
        x31 = torch.abs(self.norm(x31))
        x33 = self.s1_7(x)
        x33 = torch.abs(self.norm(x33))
        xg = torch.cat((x31, x33),1)
        xg = self.c1(xg)

        x35 = self.s1_7(x)
        x35 = torch.abs(self.norm(x35))
        x37 = self.s1_7(x)
        x37 = torch.abs(self.norm(x37))
        xh = torch.cat((x35, x37),1)
        xh = self.c1(xh)
        
        # C1 to S2
        xa1 = self.s2(xa)
        xb1 = self.s2(xb)
        xc1 = self.s2(xc)
        xd1 = self.s2(xd)
        xe1 = self.s2(xe)
        xf1 = self.s2(xf)
        xg1 = self.s2(xg)
        xh1 = self.s2(xh)

        # S2 to C2
        xa2 = torch.cat((xa1, xb1),1)
        xa2 = self.c2(xa2)
        xb2 = torch.cat((xc1, xd1),1)
        xb2 = self.c2(xb2)
        xc2 = torch.cat((xe1, xf1),1)
        xc2 = self.c2(xc2)
        xd2 = torch.cat((xg1, xh1),1)
        xd2 = self.c2(xd2)

        # C2 to S3
        xa2 = self.s3(xa2)
        xb2 = self.s3(xb2)
        xc2 = self.s3(xc2)
        xd2 = self.s3(xd2)

        # S3 to C3
        xa3 = torch.cat((xa2, xb2),1)
        xa3 = self.c3(xa3)
        xb3 = torch.cat((xc2, xd2),1)
        xb3 = self.c3(xb3)

        # C1 to S2b 
        xa2b = self.s2b(xa)
        xb2b = self.s2b(xb)
        xc2b = self.s2b(xc)
        xd2b = self.s2b(xd)
        xe2b = self.s2b(xe)
        xf2b = self.s2b(xf)
        xg2b = self.s2b(xg)
        xh2b = self.s2b(xh)

        # S2b to C2b
        xa2b = torch.cat((xa2b, xb2b),1)
        xa2b = self.c2b(xa2b)
        xb2b = torch.cat((xc2b, xd2b),1)
        xb2b = self.c2b(xb2b)
        xc2b = torch.cat((xe2b, xf2b),1)
        xc2b = self.c2b(xc2b)
        xd2b = torch.cat((xg2b, xh2b),1)
        xd2b = self.c2b(xd2b)

        # C2b to S4
        x_cat2 = torch.cat((xa2b, xb2b, xc2b, xd2b), 1)
        x_cat2 = self.x_skip(x_cat2)

        # Skip Connection 
        x_c2 = torch.cat((xa2, xb2, xc2, xd2), 1)
        x_skip = self.x_skip(x_c2)

        # C3, C2b, Skip to S4
        x_cat1 = torch.cat((xa3, xb3),1)
        x_cat1 = self.x_skip(x_cat1)
        x_cat = torch.cat((x_cat1, x_skip, x_cat2), 1)
        x_s4 = self.s4(x_cat)
        
        # Fully Connected Layers
        x_out = x_s4.view(-1, 256*8*8)
        x_out = self.drop(x_out)
        x_out = self.fc1(x_out)
        x_out = self.drop(x_out)
        x_out = self.fc2(x_out)
        x_out = self.fc3(x_out)
        
        return x_out             


best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        print('nprocs'.format(ngpus_per_node))
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = DeepHMAX()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            print('1')
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    traindir = os.path.join("/gpfs/scratch/azerroug/ILSVRC/Data/CLS-LOC", "train")
    valdir = os.path.join("/gpfs/scratch/azerroug/ILSVRC/Data/CLS-LOC", "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='hmax_no_invar_checkpoints_051721.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()


  