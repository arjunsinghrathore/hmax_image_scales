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
from hmax_models.hmax import * 
import neptune.new as neptune

# run = neptune.init(project = 'Serre-Lab/monkey-ai',
#                          source_files=['*.py'])


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument("--local_rank", type=int)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
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
parser.add_argument('--dir', default='model_ckpt', type=str, help='directory s.')

parser.add_argument('--dataset', default='imagenet', type=str,choices=['imagenet','horizontal','vertical','horizontal_small','mnist_scale'],help='dataset')

parser.add_argument('--model', default='hmax', type=str,choices=['hmax','resnet50','hmax_trainable','deep_hmax','deep_hmax_trainable','invsc'],
                    help='dataset')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

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
    print(ngpus_per_node)
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = args.dir
    os.makedirs(directory,exist_ok=True)
    
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
    # Data loading code
    if args.dataset == 'imagenet':
        traindir = os.path.join("/users/irodri15/data/irodri15/DATA/ILSVRC/Data/CLS-LOC", "train")
        valdir = os.path.join("/users/irodri15/data/irodri15/DATA/ILSVRC/Data/CLS-LOC", "val")
        classes =  1000
    elif args.dataset == 'horizontal':
        traindir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_20px_noise_0.50", "train")
        valdir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_20px_noise_0.50", "test")
        classes = 60
    elif args.dataset == 'horizontal_small':
        traindir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_small", "train")
        valdir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_small", "test")
        classes = 60
    elif args.dataset == 'horizontal_small_1':
        traindir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_small_1", "train")
        valdir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_small_1", "test")
        classes = 60
    elif args.dataset == 'horizontal_small_2':
        traindir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_small_2", "train")
        valdir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_small_2", "test")
        classes = 60
    elif args.dataset == 'horizontal_small_3_2':
        traindir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_small__3_2", "train")
        valdir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_horizontal_50_50_small__3_2", "test")
        classes = 60
    elif args.dataset == 'vertical':
        traindir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_vertical_50_50", "train")
        valdir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/baseline_vertical_50_50", "test")
        classes=60
    elif args.dataset == 'mnist_scale':
        traindir = os.path.join("/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/scale18", "train")
        valdir = "/cifs/data/tserre_lrs/projects/prj_hmax/data/mnist_scale/scale"
        classes=10
    else:
        print('Not implemented Error')
        pass
    n_ori=4
    # # create model
    if args.model == 'deep_hmax' :
        model = DeepHMAX(n_ori=n_ori,number_class=classes)
    elif args.model == 'deep_hmax_trainable':
        model = DeepHMAX(n_ori=n_ori,number_class=classes,trainable_filters=True)
    elif args.model == 'hmax' :
        model = HMAX(n_ori=n_ori,number_class=classes)
    elif args.model == 'hmax_trainable':
        model = HMAX(n_ori=n_ori,number_class=classes,trainable_filters=True)
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=False,num_classes=classes)
    elif args.model == 'vgg16':
        model = models.vgg16(pretrained=False,num_classes=classes)
    elif args.model == 'invsc':
        model = Net()

    print('models will run in the device')
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.to(dev)
            print('1')
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.to(dev)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            model = model.to(dev)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.to(dev)
    else:

        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model.to(dev))

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

    

   
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    if args.dataset =='imagenet':
        print(traindir)
        train_dataset = datasets.ImageFolder(root=
            traindir,
            transform=
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    elif args.dataset=='mnist_scale':
        print(traindir)
        train_dataset = datasets.ImageFolder(root=
            traindir,
            transform=
            transforms.Compose([
                transforms.Resize(256),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                
            ]))
        val_loader = []
        for i in [18,20,24,30,36,16,12,8,4]:
            val_loader.append(torch.utils.data.DataLoader(
                 datasets.ImageFolder(f'{valdir}{i}/test', transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
           
                ])),
                 batch_size=args.batch_size, shuffle=False,
                 num_workers=args.workers, pin_memory=False,drop_last=True))
    else: 
        print(traindir)
        train_dataset = datasets.ImageFolder(root=
            traindir,
            transform=
            transforms.Compose([
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,drop_last=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,drop_last=True)

    

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
        # run["training/test/val_acc1"].log(acc1)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            filename = os.path.join(directory,'model_%s_%s_%d.pth'%(traindir.split('/')[-2],args.model,epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best,filename)


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

    # run['parameters'] = args
    # switch to train mode
    model.train()
    print(next(model.parameters()).device)
   
    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        #print(target)
        #import pdb;pdb.set_trace()
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
        # run["training/batch/loss"].log(loss)

        # Log batch accuracy
        # run["training/batch/acc1"].log(acc1)
        # run["training/batch/acc5"].log(acc5)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    # run["training/epoch/train_acc1"].log(top1.avg)
    # run["training/epoch/train_acc5"].log(top5.avg)

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
    scales = [18,20,24,30,36,16,12,8,4]
    with torch.no_grad():
        end = time.time()
        if type(val_loader)==list: 
            for j,val in enumerate(val_loader):
                print(j)
                print(f'testing scale: {scales[j]}')
                for i, (images, target) in enumerate(val):
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
                    # run[f"training/batch/scale{scales[j]}/val_loss"].log(loss)
                    # run[f"training/batch/scale{scales[j]}/val_acc1"].log(acc1)
                    # run[f"training/batch/scale{scales[j]}/val_acc5"].log(acc5)
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % args.print_freq == 0:
                        progress.display(i)

                # TODO: this should also be done with the ProgressMeter
                print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
                # run[f"training/epoch/scale{scales[j]}/val_acc1"].log(top1.avg)
                # run[f"training/epoch/scale{scales[j]}/val_acc5"].log(top5.avg)
        else: 
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
                # run["training/batch/val_loss"].log(loss)
                # run["training/batch/val_acc1"].log(acc1)
                # run["training/batch/val_acc5"].log(acc5)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
    # run["training/epoch/val_acc1"].log(top1.avg)
    # run["training/epoch/val_acc5"].log(top5.avg)
    return top1.avg


def save_checkpoint(state, is_best, filename='models_clips_02/resnet_checkpoints_043021.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('/'.join(filename.split('/')[:-1]),'model_best.pth.tar'))


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


  
