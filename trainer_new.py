import torch
# torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
# np.random.seed(1)
import cv2
import time
import _pickle as pickle
from torch.utils.data import random_split, DataLoader, dataset
from dataloader_pytorch import get_data
import os
import shutil
import pickle
import random
import argparse

from utils.rdm_corr import rdm_corr_scales_func, rdm_corr_func, direct_corr_func
from utils.parser_utils import parser_dataset, parser_model, parser_experiment

from hmax_models.deepnet_models import DeepNet_Models
from hmax_models.ENN_han_paper import ENN_YH
from hmax_models.hmaxify_models import HMAXify_Models
from hmax_models.hmax_2_streams import HMAX_2_streams

from utils.plot_filters import s1_kernel_viz_func
from utils.monitoring_utils import make_directories, argmax_plot_hist

#####################################################################################################
# Setting args
parser = argparse.ArgumentParser("HMAX Based Models")
parser.add_argument('--device', type=str, default='cuda:0', help='set the cuda device')
parser = parser_dataset(parser)
parser = parser_model(parser)
parser = parser_experiment(parser)
args = parser.parse_args()

# Building the save directories
args = make_directories(args)

#####################################################################################################

if torch.cuda.is_available():
    args.ngpus = torch.cuda.device_count()
    
    print("GPUs detected: = {}".format( args.ngpus ) )
    
    for i in range(args.ngpus):
        print("_______")
        print( torch.cuda.get_device_name( i ) )
        print("_______")

# def set_gpus(n=2):
#     """
#     Finds all GPUs on the system and restricts to n of them that have the most
#     free memory.
#     """
#     if n > 0:
#         gpus = subprocess.run(shlex.split(
#             'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
#             stdout=subprocess.PIPE).stdout
#         gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
#         gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
#         if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
#             visible = [int(i)
#                        for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
#             gpus = gpus[gpus['index'].isin(visible)]
#         gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
#         os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
#         os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
#             [str(i) for i in gpus['index'].iloc[:n]])
#     else:
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# if args.ngpus > 0:
#     set_gpus(args.ngpus)

if args.ngpus > 0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'

#####################################################################################################

if __name__ == '__main__':

    # Get Dataset (args for num_classes, train_base_scale, category to be set in get_data)
    args, dload_train, dload_valid, dload_test = get_data(args)

    # Get Model
    if args.IP_bool or args.hmaxify_bool:
        model = HMAXify_Models(args)
    elif args.deepnet_models_bool:
        model = DeepNet_Models(args)

    # Optimiser, Learning Rate and Loss stuff
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,
                                                weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = [.9, .999], weight_decay=args.weight_decay)
        
    if args.lr_scheduler == 'stepLR':
        lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.step_factor,
                                                    step_size=args.step_size)
    elif args.lr_scheduler == 'plateauLR':
        lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.step_factor,
                                                                patience=args.step_size-1, threshold=0.01)
    else:
        lr_sch = None

    loss = nn.CrossEntropyLoss()

    # Save Configs
    torch.save(args, args.snap_dir + 'param.config')

    if args.train_mode:
        # Training the model
        train(model, dload_train, dload_valid, dload_test, optimizer, lr_sch, loss)
    elif args.val_mode:
        records = Val(model, dload_valid)
        print('Validation Records : ', records)
    elif args.test_mode:
        records = Val(model, dload_test)
        print('Test Records : ', records)


#####################################################################################################

def load_model(model):

    if args.ngpus > 0 and torch.cuda.device_count() > 1:
        print('We have multiple GPUs detected')
        model = model.to(device)
        model = nn.DataParallel(model)
    elif args.ngpus > 0 and torch.cuda.device_count() is 1:
        print('We run on GPU')
        model = model.to(device)
    else:
        print('No GPU detected!')
        model = model.module

    return model

def train(model, data_loader_train, data_loader_val, data_loader_test, optimizer, lr_sch, loss, \
          save_train_epochs=.2,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=1,  # how often save model weights
          save_model_secs=720 * 10  # how often save model (in sec)
          ):

    model = load_model(model)

    trainer = Train(model, data_loader_train, optimizer, lr_sch, loss)
    validator = Val(model, data_loader_val)

    start_epoch = 0
    records = []

    # if args.restore_epoch > 0:
    #     print('Restoring from previous...')
    #     ckpt_data = torch.load(os.path.join(args.restore_path, f'epoch_{args.restore_epoch:02d}.pth.tar'))
    #     start_epoch = ckpt_data['epoch']
    #     print('Loaded epoch: '+str(start_epoch))
    #     model.load_state_dict(ckpt_data['state_dict'])
    #     trainer.optimizer.load_state_dict(ckpt_data['optimizer'])
    #     results_old = pickle.load(open(os.path.join(args.restore_path, 'results.pkl'), 'rb'))
    #     for result in results_old:
    #         records.append(result)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }

    # records = []
    recent_time = time.time()

    nsteps = len(data_loader)

    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, args.num_epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, args.num_epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, args.num_epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    for epoch in tqdm.trange(start_epoch, args.num_epochs + 1, initial=0, desc='epoch'):
        print(epoch)
        data_load_start = np.nan

        data_loader_iter = data_loader

        for step, data in enumerate(tqdm.tqdm(data_loader_iter, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * nsteps + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    if args.lr_scheduler == 'plateauLR' and step == 0:
                        trainer.lr.step(results[validator.name]['loss'])
                    trainer.model.train()
                    print('LR: ', trainer.optimizer.param_groups[0]["lr"])

            if args.save_model_dir is not None:
                if not (os.path.isdir(args.save_model_dir)):
                    os.mkdir(args.save_model_dir)

                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(args.save_model_dir, 'results.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['args'] = args.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(args.save_model_dir,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(args.save_model_dir,
                                                           f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < args.num_epochs:
                frac_epoch = (global_step + 1) / nsteps
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()

class Train(object):

    def __init__(self, model, data_loader, optimizer, lr_sch, loss):
        self.name = 'train'
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        
        self.lr_sch = lr_sch
        
        self.loss = loss

        if args.ngpus > 0:
            self.loss = self.loss.cuda()

    def __call__(self, frac_epoch, inp, target):
        start = time.time()
        if FLAGS.optimizer == 'stepLR':
            self.lr_sch.step(epoch=frac_epoch)
        target = target.to(device)

        if args.ENN_bool or args.deepnet_models_bool:
            output = self.model(inp)
        else:
            output, c2b_maps, max_scale_index, correct_scale_loss = self.model(inp)

        record = {}
        loss = self.loss(output, target)

        if args.c2b_scale_loss_bool:
            loss = loss + (correct_scale_loss*args.c2b_scale_loss_lambda)

        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        # record['learning_rate'] = self.lr.get_lr()[0]
        record['learning_rate'] = self.optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record

class Val(object):

    def __init__(self, model, data_loader):
        if args.test_mode:
            self.name = 'test'
        else:
            self.name = 'val'
        self.model = model
        self.data_loader = data_loader
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)


    def __call__(self):
        self.model.eval()

        if (args.val_mode or args.test_mode) and args.argmax_bool:
            overall_max_scale_index = []

        if args.s1_kernels_viz:
            s1_kernel_viz_func(args, self.model, 'original')
            s1_kernel_viz_func(args, self.model, 'learned')

        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                target = target.to(device)
                
                if args.ENN_bool or args.deepnet_models_bool:
                    output = self.model(inp)
                else:
                    output, c2b_maps, max_scale_index, correct_scale_loss = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

                if (args.val_mode or args.test_mode) and args.argmax_bool:
                    overall_max_scale_index += max_scale_index

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        if (args.val_mode or args.test_mode) and args.argmax_bool:
            argmax_plot_hist(args, overall_max_scale_index)

        return record


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res

