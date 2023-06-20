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

    # Test Cases
    if args.rdm_corr:
        rdm_corr_func(args)
    if args.test_scale_inv:
        test_scale_inv(args)
    if args.feature_viz:
        feature_viz_func(args)


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

class Test(object):

    def __init__(self, model, data_loader):
        self.name = 'test'
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


#####################################################################################################

def rdm_corr_func(args):

    # Get Dataset (args for num_classes, train_base_scale, category to be set in get_data)
    args, dload_train, dload_valid, dload_test = get_data(args)

    # Get Model
    if args.IP_bool or args.hmaxify_bool:
        model = HMAXify_Models(args)
    elif args.deepnet_models_bool:
        model = DeepNet_Models(args)

    # Paralelizing Model
    model = load_model(model)

    ########################
    prj_name_save = args.model_signature

    # ########################
    # Change Path into own folder
    # Scale 1
    job_dir = os.path.join(args.fig_dir, "rdm_corr")
    os.makedirs(job_dir, exist_ok=True)
    file_name = os.path.join(job_dir, f"filters_data_{args.scale_test_list[0]}.pkl")

    open_file = open(file_name, "wb")
    pickle.dump({'Empty1':0}, open_file)
    open_file.close()

    # Scale 2
    file_name = os.path.join(job_dir, f"filters_data_{args.scale_test_list[1]}.pkl")

    open_file = open(file_name, "wb")
    pickle.dump({'Empty2':0}, open_file)
    open_file.close()

    if args.my_data:
        for s_i, s_data in enumerate(args.scale_test_list):

                print('###################################################')
                print('###################################################')
                print('This is scale : ',s_data)

                args.train_base_scale = s_data
                model.train_base_scale = s_data

                print('\n###################################################')
                print('Non Oracle Version')
                # Calling the dataloader

                for c_i in range(args.n_classes):

                    print('###################################################')
                    print('This is category : ',c_i)

                    args.category = c_i
                    model.category = c_i

                    # Get Dataset (args for num_classes, train_base_scale, category to be set in get_data)
                    args, dload_train, dload_valid, dload_test = get_data(args)

                    # Test Class
                    tester = Test(model, dload_test)
                    
                    records = tester()

    print('###################################################')
    print('###################################################')
    print('Now Loading the Data for sending to RDM Corr')

    # Change Path into own folder
    job_dir = os.path.join(args.fig_dir, "rdm_corr")
    file_name = os.path.join(job_dir, f"filters_data_{args.scale_test_list[0]}.pkl")

    open_file = open(file_name, "rb")
    filters_data_1 = pickle.load(open_file)
    # print('filters_data : ',filters_data.keys())
    open_file.close()

    # Change Path into own folder
    file_name = os.path.join(job_dir, f"filters_data_{args.scale_test_list[1]}.pkl")

    open_file = open(file_name, "rb")
    filters_data_2 = pickle.load(open_file)
    # print('filters_data : ',filters_data.keys())
    open_file.close()

    filters_data = {**filters_data_1, **filters_data_2}
    print('filters_data : ',filters_data.keys())

    
    stage_list = args.save_rdms_list

    spearman_corr_list = []
        for stage in stage_list:
            small_scale = []
            large_scale = []
            for s_i, s_data in enumerate(args.scale_test_list):
                for c_i in range(args.n_classes):
                    if not(args.linderberg_bool or args.my_data):
                        key_name = stage + '_scale_' + str(s_data) + '_cat_' + str(c_i)
                    else:
                        key_name = stage + '_scale_' + str(int(s_data*1000)) + '_cat_' + str(c_i)
                    temp_data = filters_data[key_name][:]

                    print('###################################################')
                    print('Key_name : ', key_name, ' : Shape : ', temp_data.shape)

                    if s_i == 0:
                        small_scale.append(temp_data)
                    else:
                        large_scale.append(temp_data)

            small_scale = np.stack(small_scale, axis=0)
            large_scale = np.stack(large_scale, axis=0)

            print('###################################################')
            print('Going to spearman_corr : ',stage)

            spearman_corr = rdm_corr_func(small_scale, large_scale, job_dir, stage)
            # spearman_corr = direct_corr_func(small_scale, large_scale, job_dir, stage)

            spearman_corr_list.append(spearman_corr)

        # Plot
        fig, ax = plt.subplots(1,1) 
        ax.scatter(range(1,len(spearman_corr_list)+1),spearman_corr_list)

        # Set number of ticks for x-axis
        ax.set_xticks(range(1,len(spearman_corr_list)+1))
        # Set ticks labels for x-axis
        ax.set_xticklabels(stage_list, rotation='vertical', fontsize=18)

        ax.set_xlabel("Pooling Stages", fontweight="bold")
        ax.set_ylabel("RDM Correlation", fontweight="bold")

        ax.set_ylim(0, 1)

        ax.grid()

        # Change Path into own folder
        fig.savefig(os.path.join(job_dir, "rdm_correlation_plot.png"), dpi=199)


def test_scale_inv(args):

    # Get Dataset (args for num_classes, train_base_scale, category to be set in get_data)
    args, dload_train, dload_valid, dload_test = get_data(args)

    # Get Model
    if args.IP_bool or args.hmaxify_bool:
        model = HMAXify_Models(args)
    elif args.deepnet_models_bool:
        model = DeepNet_Models(args)

    # Paralelizing Model
    model = load_model(model)

    scale_accs = {}
    for s_i, s_data in enumerate(args.scale_test_list):

        print('###################################################')
        print('###################################################')
        print('This is scale : ',s_data)

        model.train_base_scale = s_data
        args.train_base_scale = s_data

        # Get Dataset (args for num_classes, train_base_scale, category to be set in get_data)
        args, dload_train, dload_valid, dload_test = get_data(args)

        # Test Class
        tester = Test(model, dload_test)
        
        records = tester()

        scale_accs[s_data] = records['top1']

    # SOrting the dictionary based on scale
    myKeys = list(scale_accs.keys())
    myKeys.sort()
    scale_accs = {i: scale_accs[i] for i in myKeys}

    # Gwetting sorted Scales and Accuracies
    scales_list = list(scale_accs.keys())
    test_accs = list(scale_accs.values())

    # PLotting
    test_accs = np.array(test_accs)

    fig = plt.figure(figsize=(45,30), dpi=250)
    ax = fig.add_subplot(111)

    ax.plot(scales_list, test_accs, c='b', label='Max Over Scales')

    ax.legend(loc='lower right', fontsize=30)
    ax.set_xscale('log')
    ax.set_xticks(list(scales_list))
    ax.set_xticklabels(list(scales_list))
    ax.set_xlabel(
    "Scales",
    fontweight="bold",
    fontsize=40.0)
    ax.set_ylabel(
    "Accuracy",
    fontweight="bold",
    fontsize=40.0,)
    ax.tick_params(axis='both', labelsize=25)

    ax.grid()
    # Change Path into own folder
    fig.savefig(os.path.join(args.fig_dir, "scale_invariance_plot"))

def feature_viz_func(args):

    # Get Dataset (args for num_classes, train_base_scale, category to be set in get_data)
    args, dload_train, dload_valid, dload_test = get_data(args)

    # Get Model
    if args.IP_bool or args.hmaxify_bool:
        model = HMAXify_Models(args)
    elif args.deepnet_models_bool:
        model = DeepNet_Models(args)

    # Paralelizing Model
    model = load_model(model)

    for s_i, s_data in enumerate(args.scale_test_list):

        print('###################################################')
        print('###################################################')
        print('This is scale : ',s_data)

        model.train_base_scale = s_data
        args.train_base_scale = s_data

        # Get Dataset (args for num_classes, train_base_scale, category to be set in get_data)
        args, dload_train, dload_valid, dload_test = get_data(args)

        # Test Class
        tester = Test(model, dload_test)
        
        records = tester()

    for filt in args.plt_filters_list:
        # C1
        out_dir = os.path.join(args.fig_dir, 'visualize_filters/' + filt)
        images_list = os.listdir(out_dir)
        
        filtered_images_list = []
        for il in images_list:
            if il.split('.')[-1] == 'npy':
                filtered_images_list.append(il)

        scales_list = np.array([int(fil.split('_')[0]) for fil in filtered_images_list])
        print('scales_list : ',scales_list)
        index_sort = np.argsort(scales_list)
        sorted_images_list = [filtered_images_list[i_s] for i_s in index_sort]

        sorted_images_list = [os.path.join(out_dir, sil) for sil in sorted_images_list]

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

        out_path = os.path.join(out_dir, "filters_all.png")
        cv2.imwrite(out_path, combined_image)

        plt.figure(figsize = (50, 100))
        plt.imshow(combined_image)
        plt.savefig(out_path.split('.')[0] + '_plt.png')

    

