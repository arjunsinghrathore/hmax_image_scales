import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import datetime
import os

# For HMAX
def make_directories(args):
    # Model
    args.model_signature = args.model_backbone
    if args.IP_bool:
        args.model_signature += '_IP'
    elif args.IP_2_streams:
        args.model_signature += '_IP_2_streams'
    elif args.hmaxify_bool:
        args.model_signature += '_HMAXify'
    elif args.ENN_bool:
        args.model_signature += '_ENN'
    elif args.deepnet_models_bool:
        args.model_signature += '_deepnet_models'

    # Dataset
    args.model_signature += '_' + args.dataset_name
    if args.linderberg_bool:
        args.model_signature += '_linderberg'
    elif args.my_data:
        args.model_signature += '_my_data'
    elif args.orginal_mnist_bool:
        args.model_signature += '_orginal'

    # Date-Time
    data_time = str(datetime.datetime.now())[0:19].replace(' ', '_')
    args.model_signature += '_' + data_time.replace(':', '_')

    # HyperParams
    args.model_signature += '_lr' + str(args.lr)
    args.model_signature += '_wd' + str(args.weight_decay)
    args.model_signature += '_bspg' + str(args.batch_size_per_gpu)

    if not args.deepnet_models_bool:
        args.model_signature += '_ips' + str(args.ip_scales)
        args.model_signature += '_sf' + str(args.scale_factor)
        args.model_signature += '_s1' + str(args.s1_scale)
        args.model_signature += '_stride' + str(args.s1_stride)
        args.model_signature += '_n_ori' + str(args.n_ori)
        args.model_signature += '_n_phi' + str(args.n_phi)
        if args.s1_trainable_filters:
            args.model_signature += '_trainable'
        
        if args.c1_use_bool:
            args.model_signature += '_c1' + str(args.c1_sp_kernel_sizes[0])
            args.model_signature += '_stride' + str(args.c1_spatial_sride_factor)
            args.model_signature += '_cs_stride' + str(args.c_scale_stride)
            args.model_signature += '_cs_num_pool' + str(args.c_num_scales_pooled)
        else:
            args.model_signature += '_no_c1'

        if args.c2b_scale_loss_bool:
            args.model_signature += '_scale_loss' + str(args.c2b_scale_loss_lambda)
        if args.c2b_attention_weights_bool:
            args.model_signature += '_attn_weights'

        if args.force_const_size_bool:
            args.model_signature += '_fcsb_baseImSz' + str(base_image_size) + '_PAD_' + args.pad_mode 

        if args.oracle_bool:
            args.model_signature += '_oracle'
        if args.argmax_bool:
            args.model_signature += '_argmax'
    

    snapshots_path = os.path.join(args.out_dir, args.dataset_name, args.model_backbone)

    args.snap_dir = snapshots_path + '/' + args.model_signature + '/'

    
    os.makedirs(snapshots_path, exist_ok=True)
    os.makedirs(args.snap_dir, exist_ok=True)
    args.fig_dir = args.snap_dir + 'fig/'
    os.makedirs(args.fig_dir, exist_ok=True)
    args.save_model_dir = args.snap_dir + 'save_model/'
    os.makedirs(args.save_model_dir, exist_ok=True)
    
    return args

def argmax_plot_hist(args, overall_max_scale_index):
    #################################
    # Creating histogram
    fig, axs = plt.subplots(1, 1,
                            figsize =(10, 7),
                            tight_layout = True)

    print('overall_max_scale_index len : ',len(overall_max_scale_index))
    
    # axs.hist(self.overall_max_scale_index, bins = 20)
    axs.hist(overall_max_scale_index, bins = list(range(args.ip_scales)))

    # Change Path into own folder
    job_dir = os.path.join(args.fig_dir, 'scale_selection_hist')
    os.makedirs(job_dir, exist_ok=True)
    file_name = os.path.join(job_dir, f'scale_{args.MNIST_Scale}.png')
    fig.savefig(os.path.join(job_dir, file_name))
    #################################

