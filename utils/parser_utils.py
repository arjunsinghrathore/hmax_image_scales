from utils.monitoring_utils import str2bool

##########################################################################################################
############################################ HMAX & Related Models #######################################
##########################################################################################################

def parser_model(parser):
    parser.add_argument("--IP_bool", type=str2bool, const=True, default=False, help="Use Image Pyramid HMAX architecture.")
    parser.add_argument("--IP_2_streams", type=str2bool, const=True, default=False, help="Use Image Pyramid 2 stream HMAX architecture i.e., DuoScale HMAX.")
    parser.add_argument("--hmaxify_bool", type=str2bool, const=True, default=False, help="Use Image Pyramid HMAXify architecture.")
    parser.add_argument("--ENN_bool", type=str2bool, const=True, default=False, help="Use ENN architecture.")
    parser.add_argument("--deepnet_models_bool", type=str2bool, const=True, default=False, help="Use deepnet architectures")

    # IP_full_bool = False
    # capsnet_bool = False
    # IP_capsnet_bool = False
    # IP_contrastive_bool = False
    # IP_contrastive_finetune_bool = False
    # lindeberg_fov_max_bool = False
    # IP_bool_recon = False

    parser.add_argument("--ip_scales", type=int, default=9, help="The number of scale channels in the Image Pyramid")
    parser.add_argument("--scale_factor", type=int, default=4, help="The scale factor for deciding the scales in the Image Pyramid")
    parser.add_argument("--base_image_size", type=int, default=None, help="The base image size of the Image Pyramid")

    parser.add_argument("--s1_scale", type=int, default=21, help="The size of S1 stage kernel sizes")  #25 #23 #21 #19 #17 #15 #13 #11 # 7, #5
    parser.add_argument("--s1_la", type=float, default=11.5, help="S1 stage gabors lambda value") #14.1 #12.7 #11.5 #10.3 #9.1 #7.9 #6.8 #5.6 # 3.5, # 2.5
    parser.add_argument("--s1_si", type=float, default=9.2, help="S1 stage gabors sigma value") #11.3 #10.2 #9.2 #8.2 #7.3 #6.3 #5.4 #4.5 # 2.8, # 2
    parser.add_argument("--s1_stride", type=int, default=4, help="The stride of S1 stage kernels")
    parser.add_argument("--n_ori", type=int, default=16, help="The number of gabor orientations in the S1 stage")
    parser.add_argument("--n_phi", type=int, default=8, help="The number of gabor phases in the S1 stage")
    parser.add_argument("--s1_trainable_filters", type=str2bool, const=True, default=False, help="To set if the S1 stage filters are trainable")

    parser.add_argument("--c1_sp_kernel_sizes", nargs="+", type=int, default = [], help="C1 kernel sizes")
    parser.add_argument("--c1_spatial_sride_factor", type=float, default=0.5, help="Stride factor for C1 kernel sizes")
    parser.add_argument("--c1_use_bool", type=str2bool, const=True, default=False, help="To use or not to use the C1 stage.")
    parser.add_argument("--c_scale_stride", type=int, default=1, help="The stirde for scale max pooling in the C1 stage")
    parser.add_argument("--c_num_scales_pooled", type=int, default=2, help="The number of scales to pool at a time in the C1 stage")

    parser.add_argument("--s2b_channels_out", type=int, default=128, help="The number of channels out for each kernel size in the S2b stage")
    parser.add_argument("--s2b_kernel_size", nargs="+", type=int, default = [4,8,12,16], help="S2b kernel sizes")
    parser.add_argument("--s2b_stride", type=int, default=1, help="S2b kernels stride")

    parser.add_argument("--c2b_scale_loss_bool", type=str2bool, const=True, default=False, help="To use or not to use the scale loss in the C2b stage.")
    parser.add_argument("--c2b_scale_loss_lambda", type=float, default=0, help="Lambda value for C2b Scale Loss")
    parser.add_argument("--c2b_attention_weights_bool", type=str2bool, const=True, default=False, help="To use or not to use the attention method in the C2b stage.")
   

    parser.add_argument("--force_const_size_bool", type=str2bool, const=True, default=False, help="To force constant sized images (base_image_size) in the image pyramid.")
    parser.add_argument("--pad_mode", type=str, default='constant', help="To use constant padding or reflect padding for image pyramid")

    parser.add_argument("--model_backbone", type=str, help="Which model backbone to use while HMAXifying", \
                        choices = ['HMAX', 'AlexNet', 'VGG16_BN', 'ResNet18', 'ResNet50'])

    # MNIST_Scale = None, --> To be decided at runtime
    # category = None, --> To be decided at runtime

    

def parser_dataset(parser):
    parser.add_argument("--linderberg_bool", type=str2bool, const=True, default=False, help="Use Lindeberg's MNIST data.")
    parser.add_argument("--my_data", type=str2bool, const=True, default=False, help="Use MNIST dataset created by me.")
    # parser.add_argument("--all_scales_train_bool", type=str2bool, const=True, default=False, help="Train on all scales of MNIST of my dataset")
    parser.add_argument("--orginal_mnist_bool", type=str2bool, const=True, default=False, help="Use Original MNIST Dataset")
    parser.add_argument("--cifar10_data_bool", type=str2bool, const=True, default=False, help="Use Cifar10 dataset")
    parser.add_argument("--imagenette_data_bool", type=str2bool, const=True, default=False, help="Use Imagenet subset called Imagenette dataset")

    parser.add_argument("--dataset_name", type=str, help="Which dataset to use", \
                        choices = ['MNIST', 'Cifar10', 'Imagenette'])

    parser.add_argument("--image_size", type=int, default=224)

    # num_classes=10, --> Automatize this based on dataset

def parser_experiment(parser)

    # prj_name = None,--> Do this automatically based on the params

    parser.add_argument("--optimizer", type=str, help="Which optimizer to use", \
                        choices = ['SGD', 'Adam'])
    parser.add_argument('--lr_scheduler', choices=['stepLR', 'plateauLR', 'None'], default='stepLR',
                    help='lr_scheduler')
    parser.add_argument('--step_size', default=20, type=int,
                    help='after how many epochs learning rate should be decreased by step_factor')
    parser.add_argument('--step_factor', default=0.1, type=float,
                        help='factor by which to decrease the learning rate')

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size_per_gpu", type=int, default=64)

    parser.add_argument('-od', '--out_dir', type=str, default='/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/new_logs/',
                        metavar='OUT_DIR', help='output directory for model snapshots etc.')

    parser.add_argument("--train_base_scale", type=int, default=2)
    parser.add_argument("--train_scale_aug_range", nargs="+", type=float, default=[2])

    parser.add_argument("--category", type=int, default=None)

    parser.add_argument("--warp_image_bool", type=str2bool, const=True, default=False, help="Warp Images based on DiCarlo Fisheye transformation")
    # parser.add_argument("--scale_data_aug", type=str2bool, const=True, default=False, help="Do scale data augmentation")

    parser.add_argument("--train_mode", type=str2bool, const=True, default=False)
    parser.add_argument("--test_mode", type=str2bool, const=True, default=False)
    parser.add_argument("--val_mode", type=str2bool, const=True, default=False)

    parser.add_argument("--visualize_mode", type=str2bool, const=True, default=False)

    parser.add_argument("--rdm_corr", type=str2bool, const=True, default=False)
    parser.add_argument("--rdm_thomas", type=str2bool, const=True, default=False)
    parser.add_argument("--test_scale_inv", type=str2bool, const=True, default=False)

    parser.add_argument("--feature_viz", type=str2bool, const=True, default=False)
    parser.add_argument("--s1_kernels_viz", type=str2bool, const=True, default=False)
    parser.add_argument("--same_scale_viz", type=str2bool, const=True, default=False)

    parser.add_argument("--save_rdms_list", nargs="+", type=str, default=['c1'], help="If rdm_corr and test_mode then save the RDMs for these stages")
    parser.add_argument("--plt_filters_list", nargs="+", type=str, default=['c1'], help="If feature_viz and test_mode then save the feature map outputs for these stages")
    parser.add_argument("--scale_test_list", nargs="+", type=int, default=[2], help="Scales on which to test for scale invariance")
    # parser.add_argument("--plt_kernels_list", nargs="+", type=str, default=['c1'], help="If kernels_viz and test_mode/val_mode then save the learned kernels for the exactly specified layers")

    parser.add_argument("--visualize_mode", type=str2bool, const=True, default=False)
    parser.add_argument("--argmax_bool", type=str2bool, const=True, default=False)
    parser.add_argument("--orcale_bool", type=str2bool, const=True, default=False)

    parser.add_argument("--oracle_plot_overlap_bool", type=str2bool, const=True, default=False)
    parser.add_argument("--argmax_plot_overlap_bool", type=str2bool, const=True, default=False)
    parser.add_argument("--oracle_argmax_plot_overlap_bool", type=str2bool, const=True, default=False)

# def load_args_conf(args):
#     if args.conf == 'HMAX_mnist':
#     elif args.conf == 'HMAX_multi_5_mnist':
#     elif args.conf == 'HMAX_multi_5_aug_mnist':
#     elif args.conf == 'HMAXify_mnist':
#     elif args.conf == 'HMAXify_multi_5_mnist':
#     elif args.conf == 'HMAXify_multi_5_aug_mnist':