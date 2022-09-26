import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torchvision
import cv2
import os
from pytorch_lightning import Trainer, seed_everything
import _pickle as pickle

seed_everything(42, workers=True)

def visualize_map(map):
    map = map.detach().numpy()
    plt.imshow(map)


def get_gabor(l_size, la, si, n_ori, aspect_ratio):
    """generate the gabor filters

    Args
    ----
        l_size: float
            gabor sizes
        la: float
            lambda
        si: float
            sigma
        n_ori: type integer
            number of orientations
        aspect_ratio: type float
            gabor aspect ratio

    Returns
    -------
        gabor: type nparray
            gabor filter

    """

    gs = l_size

    # TODO: inverse the axes in the begining so I don't need to do swap them back
    # thetas for all gabor orientations
    th = np.array(range(n_ori)) * np.pi / n_ori + np.pi / 2.
    th = th[sp.newaxis, sp.newaxis, :]
    hgs = (gs - 1) / 2.
    yy, xx = sp.mgrid[-hgs: hgs + 1, -hgs: hgs + 1]
    xx = xx[:, :, sp.newaxis];
    yy = yy[:, :, sp.newaxis]

    x = xx * np.cos(th) - yy * np.sin(th)
    y = xx * np.sin(th) + yy * np.cos(th)

    filt = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * si ** 2)) * np.cos(2 * np.pi * x / la)
    filt[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0

    # gabor normalization (following cns hmaxgray package)
    for ori in range(n_ori):
        filt[:, :, ori] -= filt[:, :, ori].mean()
        filt_norm = fastnorm(filt[:, :, ori])
        if filt_norm != 0: filt[:, :, ori] /= filt_norm
    filt_c = np.array(filt, dtype='float32').swapaxes(0, 2).swapaxes(1, 2)

    filt_c = torch.Tensor(filt_c)
    filt_c = filt_c.view(n_ori, 1, gs, gs)
    filt_c = filt_c.repeat((1, 3, 1, 1))

    return filt_c


def fastnorm(in_arr):
    arr_norm = np.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)
    return arr_norm

def fastnorm_tensor(in_arr):
    arr_norm = torch.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)
    return arr_norm


def get_sp_kernel_sizes_C(scales, num_scales_pooled, scale_stride):
    '''
    Recursive function to find the right relative kernel sizes for the spatial pooling performed in a C layer.
    The right relative kernel size is the average of the scales that will be pooled. E.g, if scale 7 and 9 will be
    pooled, the kernel size for the spatial pool is 8 x 8

    Parameters
    ----------
    scales
    num_scales_pooled
    scale_stride

    Returns
    -------
    list of sp_kernel_size

    '''

    if len(scales) < num_scales_pooled:
        return []
    else:
        average = int(sum(scales[0:num_scales_pooled]) / len(scales[0:num_scales_pooled]))
        return [average] + get_sp_kernel_sizes_C(scales[scale_stride::], num_scales_pooled, scale_stride)


def pad_to_size(a, size):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom))

    return a


class S1(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, scale, n_ori, padding, trainable_filters, la, si, visualize_mode = False, prj_name = None, MNIST_Scale = None):

        super(S1, self).__init__()

        self.scale = scale
        self.la = la
        self.si = si
        self.visualize_mode = visualize_mode
        self.prj_name = prj_name
        self.MNIST_Scale = MNIST_Scale

        setattr(self, f's_{scale}', nn.Conv2d(3, n_ori, scale, padding=padding))
        s1_cell = getattr(self, f's_{scale}')
        gabor_filter = get_gabor(l_size=scale, la=la, si=si, n_ori=n_ori, aspect_ratio=0.3)  # ??? What is aspect ratio
        s1_cell.weight = nn.Parameter(gabor_filter, requires_grad=trainable_filters)

        # For normalization
        setattr(self, f's_uniform_{scale}', nn.Conv2d(3, n_ori, scale, bias=False))
        s1_uniform = getattr(self, f's_uniform_{scale}')
        nn.init.constant_(s1_uniform.weight, 1)
        for param in s1_uniform.parameters():
            param.requires_grad = False
        self.batchnorm = nn.BatchNorm2d(n_ori, 1e-3)

    def forward(self, x_pyramid, MNIST_Scale = None, batch_idx = None):
        self.MNIST_Scale = MNIST_Scale
        s1_maps = []
        # Loop over scales, normalizing.
        for p_i in range(len(x_pyramid)):

            x = x_pyramid[p_i]

            s1_cell = getattr(self, f's_{self.scale}')
            s1_map = torch.abs(s1_cell(x))  # adding absolute value

            # s1_unorm = getattr(self, f's_uniform_{self.scale}')
            # # s1_unorm = torch.sqrt(s1_unorm(x**2))
            # s1_unorm = torch.sqrt(s1_unorm(x))
            # s1_unorm.data[s1_unorm == 0] = 1  # To avoid divide by zero
            # s1_map /= s1_unorm
            s1_map = self.batchnorm(s1_map)

            s1_maps.append(s1_map)

            # Padding (to get s1_maps in same size)
            # TODO: figure out if we'll ever be in a scenario where we don't need the same padding left/right or top/down
            # Or if the size difference is an odd number
            ori_size = (x.shape[-2], x.shape[-1])
            s1_maps[p_i] = pad_to_size(s1_maps[p_i], ori_size)


            # s1_maps = torch.stack(s1_maps, dim=4)

        # if self.visualize_mode:
        # if True:
        #     combined_image = np.empty((1))
        #     ori_size = (x_pyramid[0][0].shape[-1], x_pyramid[0][0].shape[-1])
        #     s1_maps_reverse = list(reversed(s1_maps))
        #     for s_i, scale_maps in enumerate(s1_maps_reverse):
        #         scale_maps = pad_to_size(scale_maps, ori_size)
        #         scale_maps_clone = scale_maps.clone()
        #         scale_maps_clone = scale_maps_clone[0]

        #         # combined_vertical_image = np.empty((1))
        #         x_pyramid_reverse = list(reversed(x_pyramid))
        #         combined_vertical_image = x_pyramid_reverse[s_i][0]
        #         combined_vertical_image = pad_to_size(combined_vertical_image, ori_size)
        #         combined_vertical_image = combined_vertical_image.cpu().numpy()
        #         # CxHxW --> HxWxC
        #         combined_vertical_image = combined_vertical_image.transpose(1,2,0)
        #         combined_vertical_image = combined_vertical_image[:,:,0:1]*255.0
        #         combined_vertical_image = combined_vertical_image.astype('uint8')
        #         combined_vertical_image = cv2.copyMakeBorder(combined_vertical_image,3,3,3,3,cv2.BORDER_CONSTANT,value=[255,255,255])

        #         for f_i, filter_maps in enumerate(scale_maps_clone):
        #             filter_maps_numpy = filter_maps.cpu().numpy()
        #             # CxHxW --> HxWxC
        #             filter_maps_numpy = filter_maps_numpy.reshape(1, *filter_maps_numpy.shape)
        #             filter_maps_numpy = filter_maps_numpy.transpose(1,2,0)
        #             filter_maps_numpy = filter_maps_numpy*255.0
        #             filter_maps_numpy = filter_maps_numpy.astype('uint8')
        #             filter_maps_numpy = cv2.copyMakeBorder(filter_maps_numpy,3,3,3,3,cv2.BORDER_CONSTANT,value=[255,255,255])

        #             combined_vertical_image = cv2.vconcat([combined_vertical_image, filter_maps_numpy])

        #         if len(combined_image.shape) == 1:
        #             combined_image = combined_vertical_image
        #         else:
        #             # print('combined_image : ',combined_image.shape)
        #             # print('combined_vertical_image : ',combined_vertical_image.shape)
        #             combined_image = cv2.hconcat([combined_image, combined_vertical_image])

        #     main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters/S1'
        #     os.makedirs(main_dir, exist_ok=True)
        #     job_dir = os.path.join(main_dir, "prj_{}".format(self.prj_name))
        #     os.makedirs(job_dir, exist_ok=True)

        #     out_path = os.path.join(job_dir, "filters.png")
        #     cv2.imwrite(out_path, combined_image)

        ##############################################

        # if True and batch_idx == 0:
        #     combined_vertical_image = x_pyramid[0][0]
        #     combined_vertical_image = combined_vertical_image.cpu().numpy()
        #     # CxHxW --> HxWxC
        #     combined_vertical_image = combined_vertical_image.transpose(1,2,0)
        #     combined_vertical_image = combined_vertical_image[:,:,0:1]*255.0
        #     combined_vertical_image = combined_vertical_image.astype('uint8')
        #     combined_vertical_image = cv2.copyMakeBorder(combined_vertical_image,3,3,3,3,cv2.BORDER_CONSTANT,value=[255,255,255])
            
        #     ori_size = (x_pyramid[0][0].shape[-1], x_pyramid[0][0].shape[-1])
            
        #     for s_i, scale_maps in enumerate(s1_maps):
        #         scale_maps = pad_to_size(scale_maps, ori_size)
        #         scale_maps_clone = scale_maps.clone()
        #         scale_maps_clone = scale_maps_clone[0]

        #         for f_i, filter_maps in enumerate(scale_maps_clone):
        #             filter_maps_numpy = filter_maps.cpu().numpy()
        #             # CxHxW --> HxWxC
        #             filter_maps_numpy = filter_maps_numpy.reshape(1, *filter_maps_numpy.shape)
        #             filter_maps_numpy = filter_maps_numpy.transpose(1,2,0)
        #             filter_maps_numpy = filter_maps_numpy*255.0
        #             filter_maps_numpy = filter_maps_numpy.astype('uint8')
        #             filter_maps_numpy = cv2.copyMakeBorder(filter_maps_numpy,3,3,3,3,cv2.BORDER_CONSTANT,value=[255,255,255])

        #             combined_vertical_image = cv2.vconcat([combined_vertical_image, filter_maps_numpy])

        #             break

        #     main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters/S1'
        #     os.makedirs(main_dir, exist_ok=True)
        #     job_dir = os.path.join(main_dir, "prj_{}".format(self.prj_name))
        #     os.makedirs(job_dir, exist_ok=True)

        #     # out_path = os.path.join(job_dir, "{}_filters_temp.png".format(self.MNIST_Scale))
        #     # cv2.imwrite(out_path, combined_image)

        #     out_path = os.path.join(job_dir, "{}_filters_temp.npy".format(self.MNIST_Scale))
        #     np.save(out_path, combined_vertical_image)

        return s1_maps


class C(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, global_pool, sp_kernel_size=[10, 8], sp_stride_factor=None, n_in_sbands=None,
                 num_scales_pooled=2, scale_stride=1,image_subsample_factor=1, visualize_mode = False, \
                 c1_bool = False, prj_name = None, MNIST_Scale = None, c2_bool = False, c3_bool = False, \
                 c2b_bool = False):

        super(C, self).__init__()
        print('c1_bool : ',c1_bool)
        print('c2_bool : ',c2_bool)
        print('c3_bool : ',c3_bool)
        print('c2b_bool : ',c2b_bool)

        self.c1_bool = c1_bool
        self.c2_bool = c2_bool
        self.c3_bool = c3_bool
        self.c2b_bool = c2b_bool
        self.prj_name = prj_name
        self.MNIST_Scale = MNIST_Scale
        self.visualize_mode = visualize_mode
        self.global_pool = global_pool
        self.sp_kernel_size = sp_kernel_size
        self.num_scales_pooled = num_scales_pooled
        self.sp_stride_factor = sp_stride_factor
        self.scale_stride = scale_stride
        self.n_in_sbands = n_in_sbands
        self.n_out_sbands = int(((n_in_sbands - self.num_scales_pooled) / self.scale_stride) + 1)
        self.img_subsample = image_subsample_factor 
        if not self.global_pool:
            if self.sp_stride_factor is None:
                self.sp_stride = [1] * len(self.sp_kernel_size)
            else:
                self.sp_stride = [int(np.ceil(self.sp_stride_factor * kernel_size)) for kernel_size in self.sp_kernel_size]
                # self.sp_stride = [int(np.ceil(0.5 + kernel_size/self.sp_kernel_size[len(self.sp_kernel_size)//2])) for kernel_size in self.sp_kernel_size]

    def forward(self, x_pyramid, x_input = None, MNIST_Scale = None, batch_idx = None, category = None, \
                prj_name = None, same_scale_viz = None, base_scale = None, c1_sp_kernel_sizes = None, c2_sp_kernel_sizes = None):
        # TODO - make this whole section more memory efficient

        # print('####################################################################################')
        # print('####################################################################################')

        # print('C same_scale_viz : ',same_scale_viz)
        # print('C base_scale : ',base_scale)

        # if self.c1_bool and c1_sp_kernel_sizes != None:
        #     self.sp_kernel_size = c1_sp_kernel_sizes
        #     self.sp_stride = [int(np.ceil(self.sp_stride_factor * kernel_size)) for kernel_size in self.sp_kernel_size]
        # elif self.c2_bool and c2_sp_kernel_sizes != None:
        #     self.sp_kernel_size = c2_sp_kernel_sizes
        #     self.sp_stride = [int(np.ceil(self.sp_stride_factor * kernel_size)) for kernel_size in self.sp_kernel_size]


        c_maps = []
        if not self.global_pool:
            if same_scale_viz:
                ori_size = (base_scale, base_scale)
            else:
                ori_size = x_pyramid[0].shape[2:4]

            # print('ori_size : ',ori_size)
            count_p = 0
            for p_i in range(len(x_pyramid)-1):
                # print('############################')
                x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[p_i], self.sp_stride[p_i])
                x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[p_i+1], self.sp_stride[p_i+1])
                #
                # x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[0], self.sp_stride[0])
                # x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[1], self.sp_stride[1])

                # print('x_2 : ',x_2.shape,' :: x1 : ',x_1.shape,' :::::::: ',p_i,' :: kernel_l ',self.sp_kernel_size)

                ########## Linderberg ##################
                x_1 = pad_to_size(x_1, ori_size)
                x_2 = pad_to_size(x_2, ori_size)

                # if x_2.shape[-1] >= x_1.shape[-1]:
                #     # center_crop = torchvision.transforms.CenterCrop(x_1.shape[-1])
                #     # x_2 = center_crop(x_2)

                #     # # # x_2 = F.interpolate(x_2, size=(x_1.shape[-1], x_1.shape[-1]), mode='bilinear')
                    
                #     # x_1 = pad_to_size(x_1, (x_2.shape[-1], x_2.shape[-1]))

                # else:
                #     # center_crop = torchvision.transforms.CenterCrop(x_2.shape[-1])
                #     # x_1 = center_crop(x_1)

                #     # # # x_1 = F.interpolate(x_1, size=(x_2.shape[-1], x_2.shape[-1]), mode='bilinear')

                #     # x_2 = pad_to_size(x_2, (x_1.shape[-1], x_1.shape[-1]))

                
                    
                # assert x_2.shape[-1] >= x_1.shape[-1], "x_2 should have a bigger or equal height width"
                # x_2 = x_2[:,:,:x_1.shape[-2],:x_1.shape[-1]]

                # Maxpool over scale groups
                x = torch.stack([x_1, x_2], dim=4)

                to_append, _ = torch.max(x, dim=4)
                c_maps.append(to_append)

                count_p += 2
        else:
            scale_max = []
            for p_i in range(len(x_pyramid)):
                s_m = F.max_pool2d(x_pyramid[p_i], x_pyramid[p_i].shape[-1], 1)
                scale_max.append(s_m)

            # x = torch.stack(scale_max, dim=4)

            # # Maxpool over scale groups
            # to_append, _ = torch.max(x, dim=4)

            # c_maps.append(to_append)

            for s_i in range(len(scale_max)-1):
                # Maxpool over scale groups
                x = torch.stack([scale_max[s_i], scale_max[s_i+1]], dim=4)

                to_append, _ = torch.max(x, dim=4)
                c_maps.append(to_append)

        # #######################################

        # c_tensor = torch.cat(c_maps, dim = 1)
        # c_numpy = c_tensor.cpu().numpy()

        # job_dir = os.path.join("/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/rdm_thomas", prj_name)
        # print('self.prj_name : ', prj_name)
        # # job_dir = "/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/color_cnn_FFhGRU_center_real_hGRU_illusions_one/corr_plots"
        # # os.makedirs(job_dir, exist_ok=True)
        # file_name = os.path.join(job_dir, "filters_data.pkl")

        # open_file = open(file_name, "rb")
        # filters_data = pickle.load(open_file)
        # print('filters_data : ',filters_data.keys())
        # open_file.close()

        # if self.c1_bool:
        #     key_name = 'c1_scale_' + str(MNIST_Scale) + '_cat_' + str(category)
        #     print('key_name : ',key_name)
        # elif self.c2_bool:
        #     key_name = 'c2_scale_' + str(MNIST_Scale) + '_cat_' + str(category)
        #     print('key_name : ',key_name)
        # elif self.c3_bool:
        #     key_name = 'c3_scale_' + str(MNIST_Scale) + '_cat_' + str(category)
        #     print('key_name : ',key_name)
        # elif self.c2b_bool:
        #     key_name = 'c2b_scale_' + str(MNIST_Scale) + '_cat_' + str(category)
        #     print('key_name : ',key_name)

        # if key_name in filters_data:
        #     filters_data[key_name] = np.concatenate([filters_data[key_name], c_numpy], axis = 0)
        # else:
        #     filters_data[key_name] = c_numpy
        
        # open_file = open(file_name, "wb")
        # pickle.dump(filters_data, open_file)
        # open_file.close()


        # #######################################

        # # print('self.c1_bool : ',self.c1_bool)
        # # print('self.c2_bool : ',self.c2_bool)

        # if (self.c1_bool or self.c2_bool) and batch_idx == 0:
        #     combined_vertical_image = torch.mean(x_input[6][:], dim = 0)
        #     print('pre combined_vertical_image : ',combined_vertical_image.shape)
        #     combined_vertical_image = combined_vertical_image.cpu().numpy()
        #     # CxHxW --> HxWxC
        #     combined_vertical_image = combined_vertical_image.transpose(1,2,0)
        #     combined_vertical_image = combined_vertical_image[:,:,0:1]*255.0
        #     combined_vertical_image = combined_vertical_image.astype('uint8')
        #     combined_vertical_image = cv2.copyMakeBorder(combined_vertical_image,3,3,3,3,cv2.BORDER_CONSTANT,value=[255,255,255])
            
        #     if same_scale_viz:
        #         ori_size = (base_scale, base_scale)
        #     else:
        #         ori_size = (x_input[0][0].shape[-1], x_input[0][0].shape[-1])

        #     # print('ori_size : ',ori_size)

        #     c_maps.reverse()
            
        #     for s_i, scale_maps in enumerate(c_maps):
        #         scale_maps = pad_to_size(scale_maps, ori_size)
        #         scale_maps_clone = scale_maps.clone()
        #         scale_maps_clone = torch.mean(scale_maps_clone[:], dim = 0)


        #         for f_i, filter_maps in enumerate(scale_maps_clone):
        #             filter_maps_numpy = filter_maps.cpu().numpy()
        #             # CxHxW --> HxWxC
        #             filter_maps_numpy = filter_maps_numpy.reshape(1, *filter_maps_numpy.shape)
        #             filter_maps_numpy = filter_maps_numpy.transpose(1,2,0)
        #             filter_maps_numpy = filter_maps_numpy/np.max(filter_maps_numpy)
        #             filter_maps_numpy = filter_maps_numpy*255.0
        #             filter_maps_numpy = filter_maps_numpy.astype('uint8')
        #             filter_maps_numpy = cv2.copyMakeBorder(filter_maps_numpy,3,3,3,3,cv2.BORDER_CONSTANT,value=[255,255,255])

        #             # print('filter_maps_numpy : ',filter_maps_numpy.shape)
        #             # print('combined_vertical_image : ',combined_vertical_image.shape)

        #             combined_vertical_image = cv2.vconcat([combined_vertical_image, filter_maps_numpy])

        #             break

        #     if self.c1_bool:
        #         main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters/C1'
        #     elif self.c2_bool:
        #         main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters/C2'
        #     os.makedirs(main_dir, exist_ok=True)
        #     job_dir = os.path.join(main_dir, "prj_{}".format(prj_name))
        #     os.makedirs(job_dir, exist_ok=True)

        #     # out_path = os.path.join(job_dir, "{}_filters_temp.png".format(self.MNIST_Scale))
        #     # cv2.imwrite(out_path, combined_image)

        #     out_path = os.path.join(job_dir, "{}_filters_temp.npy".format(MNIST_Scale))
        #     # print('out_path : ',out_path)
        #     np.save(out_path, combined_vertical_image)

        return c_maps


class S2(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride):
        super(S2, self).__init__()

        ############## FastNorm Changes ##############
        if type(kernel_size) == int:
            self.kernel_size = [kernel_size]

            setattr(self, f's_0', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size, stride)))

            setattr(self, f's_uniform_0', nn.Conv2d(channels_in, channels_out, kernel_size, stride, bias=False))
            s2b_uniform = getattr(self, f's_uniform_0')
            nn.init.constant_(s2b_uniform.weight, 1)
            for param in s2b_uniform.parameters():
                param.requires_grad = False


        elif type(kernel_size) == list:
            self.kernel_size = kernel_size
            self.kernel_size.sort()
            for i in range(len(kernel_size)):
                
                setattr(self, f's_{i}', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size[i], stride)))

                setattr(self, f's_uniform_{i}', nn.Conv2d(channels_in, channels_out, kernel_size[i], stride, bias=False))
                s2b_uniform = getattr(self, f's_uniform_{i}')
                nn.init.constant_(s2b_uniform.weight, 1)
                for param in s2b_uniform.parameters():
                    param.requires_grad = False

        self.batchnorm = nn.BatchNorm2d(channels_out, 1e-3)


    def forward(self, x_pyramid):
        # Evaluate input
        

        # Convolve each kernel with each scale band
        # TODO: think of something more elegant than for loop
        s_maps_per_k = []
        for k in range(len(self.kernel_size)):
            s_maps_per_i = []
            layer = getattr(self, f's_{k}')

            # if len(self.kernel_size) > 1:
            #     s2b_unorm_norm = getattr(self, f's_uniform_{k}')

            for i in range(len(x_pyramid)):  # assuming S is last dimension

                x = x_pyramid[i]

                ############## FastNorm Changes ##############
                # ############################################
                # if True: #k != 0:
                #     # s_map = s_map / ((self.kernel_size[k]/self.kernel_size[0])**2)
                #     for p_i in range(x.shape[1]):
                #         # filter_norm = fastnorm_tensor(layer.weight[:,p_i].data)
                #         filter_norm = (self.kernel_size[k]/self.kernel_size[0])**2
                #         layer.weight[:,p_i].data = layer.weight[:,p_i].data/filter_norm
                # ############################################

                s_map = layer(x)

                # ############################################
                if k != 0:
                    # s_map = s_map / ((self.kernel_size[k]/self.kernel_size[0]))       # Sqrt
                    s_map = s_map / ((self.kernel_size[k]/self.kernel_size[0])**2)  # Square (Default)
                    # s_map = s_map / ((self.kernel_size[k]/self.kernel_size[0])**3)  # Cube

                # if len(self.kernel_size) > 1:
                #     s2b_unorm = torch.sqrt(abs(s2b_unorm_norm(x**2)))
                #     # s2b_unorm = torch.sqrt(s2b_unorm(x))
                #     s2b_unorm.data[s2b_unorm == 0] = 1  # To avoid divide by zero
                #     s_map = s_map / s2b_unorm
                ############################################

                # s_map = self.batchnorm(s_map)

                # TODO: think about whether the resolution gets too small here
                ori_size = x.shape[2:4]
                s_map = pad_to_size(s_map, (ori_size[0], ori_size[1]))
                s_maps_per_i.append(s_map)

            s_maps_per_k.append(s_maps_per_i)

        if len(s_maps_per_k) == 1:
            s_maps = s_maps_per_k[0]
        else:
            s_maps = []
            for i in range(len(x_pyramid)):
                k_list = [s_maps_per_k[j][i] for j in range(len(s_maps_per_k))]
                temp_maps = torch.cat(k_list, dim=1)
                s_maps.append(temp_maps)

        return s_maps


class S3(S2):
    # S3 does the same thing as S2
    pass


class HMAX_IP(nn.Module):
    def __init__(self,
                 ip_scales = 10,
                 s1_scale=11, #11 # 7,
                 s1_la=5.6, #5.6 # 3.5,
                 s1_si=4.5, #4.5 # 2.8,
                 n_ori=4,
                 num_classes=1000,
                 s1_trainable_filters=False,
                 visualize_mode = False,
                 prj_name = None,
                 MNIST_Scale = None,
                 category = None
                 ):
        super(HMAX_IP, self).__init__()

        # A few settings
        self.ip_scales = ip_scales
        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.scale = 5

        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled

        self.same_scale_viz = None
        self.base_scale = None
        

        ########################################################
        ########################################################
        # For scales
        base_filt_size = 20 #17
        scale = self.scale
        self.c1_sp_kernel_sizes = [int(np.ceil(base_filt_size/(2**(i/scale)))) for i in range(ip_scales)]
        # self.c1_sp_kernel_sizes = [10, 8]
        # #######################################################


        self.c2_scale_stride = self.c_scale_stride
        self.c2_num_scales_pooled = self.c_num_scales_pooled

        # For 14 scales
        base_filt_size = self.c1_sp_kernel_sizes[1]
        scale = self.scale
        self.c2_sp_kernel_sizes = [int(np.ceil(base_filt_size/(2**(i/scale)))) for i in range(ip_scales)]
        # self.c2_sp_kernel_sizes = [8, 6]
        ########################################################
        ########################################################

        ########################################################
        # self.c1_sp_kernel_sizes = get_sp_kernel_sizes_C(range(7, 37, 2), self.c_scale_stride, self.c_scale_stride)

        # self.c2_scale_stride = self.c_scale_stride
        # self.c2_num_scales_pooled = self.c_num_scales_pooled
        # self.c2_sp_kernel_sizes = get_sp_kernel_sizes_C(self.c1_sp_kernel_sizes, self.c_num_scales_pooled, self.c2_scale_stride)
        ########################################################

        # Reverese
        # self.c1_sp_kernel_sizes.reverse()
        # self.c2_sp_kernel_sizes.reverse()

        print('c1_sp_kernel_sizes : ',self.c1_sp_kernel_sizes)
        print('c2_sp_kernel_sizes : ',self.c2_sp_kernel_sizes)
    

        self.c2b_scale_stride = self.c_scale_stride
        self.c2b_num_scales_pooled = len(self.c1_sp_kernel_sizes)  # all of them
        # no kernel sizes here because global pool (spatially)

        self.c3_scale_stride = self.c_scale_stride
        self.c3_num_scales_pooled = len(self.c2_sp_kernel_sizes)  # all of them
        # no kernel sizes here because global pool (spatially)

        ###############################################
        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters=s1_trainable_filters,
                     la=self.s1_la, si=self.s1_si, visualize_mode = visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.2, n_in_sbands=ip_scales,
                    num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride, visualize_mode = visualize_mode, \
                    c1_bool = True, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.s2 = S2(channels_in=n_ori, channels_out=100, kernel_size=3, stride=1)
        self.c2 = C(global_pool = False, sp_kernel_size=self.c2_sp_kernel_sizes, sp_stride_factor=0.2, n_in_sbands=ip_scales-1,
                    num_scales_pooled=self.c2_num_scales_pooled, scale_stride=self.c2_scale_stride, c2_bool = True, prj_name = self.prj_name)
        self.s2b = S2(channels_in=n_ori, channels_out=100, kernel_size=[4, 8, 12, 16], stride=1)
        self.s3 = S3(channels_in=100, channels_out=100, kernel_size=3, stride=1)
        self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)
        self.c3 = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-2,
                    num_scales_pooled=self.c3_num_scales_pooled, scale_stride=self.c3_scale_stride, c3_bool = True, prj_name = self.prj_name)


        ###############################################
        self.pool = nn.AdaptiveMaxPool2d(18)  # not in table, but we need to get everything in the same shape before s4

        self.s4 = nn.Sequential(nn.Conv2d(self.get_s4_in_channels(), 512, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(512, 1e-3),
                                nn.ReLU(True),
                                nn.Conv2d(512, 256, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(256, 1e-3),
                                nn.ReLU(True),
                                nn.MaxPool2d(3, 2))

        ###############################################
        # Classifier
        # self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
        #                                 nn.Linear(256 * 8 * 8, 4096),  # fc1
        #                                 nn.BatchNorm1d(4096, 1e-3),
        #                                 nn.ReLU(True),
        #                                 nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
        #                                 nn.Linear(4096, 4096),  # fc2
        #                                 nn.BatchNorm1d(4096, 1e-3),
        #                                 nn.ReLU(True),
        #                                 nn.Linear(4096, num_classes)  # fc3
        #                                 )

        # # Classifier
        self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        nn.Linear(256 * 8 * 8, 256),  # fc1
                                        # nn.BatchNorm1d(4096, 1e-3),
                                        # nn.ReLU(True),
                                        # nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        # nn.Linear(4096, 4096),  # fc2
                                        # nn.BatchNorm1d(256, 1e-3),
                                        # nn.ReLU(True),
                                        nn.Linear(256, num_classes)  # fc3
                                        )

        # self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
        #                                 nn.BatchNorm1d(256, 1e-3),
        #                                 nn.Linear(256, num_classes),  # fc1
        #                                 )

    def get_s4_in_channels(self):
        # FOr linderberg
        ip_scales = int(np.ceil(self.ip_scales/2))*2 -1

        # For Mutch
        # ip_scales = self.ip_scales

        #########################################
        c1_out = (ip_scales-1) * self.n_ori
        c2_out = (ip_scales-2) * self.s2.s_0[0].weight.shape[0]
        c3_out = (ip_scales-3) * self.s3.s_0[0].weight.shape[0]
        c2b_out = len(self.s2b.kernel_size) * (ip_scales-2) * self.s2b.s_0[0].weight.shape[0]
        # c3_out = self.s3.s_0[0].weight.shape[0]
        # c2b_out = len(self.s2b.kernel_size) * self.s2b.s_0[0].weight.shape[0]
        s4_in = c1_out + c2_out + c3_out + c2b_out

        ############## FastNorm Changes ##############
        # s4_in = 2924 # For 7 scales
        # s4_in = 4132 # For 10 scales
        # s4_in = 6548 # For 14 scales

        return s4_in

    # def make_ip(self, x, same_scale_viz = None, base_scale = None):

    #     base_image_size = int(x.shape[-1])

    #     scale = self.scale
    #     # image_scales = [np.ceil(base_image_size/(2**(i/3))) if np.ceil(base_image_size/(2**(i/3)))%2 == 0 else np.floor(base_image_size/(2**(i/3))) for i in range(self.ip_scales)]
    #     image_scales = [np.ceil(base_image_size/(2**(i/scale))) for i in range(self.ip_scales)]
    #     # image_scales = [np.ceil(base_image_size) if np.ceil(base_image_size)%2 == 0 else np.floor(base_image_size) for i in range(self.ip_scales)]

    #     self.image_scales = image_scales

    #     if same_scale_viz:
    #         base_image_size = base_scale
    #     else:
    #         base_image_size = int(x.shape[-1])

    #     image_pyramid = []
    #     for i_s in image_scales:
    #         i_s = int(i_s)
    #         interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bicubic').clamp(min=0, max=1) # ??? Wgats is the range?
    #         pad_input = pad_to_size(interpolated_img, (base_image_size, base_image_size))
    #         image_pyramid.append(pad_input) # ??? Wgats is the range?
        
    #     return image_pyramid

    def make_ip(self, x, same_scale_viz = None, base_scale = None):

        # print('make_ip same_scale_viz : ',same_scale_viz)
        # print('make_ip base_scale : ',base_scale)

        base_image_size = int(x.shape[-1]) 

        # print('base_image_size : ',base_image_size)
        scale = self.scale #5
        image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) if np.ceil(base_image_size/(2**(i/scale)))%2 == 0 else np.floor(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(self.ip_scales/2)))]
        image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) if np.ceil(base_image_size*(2**(i/scale)))%2 == 0 else np.floor(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(self.ip_scales/2)))]
        # image_scales_down = [base_image_size/(2**(i/scale)) for i in range(int(np.ceil(self.ip_scales/2)))]
        # image_scales_up = [base_image_size*(2**(i/scale)) for i in range(1, int(np.ceil(self.ip_scales/2)))]
        

        image_scales = image_scales_down + image_scales_up
        # self.image_scales = sorted(image_scales)
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]

        # print('image_scales image_scales : ',self.image_scales)

        if same_scale_viz:
            base_image_size = base_scale
        else:
            base_image_size = int(x.shape[-1]) 

        # print('base_image_size : ',base_image_size)

        image_pyramid = []
        for i_s in self.image_scales:
            i_s = int(i_s)
            interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bicubic').clamp(min=0, max=1)
            if i_s <= base_image_size:
                pad_input = pad_to_size(interpolated_img, (base_image_size, base_image_size))
                image_pyramid.append(pad_input) # ??? Wgats is the range?
            elif i_s > base_image_size:
                center_crop = torchvision.transforms.CenterCrop(base_image_size)
                image_pyramid.append(center_crop(interpolated_img))
            # else:
            #     image_pyramid.append(x)

            # print('image_pyramid : ',image_pyramid[-1].shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)

        return image_pyramid

    def forward(self, x, batch_idx = None):

        ###############################################
        x_pyramid = self.make_ip(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Out 10 Scales x BxCxHxW --> C = 3

        ###############################################
        s1_maps = self.s1(x_pyramid, self.MNIST_Scale, batch_idx) # Out 10 Scales x BxCxHxW --> C = 4
        c1_maps = self.c1(s1_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, c2_sp_kernel_sizes = self.c2_sp_kernel_sizes)  # Out 9 Scales x BxCxHxW --> C = 4

        s2_maps = self.s2(c1_maps) # Out 9 Scales x BxCxHxW --> C = 2000
        c2_maps = self.c2(s2_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, c2_sp_kernel_sizes = self.c2_sp_kernel_sizes) # Out 8 Scales x BxCxHxW --> C = 2000
        s3_maps = self.s3(c2_maps) # Out 8 Scales x BxCxHxW --> C = 2000
        c3_maps = self.c3(s3_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Overall x BxCx1x1 --> C = 2000

        ###############################################
        # ByPass Routes
        s2b_maps = self.s2b(c1_maps) # Out 9 Scales x BxCxHxW --> C = 2000
        c2b_maps = self.c2b(s2b_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Overall x BxCx1x1 --> C = 2000

        ###############################################
        # Prepare inputs for S4
        # Out 9 Scales x BxCxHxW --> BxCxHxWxS
        for c_i in range(len(c1_maps)):
            c1_maps[c_i] = self.pool(c1_maps[c_i]) # matching HxW across all inputs to S4
        c1_maps = torch.stack(c1_maps, dim = 4)
        c1_maps = torch.permute(c1_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c1_maps = c1_maps.reshape(c1_maps.shape[0], -1, *c1_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        for c_i in range(len(c2_maps)):
            c2_maps[c_i] = self.pool(c2_maps[c_i]) # matching HxW across all inputs to S4
        c2_maps = torch.stack(c2_maps, dim = 4)
        c2_maps = torch.permute(c2_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c2_maps = c2_maps.reshape(c2_maps.shape[0], -1, *c2_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        for c_i in range(len(c3_maps)):
            c3_maps[c_i] = self.pool(c3_maps[c_i]) # matching HxW across all inputs to S4
        c3_maps = torch.stack(c3_maps, dim = 4)
        c3_maps = torch.permute(c3_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c3_maps = c3_maps.reshape(c3_maps.shape[0], -1, *c3_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        for c_i in range(len(c2b_maps)):
            c2b_maps[c_i] = self.pool(c2b_maps[c_i]) # matching HxW across all inputs to S4
        c2b_maps = torch.stack(c2b_maps, dim = 4)
        c2b_maps = torch.permute(c2b_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c2b_maps = c2b_maps.reshape(c2b_maps.shape[0], -1, *c2b_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        ###############################################
        # Concatenate and pass to S4
        # print('c1_maps : ',c1_maps.shape)
        # print('c2_maps : ',c2_maps.shape)
        # print('c3_maps : ',c3_maps.shape)
        # print('c2b_maps : ',c2b_maps.shape)
        s4_maps = self.s4(torch.cat([c1_maps, c2_maps, c3_maps, c2b_maps], dim=1))

        ###############################################
        s4_maps = torch.flatten(s4_maps, 1)

        # # Classify
        output = self.classifier(s4_maps)

        # ###############################################
        # # Global Max Pooling
        # # s4_maps = F.max_pool2d(s4_maps, s4_maps.shape[-1], 1)
        # # Global Avg Pooling
        # s4_maps = F.avg_pool2d(s4_maps, s4_maps.shape[-1], 1)

        # s4_maps = s4_maps.squeeze()
        
        # # print()
        # output = self.classifier(s4_maps)


        return output

#############################################################################
#############################################################################

class HMAX_IP_sep_ideal(nn.Module):
    def __init__(self,
                 ip_scales = 10,
                 s1_scale=11, #11 # 7,
                 s1_la=5.6, #5.6 # 3.5,
                 s1_si=4.5, #4.5 # 2.8,
                 n_ori=4,
                 num_classes=1000,
                 s1_trainable_filters=False,
                 visualize_mode = False,
                 prj_name = None,
                 MNIST_Scale = None,
                 category = None
                 ):
        super(HMAX_IP_sep_ideal, self).__init__()

        # A few settings
        self.ip_scales = ip_scales
        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.scale = 5

        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled

        self.same_scale_viz = None
        self.base_scale = None
        

        ########################################################
        ########################################################
        # For scales
        base_filt_size = 20 #17
        scale = self.scale
        self.c1_sp_kernel_sizes = [int(np.ceil(base_filt_size/(2**(i/scale)))) for i in range(ip_scales)]
        # self.c1_sp_kernel_sizes = [10, 8]
        # #######################################################


        self.c2_scale_stride = self.c_scale_stride
        self.c2_num_scales_pooled = self.c_num_scales_pooled

        # For 14 scales
        base_filt_size = self.c1_sp_kernel_sizes[1]
        scale = self.scale
        self.c2_sp_kernel_sizes = [int(np.ceil(base_filt_size/(2**(i/scale)))) for i in range(ip_scales)]
        # self.c2_sp_kernel_sizes = [8, 6]
        ########################################################
        ########################################################

        ########################################################
        # self.c1_sp_kernel_sizes = get_sp_kernel_sizes_C(range(7, 37, 2), self.c_scale_stride, self.c_scale_stride)

        # self.c2_scale_stride = self.c_scale_stride
        # self.c2_num_scales_pooled = self.c_num_scales_pooled
        # self.c2_sp_kernel_sizes = get_sp_kernel_sizes_C(self.c1_sp_kernel_sizes, self.c_num_scales_pooled, self.c2_scale_stride)
        ########################################################

        # Reverese
        # self.c1_sp_kernel_sizes.reverse()
        # self.c2_sp_kernel_sizes.reverse()

        print('c1_sp_kernel_sizes : ',self.c1_sp_kernel_sizes)
        print('c2_sp_kernel_sizes : ',self.c2_sp_kernel_sizes)
    

        self.c2b_scale_stride = self.c_scale_stride
        self.c2b_num_scales_pooled = len(self.c1_sp_kernel_sizes)  # all of them
        # no kernel sizes here because global pool (spatially)

        self.c3_scale_stride = self.c_scale_stride
        self.c3_num_scales_pooled = len(self.c2_sp_kernel_sizes)  # all of them
        # no kernel sizes here because global pool (spatially)

        ###############################################
        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters=s1_trainable_filters,
                     la=self.s1_la, si=self.s1_si, visualize_mode = visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.2, n_in_sbands=ip_scales,
                    num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride, visualize_mode = visualize_mode, \
                    c1_bool = True, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.s2 = S2(channels_in=n_ori, channels_out=100, kernel_size=3, stride=1)
        self.c2 = C(global_pool = False, sp_kernel_size=self.c2_sp_kernel_sizes, sp_stride_factor=0.2, n_in_sbands=ip_scales-1,
                    num_scales_pooled=self.c2_num_scales_pooled, scale_stride=self.c2_scale_stride, c2_bool = True, prj_name = self.prj_name)
        self.s2b = S2(channels_in=n_ori, channels_out=100, kernel_size=[4, 8, 12, 16], stride=1)
        self.s3 = S3(channels_in=100, channels_out=100, kernel_size=3, stride=1)
        self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)
        self.c3 = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-2,
                    num_scales_pooled=self.c3_num_scales_pooled, scale_stride=self.c3_scale_stride, c3_bool = True, prj_name = self.prj_name)


        ###############################################
        self.pool = nn.AdaptiveMaxPool2d(18)  # not in table, but we need to get everything in the same shape before s4

        self.get_s4_in_channels()

        self.s4_c1 = nn.Sequential(nn.Conv2d(self.n_ori, self.n_ori, 1, 1),
                                # nn.Dropout(0.3),
                                # # nn.BatchNorm2d(512, 1e-3),
                                # nn.ReLU(True),
                                # # nn.MaxPool2d(3, 2),
                                nn.AdaptiveMaxPool2d(18))

        self.s4_c2 = nn.Sequential(nn.Conv2d(100, 50, 1, 1),
                                # nn.Dropout(0.3),
                                # # nn.BatchNorm2d(512, 1e-3),
                                # nn.ReLU(True),
                                nn.Conv2d(50, 25, 1, 1),
                                # nn.Dropout(0.3),
                                # # nn.BatchNorm2d(256, 1e-3),
                                # nn.ReLU(True),
                                # # nn.MaxPool2d(3, 2),
                                nn.AdaptiveMaxPool2d(18))

        ###############################################
        # Classifier
        # self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
        #                                 nn.Linear(256 * 8 * 8, 4096),  # fc1
        #                                 nn.BatchNorm1d(4096, 1e-3),
        #                                 nn.ReLU(True),
        #                                 nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
        #                                 nn.Linear(4096, 4096),  # fc2
        #                                 nn.BatchNorm1d(4096, 1e-3),
        #                                 nn.ReLU(True),
        #                                 nn.Linear(4096, num_classes)  # fc3
        #                                 )

        # # Classifier
        self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        nn.Linear(149636, 256),  # fc1
                                        # nn.BatchNorm1d(4096, 1e-3),
                                        # nn.ReLU(True),
                                        # nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        # nn.Linear(4096, 4096),  # fc2
                                        # nn.BatchNorm1d(256, 1e-3),
                                        # nn.ReLU(True),
                                        nn.Linear(256, num_classes)  # fc3
                                        )


    def get_s4_in_channels(self):
        # FOr linderberg
        ip_scales = int(np.ceil(self.ip_scales/2))*2 -1

        # For Mutch
        # ip_scales = self.ip_scales

        #########################################
        c1_out = (ip_scales-1) * self.n_ori
        c2_out = (ip_scales-2) * self.s2.s_0[0].weight.shape[0]
        c3_out = (ip_scales-3) * self.s3.s_0[0].weight.shape[0]
        c2b_out = len(self.s2b.kernel_size) * (ip_scales-2) * self.s2b.s_0[0].weight.shape[0]
        # c3_out = self.s3.s_0[0].weight.shape[0]
        # c2b_out = len(self.s2b.kernel_size) * self.s2b.s_0[0].weight.shape[0]

        self.c1_out = c1_out
        self.c2_out = c2_out
        self.c3_out = c3_out
        self.c2b_out = c2b_out


    # def make_ip(self, x, same_scale_viz = None, base_scale = None):

    #     base_image_size = int(x.shape[-1])

    #     scale = self.scale
    #     # image_scales = [np.ceil(base_image_size/(2**(i/3))) if np.ceil(base_image_size/(2**(i/3)))%2 == 0 else np.floor(base_image_size/(2**(i/3))) for i in range(self.ip_scales)]
    #     image_scales = [np.ceil(base_image_size/(2**(i/scale))) for i in range(self.ip_scales)]
    #     # image_scales = [np.ceil(base_image_size) if np.ceil(base_image_size)%2 == 0 else np.floor(base_image_size) for i in range(self.ip_scales)]

    #     self.image_scales = image_scales

    #     if same_scale_viz:
    #         base_image_size = base_scale
    #     else:
    #         base_image_size = int(x.shape[-1])

    #     image_pyramid = []
    #     for i_s in image_scales:
    #         i_s = int(i_s)
    #         interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bicubic').clamp(min=0, max=1) # ??? Wgats is the range?
    #         pad_input = pad_to_size(interpolated_img, (base_image_size, base_image_size))
    #         image_pyramid.append(pad_input) # ??? Wgats is the range?
        
    #     return image_pyramid

    def make_ip(self, x, same_scale_viz = None, base_scale = None):

        # print('make_ip same_scale_viz : ',same_scale_viz)
        # print('make_ip base_scale : ',base_scale)

        base_image_size = int(x.shape[-1]) 

        # print('base_image_size : ',base_image_size)
        scale = self.scale #5
        image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) if np.ceil(base_image_size/(2**(i/scale)))%2 == 0 else np.floor(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(self.ip_scales/2)))]
        image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) if np.ceil(base_image_size*(2**(i/scale)))%2 == 0 else np.floor(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(self.ip_scales/2)))]
        # image_scales_down = [base_image_size/(2**(i/scale)) for i in range(int(np.ceil(self.ip_scales/2)))]
        # image_scales_up = [base_image_size*(2**(i/scale)) for i in range(1, int(np.ceil(self.ip_scales/2)))]
        

        image_scales = image_scales_down + image_scales_up
        # self.image_scales = sorted(image_scales)
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]

        # print('image_scales image_scales : ',self.image_scales)

        if same_scale_viz:
            base_image_size = base_scale
        else:
            base_image_size = int(x.shape[-1]) 

        # print('base_image_size : ',base_image_size)

        image_pyramid = []
        for i_s in self.image_scales:
            i_s = int(i_s)
            interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bicubic').clamp(min=0, max=1)
            if i_s <= base_image_size:
                pad_input = pad_to_size(interpolated_img, (base_image_size, base_image_size))
                image_pyramid.append(pad_input) # ??? Wgats is the range?
            elif i_s > base_image_size:
                center_crop = torchvision.transforms.CenterCrop(base_image_size)
                image_pyramid.append(center_crop(interpolated_img))
            # else:
            #     image_pyramid.append(x)

            # print('image_pyramid : ',image_pyramid[-1].shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)

        return image_pyramid

    def forward(self, x, batch_idx = None):

        ###############################################
        x_pyramid = self.make_ip(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Out 10 Scales x BxCxHxW --> C = 3

        ###############################################
        s1_maps = self.s1(x_pyramid, self.MNIST_Scale, batch_idx) # Out 10 Scales x BxCxHxW --> C = 4
        c1_maps = self.c1(s1_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, c2_sp_kernel_sizes = self.c2_sp_kernel_sizes)  # Out 9 Scales x BxCxHxW --> C = 4

        s2_maps = self.s2(c1_maps) # Out 9 Scales x BxCxHxW --> C = 2000
        c2_maps = self.c2(s2_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, c2_sp_kernel_sizes = self.c2_sp_kernel_sizes) # Out 8 Scales x BxCxHxW --> C = 2000
        s3_maps = self.s3(c2_maps) # Out 8 Scales x BxCxHxW --> C = 2000
        c3_maps = self.c3(s3_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Overall x BxCx1x1 --> C = 2000

        ###############################################
        # ByPass Routes
        s2b_maps = self.s2b(c1_maps) # Out 9 Scales x BxCxHxW --> C = 2000
        c2b_maps = self.c2b(s2b_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Overall x BxCx1x1 --> C = 2000

        # print('c1_maps : ',c1_maps[0].shape)
        # print('c2_maps : ',c2_maps[0].shape)
        # print('c3_maps : ',c3_maps[0].shape)
        # print('c2b_maps : ',c2b_maps[0].shape)

        ###############################################
        # Prepare inputs for S4
        # Out 9 Scales x BxCxHxW --> BxCxHxWxS
        for c_i in range(len(c1_maps)):
            c1_maps[c_i] = self.s4_c1(c1_maps[c_i]) # matching HxW across all inputs to S4
        c1_maps = torch.stack(c1_maps, dim = 4)
        c1_maps = torch.permute(c1_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c1_maps = c1_maps.reshape(c1_maps.shape[0], -1, *c1_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        for c_i in range(len(c2_maps)):
            c2_maps[c_i] = self.s4_c2(c2_maps[c_i]) # matching HxW across all inputs to S4
        c2_maps = torch.stack(c2_maps, dim = 4)
        c2_maps = torch.permute(c2_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c2_maps = c2_maps.reshape(c2_maps.shape[0], -1, *c2_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        # for c_i in range(len(c3_maps)):
        #     c3_maps[c_i] = self.pool(c3_maps[c_i]) # matching HxW across all inputs to S4
        c3_maps = torch.stack(c3_maps, dim = 4)
        c3_maps = torch.permute(c3_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c3_maps = c3_maps.reshape(c3_maps.shape[0], -1, *c3_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        # for c_i in range(len(c2b_maps)):
        #     c2b_maps[c_i] = self.pool(c2b_maps[c_i]) # matching HxW across all inputs to S4
        c2b_maps = torch.stack(c2b_maps, dim = 4)
        c2b_maps = torch.permute(c2b_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c2b_maps = c2b_maps.reshape(c2b_maps.shape[0], -1, *c2b_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        ###############################################
        # Concatenate and pass to S4
        # print('c1_maps : ',c1_maps.shape)
        # print('c2_maps : ',c2_maps.shape)
        # print('c3_maps : ',c3_maps.shape)
        # print('c2b_maps : ',c2b_maps.shape)

        ###############################################
        # s4_maps = torch.flatten(s4_maps, 1)
        s4_maps = torch.cat([torch.flatten(c1_maps, 1), torch.flatten(c2_maps, 1), torch.flatten(c3_maps, 1), torch.flatten(c2b_maps, 1)], dim=1)

        # print('s4_maps : ',s4_maps.shape)

        # # Classify
        output = self.classifier(s4_maps)

        # ###############################################
        


        return output



#############################################################################
#############################################################################

class HMAX_IP_sep(nn.Module):
    def __init__(self,
                 ip_scales = 10,
                 s1_scale=11, #11 # 7,
                 s1_la=5.6, #5.6 # 3.5,
                 s1_si=4.5, #4.5 # 2.8,
                 n_ori=4,
                 num_classes=1000,
                 s1_trainable_filters=False,
                 visualize_mode = False,
                 prj_name = None,
                 MNIST_Scale = None,
                 category = None
                 ):
        super(HMAX_IP_sep, self).__init__()

        # A few settings
        self.ip_scales = ip_scales
        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.scale = 5

        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled

        self.same_scale_viz = None
        self.base_scale = None
        

        ########################################################
        ########################################################
        # For scales
        base_filt_size = 20 #17
        scale = self.scale
        self.c1_sp_kernel_sizes = [int(np.ceil(base_filt_size/(2**(i/scale)))) for i in range(ip_scales)]
        # self.c1_sp_kernel_sizes = [10, 8]
        # #######################################################


        self.c2_scale_stride = self.c_scale_stride
        self.c2_num_scales_pooled = self.c_num_scales_pooled

        # For 14 scales
        base_filt_size = self.c1_sp_kernel_sizes[1]
        scale = self.scale
        self.c2_sp_kernel_sizes = [int(np.ceil(base_filt_size/(2**(i/scale)))) for i in range(ip_scales)]
        # self.c2_sp_kernel_sizes = [8, 6]
        ########################################################
        ########################################################

        ########################################################
        # self.c1_sp_kernel_sizes = get_sp_kernel_sizes_C(range(7, 37, 2), self.c_scale_stride, self.c_scale_stride)

        # self.c2_scale_stride = self.c_scale_stride
        # self.c2_num_scales_pooled = self.c_num_scales_pooled
        # self.c2_sp_kernel_sizes = get_sp_kernel_sizes_C(self.c1_sp_kernel_sizes, self.c_num_scales_pooled, self.c2_scale_stride)
        ########################################################

        # Reverese
        # self.c1_sp_kernel_sizes.reverse()
        # self.c2_sp_kernel_sizes.reverse()

        print('c1_sp_kernel_sizes : ',self.c1_sp_kernel_sizes)
        print('c2_sp_kernel_sizes : ',self.c2_sp_kernel_sizes)
    

        self.c2b_scale_stride = self.c_scale_stride
        self.c2b_num_scales_pooled = len(self.c1_sp_kernel_sizes)  # all of them
        # no kernel sizes here because global pool (spatially)

        self.c3_scale_stride = self.c_scale_stride
        self.c3_num_scales_pooled = len(self.c2_sp_kernel_sizes)  # all of them
        # no kernel sizes here because global pool (spatially)

        ###############################################
        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters=s1_trainable_filters,
                     la=self.s1_la, si=self.s1_si, visualize_mode = visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.2, n_in_sbands=ip_scales,
                    num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride, visualize_mode = visualize_mode, \
                    c1_bool = True, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.s2 = S2(channels_in=n_ori, channels_out=100, kernel_size=3, stride=1)
        self.c2 = C(global_pool = False, sp_kernel_size=self.c2_sp_kernel_sizes, sp_stride_factor=0.2, n_in_sbands=ip_scales-1,
                    num_scales_pooled=self.c2_num_scales_pooled, scale_stride=self.c2_scale_stride, c2_bool = True, prj_name = self.prj_name)
        self.s2b = S2(channels_in=n_ori, channels_out=100, kernel_size=[4, 8, 12, 16], stride=1)
        self.s3 = S3(channels_in=100, channels_out=100, kernel_size=3, stride=1)
        self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)
        self.c3 = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-2,
                    num_scales_pooled=self.c3_num_scales_pooled, scale_stride=self.c3_scale_stride, c3_bool = True, prj_name = self.prj_name)


        ###############################################
        self.pool = nn.AdaptiveMaxPool2d(18)  # not in table, but we need to get everything in the same shape before s4

        self.get_s4_in_channels()

        self.s4_c1 = nn.Sequential(nn.Conv2d(self.c1_out, 128, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(512, 1e-3),
                                nn.ReLU(True),
                                nn.Conv2d(128, 64, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(256, 1e-3),
                                nn.ReLU(True),
                                nn.MaxPool2d(3, 2))

        self.s4_c2 = nn.Sequential(nn.Conv2d(self.c2_out, 128, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(512, 1e-3),
                                nn.ReLU(True),
                                nn.Conv2d(128, 64, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(256, 1e-3),
                                nn.ReLU(True),
                                nn.MaxPool2d(3, 2))

        self.s4_c3 = nn.Sequential(nn.Conv2d(self.c3_out, 128, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(512, 1e-3),
                                nn.ReLU(True),
                                nn.Conv2d(128, 64, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(256, 1e-3),
                                nn.ReLU(True),
                                nn.MaxPool2d(3, 2))

        self.s4_c2b = nn.Sequential(nn.Conv2d(self.c2b_out, 128, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(512, 1e-3),
                                nn.ReLU(True),
                                nn.Conv2d(128, 64, 1, 1),
                                nn.Dropout(0.3),
                                # nn.BatchNorm2d(256, 1e-3),
                                nn.ReLU(True),
                                nn.MaxPool2d(3, 2))

        ###############################################
        # Classifier
        # self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
        #                                 nn.Linear(256 * 8 * 8, 4096),  # fc1
        #                                 nn.BatchNorm1d(4096, 1e-3),
        #                                 nn.ReLU(True),
        #                                 nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
        #                                 nn.Linear(4096, 4096),  # fc2
        #                                 nn.BatchNorm1d(4096, 1e-3),
        #                                 nn.ReLU(True),
        #                                 nn.Linear(4096, num_classes)  # fc3
        #                                 )

        # # Classifier
        self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        nn.Linear(256 * 8 * 8, 256),  # fc1
                                        # nn.BatchNorm1d(4096, 1e-3),
                                        # nn.ReLU(True),
                                        # nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        # nn.Linear(4096, 4096),  # fc2
                                        # nn.BatchNorm1d(256, 1e-3),
                                        # nn.ReLU(True),
                                        nn.Linear(256, num_classes)  # fc3
                                        )


    def get_s4_in_channels(self):
        # FOr linderberg
        ip_scales = int(np.ceil(self.ip_scales/2))*2 -1

        # For Mutch
        # ip_scales = self.ip_scales

        #########################################
        c1_out = (ip_scales-1) * self.n_ori
        c2_out = (ip_scales-2) * self.s2.s_0[0].weight.shape[0]
        c3_out = (ip_scales-3) * self.s3.s_0[0].weight.shape[0]
        c2b_out = len(self.s2b.kernel_size) * (ip_scales-2) * self.s2b.s_0[0].weight.shape[0]
        # c3_out = self.s3.s_0[0].weight.shape[0]
        # c2b_out = len(self.s2b.kernel_size) * self.s2b.s_0[0].weight.shape[0]

        self.c1_out = c1_out
        self.c2_out = c2_out
        self.c3_out = c3_out
        self.c2b_out = c2b_out


    # def make_ip(self, x, same_scale_viz = None, base_scale = None):

    #     base_image_size = int(x.shape[-1])

    #     scale = self.scale
    #     # image_scales = [np.ceil(base_image_size/(2**(i/3))) if np.ceil(base_image_size/(2**(i/3)))%2 == 0 else np.floor(base_image_size/(2**(i/3))) for i in range(self.ip_scales)]
    #     image_scales = [np.ceil(base_image_size/(2**(i/scale))) for i in range(self.ip_scales)]
    #     # image_scales = [np.ceil(base_image_size) if np.ceil(base_image_size)%2 == 0 else np.floor(base_image_size) for i in range(self.ip_scales)]

    #     self.image_scales = image_scales

    #     if same_scale_viz:
    #         base_image_size = base_scale
    #     else:
    #         base_image_size = int(x.shape[-1])

    #     image_pyramid = []
    #     for i_s in image_scales:
    #         i_s = int(i_s)
    #         interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bicubic').clamp(min=0, max=1) # ??? Wgats is the range?
    #         pad_input = pad_to_size(interpolated_img, (base_image_size, base_image_size))
    #         image_pyramid.append(pad_input) # ??? Wgats is the range?
        
    #     return image_pyramid

    def make_ip(self, x, same_scale_viz = None, base_scale = None):

        # print('make_ip same_scale_viz : ',same_scale_viz)
        # print('make_ip base_scale : ',base_scale)

        base_image_size = int(x.shape[-1]) 

        # print('base_image_size : ',base_image_size)
        scale = self.scale #5
        image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) if np.ceil(base_image_size/(2**(i/scale)))%2 == 0 else np.floor(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(self.ip_scales/2)))]
        image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) if np.ceil(base_image_size*(2**(i/scale)))%2 == 0 else np.floor(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(self.ip_scales/2)))]
        # image_scales_down = [base_image_size/(2**(i/scale)) for i in range(int(np.ceil(self.ip_scales/2)))]
        # image_scales_up = [base_image_size*(2**(i/scale)) for i in range(1, int(np.ceil(self.ip_scales/2)))]
        

        image_scales = image_scales_down + image_scales_up
        # self.image_scales = sorted(image_scales)
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]

        # print('image_scales image_scales : ',self.image_scales)

        if same_scale_viz:
            base_image_size = base_scale
        else:
            base_image_size = int(x.shape[-1]) 

        # print('base_image_size : ',base_image_size)

        image_pyramid = []
        for i_s in self.image_scales:
            i_s = int(i_s)
            interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bicubic').clamp(min=0, max=1)
            if i_s <= base_image_size:
                pad_input = pad_to_size(interpolated_img, (base_image_size, base_image_size))
                image_pyramid.append(pad_input) # ??? Wgats is the range?
            elif i_s > base_image_size:
                center_crop = torchvision.transforms.CenterCrop(base_image_size)
                image_pyramid.append(center_crop(interpolated_img))
            # else:
            #     image_pyramid.append(x)

            # print('image_pyramid : ',image_pyramid[-1].shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)

        return image_pyramid

    def forward(self, x, batch_idx = None):

        ###############################################
        x_pyramid = self.make_ip(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Out 10 Scales x BxCxHxW --> C = 3

        ###############################################
        s1_maps = self.s1(x_pyramid, self.MNIST_Scale, batch_idx) # Out 10 Scales x BxCxHxW --> C = 4
        c1_maps = self.c1(s1_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, c2_sp_kernel_sizes = self.c2_sp_kernel_sizes)  # Out 9 Scales x BxCxHxW --> C = 4

        s2_maps = self.s2(c1_maps) # Out 9 Scales x BxCxHxW --> C = 2000
        c2_maps = self.c2(s2_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, c2_sp_kernel_sizes = self.c2_sp_kernel_sizes) # Out 8 Scales x BxCxHxW --> C = 2000
        s3_maps = self.s3(c2_maps) # Out 8 Scales x BxCxHxW --> C = 2000
        c3_maps = self.c3(s3_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Overall x BxCx1x1 --> C = 2000

        ###############################################
        # ByPass Routes
        s2b_maps = self.s2b(c1_maps) # Out 9 Scales x BxCxHxW --> C = 2000
        c2b_maps = self.c2b(s2b_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Overall x BxCx1x1 --> C = 2000

        # print('c1_maps : ',c1_maps[0].shape)
        # print('c2_maps : ',c2_maps[0].shape)
        # print('c3_maps : ',c3_maps[0].shape)
        # print('c2b_maps : ',c2b_maps[0].shape)

        ###############################################
        # Prepare inputs for S4
        # Out 9 Scales x BxCxHxW --> BxCxHxWxS
        for c_i in range(len(c1_maps)):
            c1_maps[c_i] = self.pool(c1_maps[c_i]) # matching HxW across all inputs to S4
        c1_maps = torch.stack(c1_maps, dim = 4)
        c1_maps = torch.permute(c1_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c1_maps = c1_maps.reshape(c1_maps.shape[0], -1, *c1_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        for c_i in range(len(c2_maps)):
            c2_maps[c_i] = self.pool(c2_maps[c_i]) # matching HxW across all inputs to S4
        c2_maps = torch.stack(c2_maps, dim = 4)
        c2_maps = torch.permute(c2_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c2_maps = c2_maps.reshape(c2_maps.shape[0], -1, *c2_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        for c_i in range(len(c3_maps)):
            c3_maps[c_i] = self.pool(c3_maps[c_i]) # matching HxW across all inputs to S4
        c3_maps = torch.stack(c3_maps, dim = 4)
        c3_maps = torch.permute(c3_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c3_maps = c3_maps.reshape(c3_maps.shape[0], -1, *c3_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        for c_i in range(len(c2b_maps)):
            c2b_maps[c_i] = self.pool(c2b_maps[c_i]) # matching HxW across all inputs to S4
        c2b_maps = torch.stack(c2b_maps, dim = 4)
        c2b_maps = torch.permute(c2b_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
        c2b_maps = c2b_maps.reshape(c2b_maps.shape[0], -1, *c2b_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW

        ###############################################
        # Concatenate and pass to S4
        # print('c1_maps : ',c1_maps.shape)
        # print('c2_maps : ',c2_maps.shape)
        # print('c3_maps : ',c3_maps.shape)
        # print('c2b_maps : ',c2b_maps.shape)

        # s4_maps = self.s4(torch.cat([c1_maps, c2_maps, c3_maps, c2b_maps], dim=1))

        s4_maps_c1 = self.s4_c1(c1_maps)
        s4_maps_c2 = self.s4_c2(c2_maps)
        s4_maps_c3 = self.s4_c3(c3_maps)
        s4_maps_c2b = self.s4_c2b(c2b_maps)
        s4_maps = torch.cat([s4_maps_c1, s4_maps_c2, s4_maps_c3, s4_maps_c2b], dim=1)

        ###############################################
        s4_maps = torch.flatten(s4_maps, 1)

        # # Classify
        output = self.classifier(s4_maps)

        # ###############################################
        


        return output
