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
    # filt_c = filt_c.repeat((1, 3, 1, 1))

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

        setattr(self, f's_{scale}', nn.Conv2d(1, n_ori, scale, padding=padding))
        s1_cell = getattr(self, f's_{scale}')
        gabor_filter = get_gabor(l_size=scale, la=la, si=si, n_ori=n_ori, aspect_ratio=0.3)  # ??? What is aspect ratio
        s1_cell.weight = nn.Parameter(gabor_filter, requires_grad=trainable_filters)

        # For normalization
        setattr(self, f's_uniform_{scale}', nn.Conv2d(1, n_ori, scale, bias=False))
        s1_uniform = getattr(self, f's_uniform_{scale}')
        nn.init.constant_(s1_uniform.weight, 1)
        for param in s1_uniform.parameters():
            param.requires_grad = False

        self.batchnorm = nn.BatchNorm2d(n_ori, 1e-3)

    def forward(self, x_pyramid, MNIST_Scale = None, batch_idx = None, prj_name = None):
        self.MNIST_Scale = MNIST_Scale
        s1_maps = []
        # Loop over scales, normalizing.
        for p_i in range(len(x_pyramid)):

            x = x_pyramid[p_i]

            s1_cell = getattr(self, f's_{self.scale}')
            s1_map = torch.abs(s1_cell(x))  # adding absolute value

            # Normalization
            # s1_unorm = getattr(self, f's_uniform_{self.scale}')
            # s1_unorm = torch.sqrt(s1_unorm(x**2))
            # # s1_unorm = torch.sqrt(s1_unorm(x))
            # s1_unorm.data[s1_unorm == 0] = 1  # To avoid divide by zero
            # s1_map /= s1_unorm
            s1_map = self.batchnorm(s1_map)

            s1_maps.append(s1_map)

            # Padding (to get s1_maps in same size) ---> But not necessary for us
            ori_size = (x.shape[-2], x.shape[-1])
            s1_maps[p_i] = pad_to_size(s1_maps[p_i], ori_size)

        # if True and batch_idx == 0:
        #     ori_size = (x_pyramid[0][0].shape[-1], x_pyramid[0][0].shape[-1])

        #     combined_vertical_image = x_pyramid[-5][0].clone()

        #     combined_vertical_image = combined_vertical_image - torch.min(combined_vertical_image)
        #     combined_vertical_image = (combined_vertical_image/torch.max(combined_vertical_image))

        #     combined_vertical_image = pad_to_size(combined_vertical_image, ori_size)
        #     combined_vertical_image = combined_vertical_image.cpu().numpy()
        #     # CxHxW --> HxWxC
        #     combined_vertical_image = combined_vertical_image.transpose(1,2,0)
        #     combined_vertical_image = combined_vertical_image[:,:]*255.0
        #     combined_vertical_image = combined_vertical_image.astype('uint8')
        #     combined_vertical_image = cv2.copyMakeBorder(combined_vertical_image,3,3,3,3,cv2.BORDER_CONSTANT,value=255)

            
        #     for s_i in range(1, len(s1_maps)+1):
        #         scale_maps = s1_maps[-s_i].clone()

        #         # scale_maps = scale_maps - torch.min(scale_maps)
        #         # scale_maps = (scale_maps/torch.max(scale_maps))

        #         scale_maps = pad_to_size(scale_maps, ori_size)
        #         # print('scale_maps : ',scale_maps.shape)
        #         scale_maps_clone = scale_maps.clone()
        #         scale_maps_clone = scale_maps_clone[0]

        #         for f_i, filter_maps in enumerate(scale_maps_clone):
        #             filter_maps = filter_maps - torch.min(filter_maps)
        #             filter_maps = (filter_maps/torch.max(filter_maps))

        #             filter_maps_numpy = filter_maps.cpu().numpy()
        #             # CxHxW --> HxWxC
        #             filter_maps_numpy = filter_maps_numpy.reshape(1, *filter_maps_numpy.shape)
        #             filter_maps_numpy = filter_maps_numpy.transpose(1,2,0)
        #             # filter_maps_numpy = filter_maps_numpy - np.min(filter_maps_numpy)
        #             # filter_maps_numpy = (filter_maps_numpy/np.max(filter_maps_numpy))*255.0
        #             filter_maps_numpy = filter_maps_numpy*255.0
        #             filter_maps_numpy = filter_maps_numpy.astype('uint8')
        #             filter_maps_numpy = cv2.copyMakeBorder(filter_maps_numpy,3,3,3,3,cv2.BORDER_CONSTANT,value=255)

        #             combined_vertical_image = cv2.vconcat([combined_vertical_image, filter_maps_numpy])

        #             break

        #     main_dir = '/cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/visualize_filters/S1'
        #     os.makedirs(main_dir, exist_ok=True)
        #     job_dir = os.path.join(main_dir, "prj_{}".format(prj_name))
        #     os.makedirs(job_dir, exist_ok=True)

        #     # out_path = os.path.join(job_dir, "{}_filters_temp.png".format(self.MNIST_Scale))
        #     # cv2.imwrite(out_path, combined_image)

        #     out_path = os.path.join(job_dir, "{}_filters_temp.npy".format(int(1000*MNIST_Scale)))
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
                prj_name = None, same_scale_viz = None, base_scale = None, c1_sp_kernel_sizes = None, \
                c2_sp_kernel_sizes = None, image_scales = None, overall_max_scale_index = False):
        # TODO - make this whole section more memory efficient

        # print('####################################################################################')
        # print('####################################################################################')

        c_maps = []
        if same_scale_viz:
            ori_size = (base_scale, base_scale)
        else:
            ori_size = x_pyramid[0].shape[2:4]

        # Single scale band case --> While Training
        if len(x_pyramid) == 1:
            if not self.global_pool:
                x = F.max_pool2d(x_pyramid[0], self.sp_kernel_size[0], self.sp_stride[0])
                x = pad_to_size(x, ori_size)

                c_maps.append(x)
            else:
                s_m = F.max_pool2d(x_pyramid[0], x_pyramid[0].shape[-1], 1)
                c_maps.append(s_m)

        # Multi Scale band case ---> While Testing
        else:
            #####################################################
            if not self.global_pool:
                if same_scale_viz:
                    ori_size = (base_scale, base_scale)
                else:
                    if len(x_pyramid) == 2:
                        ori_size = x_pyramid[0].shape[2:4]
                    else:
                        ori_size = x_pyramid[-5].shape[2:4]

                ####################################################
                # MaxPool for C1 with 2 scales being max pooled over at a time
                for p_i in range(len(x_pyramid)-1):
                    # print('############################')
                    x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[0], self.sp_stride[0])
                    x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[1], self.sp_stride[1])


                    # First interpolating such that feature points match spatially
                    if x_1.shape[-1] > x_2.shape[-1]:
                        # x_2 = pad_to_size(x_2, x_1.shape[-2:])
                        x_2 = F.interpolate(x_2, size = x_1.shape[-2:], mode = 'bilinear')
                    else:
                        # x_1 = pad_to_size(x_1, x_2.shape[-2:])
                        x_1 = F.interpolate(x_1, size = x_2.shape[-2:], mode = 'bilinear')

                    # Then padding
                    x_1 = pad_to_size(x_1, ori_size)
                    x_2 = pad_to_size(x_2, ori_size)

                    ##################################
                    # Maxpool over scale groups
                    x = torch.stack([x_1, x_2], dim=4)

                    to_append, _ = torch.max(x, dim=4)
                    c_maps.append(to_append)

                ###################################################
                # MaxPool for C1 with 1 scale only being max pooled over at a time
                # for p_i in range(len(x_pyramid)):
                #     # print('############################')
                #     # print('x_pyramid[p_i] : ',x_pyramid[p_i].shape)
                #     # print('x_pyramid[p_i+1] : ',x_pyramid[p_i+1].shape)
                #     # What is the ideal case when x_1 and x_2 will have same dimensions

                #     # x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[p_i], self.sp_stride[p_i])
                #     # x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[p_i+1], self.sp_stride[p_i+1])
                #     x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[0], self.sp_stride[0])

                #     x_1 = pad_to_size(x_1, ori_size)

                #     c_maps.append(x_1)

                ####################################################
                # ArgMax case for C1
                # for p_i in range(len(x_pyramid)-1):
                #     scale_max = []

                #     # x_1 = F.max_pool2d(x_pyramid[p_i], x_pyramid[p_i].shape[-1])

                #     x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[p_i], self.sp_stride[p_i])


                #     # x_2 = F.max_pool2d(x_pyramid[p_i+1], x_pyramid[p_i+1].shape[-1])

                #     x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[p_i+1], self.sp_stride[p_i+1])

                #     if x_1.shape[-1] > x_2.shape[-1]:
                #         # x_2 = pad_to_size(x_2, x_1.shape[-2:])
                #         x_2 = F.interpolate(x_2, size = x_1.shape[-2:], mode = 'bilinear')
                #     else:
                #         # x_1 = pad_to_size(x_1, x_2.shape[-2:])
                #         x_1 = F.interpolate(x_1, size = x_2.shape[-2:], mode = 'bilinear')

                    
                #     x_1 = torch.sum(x_1, dim = (2,3)) #/image_scales[p_i]
                #     x_1 = x_1.reshape(*x_pyramid[p_i].shape[:2])
                #     x_1 = torch.sort(x_1, dim = -1)[0]
                #     scale_max.append(x_1)

                #     x_2 = torch.sum(x_2, dim = (2,3)) #/image_scales[p_i]
                #     x_2 = x_2.reshape(*x_pyramid[p_i+1].shape[:2])
                #     x_2 = torch.sort(x_2, dim = -1)[0]
                #     scale_max.append(x_2)

                #     # scale_max shape --> Scale x B x C
                #     scale_max = torch.stack(scale_max, dim=0) # Shape --> Scale x B x C

                #     x_max = []
                #     for b_i in range(scale_max.shape[1]):

                #         scale_max_argsort = torch.argsort(torch.stack([scale_max[0][b_i], scale_max[1][b_i]], dim=0), dim = 0) # Shape --> 2 x C
                #         # print('x_pyramid_flatten_argsort : ',x_pyramid_flatten_argsort)
                #         # Sum across the (CxHxW) dimension
                #         sum_scale_batch = torch.sum(scale_max_argsort, dim = 1) # SHape --> 2 x 1]
                #         # sum_scale_batch = sum_scale_batch.cpu().numpy()
                #         # sum_scale_batch = sum_scale_batch.astype(np.float32)
                #         # sum_scale_batch[0] = sum_scale_batch[0]/image_scales[max_scale_index[b_i]]
                #         # sum_scale_batch[1] = sum_scale_batch[1]/image_scales[p_i]
                #         # print('max_scale_index[b_i] : ', max_scale_index[b_i], ':: p_i : ', p_i, ' :: sum_scale_batch : ',sum_scale_batch)

                #         if sum_scale_batch[0] < sum_scale_batch[1]:
                #             # print('Smaller Scale : ')
                #             x_max_batch = F.max_pool2d(x_pyramid[p_i+1][b_i], self.sp_kernel_size[1], self.sp_stride[1])
                #         else:
                #             # print('Larger Scale : ')
                #             x_max_batch = F.max_pool2d(x_pyramid[p_i][b_i], self.sp_kernel_size[0], self.sp_stride[0])

                #         x_max_batch = pad_to_size(x_max_batch, ori_size)

                #         x_max.append(x_max_batch)

                #         x_1 = None
                #         x_2 = None
                #         x_max_batch = None

                #     x_max = torch.stack(x_max, dim=0)

                #     c_maps.append(x_max)
                
                ####################################################

            ####################################################
            # MaxPool over all positions first then scales (C2b)
            else:
                scale_max = []
                # Global MaxPool over positions for each scale separately
                for p_i in range(len(x_pyramid)):
                    s_m = F.max_pool2d(x_pyramid[p_i], x_pyramid[p_i].shape[-1], 1)
                    scale_max.append(s_m)

                # Option 1:: Global Max Pooling over Scale i.e, Maxpool over scale groups
                x = torch.stack(scale_max, dim=4)
                to_append, _ = torch.max(x, dim=4)
                c_maps.append(to_append)

                # # Option 2:: No Global Max Pooling over Scale
                # for s_i in range(len(scale_max)-1):
                #     # Maxpool over scale groups
                #     x = torch.stack([scale_max[s_i], scale_max[s_i+1]], dim=4)

                #     to_append, _ = torch.max(x, dim=4)
                #     c_maps.append(to_append)

                # # Option 3: Give scale_max as c_maps
                # c_maps = scale_max

            # #####################################################
            # # Argmax
            # else:
            #     #####################################################
            #     # x_pyramid shape --> Scale x BxCxHxW
            #     x_pyramid_flatten = [torch.sort(x_pyramid[p_i].reshape(x_pyramid[p_i].shape[0],-1), dim =- 1)[0] for p_i in range(len(x_pyramid))]
            #     x_pyramid_flatten = torch.stack(x_pyramid_flatten, dim=0) # Shape --> Scale x B x (CxHxW)

            #     # print('x_pyramid_flatten : ',x_pyramid_flatten.shape)

            #     max_scale_index = [0]*x_pyramid_flatten.shape[1]
            #     # max_scale_index = torch.tensor(max_scale_index).cuda()
            #     for p_i in range(1, len(x_pyramid)):
            #         # Getting ready the max scale indexes x_pyramid till now
            #         # x_pyramid_flatten_maxes = []
            #         # for b_i in range(x_pyramid_flatten.shape[1]):
            #         #     x_pyramid_flatten_maxes.append(x_pyramid_flatten[max_scale_index[b_i]][b_i])
            #         # x_pyramid_flatten_maxes = torch.stack(x_pyramid_flatten_maxes, dim = 0) # shape --> B x (CxHxW)
            #         # # print('x_pyramid_flatten_maxes : ',x_pyramid_flatten_maxes.shape)

            #         top_5_percentile_len = int(len(x_pyramid_flatten[0][0]) * 0.05)


            #         for b_i in range(x_pyramid_flatten.shape[1]):

            #             x_pyramid_flatten_argsort = torch.argsort(torch.stack([x_pyramid_flatten[max_scale_index[b_i]][b_i][-top_5_percentile_len:], x_pyramid_flatten[p_i][b_i][-top_5_percentile_len:]], dim=0), dim = 0) # Shape --> 2 x (CxHxW)
            #             # print('x_pyramid_flatten_argsort : ',x_pyramid_flatten_argsort)
            #             # Sum across the (CxHxW) dimension
            #             sum_scale_batch = torch.sum(x_pyramid_flatten_argsort, dim = 1) # SHape --> 2 x 1
            #             # print('sum_scale_batch : ',sum_scale_batch)

            #             if sum_scale_batch[0] < sum_scale_batch[1]:
            #                 max_scale_index[b_i] = p_i


            #     # max_scale_index = torch.argmax(sum_scale)


            #     to_append = []
            #     for b_i in range(x_pyramid_flatten.shape[1]):
            #         print('max_scale_index : ',max_scale_index[b_i])
            #         to_append_batch = F.max_pool2d(x_pyramid[max_scale_index[b_i]][b_i][None], x_pyramid[max_scale_index[b_i]][b_i][None].shape[-1], 1) # Shape --> 1 x C x 1 x 1
            #         # print('to_append_batch shape : ',to_append_batch.shape)
            #         to_append.append(to_append_batch)

            #     to_append = torch.cat(to_append, dim = 0)
            #     # print('to_append shape : ',to_append.shape)


            #     c_maps.append(to_append)

            #     #####################################################

            # # Argmax with global maxpool/sum over H, W
            # else:
            #     #####################################################
            #     scale_max = []
            #     for p_i in range(len(x_pyramid)):

            #         s_m = x_pyramid[p_i]

            #         # #####################################################
            #         # # Rescale
            #         # new_dimension = int(image_scales[-(p_i+1)]) #int(s_m.shape[-1]/(image_scales[p_i]/image_scales[4]))
            #         # # if new_dimension >= s_m.shape[-1]:
            #         # #     new_dimension = s_m.shape[-1]
            #         # print('Old Shape : ',s_m.shape[-1], ' ::: New Dimension : ',new_dimension)
            #         # s_m = F.interpolate(s_m, size = (new_dimension, new_dimension), mode = 'bilinear')
            #         # print('s_m shape : ',s_m.shape)
            #         # # s_m = s_m.reshape(*s_m.shape[:2])
            #         # #####################################################
            #         # # Crop
            #         # new_dimension = int(image_scales[-(p_i+1)]) #int(s_m.shape[-1]/(image_scales[p_i]/image_scales[4]))
            #         # if new_dimension >= s_m.shape[-1]:
            #         #     new_dimension = s_m.shape[-1]
            #         # print('Old Shape : ',s_m.shape[-1], ' ::: New Dimension : ',new_dimension)
            #         # center_crop = torchvision.transforms.CenterCrop(new_dimension)
            #         # s_m = center_crop(s_m)
            #         # print('s_m shape : ',s_m.shape)
            #         # # s_m = s_m.reshape(*s_m.shape[:2])
            #         #####################################################
            #         # MaxPool H,W
            #         s_m = F.max_pool2d(s_m, s_m.shape[-1], 1)
            #         # s_m = F.avg_pool2d(s_m, s_m.shape[-1], 1)
            #         s_m = s_m.reshape(*x_pyramid[p_i].shape[:2])
            #         # Sum H,W
            #         # s_m = torch.sum(s_m, dim = (2,3)) #/image_scales[p_i]
            #         # s_m = s_m.reshape(*x_pyramid[p_i].shape[:2])
            #         #####################################################
            #         s_m = torch.sort(s_m, dim = -1)[0]
            #         scale_max.append(s_m)

            #     # scale_max shape --> Scale x B x C
            #     scale_max = torch.stack(scale_max, dim=0) # Shape --> Scale x B x C
            #     # print('scale_max : ',scale_max.shape)

            #     # print('image_scales : ',image_scales)

            #     max_scale_index = [0]*scale_max.shape[1]
            #     # max_scale_index = torch.tensor(max_scale_index).cuda()
            #     for p_i in range(1, len(x_pyramid)):
                    
            #         for b_i in range(scale_max.shape[1]):

            #             # print(f'b_i : {b_i}, p_i {p_i}')
            #             # print('scale_max[max_scale_index[b_i]][b_i] : ',scale_max[max_scale_index[b_i]][b_i])
            #             # print('scale_max[p_i][b_i] : ',scale_max[p_i][b_i])

            #             scale_max_argsort = torch.argsort(torch.stack([scale_max[max_scale_index[b_i]][b_i], scale_max[p_i][b_i]], dim=0), dim = 0) # Shape --> 2 x C
            #             # print('x_pyramid_flatten_argsort : ',x_pyramid_flatten_argsort)
            #             # Sum across the (CxHxW) dimension
            #             sum_scale_batch = torch.sum(scale_max_argsort, dim = 1) # SHape --> 2 x 1]

            #             # sum_scale_batch = sum_scale_batch.cpu().numpy()
            #             # sum_scale_batch = sum_scale_batch.astype(np.float32)
            #             # sum_scale_batch[0] = sum_scale_batch[0]/((image_scales[max_scale_index[b_i]]/image_scales[-1])**1)
            #             # sum_scale_batch[1] = sum_scale_batch[1]/((image_scales[p_i]/image_scales[-1])**1)
            #             # print('max_scale_index[b_i] : ', max_scale_index[b_i], ':: p_i : ', p_i, ' :: sum_scale_batch : ',sum_scale_batch)

            #             if sum_scale_batch[0] < sum_scale_batch[1]:
            #                 max_scale_index[b_i] = p_i


            #     # max_scale_index = torch.argmax(sum_scale)


            #     to_append = []
            #     for b_i in range(scale_max.shape[1]):
            #         # print('max_scale_index : ',max_scale_index[b_i])
            #         to_append_batch = F.max_pool2d(x_pyramid[max_scale_index[b_i]][b_i][None], x_pyramid[max_scale_index[b_i]][b_i][None].shape[-1], 1) # Shape --> 1 x C x 1 x 1
            #         # print('to_append_batch shape : ',to_append_batch.shape)
            #         to_append.append(to_append_batch)

            #     to_append = torch.cat(to_append, dim = 0)
            #     # print('to_append shape : ',to_append.shape)


            #     c_maps.append(to_append)

                ####################################################

                
            ############################################################################
            ############################################################################
            # RDMs

            # c_tensor = torch.cat(c_maps, dim = 1)
            # c_numpy = c_tensor.cpu().numpy()
            # c_numpy = c_numpy - np.min(c_numpy)
            # c_numpy = c_numpy/np.max(c_numpy)

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
            #     key_name = 'c1_scale_' + str(int(MNIST_Scale*1000)) #+ '_cat_' + str(category)
            #     print('key_name : ',key_name)
            # elif self.c2_bool:
            #     key_name = 'c2_scale_' + str(int(MNIST_Scale*1000)) #+ '_cat_' + str(category)
            #     print('key_name : ',key_name)
            # elif self.c3_bool:
            #     key_name = 'c3_scale_' + str(int(MNIST_Scale*1000)) #+ '_cat_' + str(category)
            #     print('key_name : ',key_name)
            # elif self.c2b_bool:
            #     key_name = 'c2b_scale_' + str(int(MNIST_Scale*1000)) #+ '_cat_' + str(category)
            #     print('key_name : ',key_name)

            #     # # Option 2
            #     # c_stack = torch.stack(c_maps, dim = 4)
            #     # to_append, _ = torch.max(c_stack, dim=4)
            #     # c_maps = [to_append]

            #     # Option 3
            #     x = torch.stack(c_maps, dim=4)

            #     # Maxpool over scale groups
            #     to_append, _ = torch.max(x, dim=4)

            #     c_maps = [to_append]

            # if key_name in filters_data:
            #     filters_data[key_name] = np.concatenate([filters_data[key_name], c_numpy], axis = 0)
            # else:
            #     filters_data[key_name] = c_numpy
            
            # open_file = open(file_name, "wb")
            # pickle.dump(filters_data, open_file)
            # open_file.close()

            

            
            ############################################################################
            ############################################################################
            # Visualizing FIlters

            # if (self.c1_bool or self.c2_bool) and batch_idx == 0:
            #     # combined_vertical_image = torch.mean(x_input[8][:], dim = 0)
            #     combined_vertical_image = x_input[-5][0]

            #     combined_vertical_image = combined_vertical_image - torch.min(combined_vertical_image)
            #     combined_vertical_image = (combined_vertical_image/torch.max(combined_vertical_image))

            #     print('pre combined_vertical_image : ',combined_vertical_image.shape)
            #     combined_vertical_image = combined_vertical_image.cpu().numpy()
            #     # CxHxW --> HxWxC
            #     combined_vertical_image = combined_vertical_image.transpose(1,2,0)
            #     combined_vertical_image = combined_vertical_image[:,:]*255.0
            #     combined_vertical_image = combined_vertical_image.astype('uint8')
            #     combined_vertical_image = cv2.copyMakeBorder(combined_vertical_image,3,3,3,3,cv2.BORDER_CONSTANT,value=255)
                
            #     if same_scale_viz:
            #         ori_size = (base_scale, base_scale)
            #     else:
            #         # ori_size = (x_input[0][0].shape[-1], x_input[0][0].shape[-1])
            #         ori_size = (x_input[-5][0].shape[-1], x_input[-5][0].shape[-1])

            #     # print('ori_size : ',ori_size)

                
            #     for s_i in range(1, len(c_maps)+1):
            #         scale_maps = c_maps[-s_i]

            #         # scale_maps = scale_maps - torch.min(scale_maps)
            #         # scale_maps = scale_maps/torch.max(scale_maps)

            #         scale_maps = pad_to_size(scale_maps, ori_size)
            #         scale_maps_clone = scale_maps.clone()
            #         # scale_maps_clone = torch.mean(scale_maps_clone[:], dim = 0)
            #         scale_maps_clone = scale_maps_clone[0]


            #         for f_i, filter_maps in enumerate(scale_maps_clone):
            #             filter_maps = filter_maps - torch.min(filter_maps)
            #             filter_maps = filter_maps/torch.max(filter_maps)

            #             filter_maps_numpy = filter_maps.cpu().numpy()
            #             # CxHxW --> HxWxC
            #             filter_maps_numpy = filter_maps_numpy.reshape(1, *filter_maps_numpy.shape)
            #             filter_maps_numpy = filter_maps_numpy.transpose(1,2,0)
            #             # filter_maps_numpy = filter_maps_numpy - np.min(filter_maps_numpy)
            #             # filter_maps_numpy = filter_maps_numpy/np.max(filter_maps_numpy)
            #             filter_maps_numpy = filter_maps_numpy*255.0
            #             filter_maps_numpy = filter_maps_numpy.astype('uint8')
            #             filter_maps_numpy = cv2.copyMakeBorder(filter_maps_numpy,3,3,3,3,cv2.BORDER_CONSTANT,value=255)

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

            #     out_path = os.path.join(job_dir, "{}_filters_temp.npy".format(int(MNIST_Scale*1000)))
            #     # print('out_path : ',out_path)
            #     np.save(out_path, combined_vertical_image)


           
        if not self.global_pool: 
            return c_maps #, overall_max_scale_index
        else:
            return c_maps, max_scale_index


class S2(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride):
        super(S2, self).__init__()

        if type(kernel_size) == int:
            self.kernel_size = [kernel_size]

            setattr(self, f's_0', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size, stride),
                                                # nn.BatchNorm2d(channels_out, 1e-3),
                                                # nn.ReLU(True)
                                                ))

            # setattr(self, f's_uniform_0', nn.Conv2d(channels_in, channels_out, kernel_size, stride, bias=False))
            # s2b_uniform = getattr(self, f's_uniform_0')
            # nn.init.constant_(s2b_uniform.weight, 1)
            # for param in s2b_uniform.parameters():
            #     param.requires_grad = False


        elif type(kernel_size) == list:
            self.kernel_size = kernel_size
            self.kernel_size.sort()
            for i in range(len(kernel_size)):
                
                setattr(self, f's_{i}', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size[i], stride, dilation = 1),
                                                    #  nn.BatchNorm2d(channels_out, 1e-3),
                                                    #  nn.ReLU(True)
                                                    ))

                # setattr(self, f's_uniform_{i}', nn.Conv2d(channels_in, channels_out, kernel_size[i], stride, bias=False))
                # s2b_uniform = getattr(self, f's_uniform_{i}')
                # nn.init.constant_(s2b_uniform.weight, 1)
                # for param in s2b_uniform.parameters():
                #     param.requires_grad = False

            # self.batchnorm_s2b = nn.BatchNorm2d(channels_out*len(kernel_size), 1e-3)

        self.batchnorm = nn.BatchNorm2d(channels_out, 1e-3)


    def forward(self, x_pyramid):
        

        # Convolve each kernel with each scale band
        s_maps_per_k = []
        for k in range(len(self.kernel_size)):
            s_maps_per_i = []
            layer = getattr(self, f's_{k}')

            # if len(self.kernel_size) > 1:
            #     s2b_unorm_norm = getattr(self, f's_uniform_{k}')

            for i in range(len(x_pyramid)):  # assuming S is last dimension

                x = x_pyramid[i]
                s_map = layer(x)

                # # ############################################
                if k != 0:
                    # s_map = s_map / ((self.kernel_size[k]/self.kernel_size[0]))       # Sqrt
                    s_map = s_map / ((self.kernel_size[k]/self.kernel_size[0])**2)  # Square (Default)
                    # s_map = s_map / ((self.kernel_size[k]/self.kernel_size[0])**3)  # Cube

                # # if len(self.kernel_size) > 1:
                # #     s2b_unorm = torch.sqrt(abs(s2b_unorm_norm(x**2)))
                # #     # s2b_unorm = torch.sqrt(s2b_unorm(x))
                # #     s2b_unorm.data[s2b_unorm == 0] = 1  # To avoid divide by zero
                # #     s_map = s_map / s2b_unorm
                # ############################################

                # s_map = self.batchnorm(s_map)

                # TODO: think about whether the resolution gets too small here
                ori_size = x.shape[2:4]
                s_map = pad_to_size(s_map, ori_size)
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

        # s_maps = torch.stack(s_maps, dim = 0)
        # s_maps = s_maps.permute(1,0,2,3,4)
        # B, S, C, H, W = s_maps.shape
        # s_maps = s_maps.reshape(B, S*C, H, W)
        # s_maps = F.normalize(s_maps, p = 2, dim = 1)
        # s_maps = s_maps.reshape(B, S, C, H, W)
        # s_maps = s_maps.permute(1,0,2,3,4)

        # B, C, H, W = s_maps[0].shape
        # for s_i in range(len(s_maps)):  # 
        #     s_maps[s_i] = s_maps[s_i].reshape(B, C, H, W)
        #     s_maps[s_i] = self.batchnorm_s2b(s_maps[s_i])



        return s_maps


class S3(S2):
    # S3 does the same thing as S2
    pass


class HMAX_IP_basic_single_band(nn.Module):
    def __init__(self,
                 ip_scales = 18,
                 s1_scale=23, #25 #23 #21 #19 #17 #15 #13 #11 # 7, #5
                 s1_la=12.7, #14.1 #12.7 #11.5 #10.3 #9.1 #7.9 #6.8 #5.6 # 3.5, # 2.5
                 s1_si=10.2, #11.3 #10.2 #9.2 #8.2 #7.3 #6.3 #5.4 #4.5 # 2.8, # 2
                 n_ori=4,
                 num_classes=1000,
                 s1_trainable_filters=False,
                 visualize_mode = False,
                 prj_name = None,
                 MNIST_Scale = None,
                 category = None,
                 single_scale_bool = False,
                 ):
        super(HMAX_IP_basic_single_band, self).__init__()

        # ip_scales = 1 # 18

        # A few settings
        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters
        self.MNIST_Scale = MNIST_Scale
        self.category = category
        self.prj_name = prj_name
        self.single_scale_bool = single_scale_bool
        self.scale = 4

        self.same_scale_viz = None
        self.base_scale = None
        self.orcale_bool = None
        

        ########################################################
        ########################################################
        # For scale C1
        # # When we have 17 scales in C1 :: [31, 27, 23, 20, 18, 16, 14, 12, 10, 9, 8, 7, 6, 5, 5, 4, 4]
        # base_filt_size = 10 #17
        # # self.c1_sp_kernel_sizes = [int(np.ceil(base_filt_size/(2**(i/scale)))) for i in range(ip_scales)]
        # filt_scales_down = [np.ceil(base_filt_size/(2**(i/self.scale)))for i in range(int(np.ceil(ip_scales/2)))]
        # filt_scales_up = [np.ceil(base_filt_size*(2**(i/self.scale)))for i in range(1, int(np.ceil(ip_scales/2)))]

        # filt_scales = filt_scales_down + filt_scales_up
        # index_sort = np.argsort(filt_scales)
        # index_sort = index_sort[::-1]
        # self.c1_sp_kernel_sizes = [int(filt_scales[i_s]) for i_s in index_sort]

        # self.c1_sp_kernel_sizes = [6, 5,5,5,5,5,5,5,5,5,5,5,5,5,5]
        # self.c1_sp_kernel_sizes = [10,8,8,8,8,8,8,8,8,8,8,8,8,8]
        # self.c1_sp_kernel_sizes = [12,10,10,10,10,10,10,10,10,10,10,10,10,10]
        # self.c1_sp_kernel_sizes = [14,12,12,12,12,12,12,12,12,12,12,12,12,12]
        # self.c1_sp_kernel_sizes = [16,13,13,13,13,13,13,13,13,13,13,13,13,13]
        # self.c1_sp_kernel_sizes = [18,15,15,15,15,15,15,15,15,15,15,15,15,15]
        # self.c1_sp_kernel_sizes = [20,17,17,17,17,17,17,17,17,17,17,17,17,17]

        self.c1_sp_kernel_sizes = [22,18]
        ########################################################
        # For scale C2
        # Option 1:
        # # When we have 17 scales in C1 :: [28, 24, 21, 18, 16, 14, 12, 11, 9, 8, 7, 6, 6, 5, 4, 4, 3]
        # base_filt_size = filt_scales_down[1]
        # # self.c2_sp_kernel_sizes = [int(np.ceil(base_filt_size/(2**(i/self.scale)))) for i in range(ip_scales)]

        # filt_scales_down = [np.ceil(base_filt_size/(2**(i/self.scale)))for i in range(int(np.ceil(ip_scales/2)))]
        # filt_scales_up = [np.ceil(base_filt_size*(2**(i/self.scale)))for i in range(1, int(np.ceil(ip_scales/2)))]

        # filt_scales = filt_scales_down + filt_scales_up
        # index_sort = np.argsort(filt_scales)
        # index_sort = index_sort[::-1]
        # self.c2_sp_kernel_sizes = [int(filt_scales[i_s]) for i_s in index_sort]

        self.c2_sp_kernel_sizes = [18,15]

        # Option 2:
        #  self.c2_sp_kernel_sizes = [9]*(ip_scales-1)
        ########################################################
        ########################################################
        # Reverese
        # self.c1_sp_kernel_sizes.reverse()
        # self.c2_sp_kernel_sizes.reverse()

        # If it is a single scale band training
        if self.single_scale_bool:
            # SIngle Scale
            self.c1_sp_kernel_sizes = [self.c1_sp_kernel_sizes[0]]
            self.c2_sp_kernel_sizes = [self.c2_sp_kernel_sizes[0]]

            self.c_scale_stride = 1
            self.c_num_scales_pooled = 1

            self.ip_scales = 1

            # Single Scaleband
            # self.c1_sp_kernel_sizes = [self.c1_sp_kernel_sizes[0], self.c1_sp_kernel_sizes[1]]
            # self.c2_sp_kernel_sizes = [self.c2_sp_kernel_sizes[0], self.c2_sp_kernel_sizes[1]]

            # self.c_scale_stride = 1
            # self.c_num_scales_pooled = 2

            # self.ip_scales = 2

        else:
            self.c_scale_stride = 1
            self.c_num_scales_pooled = 2

            self.ip_scales = ip_scales

        print('c1_sp_kernel_sizes : ',self.c1_sp_kernel_sizes)
        print('c2_sp_kernel_sizes : ',self.c2_sp_kernel_sizes)
    
        ########################################################
        ########################################################
        # Setting the scale stride and number of scales pooled at a time
        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled

        self.c2_scale_stride = self.c_scale_stride
        self.c2_num_scales_pooled = self.c_num_scales_pooled
        
        # Global pooling (spatially)
        self.c2b_scale_stride = self.c_scale_stride
        self.c2b_num_scales_pooled = ip_scales-1 #len(self.c1_sp_kernel_sizes)  # all of them

        # Global pooling (spatially)
        self.c3_scale_stride = self.c_scale_stride
        self.c3_num_scales_pooled = ip_scales-2 #len(self.c2_sp_kernel_sizes)  # all of them

        ########################################################
        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters = False, #s1_trainable_filters,
                     la=self.s1_la, si=self.s1_si, visualize_mode = visualize_mode, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.2, n_in_sbands=ip_scales,
                    num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride, visualize_mode = visualize_mode, \
                    c1_bool = True, prj_name = self.prj_name, MNIST_Scale = self.MNIST_Scale)
        
        self.s2b = S2(channels_in=n_ori, channels_out=100, kernel_size=[4, 8, 12, 16], stride=1)
        # self.s2b = S2(channels_in=n_ori, channels_out=100, kernel_size=[3, 5, 7, 9], stride=1)

        self.c2b = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride, c2b_bool = True, prj_name = self.prj_name)
    
        ########################################################

        # # Classifier
        self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        nn.Linear(self.get_s4_in_channels(), 256),  # fc1
                                        # nn.BatchNorm1d(4096, 1e-3),
                                        # nn.ReLU(True),
                                        # nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        # nn.Linear(4096, 4096),  # fc2
                                        # nn.BatchNorm1d(256, 1e-3),
                                        # nn.ReLU(True),
                                        nn.Linear(256, num_classes)  # fc3
                                        )

        # self.overall_max_scale_index = []


    def get_s4_in_channels(self):
       
        c2b_out = len(self.s2b.kernel_size) * self.s2b.s_0[0].weight.shape[0]
        s4_in = c2b_out

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

        base_image_size = int(x.shape[-1]) 
        scale = self.scale #5
        
        if self.ip_scales == 1:
            image_scales_down = [base_image_size]
            image_scales_up = []
        elif self.ip_scales == 2:
            image_scales_up = [base_image_size, np.ceil(base_image_size*(2**(1/scale)))]
            image_scales_down = []
        else:
            image_scales_down = [np.ceil(base_image_size/(2**(i/scale))) for i in range(int(np.ceil(self.ip_scales/2)))]
            image_scales_up = [np.ceil(base_image_size*(2**(i/scale))) for i in range(1, int(np.ceil(self.ip_scales/2)))]
        

        image_scales = image_scales_down + image_scales_up
        index_sort = np.argsort(image_scales)
        index_sort = index_sort[::-1]
        self.image_scales = [image_scales[i_s] for i_s in index_sort]

        # print('image_scales : ',self.image_scales)

        if same_scale_viz:
            base_image_size = base_scale
        else:
            base_image_size = int(x.shape[-1]) 

        if len(self.image_scales) > 1:
            # print('Right Hereeeeeee: ', self.image_scales)
            image_pyramid = []
            for i_s in self.image_scales:
                i_s = int(i_s)
                # print('i_s : ',i_s)
                interpolated_img = F.interpolate(x, size = (i_s, i_s), mode = 'bilinear').clamp(min=0, max=1)
                # kernel_size_gauss = int(3*(i_s/base_image_size))
                # if kernel_size_gauss%2 == 0:
                #     kernel_size_gauss = kernel_size_gauss + 1
                # interpolated_img = torchvision.transforms.functional.gaussian_blur(interpolated_img, kernel_size_gauss, sigma = (7/8)*(i_s/base_image_size)).clamp(min=0, max=1)
                # if i_s <= base_image_size:
                #     pad_input = pad_to_size(interpolated_img, (base_image_size, base_image_size))
                #     image_pyramid.append(pad_input) # ??? Wgats is the range?
                # elif i_s > base_image_size:
                #     center_crop = torchvision.transforms.CenterCrop(base_image_size)
                #     image_pyramid.append(center_crop(interpolated_img))
                # # else:
                # #     image_pyramid.append(x)
                image_pyramid.append(interpolated_img)

                # # print('image_pyramid : ',image_pyramid[-1].shape,' ::: i_s : ',i_s,' ::: base_image_size : ',base_image_size)

            return image_pyramid
        else:
            # print('Hereeeeeee')
            ##############################################################################
            if self.orcale_bool:
                # FOr oracle:
                if x.shape[-1] > 224:
                    center_crop = torchvision.transforms.CenterCrop(224)
                    x = center_crop(x)
                elif x.shape[-1] < 224:
                    x = pad_to_size(x, (224, 224))
            ##############################################################################

            return [x]

    def forward(self, x, batch_idx = None):

        if x.shape[1] == 3:
            # print('0 1 : ',torch.equal(x[:,0:1], x[:,1:2]))
            # print('1 2 : ',torch.equal(x[:,1:2], x[:,2:3]))
            x = x[:,0:1]

        ###############################################
        x_pyramid = self.make_ip(x, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale) # Out 17 Scales x BxCxHxW --> C = 3

        ###############################################
        s1_maps = self.s1(x_pyramid, self.MNIST_Scale, batch_idx, prj_name = self.prj_name) # Out 17 Scales x BxCxHxW --> C = 4
        c1_maps = self.c1(s1_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, c1_sp_kernel_sizes = self.c1_sp_kernel_sizes, c2_sp_kernel_sizes = self.c2_sp_kernel_sizes, image_scales = self.image_scales)  # Out 16 Scales x BxCxHxW --> C = 4

        ###############################################
        # ByPass Route
        s2b_maps = self.s2b(c1_maps) # Out 15 Scales x BxCxHxW --> C = 2000
        c2b_maps, max_scale_index = self.c2b(s2b_maps, x_pyramid, self.MNIST_Scale, batch_idx, self.category, self.prj_name, same_scale_viz = self.same_scale_viz, base_scale = self.base_scale, image_scales = self.image_scales) # Overall x BxCx1x1 --> C = 2000

        ###############################################
        c2b_maps = torch.flatten(c2b_maps[0], 1) # Shape --> 1 x B x 400 x 1 x 1 

        # # Classify
        output = self.classifier(c2b_maps)

        # ###############################################
        # # Global Max Pooling
        # # s4_maps = F.max_pool2d(s4_maps, s4_maps.shape[-1], 1)
        # # Global Avg Pooling
        # s4_maps = F.avg_pool2d(s4_maps, s4_maps.shape[-1], 1)

        # s4_maps = s4_maps.squeeze()
        
        # # print()
        # output = self.classifier(s4_maps)


        return output , max_scale_index

#############################################################################
#############################################################################

