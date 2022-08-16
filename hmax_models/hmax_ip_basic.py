import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torchvision


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
    def __init__(self, scale, n_ori, padding, trainable_filters, la, si):

        super(S1, self).__init__()

        self.scale = scale
        self.la = la
        self.si = si

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

    def forward(self, x_pyramid):
        s1_maps = []
        # Loop over scales, normalizing.
        for p_i in range(len(x_pyramid)):

            x = x_pyramid[p_i]

            s1_cell = getattr(self, f's_{self.scale}')
            s1_map = torch.abs(s1_cell(x))  # adding absolute value

            # s1_unorm = getattr(self, f's_uniform_{self.scale}')
            # s1_unorm = torch.sqrt(s1_unorm(x** 2))
            # s1_unorm.data[s1_unorm == 0] = 1  # To avoid divide by zero
            # s1_map /= s1_unorm
            # s1_map = self.batchnorm(s1_map)

            s1_maps.append(s1_map)

            # Padding (to get s1_maps in same size)
            # TODO: figure out if we'll ever be in a scenario where we don't need the same padding left/right or top/down
            # Or if the size difference is an odd number
            ori_size = (x.shape[-2], x.shape[-1])
            s1_maps[p_i] = pad_to_size(s1_maps[p_i], ori_size)


            # s1_maps = torch.stack(s1_maps, dim=4)

        return s1_maps


class C(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, global_pool, sp_kernel_size=[10, 8], sp_stride_factor=None, n_in_sbands=None,
                 num_scales_pooled=2, scale_stride=1,image_subsample_factor=1):

        super(C, self).__init__()
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

    def forward(self, x_pyramid):
        # TODO - make this whole section more memory efficient

        # print('####################################################################################')
        # print('####################################################################################')

        c_maps = []
        if not self.global_pool:
            for p_i in range(len(x_pyramid)-1):
                # print('############################')
                # print('x_2 : ',x_pyramid[p_i+1].shape,' :: x1 : ',x_pyramid[p_i].shape,' :::::::: ',p_i,' :: kernel_l ',self.sp_kernel_size)
                x_1 = F.max_pool2d(x_pyramid[p_i], self.sp_kernel_size[p_i], self.sp_stride[p_i])
                x_2 = F.max_pool2d(x_pyramid[p_i+1], self.sp_kernel_size[p_i+1], self.sp_stride[p_i+1])
                # print('x_2 : ',x_2.shape,' :: x1 : ',x_1.shape,' :::::::: ',p_i,' :: kernel_l ',self.sp_kernel_size)
                
                if x_2.shape[-1] >= x_1.shape[-1]:
                    center_crop = torchvision.transforms.CenterCrop(x_1.shape[-1])
                    x_2 = center_crop(x_2)
                else:
                    center_crop = torchvision.transforms.CenterCrop(x_2.shape[-1])
                    x_1 = center_crop(x_1)
                
                
                # assert x_2.shape[-1] >= x_1.shape[-1], "x_2 should have a bigger or equal height width"
                # x_2 = x_2[:,:,:x_1.shape[-2],:x_1.shape[-1]]

                # Maxpool over scale groups
                x = torch.stack([x_1, x_2], dim=4)

                to_append, _ = torch.max(x, dim=4)
                c_maps.append(to_append)
        else:
            scale_max = []
            for p_i in range(len(x_pyramid)):
                s_m = F.max_pool2d(x_pyramid[p_i], x_pyramid[p_i].shape[-1], 1)
                scale_max.append(s_m)

            x = torch.stack(scale_max, dim=4)

            # Maxpool over scale groups
            to_append, _ = torch.max(x, dim=4)

            c_maps.append(to_append)

            # for s_i in range(len(scale_max)-1):
            #     # Maxpool over scale groups
            #     x = torch.stack([scale_max[s_i], scale_max[s_i+1]], dim=4)

            #     to_append, _ = torch.max(x, dim=4)
            #     c_maps.append(to_append)

        return c_maps


class S2(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride):
        super(S2, self).__init__()
        if type(kernel_size) == int:
            self.kernel_size = [kernel_size]
            setattr(self, f's_0', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size, stride),
                                                # nn.BatchNorm2d(channels_out, 1e-3),
                                                nn.ReLU(True)))
        elif type(kernel_size) == list:
            self.kernel_size = kernel_size
            self.kernel_size.sort()
            for i in range(len(kernel_size)):
                setattr(self, f's_{i}', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size[i], stride),
                                                    #   nn.BatchNorm2d(channels_out, 1e-3),
                                                      nn.ReLU(True)))

    def forward(self, x_pyramid):
        # Evaluate input
        

        # Convolve each kernel with each scale band
        # TODO: think of something more elegant than for loop
        s_maps_per_k = []
        for k in range(len(self.kernel_size)):
            s_maps_per_i = []
            layer = getattr(self, f's_{k}')
            for i in range(len(x_pyramid)):  # assuming S is last dimension
                x = x_pyramid[i]
                s_map = layer(x)
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


class HMAX_IP_basic(nn.Module):
    def __init__(self,
                 ip_scales = 10,
                 s1_scale=11,
                 s1_la=5.6,
                 s1_si=4.5,
                 n_ori=4,
                 num_classes=1000,
                 s1_trainable_filters=False):
        super(HMAX_IP_basic, self).__init__()

        # A few settings
        self.ip_scales = ip_scales
        self.s1_scale = s1_scale
        self.s1_la = s1_la
        self.s1_si = s1_si
        self.n_ori = n_ori
        self.num_classes = num_classes
        self.s1_trainable_filters = s1_trainable_filters

        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled
        # self.c1_sp_kernel_sizes = [10, 8] 
        self.c1_sp_kernel_sizes = [10, 8, 6, 4, 3, 2, 1, 1, 1]

        ###############################################
        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scale=self.s1_scale, n_ori=n_ori, padding='valid', trainable_filters=s1_trainable_filters,
                     la=self.s1_la, si=self.s1_si)
        self.c1 = C(global_pool = False, sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.2, n_in_sbands=ip_scales,
                    num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride)
        self.s2 = S2(channels_in=n_ori, channels_out=500, kernel_size=[4, 8, 12, 15], stride=1)
        self.c2 = C(global_pool = True, sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=ip_scales-1,
                     num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c_scale_stride)


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
        self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
                                        # nn.BatchNorm1d(256, 1e-3),
                                        nn.Linear(2000, 500),
                                        nn.Linear(500, num_classes),  # fc1
                                        )

    def get_s4_in_channels(self):
        c1_out = (self.ip_scales-1) * self.n_ori
        c2_out = (self.ip_scales-2) * self.s2.s_0[0].weight.shape[0]
        # c3_out = (self.ip_scales-3) * self.s3.s_0[0].weight.shape[0]
        # c2b_out = len(self.s2b.kernel_size) * (self.ip_scales-2) * self.s2b.s_0[0].weight.shape[0]
        c3_out = self.s3.s_0[0].weight.shape[0]
        c2b_out = len(self.s2b.kernel_size) * self.s2b.s_0[0].weight.shape[0]
        s4_in = c1_out + c2_out + c3_out + c2b_out
        return s4_in

    def make_ip(self, x):

        base_image_size = int(x.shape[-1])
        image_scales = [np.ceil(base_image_size/(2**(i/4))) if np.ceil(base_image_size/(2**(i/4)))%2 == 0 else np.floor(base_image_size/(2**(i/4))) for i in range(self.ip_scales)]

        image_pyramid = []
        for i_s in image_scales:
            i_s = int(i_s)
            image_pyramid.append(F.interpolate(x, size = (i_s, i_s), mode = 'bicubic').clamp(min=0, max=1)) # ??? Wgats is the range?
        
        return image_pyramid

    def forward(self, x):

        ###############################################
        x_pyramid = self.make_ip(x) # Out 10 Scales x BxCxHxW --> C = 3

        ###############################################
        s1_maps = self.s1(x_pyramid) # Out 10 Scales x BxCxHxW --> C = 4
        c1_maps = self.c1(s1_maps)  # Out 9 Scales x BxCxHxW --> C = 4

        s2_maps = self.s2(c1_maps) # Out 9 Scales x BxCxHxW --> C = 2000
        c2_maps = self.c2(s2_maps) # Out 8 Scales x BxCxHxW --> C = 2000

        ###############################################
        s4_maps = c2_maps[0]

        # Global Max Pooling
        # s4_maps = F.max_pool2d(s4_maps, s4_maps.shape[-1], 1)
        # Global Avg Pooling
        s4_maps = F.avg_pool2d(s4_maps, s4_maps.shape[-1], 1)

        s4_maps = s4_maps.squeeze()
        
        # print()
        output = self.classifier(s4_maps)


        return output
