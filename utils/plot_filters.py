import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torchvision
import cv2
import os
import _pickle as pickle

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
    
def plt_filter_func(args, x_input, filter_maps_all, prj_name, train_base_scale, stage):

    if stage == 'S1':
        ori_size = (x_input[0][0].shape[-1], x_input[0][0].shape[-1])
    else:
        ori_size = (x_input[-int(np.ceil(len(x_input)/2))][0].shape[-1], x_input[-int(np.ceil(len(x_input)/2))][0].shape[-1])

    combined_vertical_image = x_input[-int(np.ceil(len(x_input)/2))][0].clone()

    # combined_vertical_image = combined_vertical_image - torch.min(combined_vertical_image)
    # combined_vertical_image = (combined_vertical_image/torch.max(combined_vertical_image))

    if stage == 'S1':
        combined_vertical_image = pad_to_size(combined_vertical_image, ori_size)

    combined_vertical_image = combined_vertical_image.cpu().numpy()
    # CxHxW --> HxWxC
    combined_vertical_image = combined_vertical_image.transpose(1,2,0)
    combined_vertical_image = combined_vertical_image[:,:]*255.0
    combined_vertical_image = combined_vertical_image.astype('uint8')
    combined_vertical_image = cv2.copyMakeBorder(combined_vertical_image,3,3,3,3,cv2.BORDER_CONSTANT,value=255)

    
    for s_i in range(1, len(filter_maps_all)+1):
        scale_maps = filter_maps_all[-s_i].clone()

        # scale_maps = scale_maps - torch.min(scale_maps)
        # scale_maps = (scale_maps/torch.max(scale_maps))

        print('scale_maps : ',scale_maps.shape)

        scale_maps = pad_to_size(scale_maps, ori_size)
        # print('scale_maps : ',scale_maps.shape)
        scale_maps_clone = scale_maps.clone()
        scale_maps_clone = scale_maps_clone[0]

        if stage == 'S2b':
            filter_maps = torch.mean(scale_maps_clone[300:400], dim = 0)
        else:
            filter_maps = scale_maps_clone[0]

        filter_maps = filter_maps - torch.min(filter_maps)
        filter_maps = (filter_maps/torch.max(filter_maps))

        filter_maps_numpy = filter_maps.cpu().numpy()
        # CxHxW --> HxWxC
        filter_maps_numpy = filter_maps_numpy.reshape(1, *filter_maps_numpy.shape)
        filter_maps_numpy = filter_maps_numpy.transpose(1,2,0)
        # filter_maps_numpy = filter_maps_numpy - np.min(filter_maps_numpy)
        # filter_maps_numpy = (filter_maps_numpy/np.max(filter_maps_numpy))*255.0
        filter_maps_numpy = filter_maps_numpy*255.0
        filter_maps_numpy = filter_maps_numpy.astype('uint8')
        filter_maps_numpy = cv2.copyMakeBorder(filter_maps_numpy,3,3,3,3,cv2.BORDER_CONSTANT,value=255)

        combined_vertical_image = cv2.vconcat([combined_vertical_image, filter_maps_numpy])



    job_dir = os.path.join(args.fig_dir, 'visualize_filters/' + stage)
    os.makedirs(main_dir, exist_ok=True)

    # out_path = os.path.join(job_dir, "{}_filters_temp.png".format(self.train_base_scale))
    # cv2.imwrite(out_path, combined_image)

    out_path = os.path.join(job_dir, "{}_filters_temp.npy".format(int(1000*train_base_scale)))
    np.save(out_path, combined_vertical_image)


def s1_kernel_viz_func(args, model, title):

    filter_sizee_list = [args.s1_scale]

    for filter_sizee in filter_sizee_list:
        if title == 'learned':
            s1_filters = getattr(model.s1, f's_{filter_sizee}').weight.data
        elif title == 'original':
            s1_filters = model.s1.gabor_filter
        # print('s1_filters : ',s1_filters.shape)
        s1_filters = s1_filters.squeeze()
        print('s1_filters : ',s1_filters.shape)

        save_dir = os.path.join(args.fig_dir, 's1_kernels')
        os.makedirs(save_dir, exist_ok=True)

        s1_filt_list = []
        for filt_ind, s1_filt in enumerate(s1_filters):
            image_name = str(filt_ind) + '.png'
            s1_filt_3d = s1_filt.permute(1,2,0).reshape(filter_sizee, filter_sizee, 3)
            s1_filt_3d = s1_filt_3d.cpu().numpy()

            s1_filt_list.append(s1_filt_3d)
            # cv2.imwrite(os.path.join(save_dir, image_name), s1_filt_3d*255.0)

        for v_i in range(args.n_ori):
            for h_i in range(args.n_phi):
                if h_i == 0:
                    s1_filt_list_pad = cv2.copyMakeBorder(s1_filt_list[h_i*args.n_ori + v_i],1,1,1,1,cv2.BORDER_CONSTANT,value=[1,1,1])
                    hori_img = s1_filt_list_pad
                else:
                    s1_filt_list_pad = cv2.copyMakeBorder(s1_filt_list[h_i*args.n_ori + v_i],1,1,1,1,cv2.BORDER_CONSTANT,value=[1,1,1])
                    hori_img = cv2.hconcat([hori_img, s1_filt_list_pad])

            if v_i == 0: 
                vertical_img = hori_img
            else:
                vertical_img = cv2.vconcat([vertical_img, hori_img])

        print('vertical_img shape : ',vertical_img.shape)
        vertical_img = cv2.resize(vertical_img, (vertical_img.shape[1]*5,vertical_img.shape[0]*5))
        # vertical_img = vertical_img/np.max(vertical_img)
        vertical_img = (vertical_img - np.min(vertical_img)) / (np.max(vertical_img) - np.min(vertical_img)) # normalize (min-max)
        image_name = f'{title}_collage_{filter_sizee}_plt.png'

        cv2.imwrite(os.path.join(save_dir, image_name), vertical_img*255.0)