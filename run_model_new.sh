#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=20GB
#SBATCH -J hmax_pytorch
##SBATCH -C quadrortx
#SBATCH --constraint=a40
##SBATCH --constraint=v100
#SBATCH -o /cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/logs/MI_%A_%a_%J.out
#SBATCH -e /cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/logs/MI_%A_%a_%J.err
##SBATCH --account=carney-tserre-condo
##SBATCH --array=0-1

##SBATCH -p gpu

cd /cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch


module load anaconda/3-5.2.0
module load python/3.5.2
# module load opencv-python/4.1.0.25
# module load cuda
# module load cudnn/8.1.0
# module load cuda/11.1.1

source activate color_CNN

echo $SLURM_ARRAY_TASK_ID

python trainer_new.py \
            --hmaxify_bool True \
            --ip_scales 5 \
            --scale_factor 4 \
            --s1_scale 21 \
            --s1_la 11.5 \
            --s1_si 9.2 \
            --s1_stride 4 \
            --s1_trainable_filters True \
            --n_ori 4 \
            --n_phi 1 \
            --c1_use_bool False \
            --force_const_size_bool True \
            --c2b_scale_loss_bool False \
            --c2b_attention_weights_bool False \
            --pad_mode constant \
            --model_backbone AlexNet \
            --my_data True \
            --dataset_name MNIST \
            --image_size 224 \
            --optimizer Adam \
            --lr_scheduler None \
            --lr 1e-4 \
            --weight_decay 1e-4 \
            --num_epochs 500 \
            --batch_size_per_gpu 32 \
            --train_base_scale 2 \
            --train_scale_aug_range 2 \
            --warp_image_bool False \
            --train_mode True \
            --val_mode False \
            --test_mode False \
            --out_dir /cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/new_logs/ \
            # --c1_sp_kernel_sizes 10 8 \
            # --c1_spatial_sride_factor 0.5 \
            # --c_scale_stride 1 \
            # --c_num_scales_pooled 2 \
            # --s2b_channels_out 128 \
            # --s2b_kernel_size 4 8 12 16 \
            # --s2b_stride 1 \
            # --c2b_scale_loss_lambda 0 \




# python create_mydataset.py