#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -J color_CNN
##SBATCH -C quadrortx
#SBATCH --constraint=v100
#SBATCH -o /cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/logs/MI_%A_%a_%J.out
#SBATCH -e /cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/logs/MI_%A_%a_%J.err
##SBATCH --account=carney-tserre-condo
##SBATCH --array=0-1

##SBATCH -p gpu

cd /cifs/data/tserre/CLPS_Serre_Lab/aarjun1/hmax_pytorch/

module load anaconda/3-5.2.0
module load python/3.5.2
# module load opencv-python/4.1.0.25
# module load cuda
# module load cudnn/8.1.0
# module load cuda/11.1.1

source activate color_CNN

echo $SLURM_ARRAY_TASK_ID

# python -u file_management.py #--job_number $SLURM_JOB_ID --gpu_index $CUDA_VISIBLE_DEVICES #-n 1 -v single_illuminant -j $SLURM_ARRAY_TASK_ID

# python -u dataloaderr.py #--job_number $SLURM_JOB_ID --gpu_index $CUDA_VISIBLE_DEVICES #-n 1 -v single_illuminant -j $SLURM_ARRAY_TASK_ID

python -u trainer.py #--job_number $SLURM_JOB_ID --gpu_index $CUDA_VISIBLE_DEVICES #-n 1 -v single_illuminant -j $SLURM_ARRAY_TASK_ID


# python create_mydataset.py
