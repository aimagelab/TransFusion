#!/bin/bash -l
#SBATCH --job-name=dataset_distillation_angel_sign_loss
#SBATCH --partition=all_usr_prod  
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
##SBATCH --constraint="gpu_A40_48G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
##SBATCH --constraint="gpu_A40_48G"
#SBATCH --time=10:00:00
#SBATCH --account=debiasing
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate fairclip

srun python task_vector_analysis.py
