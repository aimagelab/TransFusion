#!/bin/bash -l

#SBATCH --job-name=loss-barrier-vitb-16-csv
#SBATCH --partition=all_usr_prod  
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
##SBATCH --constraint="gpu_A40_48G|gpu_RTX5000_16G"
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"
#SBATCH --time=24:00:00
#SBATCH --account=debiasing
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate fairclip

python vit-finetune/main.py fit --config /homes/frinaldi/Merge2Update/vit-finetune/configs/full/cifar10.yaml