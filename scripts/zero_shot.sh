#!/bin/bash -l

#SBATCH --job-name=CLIP_zero_shot_evaluation_cifar_openclip_xl_ta1_att_grid_k=theta_a1
#SBATCH --partition=all_usr_prod  
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --constraint="gpu_A40_48G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --time=10:00:00
#SBATCH --account=debiasing
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate fairclip


DATASET_DIR_GL=/work/debiasing/frinaldi/harvard-fairvlmed
DATASET_DIR_AMD=/work/debiasing/frinaldi/harvard-fairvision/dataset-001
MODEL_ARCH=vit-b32  # Options: vit-b16 | vit-l14
MODALITY_TYPE='slo_fundus'
LR=1e-5
BATCH_SIZE=32

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}.csv

srun python zero_shot_evaluation.py \
		--lr ${LR} \
		--model_arch ${MODEL_ARCH} \
		--interpolate_weights \
		--wandb_project Evaluation_rebasin_task_vector \
		--wandb_group_name cifar100 \
		--wandb_run_name openclip_xl_ta1_att_k=theta_a_normalized \
		# --save_best_model /work/debiasing/frinaldi/clip-finetuned-weights/task_vectors \