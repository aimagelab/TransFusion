#!/bin/bash -l

#SBATCH --job-name=ft_openclip_vitb32
#SBATCH --partition=all_usr_prod  
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --constraint="gpu_A40_48G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
##SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --time=20:00:00
#SBATCH --account=debiasing
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate fairclip


RESULT_DIR=/work/debiasing/frinaldi/clip-finetuned-weights/${DATASET}/

DATASET=resisc45
MODEL_ARCH=ViT-L-14
NUM_STEPS=2000
LR=1e-5
WD=1e-1
BATCH_SIZE=32
PRETRAINING=datacomp_l_s1b_b8k

# PERF_FILE=stats.txt

srun python src/finetune_openCLIP.py \
		--result_dir ${RESULT_DIR}/openclip_${MODEL_ARCH}-${PRETRAINING}-b${BATCH_SIZE}_s${NUM_STEPS}_lr${LR}_wd${WD} \
		--lr ${LR} \
		--batch_size ${BATCH_SIZE} \
		--num_steps ${NUM_STEPS} \
		--weight_decay ${WD} \
		--pretraining ${PRETRAINING} \
		--dataset ${DATASET} \
		--wandb_run_name openclip_${MODEL_ARCH}-${PRETRAINING}-b${BATCH_SIZE}_s${NUM_STEPS}_lr${LR}_wd${WD} \
		--wandb_project Finetuning_openclip_${DATASET} \
