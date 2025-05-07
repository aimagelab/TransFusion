. /usr/local/anaconda3/etc/profile.d/conda.sh

srun -Q --immediate=400 -wailb-login-03 --cpus-per-task=1 --mem=5G --account=debiasing --partition=all_serial --gres=gpu:1 --time 4:00:00 --pty bash
conda activate fairclip