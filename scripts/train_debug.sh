#!/bin/bash
#SBATCH --job-name=hsgan_debug
#SBATCH --output=logs/hsgan_debug_%j.out
#SBATCH --error=logs/hsgan_debug_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=00:30:00

echo "Running on node: $(hostname)"

module load python/3.11

source /users/iviwebooi/HSGAN_Project/hsgan_env/bin/activate

cd /users/iviwebooi/HSGAN_Project/HSGAN

echo "Python path: $(which python)"
python --version
python -c "import torch; print('Torch ok:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
echo "Starting training now..."
python -u train.py \
  --data_type arad \
  --process_type gen \
  --network_type hsgan \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --single_gpu \
  --gpu_ids 0 \
  --save_by_epoch 1 \
  --baseroot_train /cbio/projects/046/iviwe/NTIRE/train \
  --baseroot_val /cbio/projects/046/iviwe/NTIRE/val \
  --crop_size 256
