#!/bin/bash
#SBATCH --job-name=premier_taco_ddp
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_ddp_%j.out
#SBATCH --error=logs/slurm_ddp_%j.err

# Activate conda environment
source activate premier_taco

# Working directory
cd /share/home/u06094/zzx_workspace/premier-taco/premier-taco

# Create log directory
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi

torchrun --nproc_per_node=2 \
    train_premier_taco_droid_ddp.py \
    exp_name=premier_taco_l40_2gpu \
    num_epochs=50 \
    batch_size=256 \
    lr=2e-4

echo "End: $(date)"
