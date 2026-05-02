#!/bin/bash
#SBATCH --job-name=premier_taco
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

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

python train_premier_taco_droid.py \
    exp_name=premier_taco_l40 \
    num_epochs=50 \
    batch_size=512 \
    lr=2e-4 \
    steps_per_epoch=3750 \
    steps_per_chunk=188

echo "End: $(date)"
