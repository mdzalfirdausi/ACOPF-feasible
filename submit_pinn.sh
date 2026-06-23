#!/bin/bash
#SBATCH --job-name=acopf_pinn
#SBATCH --output=/home/g202210120/projects/ACOPF_Benchmark/logs/pinn_%j.out
#SBATCH --error=/home/g202210120/projects/ACOPF_Benchmark/logs/pinn_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu_x450
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/g202210120/projects/ACOPF_Benchmark

# 1. Load any necessary modules (Uncomment and adjust if your HPC requires it)
# module load anaconda3/2022.05

# 2. Activate your specific Python/PyTorch environment
# source activate acopf_env 
# OR
# conda activate acopf_env

# 3. Print out hardware info to the log for verification
echo "Job started on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 4. Execute the high-speed CUDA training script
python train_baseline.py