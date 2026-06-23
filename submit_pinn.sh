#!/bin/bash
#SBATCH --job-name=acopf_pinn
#SBATCH --output=/home/g202210120/projects/ACOPF-feasible/logs/pinn_%j.out
#SBATCH --error=/home/g202210120/projects/ACOPF-feasible/logs/pinn_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu_x450
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/g202210120/projects/ACOPF-feasible

# 1. Load any necessary modules (Uncomment and adjust if your HPC requires it)
module load cuda/12.9.lua
module load pytorch/24.08
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# 2. Activate your specific Python/PyTorch environment
source activate pinn_acopf

# 3. Print out hardware info to the log for verification
echo "Job started on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 4. Execute the high-speed CUDA training script
python ACOPF_pinn_baseline.py