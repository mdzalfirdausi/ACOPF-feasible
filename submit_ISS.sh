#!/bin/bash
#SBATCH --partition=main
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/dev/null   # Suppress default slurm-%j.out creation
#SBATCH --error=/dev/null    # Suppress default slurm-%j.err creation

# 1. Ensure a script was passed
if [ -z "$1" ]; then
    echo "ERROR: No script filename provided."
    exit 1
fi

# 2. Extract script name without extension (e.g., ACOPF_Hard_KKT)
SCRIPT_BASE=$(basename "$1" .py)

# 3. Create directories
mkdir -p ./model
mkdir -p ./logs

# 4. Dynamically redirect stdout & stderr using SLURM_JOB_ID
LOG_OUT="logs/${SLURM_JOB_ID}_${SCRIPT_BASE}.out"
LOG_ERR="logs/${SLURM_JOB_ID}_${SCRIPT_BASE}.err"

exec > "$LOG_OUT" 2> "$LOG_ERR"

# --- Rest of your script runs normally below ---
source ~/miniconda3/bin/activate pytorch

echo "Job ID: $SLURM_JOB_ID"
echo "Executing: $@"
echo "Node: $(hostname)"
echo "GPU Allocated: $CUDA_VISIBLE_DEVICES"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA check failed!'"
python -u "$@"