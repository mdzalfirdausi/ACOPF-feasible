#!/bin/bash
#SBATCH --job-name=acopf
#SBATCH --output=/home/g202210120/projects/ACOPF-feasible/logs/%j_%x.out
#SBATCH --error=/home/g202210120/projects/ACOPF-feasible/logs/%j_%x.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=gpu_x450
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu0002
##SBATCH --nodelist=gpu0003,gpu0004
#SBATCH --chdir=/home/g202210120/projects/ACOPF-feasible

# 1. Ensure a script was passed
if [ -z "$1" ]; then
    echo "ERROR: No script filename provided."
    echo "Usage: sbatch submit.sh <your_script_name.py> [arguments...]"
    exit 1
fi

# 2. Ensure directories exist
mkdir -p ./model
mkdir -p ./logs

# 3. Environment Setup
module load conda/25.08
source activate pytorch

# 4. Hardware/Environment Check
echo "Job: $SLURM_JOB_NAME"
echo "Executing: $@"
echo "Node: $(hostname)"
echo "GPU Allocated: $CUDA_VISIBLE_DEVICES"

# Force CUDA to trigger an error if the driver isn't responsive
python -c "import torch; assert torch.cuda.is_available(), 'CUDA check failed before execution!'"

# 5. Run the target script directly, passing ALL arguments ($@)
python "$@"