#!/bin/bash
#SBATCH --job-name=acopf
#SBATCH --output=/home/g202210120/projects/ACOPF-feasible/logs/%j_%x.out
#SBATCH --error=/home/g202210120/projects/ACOPF-feasible/logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu_x450
#SBATCH --gres=gpu:1
##SBATCH --exclude=gpu0002
##SBATCH --nodelist=gpu0003,gpu0004
#SBATCH --chdir=/home/g202210120/projects/ACOPF-feasible

# 1. Capture the filename passed after the sbatch command
# $1 is the first argument after sbatch
SCRIPT_NAME=$1

if [ -z "$SCRIPT_NAME" ]; then
    echo "ERROR: No script filename provided."
    echo "Usage: sbatch submit.sh <your_script_name.py>"
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
echo "Executing: $SCRIPT_NAME"
echo "Node: $(hostname)"
echo "GPU Allocated: $CUDA_VISIBLE_DEVICES"

# 5. Run the target script
srun conda run --no-capture-output -n pytorch python "$SCRIPT_NAME"