#!/bin/bash

#SBATCH --job-name=acopf162
#SBATCH --output=/home/g202210120/projects/ACOPF-feasible/logs/%j_%x.out
#SBATCH --error=/home/g202210120/projects/ACOPF-feasible/logs/%j_%x.err

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

#SBATCH --partition=gpu_x450
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu0002

#SBATCH --chdir=/home/g202210120/projects/ACOPF-feasible

# ============================================================
# Controlled Ablation - 162 Bus
# Generic single-run Slurm submission script
#
# Usage:
#
# sbatch --job-name=<name> submit_controlled_ablation_162.sh \
#     --projection <projection> \
#     --weights <weights> \
#     --seed <seed>
# ============================================================

set -e


# ------------------------------------------------------------
# 1. Ensure directories exist
# ------------------------------------------------------------

mkdir -p ./model
mkdir -p ./logs


# ------------------------------------------------------------
# 2. Environment setup
# ------------------------------------------------------------

module load conda/25.08

eval "$(conda shell.bash hook)"

# Activate explicit environment path
conda activate /home/g202210120/.conda/envs/pytorch

# Use explicit Python executable.
# This avoids the PATH problem where `python` resolved to
# /software/conda/bin/python instead of the pytorch environment.
PYTHON=/home/g202210120/.conda/envs/pytorch/bin/python


# ------------------------------------------------------------
# 3. Job information
# ------------------------------------------------------------

echo "============================================================"
echo "CONTROLLED ABLATION - 162 BUS"
echo "============================================================"

echo "Job ID             : $SLURM_JOB_ID"
echo "Job Name           : $SLURM_JOB_NAME"
echo "Node               : $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo
echo "Python executable  : $PYTHON"
$PYTHON --version

echo
echo "Arguments          : $@"

echo "============================================================"


# ------------------------------------------------------------
# 4. PyTorch / CUDA verification
# ------------------------------------------------------------

$PYTHON -c "
import torch

print('PyTorch version :', torch.__version__)
print('CUDA runtime    :', torch.version.cuda)
print('CUDA available  :', torch.cuda.is_available())

assert torch.cuda.is_available(), \
    'CUDA check failed before training!'

print('GPU              :', torch.cuda.get_device_name(0))
print('GPU count        :', torch.cuda.device_count())
"


# ------------------------------------------------------------
# 5. Run controlled ablation
#
# Fixed parameters:
#   case   = 162-bus
#   epochs = 10000
#
# Variable parameters supplied through sbatch:
#   --projection
#   --weights
#   --seed
# ------------------------------------------------------------

echo
echo "============================================================"
echo "STARTING TRAINING"
echo "Start time: $(date)"
echo "============================================================"


$PYTHON ACOPF_controlled_ablation.py \
    --case_name pglib_opf_case162_ieee_dtc \
    --epochs 10000 \
    "$@"


STATUS=$?


# ------------------------------------------------------------
# 6. Completion
# ------------------------------------------------------------

echo
echo "============================================================"

if [ "$STATUS" -eq 0 ]; then

    echo "TRAINING COMPLETED SUCCESSFULLY"

else

    echo "TRAINING FAILED"
    echo "Exit code: $STATUS"

fi

echo "End time: $(date)"
echo "============================================================"

exit "$STATUS"