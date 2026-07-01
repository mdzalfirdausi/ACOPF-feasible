#!/bin/bash
#SBATCH --job-name=acopf_ISS
#SBATCH --output=/nfs/mfirdausi/redsea/ACOPF-feasible/logs/%j_%x.out
#SBATCH --error=/nfs/mfirdausi/redsea/ACOPF-feasible/logs/%j_%x.err
#SBATCH --partition=main
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

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