#!/bin/bash
#SBATCH --job-name=acopf_ws
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --array=0-99%10
#SBATCH --output=/home/g202210120/projects/ACOPF-feasible/logs/ws_%A_%a.out
#SBATCH --error=/home/g202210120/projects/ACOPF-feasible/logs/ws_%A_%a.err
#SBATCH --chdir=/home/g202210120/projects/ACOPF-feasible

# ============================================================
# Arguments
# ============================================================

CASE_NAME=""
BUS_NUMBER=""
MODEL_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            CASE_NAME="$2"
            shift 2
            ;;
        --bus)
            BUS_NUMBER="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$CASE_NAME" || -z "$BUS_NUMBER" ]]; then
    echo "Usage:"
    echo "sbatch submit_warmstart_hpc.sh --case <case_name> --bus <bus_number> [--model-dir <path>]"
    exit 1
fi

# ============================================================
# Directories
# ============================================================

mkdir -p ./logs
mkdir -p ./result

# ============================================================
# Python environment
# Use the exact Python interpreter from the pytorch environment.
# No conda activation is required.
# ============================================================

PYTHON="/home/g202210120/.conda/envs/pytorch/bin/python"

echo "============================================================"
echo "Array Job ID : $SLURM_ARRAY_JOB_ID"
echo "Task ID      : $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Working dir  : $(pwd)"
echo "Case         : $CASE_NAME"
echo "Bus          : $BUS_NUMBER"
echo "============================================================"

echo "Python executable:"
echo "$PYTHON"
"$PYTHON" --version

echo "Environment check:"
"$PYTHON" -c "import torch, pandas, pyomo; \
print('PyTorch:', torch.__version__); \
print('Pandas:', pandas.__version__); \
print('Pyomo:', pyomo.__version__)"

# ============================================================
# CPU threading
# ============================================================

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# ============================================================
# Array chunk
#
# 100 tasks x 10 test instances = 1000 test instances
# ============================================================

CHUNK_SIZE=10

START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$((START_IDX + CHUNK_SIZE))
CHUNK_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")

echo "Start index : $START_IDX"
echo "End index   : $END_IDX"
echo "Chunk ID    : $CHUNK_ID"
echo "Start time  : $(date)"

# ============================================================
# Build Python command
# ============================================================

CMD=(
    "$PYTHON"
    evaluate_warmstart_qcqp_array.py
    --case_name "$CASE_NAME"
    --bus_number "$BUS_NUMBER"
    --start_idx "$START_IDX"
    --end_idx "$END_IDX"
    --chunk_id "$CHUNK_ID"
)

if [[ -n "$MODEL_DIR" ]]; then
    CMD+=(--model_dir "$MODEL_DIR")
fi

echo "Command:"
printf '%q ' "${CMD[@]}"
echo

# ============================================================
# Run
# ============================================================

"${CMD[@]}"

EXIT_CODE=$?

echo "============================================================"
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit "$EXIT_CODE"