#!/bin/bash
#SBATCH --job-name=acopf_ws
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-99
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -u

CASE_NAME=""
BUS_NUMBER=""
MODEL_DIR=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --case) CASE_NAME="${2:-}"; shift 2 ;;
        --bus) BUS_NUMBER="${2:-}"; shift 2 ;;
        --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CASE_NAME" || -z "$BUS_NUMBER" ]]; then
    echo "Usage: sbatch submit_warmstart_hpc.sh --case <case_name> --bus <bus_number> [--model-dir <path>]"
    exit 1
fi

CHUNK_SIZE=10
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$((START_IDX + CHUNK_SIZE))
CHUNK_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")

LOG_DIR="logs/warmstart_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$LOG_DIR" result
exec > "${LOG_DIR}/warmstart_${CHUNK_ID}.out" 2> "${LOG_DIR}/warmstart_${CHUNK_ID}.err"

echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Case: $CASE_NAME"
echo "Bus: $BUS_NUMBER"
echo "Range: [$START_IDX, $END_IDX)"
echo "Start: $(date)"

module purge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

CMD=(python evaluate_warmstart_qcqp_array.py --case_name "$CASE_NAME" --bus_number "$BUS_NUMBER" --start_idx "$START_IDX" --end_idx "$END_IDX" --chunk_id "$CHUNK_ID")
if [[ -n "$MODEL_DIR" ]]; then CMD+=(--model_dir "$MODEL_DIR"); fi

"${CMD[@]}"
EXIT_CODE=$?
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
exit "$EXIT_CODE"
