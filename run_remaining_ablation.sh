#!/usr/bin/env bash

set -u
set -o pipefail

# ============================================================
# Controlled ACOPF ablation queue
#
# Sequence:
#   1. Wait until existing 14/57 training has stopped
#      for 5 consecutive checks.
#   2. Run complete 162-bus controlled ablation.
#   3. Run complete 300-bus controlled ablation.
#
# Each case:
#   Projection study:
#       none, voltage, generation, full
#       x seeds 1--5
#
#   Weight study:
#       equal, inequality, physics, objective
#       with full projection
#       x seeds 1--5
#
# Total = 40 runs/case.
# ============================================================


# ----------------------------
# Configuration
# ----------------------------

PROJECT_DIR="$HOME/projects/ACOPF-feasible"

CASE162="pglib_opf_case162_ieee_dtc"
CASE300="pglib_opf_case300_ieee"

EPOCHS=10000

PROJECTIONS=(none voltage generation full)
WEIGHTS=(equal inequality physics objective)
SEEDS=(1 2 3 4 5)

LOG_DIR="$PROJECT_DIR/logs/controlled_ablation"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR" || exit 1


# ============================================================
# Helper: run one experiment
# ============================================================

run_experiment () {

    local CASE="$1"
    local PROJ="$2"
    local WEIGHT="$3"
    local SEED="$4"

    local LOGFILE
    LOGFILE="$LOG_DIR/${CASE}_${PROJ}_${WEIGHT}_seed${SEED}.log"

    echo
    echo "============================================================"
    echo "START"
    echo "Case       : $CASE"
    echo "Projection : $PROJ"
    echo "Weights    : $WEIGHT"
    echo "Seed       : $SEED"
    echo "Time       : $(date)"
    echo "Log        : $LOGFILE"
    echo "============================================================"

    python ACOPF_controlled_ablation.py \
        --case_name "$CASE" \
        --epochs "$EPOCHS" \
        --projection "$PROJ" \
        --weights "$WEIGHT" \
        --seed "$SEED" \
        2>&1 | tee "$LOGFILE"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: experiment failed."
        echo "Case       : $CASE"
        echo "Projection : $PROJ"
        echo "Weights    : $WEIGHT"
        echo "Seed       : $SEED"
        echo "Exit code  : $STATUS"
        echo "Time       : $(date)"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit "$STATUS"
    fi

    echo
    echo "FINISHED: $CASE | $PROJ | $WEIGHT | seed=$SEED | $(date)"
}


# ============================================================
# Helper: complete one bus case
# ============================================================

run_case () {

    local CASE="$1"

    echo
    echo "################################################################"
    echo "STARTING CONTROLLED ABLATION"
    echo "CASE: $CASE"
    echo "TIME: $(date)"
    echo "################################################################"


    # --------------------------------------------------------
    # Part A: Projection ablation
    # --------------------------------------------------------

    for PROJ in "${PROJECTIONS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            run_experiment \
                "$CASE" \
                "$PROJ" \
                "nominal" \
                "$SEED"
        done
    done


    # --------------------------------------------------------
    # Part B: Loss-weight sensitivity
    #
    # full + nominal has already been generated above,
    # therefore it is NOT repeated here.
    # --------------------------------------------------------

    for WEIGHT in "${WEIGHTS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            run_experiment \
                "$CASE" \
                "full" \
                "$WEIGHT" \
                "$SEED"
        done
    done


    echo
    echo "################################################################"
    echo "COMPLETED CASE: $CASE"
    echo "TIME          : $(date)"
    echo "################################################################"
}


# ============================================================
# STEP 1: Wait for existing 14/57 jobs
# ============================================================

echo
echo "============================================================"
echo "WAITING FOR CURRENT 14/57 ABLATION JOBS"
echo "Started waiting: $(date)"
echo "============================================================"

CLEAR_COUNT=0
REQUIRED_CLEAR_CHECKS=5

while true
do

    if pgrep -f "python ACOPF_controlled_ablation.py" > /dev/null
    then

        CLEAR_COUNT=0

        echo "$(date): existing ablation jobs still running"

    else

        CLEAR_COUNT=$((CLEAR_COUNT + 1))

        echo "$(date): no training process detected" \
             "($CLEAR_COUNT/$REQUIRED_CLEAR_CHECKS)"

    fi


    if [ "$CLEAR_COUNT" -ge "$REQUIRED_CLEAR_CHECKS" ]
    then

        echo
        echo "$(date): no training process detected for five consecutive checks."
        echo "Current 14/57 jobs are considered finished."

        break

    fi

    sleep 60

done


# ============================================================
# STEP 2: 162-bus
# ============================================================

echo
echo "============================================================"
echo "$(date): STARTING 162-BUS EXPERIMENTS"
echo "============================================================"

run_case "$CASE162"


# ============================================================
# STEP 3: 300-bus
# ============================================================

echo
echo "============================================================"
echo "$(date): STARTING 300-BUS EXPERIMENTS"
echo "============================================================"

run_case "$CASE300"


# ============================================================
# DONE
# ============================================================

echo
echo "================================================================"
echo "ALL REMAINING CONTROLLED ABLATION EXPERIMENTS FINISHED"
echo "162-bus : COMPLETE"
echo "300-bus : COMPLETE"
echo "Time    : $(date)"
echo "================================================================"

exit 0
