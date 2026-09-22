#!/bin/bash

SUBMIT="submit_controlled_ablation_162.sh"

echo "Submitting 162-bus projection ablation..."

for PROJ in none voltage generation full
do
    for SEED in 1 2 3 4 5
    do
        sbatch \
            --job-name="162_${PROJ}_n_s${SEED}" \
            "$SUBMIT" \
            --projection "$PROJ" \
            --weights nominal \
            --seed "$SEED"
    done
done

echo "Submitting 162-bus weight sensitivity..."

for WEIGHT in equal inequality physics objective
do
    for SEED in 1 2 3 4 5
    do
        sbatch \
            --job-name="162_f_${WEIGHT}_s${SEED}" \
            "$SUBMIT" \
            --projection full \
            --weights "$WEIGHT" \
            --seed "$SEED"
    done
done

echo "All 40 jobs submitted."
