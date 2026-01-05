#!/bin/bash

set -e
DATASETS_ROOT="../datasets"
NOISE_ROOT="$DATASETS_ROOT/noise"

echo "PWD=$(pwd)"
echo "DATASETS_ROOT=$DATASETS_ROOT"
ls "$DATASETS_ROOT"

LOG_DIR="log"
LOG_FILE="$LOG_DIR/addnoise_process.log"

mkdir -p "$LOG_DIR"

> "$LOG_FILE"

echo "=== Batch noise augmentation started: $(date) ===" | tee -a "$LOG_FILE"

# Find all clean/raw directories
find -L "$DATASETS_ROOT" -type d -path "*/clean/raw" | while read CLEAN_DIR; do

    # Example of CLEAN_DIR: datasets/music/clean/raw
    TASK_DIR=$(dirname "$(dirname "$CLEAN_DIR")")  # datasets/music
    OUTPUT_DIR="$TASK_DIR/noise"

    echo "" | tee -a "$LOG_FILE"
    echo "🔥 Processing task directory: $TASK_DIR" | tee -a "$LOG_FILE"
    echo "➡  clean_dir:   $CLEAN_DIR" | tee -a "$LOG_FILE"
    echo "➡  noise_root:  $NOISE_ROOT" | tee -a "$LOG_FILE"
    echo "➡  output_dir:  $OUTPUT_DIR" | tee -a "$LOG_FILE"

    mkdir -p "$OUTPUT_DIR"

    python addnoise.py \
        --clean_dir "$CLEAN_DIR" \
        --noise_root "$NOISE_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --snrs="-10,-5,0,5,10,20,30" \
        --skip_existing 2>&1 | tee -a "$LOG_FILE"

    echo "✔ Completed: $TASK_DIR" | tee -a "$LOG_FILE"

done

echo "" | tee -a "$LOG_FILE"
echo "🎉 All tasks have been completed! $(date)" | tee -a "$LOG_FILE"
