#!/bin/bash

set -e
DATASETS_ROOT="../datasets"
NOISE_ROOT="$DATASETS_ROOT/noise"

echo "PWD=$(pwd)"
echo "DATASETS_ROOT=$DATASETS_ROOT"
ls "$DATASETS_ROOT"
# 日志目录和文件
LOG_DIR="log"
LOG_FILE="$LOG_DIR/addnoise_process.log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 如果日志文件不存在，则创建;有则覆盖已有内容
> "$LOG_FILE"

echo "=== 批量加噪开始: $(date) ===" | tee -a "$LOG_FILE"

# 查找所有 clean/raw 目录
find -L "$DATASETS_ROOT" -type d -path "*/clean/raw" | while read CLEAN_DIR; do

    # CLEAN_DIR 例如：datasets/music/clean/raw
    TASK_DIR=$(dirname "$(dirname "$CLEAN_DIR")")  # datasets/music
    OUTPUT_DIR="$TASK_DIR/noise"

    echo "" | tee -a "$LOG_FILE"
    echo "🔥 处理任务目录: $TASK_DIR" | tee -a "$LOG_FILE"
    echo "➡  clean_dir:   $CLEAN_DIR" | tee -a "$LOG_FILE"
    echo "➡  noise_root:  $NOISE_ROOT" | tee -a "$LOG_FILE"
    echo "➡  output_dir:  $OUTPUT_DIR" | tee -a "$LOG_FILE"

    mkdir -p "$OUTPUT_DIR"

    # 运行加噪脚本，同时保存日志
    python addnoise.py \
        --clean_dir "$CLEAN_DIR" \
        --noise_root "$NOISE_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --snrs="-10,-5,0,5,10,20,30" \
        --skip_existing 2>&1 | tee -a "$LOG_FILE"

    echo "✔ 完成: $TASK_DIR" | tee -a "$LOG_FILE"

done

echo "" | tee -a "$LOG_FILE"
echo "🎉 全部任务处理完毕！ $(date)" | tee -a "$LOG_FILE"
