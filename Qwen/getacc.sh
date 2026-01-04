#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Error at method=$method dataset=$dataset noise=$noise snr=$snr"; exit 1' ERR

DATASETS=(speech sound music librispeech)
METHODS=(SEES)
SNRS=(-10 -5 0 5 10 20 30)

# ⭐ 支持多种噪声
NOISES=(crowd machine traffic wind)   # 按需增减

GPU=0

# ===== 特殊数据集配置 =====
SPECIAL_DATASET="librispeech"
SPECIAL_SCRIPT="get_wer.py"
DEFAULT_SCRIPT="get_acc.py"

# ===== 记录成功运行的命令 =====
DONE_LOG="run_done.log"
touch "$DONE_LOG"

is_done () {
  local key="$1"
  grep -Fxq "$key" "$DONE_LOG"
}

mark_done () {
  local key="$1"
  echo "$key" >> "$DONE_LOG"
}

for method in "${METHODS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for noise in "${NOISES[@]}"; do
      for snr in "${SNRS[@]}"; do

        key="method=${method},dataset=${dataset},noise=${noise},snr=${snr}"

        if is_done "$key"; then
          echo "[SKIP] $key already done"
          continue
        fi

        # ===== 选择脚本 =====
        if [[ "$dataset" == "$SPECIAL_DATASET" ]]; then
          SCRIPT="$SPECIAL_SCRIPT"
        else
          SCRIPT="$DEFAULT_SCRIPT"
        fi

        echo "[RUN ] $key -> $SCRIPT"
        python "$SCRIPT" \
          --datasets "$dataset" \
          --noise "$noise" \
          --gpu "$GPU" \
          --SNR "$snr"

        mark_done "$key"
        echo "[DONE] $key"

      done
    done
  done
done
