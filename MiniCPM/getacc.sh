#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Error at method=$method dataset=$dataset noise=$noise snr=$snr"; exit 1' ERR

DATASETS=(speech sound music librispeech)
METHODS=(SEES)
SNRS=(-10 -5 0 5 10 20 30)

# Support multiple noise types
NOISES=(crowd machine traffic)   # Add or remove as needed

GPU=0

# ===== Special dataset configuration =====
SPECIAL_DATASET="librispeech"
SPECIAL_SCRIPT="get_wer.py"
DEFAULT_SCRIPT="get_acc.py"

# ===== Record successfully executed commands =====
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
          echo "[SKIP] $key already completed"
          continue
        fi

        # ===== Select script =====
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
