#!/usr/bin/env bash
# 位置编码单变量实验：Run-A (sine) / Run-B (learned)
# 用法：
#   chmod +x scripts/train_pos_encoding_ablation.sh
#   # Run-A 基线
#   POSITION_EMBEDDING=sine RUN_NAME=pos_sine_r101 OUTPUT_DIR=./outputs/pos_sine_r101 ./scripts/train_pos_encoding_ablation.sh
#   # Run-B 单变量
#   POSITION_EMBEDDING=learned RUN_NAME=pos_learned_r101 OUTPUT_DIR=./outputs/pos_learned_r101 ./scripts/train_pos_encoding_ablation.sh
#
# 可选：export PYTHON=/root/miniconda3/envs/mdetr_env/bin/python

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DATASET_CONFIG="${DATASET_CONFIG:-${ROOT}/configs/refcoco.json}"
BACKBONE="${BACKBONE:-resnet101}"
POSITION_EMBEDDING="${POSITION_EMBEDDING:-sine}"
RUN_NAME="${RUN_NAME:-pos_${POSITION_EMBEDDING}_r101}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/${RUN_NAME}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# 与 .github/refexp.md 中 refcoco 微调示例对齐的可调项；按需增加 --load / --ema 等
exec "$PYTHON" main.py \
  --dataset_config "$DATASET_CONFIG" \
  --backbone "$BACKBONE" \
  --position_embedding "$POSITION_EMBEDDING" \
  --run_name "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --text_encoder_lr 1e-5 \
  --lr 5e-5 \
  $EXTRA_ARGS
