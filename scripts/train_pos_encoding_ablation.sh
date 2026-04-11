#!/usr/bin/env bash
# 位置编码单变量实验：Run-A (sine) / Run-B (learned)
#
# main.py 已修正：Joiner 中仅 backbone.0（CNN）用 --lr_backbone；backbone.1（learned PE）与 Transformer 等同用 --lr。
# 曾用旧代码跑的 learned 实验需重跑，才能与 sine 公平对比。
#
# 用法：
#   chmod +x scripts/train_pos_encoding_ablation.sh
#   # Run-A 基线
#   POSITION_EMBEDDING=sine RUN_NAME=pos_sine_r101 OUTPUT_DIR=./outputs/pos_sine_r101 ./scripts/train_pos_encoding_ablation.sh
#   # Run-B 单变量
#   POSITION_EMBEDDING=learned RUN_NAME=pos_learned_r101 OUTPUT_DIR=./outputs/pos_learned_r101 ./scripts/train_pos_encoding_ablation.sh
#
# 本脚本学习率与 main.py 默认不同：--lr 5e-5（默认 1e-4）、--text_encoder_lr 1e-5（默认 5e-5）；sine/learned 请共用本脚本以保证除 PE 外一致。
# DataLoader：`NUM_WORKERS`（默认 8），可按 CPU/内存改，例如 `NUM_WORKERS=12 ./scripts/...`；仍可用 `EXTRA_ARGS` 覆盖。
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
NUM_WORKERS="${NUM_WORKERS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# 与 .github/refexp.md 中 refcoco 微调示例对齐的可调项；按需增加 --load / --ema 等
LR="${LR:-5e-5}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
TEXT_ENCODER_LR="${TEXT_ENCODER_LR:-1e-5}"

exec "$PYTHON" main.py \
  --dataset_config "$DATASET_CONFIG" \
  --backbone "$BACKBONE" \
  --position_embedding "$POSITION_EMBEDDING" \
  --run_name "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --lr_backbone "$LR_BACKBONE" \
  --text_encoder_lr "$TEXT_ENCODER_LR" \
  $EXTRA_ARGS
