#!/usr/bin/env bash
# RefCOCO 微调权重 + 冻结骨干（lr_backbone=0）下的 sine / relative 对照。
# 用法：
#   REFCOCO_CKPT=/path/to/refcoco_resnet101_checkpoint.pth MODE=sine  ./scripts/train_refcoco_pair_sine_vs_relative.sh
#   REFCOCO_CKPT=/path/to/refcoco_resnet101_checkpoint.pth MODE=relative ./scripts/train_refcoco_pair_sine_vs_relative.sh
# 可选环境变量 OUT 覆盖输出目录。

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${MODE:-relative}"
REFCOCO_CKPT="${REFCOCO_CKPT:-/workapp1219/detr/mdetr/checkpoints/refcoco_resnet101_checkpoint.pth}"
if [[ -n "${OUT:-}" ]]; then
  :
elif [[ "$MODE" == "sine" ]]; then
  OUT="./outputs/refcoco_ablation_sine_bb0"
else
  OUT="./outputs/refcoco_ablation_relative_bb0"
fi

if [[ "$MODE" != "sine" && "$MODE" != "relative" ]]; then
  echo "MODE 必须是 sine 或 relative，当前: $MODE" >&2
  exit 1
fi

echo "MODE=$MODE OUT=$OUT CKPT=$REFCOCO_CKPT"

python main.py \
  --dataset_config configs/refcoco.json \
  --backbone resnet101 \
  --load "${REFCOCO_CKPT}" \
  --position_embedding "${MODE}" \
  --output-dir "${OUT}" \
  --device cuda \
  --batch_size 2 \
  --optimizer adamw \
  --lr 1e-4 \
  --lr_backbone 0 \
  --text_encoder_lr 5e-5 \
  --weight_decay 1e-4 \
  --clip_max_norm 0.1 \
  --ema \
  --num_workers 4 \
  --eval_skip 1
