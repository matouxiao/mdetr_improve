#!/usr/bin/env bash
# RefCOCO：相对位置 — 使用 --position_embedding relative（编码器 2D 相对偏置，见 models/transformer.py）
# 勿加 --contrastive_align_loss：main.py 无此参数，argparse 缩写会误匹配 --contrastive_align_loss_coef 导致报错。
# contrastive_align_loss / aux_loss 默认 True（仅 --no_contrastive_align_loss / --no_aux_loss 可关）。
# 学习率策略见 configs/refcoco.json：默认 cosine（warmup + 余弦衰减），避免 linear_with_warmup
# 下 gamma=0.1**(epoch//lr_drop) 在 lr_drop 过小时过早断崖式降 LR。
# 若改用 step/linear_with_warmup：50 轮建议 lr_drop≈30（约前 60% 满 LR）或 35（对齐 MDETR 40 轮里 35 的习惯），勿用 3。

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-./outputs/refcoco_relative_r101}"
RESUME="${2:-/workapp1219/detr/mdetr/outputs/refcoco_relative_r101/checkpoint.pth}"

# 默认断点续训：--resume 恢复 model / optimizer / epoch（与保存时结构一致）。
# 第 1 参数：output-dir；第 2 参数：可覆盖 RESUME 路径。
# 从公开预训练从头微调请另写命令使用 --load 与 pretrained 权重，勿与本 resume 混用。
python main.py \
  --dataset_config configs/refcoco.json \
  --backbone resnet101 \
  --resume "${RESUME}" \
  --position_embedding relative \
  --output-dir "${OUT}" \
  --device cuda \
  --batch_size 6 \
  --optimizer adamw \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --text_encoder_lr 5e-5 \
  --weight_decay 1e-4 \
  --clip_max_norm 0.1 \
  --ema \
  --num_workers 4 \
  --eval_skip 1
