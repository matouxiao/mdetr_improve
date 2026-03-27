# 位置编码单变量实验（第一阶段：sine vs learned）

## 实验目的

在 **其余设置不变**（含 `resnet101` backbone、数据集与超参）的前提下，仅改变 `--position_embedding`，比较：

- **Run-A（基线）**：`sine` — 正弦绝对位置编码（默认）。
- **Run-B（单变量）**：`learned` — 行/列可学习的 **绝对** 网格编码（见 `PositionEmbeddingLearned`）。

**注意**：`learned` **不是**「相对位置编码」。若论文要讨论相对 vs 绝对，需另设实验（例如使用 `--position_embedding relative` 的实现，若已启用）。

## 如何运行

使用仓库内脚本（可自行设置 `PYTHON` 指向 conda 环境，例如 `/root/miniconda3/envs/mdetr_env/bin/python`）：

```bash
export PYTHON=/root/miniconda3/envs/mdetr_env/bin/python
export DATASET_CONFIG=/path/to/configs/refcoco.json

POSITION_EMBEDDING=sine RUN_NAME=pos_sine_r101 OUTPUT_DIR=./outputs/pos_sine_r101 \
  ./scripts/train_pos_encoding_ablation.sh

POSITION_EMBEDDING=learned RUN_NAME=pos_learned_r101 OUTPUT_DIR=./outputs/pos_learned_r101 \
  ./scripts/train_pos_encoding_ablation.sh
```

请保证 `configs/refcoco.json` 中 `coco_path`、`refexp_ann_path` 等路径正确；需要预训练权重时可在 `EXTRA_ARGS` 中传入，例如：

```bash
EXTRA_ARGS="--load /path/to/pretrained_resnet101_checkpoint.pth --ema" ...
```

## 特征图尺寸与 learned 上限

`PositionEmbeddingLearned` 使用 `nn.Embedding(50, …)`，要求特征图 **H、W 均 ≤ 50**。请运行：

```bash
python scripts/verify_feature_map_hw.py
```

在默认 COCO 缩放（短边 800、长边不超过 1333）与 ResNet stride 32 下，典型特征图约 25×42，**满足** 50 上限。

## 指标对比

训练过程中 `output_dir/log.txt` 每行一条 JSON。对比两次 run：

```bash
python scripts/compare_pos_encoding_runs.py \
  ./outputs/pos_sine_r101/log.txt \
  ./outputs/pos_learned_r101/log.txt
```

关注 `test_refexp_refcoco`（RefCOCO 指代指标）及 `train_loss` 曲线；也可用 `plot_loss_curve.py` 分别绘制。

## 记录清单

- 完整命令行或脚本环境变量快照。
- 两次 run 的 `output_dir`、`log.txt`、checkpoint 策略一致（例如都保存 `BEST_checkpoint.pth` 规则相同）。
