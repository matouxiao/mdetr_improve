##训练指令
python main.py \
  --dataset_config configs/refcoco.json \
  --backbone resnet101 \
  --position_embedding relative \
  --output-dir ./outputs/pos_mix_r101_e5 \
  --batch_size 6 \
  --lr 5e-5 \
  --lr_backbone 1e-5 \
  --text_encoder_lr 1e-5 \
  --num_workers 32 \
  --resume ./outputs/pos_mix_r101_e5/checkpoint.pth \
  --swanlab \
  --swanlab_project mdetr \
  --swanlab_logging_steps 100 \
  --swanlab_run_name pos_mix_r101_e5 

##测试指令
相对筛选数据集
python main.py \
  --dataset_config configs/refcoco_relspatial_val.json \
  --backbone resnet101 \
  --position_embedding relative \
  --resume /workapp1219/detr/mdetr/outputs/pos_relative_r101_e5/checkpoint0005.pth \
  --eval \
  --eval_max_samples 1000

全量数据集
python main.py \
  --dataset_config configs/refcoco.json \
  --backbone resnet101 \
  --position_embedding sine \
  --resume /workapp1219/detr/mdetr/outputs/pos_sine_r101_e5/checkpoint0003.pth \
  --eval  \
  --eval_max_samples 1000