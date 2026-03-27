#!/usr/bin/env python3
"""
估算 ResNet backbone + COCO 风格 resize（短边 800、max_size 1333）下，
最后一层特征图的最大 H、W（stride=32），用于检查 PositionEmbeddingLearned 的 50 上限。

用法: python scripts/verify_feature_map_hw.py
"""
import math

STRIDE = 32


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def main():
    # 典型 val：800×1333；旋转后 1333×800。取各边除以 stride 后的最大值作为上界。
    max_h = 0
    max_w = 0
    for ow, oh in ((800, 1333), (1333, 800)):
        max_h = max(max_h, ceil_div(oh, STRIDE))
        max_w = max(max_w, ceil_div(ow, STRIDE))

    print("COCO 风格 resize 下 ResNet layer4 特征图（stride=32）保守上界：")
    print(f"  max H ≈ {max_h}, max W ≈ {max_w}")
    print(f"PositionEmbeddingLearned 要求 H,W <= 50: {'OK' if max_h <= 50 and max_w <= 50 else '可能越界，需检查实际输入分辨率'}")


if __name__ == "__main__":
    main()
