#!/usr/bin/env python3
# Copyright (c) MDETR users. Licensed under the Apache License 2.0.
"""
从 RefCOCO 系 val 标注中筛出「caption 含显式空间/相对关系用语」的样本，写出子集 COCO JSON。

默认输入：mdetr_annotations/finetune_refcoco_val.json
用法示例：
  python scripts/filter_refcoco_val_relspatial.py \
    --input mdetr_annotations/finetune_refcoco_val.json \
    --output mdetr_annotations/finetune_refcoco_val_relspatial.json

评估：在 configs 里设置 "refexp_val_ann_suffix": "_relspatial"，或命令行
  --refexp_val_ann_suffix _relspatial
这样会加载 finetune_refcoco_val_relspatial.json（需已用本脚本生成）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, List, Tuple


# 显式空间/相对关系（英文），偏严、误伤较少
_SPATIAL_REL_STRICT = (
    r"\bto the left\b|\bto the right\b|\bon the left\b|\bon the right\b|"
    r"\bleft of\b|\bright of\b|"
    r"\babove\b|\bbelow\b|\bunder\b|\bover\b|"
    r"\bin front of\b|\bbehind\b|\bnext to\b|\bbetween\b|\bbeside\b|"
    r"\bnear the\b|\bnear a\b|"
    r"\bclosest\b|\bfurthest\b|\bfarthest\b|"
    r"\btop\b|\bbottom\b|\bcorner\b|\bmiddle\b|\bcenter\b|"
    r"\bupper\b|\blower\b|\bforeground\b|\bbackground\b"
)


def pattern_strict() -> re.Pattern:
    """显式空间/相对关系（英文），偏严、误伤较少。"""
    return re.compile(rf"({_SPATIAL_REL_STRICT})", re.I)


def pattern_broad() -> re.Pattern:
    """更宽：在 strict 基础上再匹配单词 left / right（子集更大、更噪）。"""
    return re.compile(rf"({_SPATIAL_REL_STRICT}|\bleft\b|\bright\b)", re.I)


def filter_coco_val(
    data: dict,
    caption_match: Callable[[str], bool],
) -> Tuple[dict, List[int]]:
    """返回新 dict（images/annotations 子集）与保留的 image_id 列表。"""
    keep_ids: List[int] = []
    images_out = []
    for im in data["images"]:
        cap = im.get("caption") or ""
        if caption_match(cap):
            images_out.append(im)
            keep_ids.append(im["id"])

    keep_set = set(keep_ids)
    anns_out = [a for a in data["annotations"] if a["image_id"] in keep_set]

    if len(images_out) != len(anns_out):
        raise RuntimeError(
            f"Filtered images ({len(images_out)}) != filtered annotations ({len(anns_out)}); "
            "check annotation consistency."
        )

    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": images_out,
        "annotations": anns_out,
    }
    return out, keep_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "mdetr_annotations" / "finetune_refcoco_val.json",
        help="RefCOCO 系 val COCO JSON（images 中含 caption）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "mdetr_annotations" / "finetune_refcoco_val_relspatial.json",
        help="写出子集标注路径",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "broad"),
        default="strict",
        help="strict：短语级；broad：额外匹配单词 left/right",
    )
    parser.add_argument(
        "--id-list",
        type=Path,
        default=None,
        help="可选：将保留的 image_id 每行一个写入该文件",
    )
    args = parser.parse_args()

    pat = pattern_strict() if args.mode == "strict" else pattern_broad()
    match = lambda s: bool(pat.search(s))

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_in = len(data["images"])
    out_data, keep_ids = filter_coco_val(data, match)
    n_out = len(out_data["images"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f)

    if args.id_list is not None:
        args.id_list.parent.mkdir(parents=True, exist_ok=True)
        args.id_list.write_text("\n".join(str(i) for i in keep_ids) + "\n", encoding="utf-8")

    print(f"Input:  {args.input}")
    print(f"Mode:   {args.mode}")
    print(f"Kept:   {n_out} / {n_in} ({100.0 * n_out / max(n_in, 1):.2f}%)")
    print(f"Output: {args.output}")
    if args.id_list:
        print(f"ID list: {args.id_list}")


if __name__ == "__main__":
    main()
