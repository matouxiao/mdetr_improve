#!/usr/bin/env python3
"""
从 main.py 写入的 log.txt（每行一个 JSON）绘制训练/验证 loss 曲线。

用法:
  python scripts/plot_training_log.py outputs/pos_learned_r101/log.txt
  python scripts/plot_training_log.py path/to/log.txt -o /tmp/curves.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_KEYS = [
    ("train_loss", "Total"),
    ("train_loss_ce", "CE"),
    ("train_loss_bbox", "Bbox"),
    ("train_loss_giou", "GIoU"),
    ("train_loss_contrastive_align", "Contrastive align"),
]

TEST_KEYS = [
    ("test_refexp_loss", "Total"),
    ("test_refexp_loss_ce", "CE"),
    ("test_refexp_loss_bbox", "Bbox"),
    ("test_refexp_loss_giou", "GIoU"),
    ("test_refexp_loss_contrastive_align", "Contrastive align"),
]


def parse_log(log_path: Path) -> tuple[list[int], dict[str, list[float]], dict[str, list[float]]]:
    epochs: list[int] = []
    train: dict[str, list[float]] = {k: [] for k, _ in TRAIN_KEYS}
    test: dict[str, list[float]] = {k: [] for k, _ in TEST_KEYS}

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = row.get("epoch")
            if ep is None:
                continue
            epochs.append(int(ep))
            for key, _ in TRAIN_KEYS:
                train[key].append(float(row.get(key, float("nan"))))
            for key, _ in TEST_KEYS:
                v = row.get(key)
                test[key].append(float(v) if v is not None else float("nan"))

    return epochs, train, test


def _plot_grid(
    epochs: list[int],
    series: dict[str, list[float]],
    key_labels: list[tuple[str, str]],
    title: str,
    ncols: int = 3,
):
    n = len(key_labels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, (key, label) in zip(axes_flat, key_labels):
        y = series.get(key, [])
        if len(y) != len(epochs):
            y = [float("nan")] * len(epochs)
        ax.plot(epochs, y, "o-", linewidth=1.5, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for j in range(len(key_labels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot train/val loss from MDETR log.txt")
    p.add_argument("log_txt", type=Path, help="Path to log.txt (JSON lines)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image base path (default: <log_dir>/loss_curves_train.png and _val.png)",
    )
    args = p.parse_args(argv)

    log_path = args.log_txt.resolve()
    if not log_path.is_file():
        print(f"Error: file not found: {log_path}", file=sys.stderr)
        return 1

    epochs, train, test = parse_log(log_path)
    if not epochs:
        print("Error: no valid JSON lines with 'epoch' field", file=sys.stderr)
        return 1

    if args.output is None:
        train_png = log_path.parent / "loss_curves_train.png"
        val_png = log_path.parent / "loss_curves_val.png"
    else:
        o = args.output
        if o.suffix.lower() in (".png", ".pdf", ".svg", ".jpg", ".jpeg"):
            train_png = o.with_name(f"{o.stem}_train{o.suffix}")
            val_png = o.with_name(f"{o.stem}_val{o.suffix}")
        else:
            train_png = o.parent / f"{o.name}_train.png"
            val_png = o.parent / f"{o.name}_val.png"

    fig_t = _plot_grid(epochs, train, TRAIN_KEYS, f"Training loss ({log_path.name})")
    fig_t.savefig(train_png, dpi=200, bbox_inches="tight")
    plt.close(fig_t)
    print(f"Saved: {train_png}")

    has_val = any(v == v for k, _ in TEST_KEYS for v in test[k])  # any finite row (NaN != NaN)
    if has_val:
        fig_v = _plot_grid(epochs, test, TEST_KEYS, f"Validation loss ({log_path.name})")
        fig_v.savefig(val_png, dpi=200, bbox_inches="tight")
        plt.close(fig_v)
        print(f"Saved: {val_png}")
    else:
        print("No test_* loss fields in log; skipped val figure.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
