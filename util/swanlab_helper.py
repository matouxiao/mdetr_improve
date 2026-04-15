# Copyright (c) MDETR users. Licensed under the Apache License 2.0.
"""Optional SwanLab experiment tracking (pip install swanlab)."""
from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any, Dict


def flatten_log_stats(log_stats: Dict[str, Any]) -> Dict[str, float]:
    """Convert main.py log_stats to float-only dict for swanlab.log (lists -> key_0, key_1, ...)."""
    out: Dict[str, float] = {}
    for k, v in log_stats.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, numbers.Real):
            out[k] = float(v)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, numbers.Real):
                    out[f"{k}_{i}"] = float(x)
    return out


def swanlab_config_from_args(args) -> Dict[str, Any]:
    """Subset of hyperparameters for SwanLab config panel."""
    keys = [
        "lr",
        "lr_backbone",
        "text_encoder_lr",
        "batch_size",
        "epochs",
        "backbone",
        "position_embedding",
        "dataset_config",
        "num_queries",
        "hidden_dim",
        "enc_layers",
        "dec_layers",
        "eval_skip",
        "seed",
    ]
    cfg: Dict[str, Any] = {}
    for k in keys:
        if hasattr(args, k):
            v = getattr(args, k)
            if v is None:
                continue
            if k == "dataset_config" and v:
                cfg[k] = str(v)
            else:
                cfg[k] = v
    if getattr(args, "output_dir", None):
        cfg["output_dir"] = str(args.output_dir)
    return cfg


def init_swanlab(args) -> bool:
    """
    Initialize SwanLab on rank 0. Returns True if finish_swanlab() should be called after training.
    """
    import util.dist as dist

    if not getattr(args, "swanlab", False) or args.eval or not dist.is_main_process():
        return False

    try:
        import swanlab
    except ImportError as e:
        raise ImportError("启用 --swanlab 请先安装: pip install swanlab") from e

    out = Path(args.output_dir) if getattr(args, "output_dir", None) else Path(".")
    exp_name = (getattr(args, "swanlab_run_name", None) or "").strip() or out.name or "mdetr"
    logdir = str(out / "swanlog")

    kwargs = {
        "project": getattr(args, "swanlab_project", "mdetr"),
        "experiment_name": exp_name,
        "config": swanlab_config_from_args(args),
        "logdir": logdir,
    }
    mode = getattr(args, "swanlab_mode", "cloud")
    if mode and str(mode).lower() != "cloud":
        kwargs["mode"] = mode

    swanlab.init(**kwargs)
    return True


def log_swanlab(args, log_stats: Dict[str, Any], step: int) -> None:
    import util.dist as dist

    if not getattr(args, "swanlab", False) or not dist.is_main_process():
        return
    import swanlab

    swanlab.log(flatten_log_stats(log_stats), step=step)


def finish_swanlab() -> None:
    import util.dist as dist

    if not dist.is_main_process():
        return
    try:
        import swanlab
    except ImportError:
        return
    swanlab.finish()
