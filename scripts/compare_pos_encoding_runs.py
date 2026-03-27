#!/usr/bin/env python3
"""
对比两次训练写在 output_dir/log.txt 中的 JSON 行，提取验证集 RefCOCO 指标等。

用法:
  python scripts/compare_pos_encoding_runs.py path/to/run_a/log.txt path/to/run_b/log.txt
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def load_logs(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """取每个 epoch 的 test_refexp_refcoco（若存在），并报告最后一个含 test_* 的 epoch。"""
    best = None
    best_p = -1.0
    last_with_test = None
    for r in rows:
        k = "test_refexp_refcoco"
        if k in r and isinstance(r[k], list) and len(r[k]) >= 1:
            p = float(r[k][0])
            if p > best_p:
                best_p = p
                best = dict(epoch=r.get("epoch"), test_refexp_refcoco=r[k], train_loss=r.get("train_loss"))
        if any(x.startswith("test_") for x in r):
            last_with_test = r
    return {
        "best_by_refexp0": best,
        "last_eval_line": last_with_test,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare two training log.txt (JSON lines)")
    ap.add_argument("log_a", help="Run-A log.txt (e.g. sine baseline)")
    ap.add_argument("log_b", help="Run-B log.txt (e.g. learned)")
    args = ap.parse_args()

    sa = summarize(load_logs(args.log_a))
    sb = summarize(load_logs(args.log_b))
    print("=== Run-A ===", args.log_a)
    print(json.dumps(sa, indent=2, ensure_ascii=False))
    print("=== Run-B ===", args.log_b)
    print(json.dumps(sb, indent=2, ensure_ascii=False))
    if sa and sb and sa.get("best_by_refexp0") and sb.get("best_by_refexp0"):
        a0 = sa["best_by_refexp0"]["test_refexp_refcoco"]
        b0 = sb["best_by_refexp0"]["test_refexp_refcoco"]
        print("=== 对比 test_refexp_refcoco [P@0.5, P@0.7, overall] ===")
        print(f"  A: {a0}")
        print(f"  B: {b0}")


if __name__ == "__main__":
    main()
    sys.exit(0)
