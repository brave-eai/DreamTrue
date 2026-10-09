"""Pools results.json across variants per checkpoint.

Sample-level pooling per dimension, then simple average over dimensions;
writes summary.json per checkpoint and summary_per_ckpt.csv.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from embodied.taxonomy import DIM_BY_ID

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", default="eval_results", help="eval_results root dir (default: eval_results)")
    parser.add_argument("--out", default=None, help="Summary CSV path (default: <eval-dir>/summary_per_ckpt.csv)")
    return parser.parse_args()

def locate_key(rel: str) -> tuple[str, str] | None:
    parts = rel.split(os.sep)
    if "test" in parts:
        i = parts.index("test")
        if i >= 2:
            run, ckpt = os.sep.join(parts[:i - 1]), parts[i - 1]
            return run, ckpt
    return None

def macro_f1(gts: list[str], preds: list[str | None]) -> float:
    classes = {g for g in gts} | {p for p in preds if p is not None}
    if not classes:
        return float("nan")
    f1s = []
    for c in sorted(classes):
        tp = sum(1 for g, p in zip(gts, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(gts, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(gts, preds) if g == c and p != c)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)

def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    dims = sorted({d for r in rows for d in r.get("dims", [])})
    per_dim: dict[str, dict] = {}
    row_maes: list[float] = []
    for d in dims:
        sub = [r for r in rows if d in r.get("per_dim", {})]
        gts = [r["labels"][d] for r in sub]
        preds = [r.get("preds", {}).get(d) for r in sub]
        w = DIM_BY_ID[d]["weights"]
        sev_pairs = [(r["labels"][d], r["soft"][d]["E_severity"])
                     for r in sub if r.get("soft") and d in r["soft"]]
        per_dim[d] = {
            "n": len(sub),
            "acc": sum(r["per_dim"][d] for r in sub) / max(len(sub), 1),
            "f1": macro_f1(gts, preds),
            "mae": (sum(abs(e - w[g]) for g, e in sev_pairs) / len(sev_pairs)) if sev_pairs else None,
            "n_sev": len(sev_pairs),
        }
    for r in rows:
        errs = [abs(r["soft"][d]["E_severity"] - DIM_BY_ID[d]["weights"][r["labels"][d]])
                for d in r.get("dims", [])
                if r.get("soft") and d in r["soft"] and r["labels"].get(d) in DIM_BY_ID[d]["weights"]]
        if errs:
            row_maes.append(sum(errs) / len(errs))
    return {
        "n": n,
        "parse_fail": sum(not r.get("parse_ok") for r in rows),
        "acc": _mean([v["acc"] for v in per_dim.values()]),
        "f1": _mean([v["f1"] for v in per_dim.values()]),
        "mae": sum(row_maes) / len(row_maes) if row_maes else None,
        "per_dim": per_dim,
    }

def _mean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None

def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"

def main() -> None:
    args = parse_args()
    args.out = args.out or os.path.join(args.eval_dir, "summary_per_ckpt.csv")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_variants: dict[tuple[str, str], int] = defaultdict(int)
    n_files = 0
    for dirpath, _, filenames in os.walk(args.eval_dir):
        if "results.json" not in filenames:
            continue
        rel = os.path.relpath(os.path.join(dirpath, "results.json"), args.eval_dir)
        key = locate_key(rel)
        if key is None:
            print(f"[skip] cannot parse (run, ckpt): {rel}")
            continue
        with open(os.path.join(dirpath, "results.json"), encoding="utf-8") as f:
            rows = json.load(f).get("rows", [])
        grouped[key].extend(rows)
        n_variants[key] += 1
        n_files += 1

    if not grouped:
        print("no results.json to aggregate")
        return

    csv_rows = []
    unwritable: list[str] = []
    for (run, ckpt), rows in sorted(grouped.items()):
        m = summarize(rows)
        dims = sorted(m["per_dim"])
        print(f"\n{'=' * 100}\n{run}  {ckpt}   (variants={n_variants[(run, ckpt)]}"
              f" → n={m['n']}, parse_fail={m['parse_fail']})")
        header = "  dim    n      acc      F1      MAE     n_sev"
        print(header)
        for d in dims:
            v = m["per_dim"][d]
            print(f"  {d:<5} {v['n']:<6} {fmt(v['acc']):<7} {fmt(v['f1']):<7} {fmt(v['mae']):<7} "
                  f"{v['n_sev']:<7}")
        print(f"  {'ALL':<5} {m['n']:<6} {fmt(m['acc']):<7} {fmt(m['f1']):<7} {fmt(m['mae']):<7}")
        row = {"run": run, "ckpt": ckpt, "n": m["n"], "parse_fail": m["parse_fail"],
               "acc": m["acc"], "F1": m["f1"], "MAE": m["mae"]}
        for d in dims:
            v = m["per_dim"][d]
            row |= {f"acc_{d}": v["acc"], f"F1_{d}": v["f1"], f"MAE_{d}": v["mae"],
                    f"n_sev_{d}": v["n_sev"]}
        csv_rows.append(row)

        summary_path = os.path.join(args.eval_dir, run, ckpt, "test", "summary.json")
        payload = {"run": run, "ckpt": ckpt, "variants": n_variants[(run, ckpt)],
                   "n": m["n"], "parse_fail": m["parse_fail"], "acc": m["acc"],
                   "F1": m["f1"], "MAE": m["mae"],
                   "per_dim": m["per_dim"]}
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"  → {summary_path}")
        except PermissionError:
            unwritable.append(summary_path)

    fieldnames = ["run", "ckpt", "n", "parse_fail", "acc", "F1", "MAE"]
    for row in csv_rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n{len(csv_rows)} ckpts summarized -> {args.out} (from {n_files} results.json)")
    if unwritable:
        print(f"[warn] {len(unwritable)} summary.json could not be written (read-only dirs), "
              f"e.g. {unwritable[0]}")

if __name__ == "__main__":
    main()
