"""Free-generation inference over unlabeled video directories.

Appends one JSONL line per video; reruns skip videos already present in infer.jsonl.

Usage (from the repository root):
    VIEWAUG_CONFIG_EVAL=configs/embodied/eval_l123.yaml VIDEO_HEIGHT=240 FPS=5 \
    CUDA_VISIBLE_DEVICES=0 python -m scripts.dataconv.infer_videos \
        --model <MODEL_DIR> --dirs /path/to/videos --dim L123 \
        --save-dir eval_results/infer/run1 --batch-size 1
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from embodied.scoring import ParseError, parse_answer, parse_value, soft_probs
from scripts.dataconv.eval_freegen import (_viewaug, answer_after_think, infer_batch, load_engine,
                                           single_value_soft)
from scripts.data.ours.build_1dim import probe_all, rec

ROOT = Path(__file__).resolve().parents[2]
FPS = float(os.getenv("FPS", "5"))

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Model / checkpoint directory")
    ap.add_argument("--dirs", nargs="+", required=True, help="Video directories (non-recursive *.mp4 scan)")
    ap.add_argument("--dim", default="L123", help="Dimension combo; reads configs/embodied/eval_<dim>.yaml")
    ap.add_argument("--save-dir", required=True, help="Output directory (infer.jsonl / skipped.txt)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N remaining videos when > 0 (smoke tests)")
    ap.add_argument("--workers", type=int, default=16, help="Probe worker processes")
    return ap.parse_args()

def scan(dirs: list[str]) -> list[str]:
    paths: list[str] = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            raise SystemExit(f"--dirs is not a directory: {d}")
        found = sorted(str(f) for f in p.glob("*.mp4"))
        print(f"[scan] {d}: {len(found)} mp4", flush=True)
        paths += found
    paths = list(dict.fromkeys(paths))
    print(f"[scan] total {len(paths)} videos", flush=True)
    return paths

def build_rows(paths: list[str], info: dict, bad: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        nfr, w, h = info.get(p, (-1, 0, 0))
        if nfr <= 0 or w <= 0 or h <= 0:
            bad.append(f"{p}\tprobe failed frames={nfr} {w}x{h}")
            continue
        rows.append(rec(p, {"fps": FPS, "dims": {}, "nfr": nfr, "cell": [w, h],
                            "ncell": 1, "head": [0], "rep": 0, "aug": "off"}))
    return rows

def record(row: dict, prompt: str, dims: list[str], text: str, lp_content: list) -> dict:
    preds, reason = {}, None
    try:
        preds = {dims[0]: parse_value(text, dims[0])} if len(dims) == 1 else parse_answer(text, dims)
    except ParseError as e:
        reason = e.reason
    parse_ok = reason is None and bool(dims)

    soft: dict = {}
    if lp_content and parse_ok:
        if len(dims) == 1:
            one = single_value_soft(dims[0], lp_content)
            soft = {dims[0]: one} if one is not None else {}
        else:
            soft = soft_probs(text, lp_content, dims)

    return {
        "video": row["videos"][0],
        "nfr": row["meta"]["nfr"],
        "cell": list(row["meta"]["cell"]),
        "dims": list(dims),
        "prompt": prompt,
        "gen": text,
        "pred": answer_after_think(text),
        "preds": preds,
        "parse_ok": parse_ok,
        "parse_fail_reason": reason,
        "soft": soft,
    }

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "infer.jsonl"

    paths = scan(args.dirs)
    bad: list[str] = []
    rows = build_rows(paths, probe_all(paths, args.workers), bad)

    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["video"])
        print(f"[resume] {out_path}: {len(done)} done, skipping", flush=True)

    aug = _viewaug()
    cfg = aug.load_config(str(ROOT / "configs" / "embodied" / f"eval_{args.dim.lower()}.yaml"))
    todo: list[tuple[dict, object]] = []
    for r in rows:
        if r["videos"][0] in done:
            continue
        view = aug.view_of(r, cfg)
        if view is None or not view.dims:
            bad.append(f"{r['videos'][0]}\tview_of is None")
            continue
        todo.append((r, view))
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"[plan] {len(todo)} to infer (total {len(paths)}, done {len(done)}, skipped {len(bad)})", flush=True)
    if bad:
        (save_dir / "skipped.txt").write_text("\n".join(bad) + "\n", encoding="utf-8")
        print(f"[warn] skipped {len(bad)}, see {save_dir / 'skipped.txt'}", flush=True)
    if not todo:
        print("[done] nothing to infer")
        return

    ns = SimpleNamespace(model=args.model, adapter=None, enable_thinking=False, batch_size=args.batch_size)
    engine = load_engine(ns)

    n, bs = len(todo), max(1, args.batch_size)
    n_fail = 0
    with open(out_path, "a", encoding="utf-8") as f, tqdm(total=n, desc="infer", unit="vid", mininterval=10) as pbar:
        for start in range(0, n, bs):
            chunk = todo[start:start + bs]
            prompts = [v.prompt for _r, v in chunk]
            vpaths = [[r["videos"][0]] for r, _v in chunk]
            results = infer_batch(engine, vpaths, prompts, args.max_tokens)
            for (row, view), (text, lp) in zip(chunk, results):
                obj = record(row, view.prompt, list(view.dims), text, lp)
                n_fail += not obj["parse_ok"]
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(len(chunk))

    print(f"[done] wrote {n} rows -> {out_path}; parse errors {n_fail}; skipped {len(bad)}", flush=True)

if __name__ == "__main__":
    main()
