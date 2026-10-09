"""Free-generation evaluation over one split file.

Rows carrying `meta` get their prompt/answer rendered by plugins/qwen_video_aug
(same code path as training); rows with a baked prompt are used as-is.
Writes results.json (per-row labels/preds/soft, consumed by summarize_eval).

Usage (from the repository root):
    VIEWAUG_CONFIG_EVAL=configs/embodied/eval_l123.yaml VIDEO_HEIGHT=240 FPS=5 \
    CUDA_VISIBLE_DEVICES=0 python -m scripts.dataconv.eval_freegen \
        --model <MODEL_DIR> --adapter <ADAPTER_DIR> --nothink \
        --json-file <SPLIT_JSON> --data-root <DATA_ROOT> --limit 5
"""

import argparse
import json
import os
from collections import Counter
from typing import Any

from swift.arguments import InferArguments
from swift.infer_engine import InferRequest, RequestConfig
from swift.infer_engine.transformers_engine import TransformersEngine
from swift.pipelines.utils import prepare_model_template
from tqdm import tqdm

from embodied.data import load_datasets
from embodied.scoring import (CANDIDATE_TOP_LOGPROBS, ParseError, candidate_logprobs,
                              parse_answer, parse_value, soft_from_entry, soft_probs)
from embodied.taxonomy import severity_token_ids

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--adapter", default=None, help="Adapter checkpoint (optional; base model if omitted)")
    parser.add_argument("--json-file", nargs="+", required=True, help="One or more JSON datasets")
    parser.add_argument("--data-root", default=".", help="Root dir for video paths (default: .)")
    parser.add_argument("--save-dir", default="eval_results", help="Output directory (default: eval_results)")
    parser.add_argument("--batch-size", type=int, default=8, help="Videos per inference batch (default: 8)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Generation cap (default: 1024)")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N rows when > 0 (smoke tests)")
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--think", dest="enable_thinking", action="store_true", default=None,
                             help="Prefill '<think>\\n' and generate a real reasoning chain")
    think_group.add_argument("--nothink", dest="enable_thinking", action="store_false", default=None,
                             help="Prefill empty think and answer directly")
    return parser.parse_args()

def load_engine(args: argparse.Namespace) -> Any:
    print(f"[load] {args.model} (adapter={args.adapter}, enable_thinking={args.enable_thinking}, bs={args.batch_size})...", flush=True)
    infer_args = InferArguments(
        model=args.model,
        adapters=[args.adapter] if args.adapter else [],
        external_plugins=[os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "plugins", "qwen_video_patch.py")],
        max_length=int(os.getenv("MAX_LENGTH", "128000")),
        torch_dtype="bfloat16",
        attn_impl="sdpa",
        infer_backend="pt",
        enable_thinking=args.enable_thinking,
    )
    model, template = prepare_model_template(infer_args)
    model.eval()

    cand_ids = sorted(set(severity_token_ids(template.processor.tokenizer).values()))
    TransformersEngine.preprocess_logits = staticmethod(lambda batched_logits, batched_generate_ids, top_logprobs: candidate_logprobs(batched_logits, batched_generate_ids, cand_ids))
    return TransformersEngine(model, template=template, max_batch_size=args.batch_size)

def answer_after_think(text: str) -> str:
    i = text.rfind("</think>")
    return (text[i + len("</think>"):] if i != -1 else text).strip()

def _turns(sample: dict) -> list[tuple[str, str]]:
    if "messages" in sample:
        return [("user" if m["role"] == "user" else m["role"], m["content"])
                for m in sample["messages"]]
    return [("user" if c["from"] == "human" else "assistant", c["value"])
            for c in sample["conversations"]]

def get_prompt(sample: dict) -> str:
    for role, text in _turns(sample):
        if role == "user":
            return text
    raise KeyError(f"no user turn: {sample.keys()}")

def _viewaug():
    import sys
    plugins = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "plugins")
    if plugins not in sys.path:
        sys.path.insert(0, plugins)
    import qwen_video_aug
    return qwen_video_aug

def resolve(sample: dict) -> tuple[str, dict, list]:
    if not sample.get("meta"):
        labels = sample.get("labels") or {}
        dims = [next(iter(labels))] if labels else []
        return get_prompt(sample), dict(labels), dims
    view = _viewaug().view_of(sample)
    return view.prompt, dict(view.gts), list(view.dims)

def find_answer_entry(lp_content: list[dict[str, Any]]) -> dict[str, Any] | None:
    ti = next((i for i, e in enumerate(lp_content) if (e.get("token") or "").strip() == "</think>"), None)
    search = lp_content[ti + 1:] if ti is not None else lp_content
    for e in search:
        if (e.get("token") or "").strip():
            return e
    return None

def single_value_soft(dim: str, lp_content: list[dict[str, Any]]) -> dict[str, Any] | None:
    entry = find_answer_entry(lp_content)
    return soft_from_entry(entry, dim) if entry is not None else None

def infer_batch(engine: Any, video_paths: list[str], prompts: list[str], max_tokens: int) -> list[tuple[str, list]]:
    infer_requests = [InferRequest(messages=[{"role": "user", "content": pr}], videos=p) for pr, p in zip(prompts, video_paths)]
    request_config = RequestConfig(max_tokens=max_tokens, temperature=0, logprobs=True, top_logprobs=CANDIDATE_TOP_LOGPROBS)
    resp_list = engine.infer(infer_requests, request_config, use_tqdm=False)
    outputs = []
    for resp in resp_list:
        choice = resp.choices[0]
        content = (choice.logprobs or {}).get("content", [])
        outputs.append((choice.message.content, content))
    return outputs

def main() -> None:
    args = parse_args()
    engine = load_engine(args)

    dataset = load_datasets(args.json_file)
    if args.limit > 0:
        dataset = dataset[:args.limit]

    rows: list[dict[str, Any]] = []
    n_rows = 0
    parse_fail = 0
    fail_reasons: Counter = Counter()
    per_dim_hit: Counter = Counter()
    per_dim_total: Counter = Counter()
    n, bs = len(dataset), max(1, args.batch_size)
    for start in tqdm(range(0, n, bs)):
        chunk = dataset[start:start + bs]
        paths = [[os.path.join(args.data_root, v) for v in s["videos"]] for s in chunk]
        resolved = [resolve(s) for s in chunk]
        prompts = [r[0] for r in resolved]
        results = infer_batch(engine, paths, prompts, args.max_tokens)

        for sample, (_prompt, gts, dims), (text, lp_content) in zip(chunk, resolved, results):
            preds, reason = {}, None
            try:
                if len(dims) == 1:
                    preds = {dims[0]: parse_value(text, dims[0])}
                elif dims:
                    preds = parse_answer(text, dims)
            except ParseError as e:
                reason = e.reason
            parse_ok = reason is None and bool(dims)
            if not parse_ok:
                parse_fail += 1
                fail_reasons[reason or "no_dims"] += 1

            per_dim = {d: preds.get(d) == gts[d] for d in dims}
            for d in dims:
                per_dim_total[d] += 1
                per_dim_hit[d] += per_dim[d]
            n_rows += 1

            gt_answer = gts[dims[0]] if len(dims) == 1 else json.dumps({d: gts[d] for d in dims}, ensure_ascii=False)
            pred = answer_after_think(text)
            print(f"\n{'=' * 70}\nVIDEO: {sample['videos'][0]}")
            print(f"  GEN : {text}")
            print(f"  PRED: {pred!r}")
            print(f"  GT  : {gt_answer!r}")
            if not parse_ok:
                print(f"  [parse error: {reason}]")

            soft: dict[str, Any] = {}
            if lp_content and parse_ok:
                if len(dims) == 1:
                    one = single_value_soft(dims[0], lp_content)
                    soft = {dims[0]: one} if one is not None else {}
                else:
                    soft = soft_probs(text, lp_content, dims)
            for d, s in soft.items():
                print(f"  E_sev[{d}]: {s['E_severity']:.4f}  argmax={s['argmax']!r}  "
                      f"probs={ {v: round(p, 4) for v, p in s['probs'].items()} }")

            rows.append({
                "videos": sample["videos"],
                "gen": text, "pred": pred, "gt": gt_answer,
                "labels": gts,
                "dims": dims, "preds": preds, "per_dim": per_dim,
                "parse_ok": parse_ok, "parse_fail_reason": reason,
                "soft": {d: {"E_severity": s["E_severity"], "argmax": s["argmax"], "probs": s["probs"]}
                         for d, s in soft.items()},
            })

    if n_rows:
        print(f"\n{'=' * 70}\n  parse errors: {parse_fail}/{n_rows} = {parse_fail / n_rows:.4f} {dict(fail_reasons) or ''}")
        for d in sorted(per_dim_total):
            print(f"  acc[{d}]: {per_dim_hit[d]}/{per_dim_total[d]} = {per_dim_hit[d] / per_dim_total[d]:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    res_path = os.path.join(args.save_dir, "results.json")
    dims_all = sorted(per_dim_total)
    payload: dict[str, Any] = {
        "model": args.model, "adapter": args.adapter,
        "json_file": args.json_file, "enable_thinking": args.enable_thinking,
        "n": n_rows,
        "dims": dims_all,
        "parse_fail": parse_fail,
        "parse_fail_rate": parse_fail / max(n_rows, 1),
        "parse_fail_reasons": dict(fail_reasons),
    }
    for d in dims_all:
        payload[f"acc_{d}"] = per_dim_hit[d] / per_dim_total[d]
    payload["rows"] = rows
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"  rows written to: {res_path} ({len(rows)})")

if __name__ == "__main__":
    main()
