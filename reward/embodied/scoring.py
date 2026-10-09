"""Prediction parsing and scoring shared by offline evaluation.

Weights and dimension order come from embodied.taxonomy (single source of truth).
"""

import json
import math
import re
from typing import Any

import torch

from embodied.taxonomy import DIM_BY_ID, all_values, dimension_ids

# cap >= candidate count + 1 so patched top_logprobs never truncates candidates (see README)
CANDIDATE_TOP_LOGPROBS = max(32, len(all_values()) + 1)

# exact log_softmax for chosen + candidate ids; single .cpu() per batch (no per-item sync)
def candidate_logprobs(
    batched_logits: list["torch.Tensor"] | None,
    batched_generate_ids: "torch.Tensor",
    cand_ids: list[int],
) -> list[list[dict[int, float]]] | None:
    if batched_logits is None:
        return None
    batch_size = batched_generate_ids.shape[0]
    seq_len = len(batched_logits)
    if seq_len == 0:
        return [[] for _ in range(batch_size)]

    device = batched_logits[0].device
    gen = batched_generate_ids.to(device)  # [B, T]
    cand = torch.tensor(cand_ids, device=device)  # [C]
    cand_lp_steps: list["torch.Tensor"] = []
    chosen_lp_steps: list["torch.Tensor"] = []
    for j, logits in enumerate(batched_logits):
        lp = torch.log_softmax(logits.float(), dim=-1)
        cand_lp_steps.append(lp.index_select(1, cand))
        chosen_lp_steps.append(lp.gather(1, gen[:, j:j + 1])[:, 0])

    cand_lp = torch.stack(cand_lp_steps, dim=1).cpu().tolist()
    chosen_lp = torch.stack(chosen_lp_steps, dim=1).cpu().tolist()  # [B][T]
    gen_ids = gen.cpu().tolist()  # [B][T]

    batched_logprobs: list[list[dict[int, float]]] = []
    for i in range(batch_size):
        logprobs_list: list[dict[int, float]] = []
        for j in range(seq_len):
            d = {cid: cand_lp[i][j][c] for c, cid in enumerate(cand_ids)}
            d[gen_ids[i][j]] = chosen_lp[i][j]
            logprobs_list.append(d)
        batched_logprobs.append(logprobs_list)
    return batched_logprobs

def soft_from_entry(entry: dict[str, Any], dim_id: str) -> dict[str, Any]:
    chosen = (entry.get("token") or "").strip()
    lp: dict[str, float] = {(tl.get("token") or "").strip(): tl["logprob"] for tl in entry.get("top_logprobs", [])}
    lp.setdefault(chosen, entry.get("logprob", float("-inf")))

    cands = DIM_BY_ID[dim_id]["values"]
    cand_lps = [lp.get(v, float("-inf")) for v in cands]
    finite = [x for x in cand_lps if x != float("-inf")]
    if finite:
        m = max(finite)
        exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in cand_lps]
        s = sum(exps) or 1.0
        probs = {v: ex / s for v, ex in zip(cands, exps)}
    else:
        probs = {v: (1.0 if v == chosen else 0.0) for v in cands}
    e_sev = sum(probs[v] * DIM_BY_ID[dim_id]["weights"][v] for v in cands)
    return {"argmax": chosen, "probs": probs, "E_severity": e_sev}

class ParseError(ValueError):

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail

_PAIR_RE = re.compile(r'"(\w+)"\s*:\s*"([^"]+)"')

def _after_think(text: str) -> str:
    i = text.rfind("</think>")
    if i == -1:
        raise ParseError("no_think_end")
    return text[i + len("</think>"):].strip()

def parse_value(text: str, dim: str) -> str:
    ans = _after_think(text)
    if ans not in DIM_BY_ID[dim]["values"]:
        raise ParseError("not_a_value", ans[:40])
    return ans

# strict JSON parse: no fallback, no guessing
def parse_answer(text: str, dims: list[str] | None = None) -> dict[str, str]:
    body = _after_think(text)

    def no_dup(pairs):
        seen = set()
        for k, _v in pairs:
            if k in seen:
                raise ParseError("duplicate_key", k)
            seen.add(k)
        return dict(pairs)

    try:
        parsed = json.loads(body, object_pairs_hook=no_dup)
    except json.JSONDecodeError as e:
        raise ParseError("not_json_object", f"{e.msg} @ {body[:40]}") from e
    if not isinstance(parsed, dict):
        raise ParseError("not_json_object", type(parsed).__name__)

    want = set(dims) if dims is not None else set(parsed)
    got = set(parsed)
    if got != want:
        raise ParseError("key_mismatch", f"缺 {sorted(want - got)} 多 {sorted(got - want)}")
    for k, v in parsed.items():
        if k not in DIM_BY_ID:
            raise ParseError("key_mismatch", f"未知维度 {k}")
        if not isinstance(v, str) or v not in DIM_BY_ID[k]["values"]:
            raise ParseError("bad_value", f"{k}={v!r}")
    return {k: str(v) for k, v in parsed.items()}

def dim_value_entries(gen_text: str, logprob_content: list[dict[str, Any]],
                      dims: list[str] | None = None) -> dict[str, dict[str, Any]]:
    parse_answer(gen_text, dims)
    if not logprob_content:
        raise ParseError("token_unlocatable", "logprob_content 为空")

    offsets, concat = [], ""
    for e in logprob_content:
        offsets.append(len(concat))
        concat += e.get("token") or ""

    ti = concat.rfind("</think>")
    region = ti + len("</think>") if ti != -1 else 0
    out: dict[str, dict[str, Any]] = {}
    for m in _PAIR_RE.finditer(concat, region):
        dim = m.group(1)
        if dim not in DIM_BY_ID or (dims is not None and dim not in dims):
            continue
        vs, ve = m.span(2)
        covering = [e for e, off in zip(logprob_content, offsets) if off <= vs and off + len(e.get("token") or "") >= ve]
        if len(covering) != 1:
            why = "被切成多个 token" if not covering else f"有 {len(covering)} 个 token 覆盖"
            raise ParseError("token_unlocatable", f"{dim} 取值{why}")
        out[dim] = covering[0]
    missing = (set(dims) if dims is not None else set()) - set(out)
    if missing:
        raise ParseError("token_unlocatable", f"未定位到 {sorted(missing)}")
    return out

# renormalized softmax over the dimension values -> {argmax, probs, E_severity}
def soft_probs(gen_text: str, logprob_content: list[dict[str, Any]],
               dims: list[str] | None = None) -> dict[str, dict[str, Any]]:
    try:
        entries = dim_value_entries(gen_text, logprob_content, dims)
    except ParseError:
        return {}
    return {dim: soft_from_entry(e, dim) for dim, e in entries.items()}

