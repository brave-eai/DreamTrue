"""Reward taxonomy: dimensions, severity values/weights, prompt construction.

GROUP_DIMS are answer-folding pseudo-dimensions whose value is the worst member value.
"""
import json
from typing import Any

Dim = dict[str, Any]

BINARY: dict[str, Any] = {"values": ["无", "有"], "weights": {"无": 0.0, "有": -2.0}}
GRADED: dict[str, Any] = {"values": ["无", "轻微", "严重"], "weights": {"无": 0.0, "轻微": -1.0, "严重": -2.0}}

GROUPS: dict[str, str] = {
    "L1": "本体",
    "L2": "物体",
    "L3": "过程",
}

DIMENSIONS: list[Dim] = [
    {
        "id": "L1a",
        "group": "L1",
        "name": "本体副本/重影",
        **BINARY,
    },
    {
        "id": "L1b",
        "group": "L1",
        "name": "本体消失/淡出",
        **BINARY,
    },
    {
        "id": "L1c",
        "group": "L1",
        "name": "结构或材质异常",
        **GRADED,
    },
    {
        "id": "L2a",
        "group": "L2",
        "name": "物体副本/重影/分裂/凭空生成",
        **BINARY,
    },
    {
        "id": "L2b",
        "group": "L2",
        "name": "物体消失/淡出",
        **BINARY,
    },
    {
        "id": "L2c",
        "group": "L2",
        "name": "不合理形变/材质/纹理异常",
        **GRADED,
        "question": "回答视频中物体有无不合理形变/材质/纹理异常",
    },
    {
        "id": "L3a",
        "group": "L3",
        "name": "接触与动作不符",
        **BINARY,
        "question": "回答机械臂与物体有无接触与动作不符的异常",
    },
    {
        "id": "L3b",
        "group": "L3",
        "name": "物理违例",
        **BINARY,
    },
    {
        "id": "gripper",
        "group": "gripper",
        "name": "夹爪开闭",
        **BINARY,
    },
]

# group value = worst member value (folded at answer time)
GROUP_DIMS: dict[str, Dim] = {
    "L1": {
        "id": "L1",
        "group": "L1",
        "name": "本体异常",
        "members": ["L1a", "L1b", "L1c"],
        **GRADED,
        "question": "回答视频中本体（机械臂自身）有无异常，比如副本、重影、消失、淡出、结构或材质异常",
    },
    "L2": {
        "id": "L2",
        "group": "L2",
        "name": "物体异常",
        "members": ["L2a", "L2b", "L2c"],
        **GRADED,
        "question": "回答视频中物体有无异常，比如物体副本、重影、分裂、凭空生成、消失、不合理形变、材质或纹理异常",
    },
    "L3": {
        "id": "L3",
        "group": "L3",
        "name": "过程异常",
        "members": ["L3a", "L3b"],
        **BINARY,
        "question": "回答视频中过程有无接触与动作不符或者物理违例",
    },
}

DIM_BY_ID: dict[str, Dim] = {d["id"]: d for d in DIMENSIONS}
DIM_BY_ID.update(GROUP_DIMS)

def dimension_ids() -> list[str]:
    return [d["id"] for d in DIMENSIONS]

def group_value(gid: str, dims_dict: dict[str, str]) -> str:
    g = GROUP_DIMS[gid]
    worst = min(DIM_BY_ID[m]["weights"][dims_dict.get(m, "无")] for m in g["members"])
    return {w: v for v, w in g["weights"].items()}[worst]

def all_values() -> set[str]:
    vals: set[str] = set()
    for d in DIMENSIONS:
        vals.update(d["values"])
    return vals

def severity_token_ids(tokenizer: Any) -> dict[str, int]:
    ids: dict[str, int] = {}
    for value in all_values():
        enc = tokenizer.encode(value, add_special_tokens=False)
        assert len(enc) == 1, f"{value!r} tokenizes to {enc}, expected single token"
        ids[value] = enc[0]
    return ids

ZOOM_PREFIX = "先在视频中定位夹爪动作的时空区域，再"

def _selected(dims: list[str] | None) -> list[Dim]:
    if dims is None:
        return DIMENSIONS
    out = [DIM_BY_ID[d] for d in dims]
    orphan = [d["id"] for d in out if d["group"] not in GROUPS]
    if orphan:
        raise ValueError(f"维度 {orphan} 的 group 不在 GROUPS 里，无法渲染进 prompt")
    return out

def _dims_block(dims: list[str] | None = None) -> str:
    selected = _selected(dims)
    lines: list[str] = []
    for gid, gname in GROUPS.items():
        in_group = [d for d in selected if d["group"] == gid]
        if not in_group:
            continue
        lines.append(f"  {gid} {gname}:")
        for d in in_group:
            lines.append(f"    {d['id']} {d['name']}: {' / '.join(d['values'])}")
    return "\n".join(lines)

def _example_json(dims: list[str] | None = None) -> str:
    parts = [f'"{d["id"]}": "无"' for d in _selected(dims)]
    return "{" + ", ".join(parts) + "}"

def build_prompt(thinking: bool = False, with_video_tag: bool = False,
                 dims: list[str] | None = None, zoom: bool = False) -> str:
    n = len(_selected(dims))
    head = f"{ZOOM_PREFIX if zoom else ''}逐项评分，各项取值如下：\n"
    block = _dims_block(dims)
    if thinking:
        tail = ("\n先在 <think>...</think> 中分析：说明操作物体名称，"
                "再逐项分析，必要时引用时间(秒)；"
                f"然后仅以 JSON 输出这 {n} 个失效项，格式如:\n{_example_json(dims)}")
    else:
        tail = f"\n仅以 JSON 输出这 {n} 个失效项，格式如:\n{_example_json(dims)}\n不要输出其他内容。"
    prompt = head + block + tail
    if with_video_tag:
        prompt = "<video>\n" + prompt
    return prompt

def dim_prompt(dim_id: str, zoom: bool = False, with_video_tag: bool = True) -> str:
    d = DIM_BY_ID[dim_id]
    assert "question" in d, f"维度 {dim_id} 未定义 question 字段"
    vals = "、".join(d["values"][:-1]) + f" 或 {d['values'][-1]}"
    head = ZOOM_PREFIX if zoom else ""
    p = f"{head}{d['question']}，值可以是 {vals}"
    return f"<video>\n{p}" if with_video_tag else p
