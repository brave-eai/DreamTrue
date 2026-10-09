"""Dataset loading plus import-time vision sampling alignment.

Patches qwen_vl_utils.vision_process from environment variables the same way
ms-swift does, so evaluation frame sampling matches training.
"""

import json
import os
from typing import Any

from qwen_vl_utils import vision_process

_VISION_ENV_KEYS: list[str] = [
    "image_factor",
    "min_pixels",
    "max_pixels",
    "video_min_pixels",
    "video_max_pixels",
    "video_total_pixels",
    "max_ratio",
    "frame_factor",
    "fps",
    "fps_min_frames",
    "fps_max_frames",
    # qwen3_vl
    "image_max_token_num",
    "image_min_token_num",
    "spatial_merge_size",
    "video_max_token_num",
    "video_min_token_num",
]

def __patch_vision_process_from_env() -> dict[str, float]:
    # Same fallback as swift; see https://github.com/QwenLM/Qwen2.5-VL/issues/1120
    if os.getenv("VIDEO_MAX_PIXELS") and not os.getenv("VIDEO_TOTAL_PIXELS"):
        os.environ["VIDEO_TOTAL_PIXELS"] = str(int(128000 * 28 * 28 * 0.9))

    applied: dict[str, float] = {}
    for key in _VISION_ENV_KEYS:
        env_name = key.upper()
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        if getattr(vision_process, env_name, None) is None:  # unsupported by this version
            continue
        type_func = float if key == "fps" else int
        val = type_func(raw)
        setattr(vision_process, env_name, val)
        applied[env_name] = val
    if applied:
        print(f"[vision] Patched vision_process from env: {applied}", flush=True)
    return applied

# Import-time patch: env vars are set by the shell before python starts, so the
# timing is correct. No relevant env -> pure no-op. Idempotent.
__patch_vision_process_from_env()

def load_datasets(paths: list[str]) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for jf in paths:
        with open(jf, encoding="utf-8") as f:
            part = json.load(f)
        print(f"[data] Loaded {len(part)} samples from {jf}", flush=True)
        dataset.extend(part)
    return dataset
