"""VACE train/inference shared contract.

Single owner of the semantics that training and inference must agree on
bit-for-bit: three-view ordering, temporal mask construction, condition mode
normalization, and padding-trim rules.

Dependencies are deliberately light (stdlib + PIL only, no torch), so any
tool in the repo can import this module cheaply.

Canonical normalization behavior = the historical demo implementations plus
the two lenient forms accepted by prepare_cache.py ("condition.h5" and
literal filenames like "condition-xxx.h5"). Deliberately NOT migrated to
this module (they keep their own copies):
  - Test/cache_utils/validate_test_cache.py (stricter on purpose: validation)
  - Test/build_test_cache.py, Test/build_test_cache_challenge.py (zero-dependency
    standalone tools)
"""

import argparse
import os
import re

from PIL import Image

# ---------------------------------------------------------------------------
# Three-view ordering
# ---------------------------------------------------------------------------

# Order is load-bearing: the dataset emits head/hand_left/hand_right_* keys,
# training concatenates view latents in this order, and inference must match.
VIEW_NAMES = ("head", "hand_left", "hand_right")


def view_keys(suffix):
    """view_keys("video") -> ("head_video", "hand_left_video", "hand_right_video")."""
    return tuple(f"{name}_{suffix}" for name in VIEW_NAMES)


# ---------------------------------------------------------------------------
# Temporal mask construction
# ---------------------------------------------------------------------------

def temporal_mask_pattern(num_frames, n_real_frames=None, pad_mode="front"):
    """Temporal 0/1 mask pattern (False = preserve, True = generate).

    Default (no padding): [0, 1, 1, ..., 1] - first frame is the reference.

    pad_short_actions with pad_mode="front" and n_real_frames < num_frames:
        [0] * (num_frames - n_real + 1) + [1] * (n_real - 1)
        leading padding frames and the reference frame are preserved.

    pad_mode="back": identical to the default mask; the tail-repeated
    condition frames only act as guidance.
    """
    if n_real_frames is not None and n_real_frames < num_frames:
        if pad_mode != "back":
            n_preserve = num_frames - n_real_frames + 1
            n_generate = n_real_frames - 1
            return [False] * n_preserve + [True] * n_generate
    return [False] + [True] * (num_frames - 1)


def render_mask_frames(pattern, width, height):
    """Render a temporal pattern as PIL "L" frames (0 = black, 255 = white).

    The pipeline requires 2D (H, W) mask frames, hence mode "L"."""
    black_frame = Image.new("L", (width, height), 0)
    white_frame = Image.new("L", (width, height), 255)
    return [white_frame if generate else black_frame for generate in pattern]


def build_mask_frames(num_frames, width, height, n_real_frames=None, pad_mode="front"):
    """Convenience wrapper: temporal_mask_pattern + render_mask_frames."""
    return render_mask_frames(
        temporal_mask_pattern(num_frames, n_real_frames=n_real_frames, pad_mode=pad_mode),
        width,
        height,
    )


# ---------------------------------------------------------------------------
# Condition mode normalization
# ---------------------------------------------------------------------------

_H5_CONDITION_MODE_PATTERN = re.compile(r"^condition[a-z0-9_-]*_h5$")
_H5_CONDITION_FILENAME_PATTERN = re.compile(r"^(condition[a-z0-9_-]*)\.h5$")


def normalize_vace_condition_mode(mode):
    mode = str(mode or "").strip().lower()
    if mode in ("normal", "normal_png", "png"):
        return "normal_png"
    if mode in ("condition_h5", "h5", "condition.h5"):
        return "condition_h5"
    if _H5_CONDITION_MODE_PATTERN.fullmatch(mode):
        return mode
    # Accept the literal h5 filename form (e.g. "condition-agibot_action.h5")
    # by rewriting the trailing ".h5" into the canonical "_h5" suffix.
    m = _H5_CONDITION_FILENAME_PATTERN.fullmatch(mode)
    if m:
        return f"{m.group(1)}_h5"
    if mode in ("", "0", "false", "off", "none", "disabled"):
        return "disabled"
    raise ValueError(
        f"Unsupported --vace_condition_mode: {mode}. "
        "Supported: normal_png, condition_h5, condition*_h5, condition*.h5, disabled"
    )


def parse_vace_condition_mode(value):
    try:
        return normalize_vace_condition_mode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def is_h5_condition_mode(mode):
    return mode == "condition_h5" or bool(_H5_CONDITION_MODE_PATTERN.fullmatch(mode))


# ---------------------------------------------------------------------------
# Padding trim rules (pure part; file IO stays with the callers)
# ---------------------------------------------------------------------------

def infer_n_real_frames_from_indices(sample_indices, pad_mode):
    """Infer the number of real (non-padding) frames from sample indices.

    front padding repeats the first index at the head; back padding repeats
    the last index at the tail. One occurrence is the real frame itself."""
    if not sample_indices:
        return None
    try:
        indices = [int(x) for x in sample_indices]
    except (TypeError, ValueError):
        return None
    if not indices:
        return None

    total_frames = len(indices)
    if pad_mode == "back":
        last_index = indices[-1]
        duplicate_count = 0
        for frame_index in reversed(indices):
            if frame_index != last_index:
                break
            duplicate_count += 1
    else:
        first_index = indices[0]
        duplicate_count = 0
        for frame_index in indices:
            if frame_index != first_index:
                break
            duplicate_count += 1

    padding_count = max(0, duplicate_count - 1)
    n_real_frames = total_frames - padding_count
    return n_real_frames if 0 < n_real_frames <= total_frames else None


def to_positive_int(value):
    if value is None:
        return None
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return None
    return int_value if int_value > 0 else None


def resolve_padding_trim_info(sample, clip_info, total_frames, default_pad_mode):
    """Decide how many frames to keep when trimming padding from outputs."""
    clip_info = clip_info or {}
    pad_mode = sample.get("pad_mode") or clip_info.get("pad_mode") or default_pad_mode or "front"
    if pad_mode not in {"front", "back"}:
        pad_mode = "front"

    n_real_frames = to_positive_int(sample.get("n_real_frames"))
    if n_real_frames is None:
        n_real_frames = to_positive_int(clip_info.get("n_real_frames"))
    if n_real_frames is None:
        n_real_frames = infer_n_real_frames_from_indices(
            clip_info.get("sample_indices"),
            pad_mode,
        )

    original_num_frames = int(total_frames or 0)
    trim_padding_applied = False
    saved_num_frames = original_num_frames
    trimmed_count = 0

    if n_real_frames is not None and 0 < n_real_frames < original_num_frames:
        trim_padding_applied = True
        saved_num_frames = n_real_frames
        trimmed_count = original_num_frames - saved_num_frames

    return {
        "pad_mode": pad_mode,
        "n_real_frames": n_real_frames,
        "original_num_frames": original_num_frames,
        "saved_num_frames": saved_num_frames,
        "trim_padding_applied": trim_padding_applied,
        "trimmed_count": trimmed_count,
    }


def trim_padding_frames(frames, trim_info):
    """Drop padding frames according to trim_info (front: keep tail; back: keep head)."""
    if frames is None:
        return None
    frame_list = list(frames)
    if not frame_list or not trim_info.get("trim_padding_applied"):
        return frame_list

    saved_num_frames = min(trim_info["saved_num_frames"], len(frame_list))
    if saved_num_frames <= 0:
        return []
    if trim_info.get("pad_mode") == "back":
        return frame_list[:saved_num_frames]
    return frame_list[-saved_num_frames:]


# ---------------------------------------------------------------------------
# Environment variable channel (read side only; writers stay untouched)
# ---------------------------------------------------------------------------

CONDITION_MODE_ENV = "AGIBOT_VACE_CONDITION_MODE"
CONDITION_MODE_ENV_FALLBACK = "DATA_VACE_CONDITION_MODE"


def raw_condition_mode_env():
    """Raw value from the env channel (primary then fallback), no normalization."""
    return os.environ.get(CONDITION_MODE_ENV) or os.environ.get(CONDITION_MODE_ENV_FALLBACK)


def read_condition_mode_env(default="disabled"):
    return normalize_vace_condition_mode(raw_condition_mode_env() or default)
