"""V2V_VACE Demo 推理脚本：基于预提取 cache.pkl 推理固定样本。

V2V-VACE 纯推理入口，支持固定长度三视角 case。
- DIT 来自 VACE-14B (T2V, in_dim=16)，DIFFSYNTH_SKIP_VACE_DIT=false
- 使用 dual LoRA checkpoint (pipe.dit.* + pipe.vace.*)
- 首帧通过 vace_reference_image 注入（而非 I2V input_image）
- 不需要 CLIP 图像编码器
- Mask 自动生成: 首帧=0 (保留), 其余=1 (生成)

Usage:
    python Script/Demo/inference_demo_v2v_vace.py \\
        --cache_files Test/challenge_test_cache.pkl \\
        --dataset_base_path /path/to/test \\
        --dual_lora_checkpoint /path/to/checkpoint.safetensors \\
        --model_id_with_origin_paths "..." \\
        --output_path ./outputs/inference/v2v_vace_challenge \\
        ...
"""

# ── 在所有 import 之前设置离线模式，防止 ModelScope/HuggingFace 联网 ──
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import datetime
import json
import os
import re
import socket
import sys
import atexit
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, InitProcessGroupKwargs
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
# 无论 _PROJECT_ROOT 是否已在 sys.path 中，都强制置于最前，
# 避免远端机器的 PYTHONPATH 中存在同名旧版本包而被优先加载。
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video
from examples.wanvideo.model_training.data.datasets import AgibotWorldDataset
from examples.wanvideo.model_training import vace_contract
from examples.wanvideo.model_training.vace_contract import (
    is_h5_condition_mode,
    normalize_vace_condition_mode,
    parse_vace_condition_mode,
)


def _file_based_barrier(output_dir, rank, world_size, timeout=7200, poll_interval=5, tag="default"):
    """基于文件的进程同步屏障，替代 NCCL barrier 避免超时。

    当各进程的推理样本数不均或模型加载耗时不均时，先完成的进程
    会在 NCCL barrier 处等待，若等待超过 NCCL timeout（默认 600s），
    进程会被 watchdog 杀掉。此函数通过文件轮询实现等待，不受 NCCL
    timeout 限制。

    注意：此函数不会清理标记文件，避免竞态条件。
    请在所有 barrier 完成后调用 _cleanup_barrier_files() 统一清理。

    Args:
        tag: 屏障标识，同一次运行中不同阶段的屏障使用不同的 tag
             以避免标记文件冲突。
    """
    marker = os.path.join(output_dir, f".barrier_{tag}_rank_{rank}")
    with open(marker, "w") as f:
        f.write(f"done {time.time()}")

    start = time.time()
    while True:
        all_done = all(
            os.path.exists(os.path.join(output_dir, f".barrier_{tag}_rank_{i}"))
            for i in range(world_size)
        )
        if all_done:
            break
        elapsed = time.time() - start
        if elapsed > timeout:
            missing = [
                i for i in range(world_size)
                if not os.path.exists(os.path.join(output_dir, f".barrier_{tag}_rank_{i}"))
            ]
            raise TimeoutError(
                f"File-based barrier '{tag}' timed out after {timeout}s. "
                f"Missing ranks: {missing}"
            )
        time.sleep(poll_interval)


def _cleanup_barrier_files(output_dir):
    """清理输出目录中所有 .barrier_* 标记文件。"""
    import glob as _glob
    for p in _glob.glob(os.path.join(output_dir, ".barrier_*")):
        try:
            os.remove(p)
        except OSError:
            pass


# ─── 日志工具 ─────────────────────────────────────────────────────────────

_STEP_LINE_PATTERN = re.compile(
    r"\b(step|steps|iter|iteration|epoch|progress|loss)\b|\b\d+/\d+\b|%",
    re.IGNORECASE,
)
_ORIGINAL_STDOUT = sys.__stdout__
_ORIGINAL_STDERR = sys.__stderr__
_RUN_LOG_FILE = None
_RUN_LOG_PATH = None
_RUN_LOG_HOOKED = False


def _is_step_line(text: str) -> bool:
    return bool(_STEP_LINE_PATTERN.search(text))


class _LogOnlyStream:
    def __init__(self, log_file, fallback_stream):
        self.log_file = log_file
        self.fallback_stream = fallback_stream

    @property
    def encoding(self):
        return getattr(self.fallback_stream, "encoding", "utf-8")

    def writable(self):
        return True

    def isatty(self):
        return False

    def fileno(self):
        return self.fallback_stream.fileno()

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.log_file.flush()


class _StepOnlyStdout(_LogOnlyStream):
    def __init__(self, log_file, fallback_stream):
        super().__init__(log_file, fallback_stream)
        self._line_buffer = ""

    def _mirror_segment(self, segment: str):
        if segment and _is_step_line(segment):
            self.fallback_stream.write(segment + "\n")
            self.fallback_stream.flush()

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        self.log_file.write(text)
        self.log_file.flush()
        self._line_buffer += text
        while True:
            newline_pos = self._line_buffer.find("\n")
            carriage_pos = self._line_buffer.find("\r")
            split_pos = -1
            if newline_pos >= 0 and carriage_pos >= 0:
                split_pos = min(newline_pos, carriage_pos)
            elif newline_pos >= 0:
                split_pos = newline_pos
            elif carriage_pos >= 0:
                split_pos = carriage_pos
            if split_pos < 0:
                break
            segment = self._line_buffer[:split_pos]
            self._line_buffer = self._line_buffer[split_pos + 1:]
            self._mirror_segment(segment)
        return len(text)

    def flush(self):
        if self._line_buffer:
            self._mirror_segment(self._line_buffer)
            self._line_buffer = ""
        super().flush()
        self.fallback_stream.flush()


def _close_run_log_file():
    global _RUN_LOG_FILE
    if _RUN_LOG_FILE is not None:
        try:
            _RUN_LOG_FILE.flush()
            _RUN_LOG_FILE.close()
        except Exception:
            pass
        _RUN_LOG_FILE = None


def setup_run_log_streams(run_output_path: str, log_name: str = "run.log") -> str:
    global _RUN_LOG_FILE, _RUN_LOG_PATH, _RUN_LOG_HOOKED
    os.makedirs(run_output_path, exist_ok=True)
    target_log_path = os.path.join(run_output_path, log_name)
    if _RUN_LOG_PATH == target_log_path and _RUN_LOG_FILE is not None:
        return target_log_path
    _close_run_log_file()
    _RUN_LOG_FILE = open(target_log_path, "a", encoding="utf-8", buffering=1)
    _RUN_LOG_PATH = target_log_path
    sys.stdout = _StepOnlyStdout(_RUN_LOG_FILE, _ORIGINAL_STDOUT)
    sys.stderr = _LogOnlyStream(_RUN_LOG_FILE, _ORIGINAL_STDERR)
    if not _RUN_LOG_HOOKED:
        atexit.register(_close_run_log_file)
        _RUN_LOG_HOOKED = True
    return target_log_path


# ─── 工具函数 ─────────────────────────────────────────────────────────────

def sanitize_filename(text: str) -> str:
    if text is None:
        return "action"
    text = text.strip()
    if not text:
        return "action"
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "action"


def _to_rgb_pil(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    if isinstance(frame, np.ndarray):
        array = frame
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim != 3:
            raise ValueError(f"Unsupported frame shape: {array.shape}")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]
        elif array.shape[-1] != 3:
            raise ValueError(f"Unsupported frame channels: {array.shape[-1]}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")
    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _stack_video_rows(row_videos):
    valid_rows = [(name, frames) for name, frames in row_videos if frames is not None]
    if not valid_rows:
        raise ValueError("Cannot stack videos without any frame sequences")
    row_lengths = {name: len(frames) for name, frames in valid_rows}
    stacked_len = min(row_lengths.values())
    if stacked_len <= 0:
        length_desc = ", ".join(f"{name}={length}" for name, length in row_lengths.items())
        raise ValueError(f"Cannot stack videos with empty frames: {length_desc}")
    stacked_frames = []
    for frame_index in range(stacked_len):
        row_images = [_to_rgb_pil(frames[frame_index]) for _, frames in valid_rows]
        width, height = row_images[0].size
        normalized_images = []
        for image in row_images:
            if image.size != (width, height):
                image = image.resize((width, height), Image.BICUBIC)
            normalized_images.append(image)
        stacked_image = Image.new("RGB", (width, height * len(normalized_images)))
        for row_index, image in enumerate(normalized_images):
            stacked_image.paste(image, (0, row_index * height))
        stacked_frames.append(stacked_image)
    return stacked_frames


def _concat_views_horizontally(per_view_frames):
    """Per-frame 横向 concat 多视角帧 (沿宽度方向, 顺序按 per_view_frames 列表)。

    输入: per_view_frames = [view0_frames, view1_frames, ...], 每个 view 是 list of frames。
    输出: list of PIL.Image, 每帧形状 (H, W*num_views)。

    与训练侧 WanVideoUnit_VACE 的 torch.cat(view_contexts, dim=-1) 同向。
    高度若不一致, 以第 1 个视角为准对其他视角做 BICUBIC 缩放。
    任一视角为 None / 空时, 返回 None (调用方可据此跳过保存)。
    """
    if not per_view_frames or any(v is None or len(v) == 0 for v in per_view_frames):
        return None
    num_frames = min(len(v) for v in per_view_frames)
    if num_frames <= 0:
        return None
    concat_frames = []
    for fi in range(num_frames):
        view_imgs = [_to_rgb_pil(v[fi]) for v in per_view_frames]
        target_h = view_imgs[0].height
        normalized = []
        for img in view_imgs:
            if img.height != target_h:
                ratio = target_h / img.height
                new_w = max(1, int(round(img.width * ratio)))
                img = img.resize((new_w, target_h), Image.BICUBIC)
            normalized.append(img)
        total_w = sum(img.width for img in normalized)
        canvas = Image.new("RGB", (total_w, target_h))
        x_off = 0
        for img in normalized:
            canvas.paste(img, (x_off, 0))
            x_off += img.width
        concat_frames.append(canvas)
    return concat_frames


def _build_auto_mask_frames(num_frames: int, height: int, width: int,
                            n_real_frames: int = None, pad_mode: str = "front"):
    """构建自动 mask 帧序列，支持前 padding 和后 padding 模式。

    当 n_real_frames 指定且 < num_frames 时：
        pad_mode="front" (前 padding):
            mask = [0] × (num_frames - n_real + 1) + [1] × (n_real - 1)
            前 (num_frames - n_real) 帧为 padding（mask=0），
            第 (num_frames - n_real) 帧为 reference（mask=0），
            最后 (n_real - 1) 帧为生成区域（mask=1）。

        pad_mode="back" (后 padding):
            mask = [0] + [1] × (num_frames - 1)
            与无 padding 时的默认 mask 相同。
            首帧=reference（mask=0），其余帧全部生成（mask=1），
            padding 区域的条件帧（尾帧重复）仅作为条件引导。

    否则使用默认行为:
        mask = [0, 1, 1, ..., 1]（首帧=0，其余=1）

    使用 "L" 灰度模式生成 2D mask，与训练时 train_v2v_vace.py 一致，
    pipeline 验证要求 mask 帧必须为 2D (H, W)。
    """
    return vace_contract.build_mask_frames(
        num_frames, width, height, n_real_frames=n_real_frames, pad_mode=pad_mode
    )


def _to_positive_int(value):
    return vace_contract.to_positive_int(value)


def _infer_n_real_frames_from_indices(sample_indices, pad_mode):
    return vace_contract.infer_n_real_frames_from_indices(sample_indices, pad_mode)


def _resolve_padding_trim_info(sample, clip_info, total_frames, default_pad_mode):
    return vace_contract.resolve_padding_trim_info(
        sample, clip_info, total_frames, default_pad_mode
    )


def _trim_padding_frames(frames, trim_info):
    return vace_contract.trim_padding_frames(frames, trim_info)


def _select_eval_indices(dataset_len, samples_per_source):
    if dataset_len <= 0:
        return []
    if samples_per_source is None or int(samples_per_source) <= 0:
        return range(dataset_len)
    count = min(int(samples_per_source), dataset_len)
    if count == 1:
        return [0]
    return np.linspace(0, dataset_len - 1, count, dtype=int).tolist()


def _dataset_config_source_order(dataset_config, requested_sources=None):
    if requested_sources:
        sources = list(dict.fromkeys(requested_sources))
    else:
        sources = []
        for entry in dataset_config.get("train_mix", []):
            ref = entry.get("ref")
            if ref and ref not in sources:
                sources.append(ref)
        for name in dataset_config.get("datasets", {}):
            if name not in sources:
                sources.append(name)
    missing = [name for name in sources if name not in dataset_config.get("datasets", {})]
    if missing:
        raise ValueError(
            f"--source_names contains unknown dataset ref(s): {missing}. "
            f"Available: {list(dataset_config.get('datasets', {}))}"
        )
    return sources


def _select_candidate_positions(total_count, max_samples, random_sample, seed):
    if total_count <= 0:
        return []
    if max_samples is None or int(max_samples) <= 0:
        return list(range(total_count))

    count = min(int(max_samples), total_count)
    if count >= total_count:
        return list(range(total_count))
    if random_sample:
        sample_seed = None
        if seed is not None and int(seed) >= 0:
            sample_seed = int(seed)
        rng = np.random.default_rng(sample_seed)
        return rng.choice(total_count, size=count, replace=False).tolist()
    return list(range(count))


def _allocate_by_target_size(source_entries, target_size_map, max_samples,
                              random_sample, seed):
    """按 target_size 比例分配 max_samples 到各 source，再从每个 source 中抽样。"""
    if max_samples is None or int(max_samples) <= 0:
        eval_items = []
        budgets = {}
        global_index = 0
        for entry in source_entries:
            budgets[entry["source_name"]] = entry["candidate_count"]
            for local_idx in range(entry["candidate_count"]):
                data_index = int(entry["indices"][local_idx])
                eval_items.append({
                    "global_index": global_index,
                    "pool_index": global_index,
                    "source_name": entry["source_name"],
                    "dataset": entry["dataset"],
                    "data_index": data_index,
                })
                global_index += 1
        return eval_items, budgets

    max_samples = int(max_samples)

    weights = []
    for entry in source_entries:
        ts = target_size_map.get(entry["source_name"], entry["candidate_count"])
        weights.append(max(ts, 1))
    total_weight = sum(weights)

    budgets_list = [0] * len(source_entries)
    for i in range(len(source_entries)):
        raw = max_samples * weights[i] / total_weight
        budgets_list[i] = min(int(raw), source_entries[i]["candidate_count"])

    remainder = max_samples - sum(budgets_list)
    if remainder > 0:
        fractional = []
        for i in range(len(source_entries)):
            raw = max_samples * weights[i] / total_weight
            frac = raw - int(raw)
            has_room = budgets_list[i] < source_entries[i]["candidate_count"]
            fractional.append((frac if has_room else -1, i))
        fractional.sort(key=lambda x: -x[0])
        for _, i in fractional:
            if remainder <= 0:
                break
            if budgets_list[i] < source_entries[i]["candidate_count"]:
                budgets_list[i] += 1
                remainder -= 1

    sample_seed = None
    if seed is not None and int(seed) >= 0:
        sample_seed = int(seed)
    rng = np.random.default_rng(sample_seed) if random_sample else None

    eval_items = []
    per_source_budgets = {}
    global_index = 0
    for i, entry in enumerate(source_entries):
        budget = budgets_list[i]
        per_source_budgets[entry["source_name"]] = budget
        candidate_count = entry["candidate_count"]

        if budget <= 0:
            continue
        if budget >= candidate_count:
            selected_local = list(range(candidate_count))
        elif rng is not None:
            selected_local = rng.choice(
                candidate_count, size=budget, replace=False,
            ).tolist()
        else:
            selected_local = list(range(budget))

        for local_idx in selected_local:
            data_index = int(entry["indices"][local_idx])
            eval_items.append({
                "global_index": global_index,
                "pool_index": global_index,
                "source_name": entry["source_name"],
                "dataset": entry["dataset"],
                "data_index": data_index,
            })
            global_index += 1

    return eval_items, per_source_budgets


def _summarize_dataset_config_selection(source_entries, eval_items,
                                         target_size_map=None, budgets=None):
    selected_by_source = {entry["source_name"]: [] for entry in source_entries}
    for item in eval_items:
        selected_by_source.setdefault(item["source_name"], []).append(int(item["data_index"]))

    summaries = []
    for entry in source_entries:
        selected_indices = selected_by_source.get(entry["source_name"], [])
        summary = {
            "source_name": entry["source_name"],
            "type": entry["type"],
            "dataset_len": entry["dataset_len"],
            "candidate_count": entry["candidate_count"],
            "selected_count": len(selected_indices),
        }
        if target_size_map:
            summary["target_size"] = target_size_map.get(entry["source_name"])
        if budgets:
            summary["allocated_budget"] = budgets.get(entry["source_name"])
        if len(selected_indices) <= 10000:
            summary["selected_indices"] = selected_indices
        else:
            summary["selected_indices_preview"] = selected_indices[:10000]
            summary["selected_indices_truncated"] = True
        summaries.append(summary)
    return summaries


def _build_dataset_config_items(args):
    import bisect

    from examples.wanvideo.model_training.data.dataset_factory import (
        build_single_dataset,
        load_dataset_config,
    )

    dataset_config = load_dataset_config(args.dataset_config)
    source_order = _dataset_config_source_order(dataset_config, args.source_names)

    target_size_map = {
        entry["ref"]: int(entry["target_size"])
        for entry in dataset_config.get("train_mix", [])
        if "target_size" in entry
    }

    common_kwargs = {
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "repeat": args.dataset_repeat,
        "min_interval": args.min_interval,
        "max_interval": args.max_interval,
        "load_workers": args.load_workers,
        "pad_short_actions": args.pad_short_actions,
        "random_sample_start": args.random_sample_start,
        "detail_prompt": args.detail_prompt,
        "use_plucker": args.use_plucker,
    }

    source_entries = []
    total_candidates = 0
    for source_name in source_order:
        source_cfg = dataset_config["datasets"][source_name]
        source_kwargs = dict(common_kwargs)
        print(
            f"[DatasetConfig] Building source '{source_name}' "
            f"(type={source_cfg.get('type')})..."
        )
        dataset = build_single_dataset(source_cfg, source_kwargs)
        dataset_len = len(dataset)
        indices = _select_eval_indices(dataset_len, args.samples_per_source)
        candidate_count = len(indices)
        total_candidates += candidate_count
        source_entries.append(
            {
                "source_name": source_name,
                "type": source_cfg.get("type"),
                "dataset": dataset,
                "dataset_len": dataset_len,
                "indices": indices,
                "candidate_count": candidate_count,
            }
        )
        ts_info = f", target_size={target_size_map[source_name]}" if source_name in target_size_map else ""
        print(
            f"[DatasetConfig] source='{source_name}' len={dataset_len}, "
            f"candidates={candidate_count}{ts_info}"
        )

    use_target_weight = bool(target_size_map) and all(
        name in target_size_map for name in source_order
    )

    if use_target_weight:
        eval_items, budgets = _allocate_by_target_size(
            source_entries, target_size_map,
            args.max_samples, args.random_sample, args.seed,
        )
    else:
        budgets = None
        source_offsets = []
        offset = 0
        for entry in source_entries:
            source_offsets.append(offset)
            offset += entry["candidate_count"]

        selected_positions = _select_candidate_positions(
            total_candidates,
            args.max_samples,
            args.random_sample,
            args.seed,
        )

        eval_items = []
        for sample_index, candidate_pos in enumerate(selected_positions):
            source_idx = bisect.bisect_right(source_offsets, candidate_pos) - 1
            source_entry = source_entries[source_idx]
            source_local_offset = candidate_pos - source_offsets[source_idx]
            data_index = int(source_entry["indices"][source_local_offset])
            eval_items.append(
                {
                    "global_index": sample_index,
                    "pool_index": int(candidate_pos),
                    "source_name": source_entry["source_name"],
                    "dataset": source_entry["dataset"],
                    "data_index": data_index,
                }
            )

    if not eval_items:
        raise ValueError(
            f"No samples selected from dataset_config={args.dataset_config}. "
            "Check manifest contents, --samples_per_source, and --max_samples."
        )

    source_summaries = _summarize_dataset_config_selection(
        source_entries, eval_items,
        target_size_map=target_size_map if use_target_weight else None,
        budgets=budgets,
    )
    weight_mode = "target_size_weighted" if use_target_weight else "uniform"
    print(
        f"[DatasetConfig] total_candidates={total_candidates}, "
        f"selected={len(eval_items)}, max_samples={args.max_samples}, "
        f"random_sample={args.random_sample}, weight_mode={weight_mode}"
    )
    return eval_items, source_summaries


def _build_case_manifest_items(args):
    from examples.wanvideo.model_training.data.dataset_factory import (
        build_mixed_dataset,
        load_dataset_config,
    )

    with open(args.case_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError(f"No cases found in manifest: {args.case_manifest}")

    manifest_config_path = manifest.get("dataset_config")
    dataset_config_path = args.dataset_config or manifest_config_path
    if not dataset_config_path:
        raise ValueError("A dataset config is required for manifest-driven inference")
    if manifest_config_path and os.path.realpath(dataset_config_path) != os.path.realpath(manifest_config_path):
        raise ValueError(
            f"Dataset config does not match manifest: {dataset_config_path} != {manifest_config_path}"
        )
    dataset_config = load_dataset_config(dataset_config_path)
    dataset = build_mixed_dataset(
        dataset_config,
        {
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "repeat": args.dataset_repeat,
            "min_interval": args.min_interval,
            "max_interval": args.max_interval,
            "load_workers": args.load_workers,
            "pad_short_actions": args.pad_short_actions,
            "random_sample_start": args.random_sample_start,
            "detail_prompt": args.detail_prompt,
            "use_plucker": args.use_plucker,
        },
    )
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)

    eval_items = []
    for case in cases:
        data_index = int(case["global_dataset_index"])
        if data_index < 0 or data_index >= len(dataset):
            raise IndexError(
                f"Manifest index {data_index} is outside dataset length {len(dataset)}"
            )
        eval_items.append(
            {
                "global_index": int(case["case_id"]),
                "pool_index": data_index,
                "source_name": case.get("source_name"),
                "dataset": dataset,
                "data_index": data_index,
                "condition_id": str(case["condition_id"]),
                "seed": int(case["seed"]),
                "n_real_frames": case.get("n_real_frames"),
            }
        )

    return eval_items, [{"mode": "case_manifest", "path": args.case_manifest, "count": len(eval_items)}]


def _build_cache_items(dataset):
    return [
        {
            "global_index": idx,
            "source_name": None,
            "dataset": dataset,
            "data_index": idx,
        }
        for idx in range(len(dataset))
    ]


def _clip_info_for_dataset_index(dataset, data_index):
    scenes = getattr(dataset, "scenes", None)
    if not scenes:
        return {}
    scene_index = int(data_index) % len(scenes)
    return scenes[scene_index]


# ─── V2V_VACE 推理模块 ───────────────────────────────────────────────────

class WanV2VVaceInferenceModule:
    """
    V2V_VACE 推理模块。

    - DIT 来自 VACE-14B (T2V, in_dim=16)，DIFFSYNTH_SKIP_VACE_DIT=false
    - 使用 dual LoRA checkpoint (pipe.dit.* + pipe.vace.*)
    - 首帧通过 vace_reference_image 注入
    """

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        dual_lora_checkpoint=None,
        dual_lora_alpha=1.0,
        dit_lora_path=None,
        dit_lora_alpha=1.0,
        vace_lora_path=None,
        vace_lora_alpha=1.0,
        vram_limit=None,
        torch_dtype=torch.bfloat16,
        device="cpu",
        use_plucker=False,
    ):
        self.use_plucker = bool(use_plucker)
        model_configs = []
        if model_paths is not None:
            model_configs.extend(ModelConfig(path=path) for path in json.loads(model_paths))
        if model_id_with_origin_paths is not None:
            for item in model_id_with_origin_paths.split(","):
                model_id, origin_pattern = item.split(":", 1)
                # 当 origin_pattern 是绝对路径时，直接用 glob 解析本地文件，
                # 跳过 ModelScope 下载（否则 ModelConfig 会调用 snapshot_download
                # 联系 ModelScope 服务器，可能挂起或失败）
                if os.path.isabs(origin_pattern):
                    import glob as _glob
                    resolved = sorted(_glob.glob(origin_pattern))
                    if not resolved:
                        raise FileNotFoundError(
                            f"No files matched pattern: {origin_pattern}"
                        )
                    if len(resolved) == 1:
                        model_configs.append(ModelConfig(path=resolved[0]))
                    else:
                        model_configs.append(ModelConfig(path=resolved))
                else:
                    model_configs.append(
                        ModelConfig(model_id=model_id, origin_file_pattern=origin_pattern)
                    )

        # tokenizer: 仅在提供本地路径时加载，否则不联网下载
        if tokenizer_path is not None:
            tokenizer_config = ModelConfig(tokenizer_path)
        else:
            tokenizer_config = None
            print("WARNING: tokenizer_path not provided, tokenizer will not be loaded.")

        # audio_processor: V2V_VACE 不需要音频处理器，仅在显式指定时加载
        if audio_processor_path is not None and os.path.exists(audio_processor_path):
            audio_processor_config = ModelConfig(audio_processor_path)
        else:
            audio_processor_config = None

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            vram_limit=vram_limit,
        )

        # plücker: 必须在加载 ckpt 之前 enable, 这样 ckpt 中 mask_cam_embedding.* 才有
        # 对应 module 可以被 load_state_dict(strict=False) 命中。
        if self.use_plucker:
            attached = []
            for vace_name in ("vace", "vace2"):
                vace_model = getattr(self.pipe, vace_name, None)
                if vace_model is None or not hasattr(vace_model, "enable_plucker"):
                    continue
                vace_model.enable_plucker()
                param_ref = next(
                    (p for p in vace_model.vace_patch_embedding.parameters() if p is not None),
                    None,
                )
                if param_ref is not None and vace_model.mask_cam_embedding is not None:
                    vace_model.mask_cam_embedding.to(
                        dtype=param_ref.dtype, device=param_ref.device,
                    )
                attached.append(vace_name)
            print(f"[plücker] enable_plucker() attached on: {attached or 'NONE'}")

        # 加载 LoRA 权重
        if dual_lora_checkpoint:
            self._load_dual_lora(dual_lora_checkpoint, dual_lora_alpha)
        else:
            if dit_lora_path:
                self._load_single_lora("dit", dit_lora_path, dit_lora_alpha)
            if vace_lora_path:
                self._load_single_lora("vace", vace_lora_path, vace_lora_alpha)

    def _load_dual_lora(self, checkpoint_path: str, alpha: float):
        """加载双 LoRA 检查点 (pipe.dit.* + pipe.vace.*) + 可选的 plücker 非 LoRA 模块。

        Checkpoint 分两类键:
          (1) LoRA 适配器 (含 'lora_A' 或 'lora_B') → 通过 pipe.load_lora 融合到 base.weight
          (2) 非 LoRA 全量参数 (plücker 训练新增的 mask_cam_embedding.*, vace_patch_embedding.*)
              → 通过 module.load_state_dict(strict=False) 直接 copy 到对应模块。
        当 ckpt 来自 plücker 训练而推理未开 --use_plucker, mask_cam_embedding 键会被 drop,
        触发 WARNING; 反之同理。
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Dual LoRA checkpoint not found: {checkpoint_path}")

        full_state_dict = load_state_dict(checkpoint_path)
        dit_prefix = "pipe.dit."
        vace_prefix = "pipe.vace."

        # 4 桶: (prefix, lora?) → state dict
        dit_lora, vace_lora, dit_other, vace_other = {}, {}, {}, {}
        unmatched_keys = []
        for key, value in full_state_dict.items():
            is_lora = ("lora_A" in key) or ("lora_B" in key)
            if key.startswith(dit_prefix):
                stripped = key[len(dit_prefix):]
                (dit_lora if is_lora else dit_other)[stripped] = value
            elif key.startswith(vace_prefix):
                stripped = key[len(vace_prefix):]
                (vace_lora if is_lora else vace_other)[stripped] = value
            else:
                unmatched_keys.append(key)

        if unmatched_keys:
            print(
                f"WARNING: {len(unmatched_keys)} keys don't match pipe.dit.*/pipe.vace.* prefix. "
                f"First 5: {unmatched_keys[:5]}"
            )

        # (1) LoRA 融合
        if dit_lora:
            self.pipe.load_lora(self.pipe.dit, state_dict=dit_lora, alpha=alpha)
            print(f"Loaded DIT LoRA from dual checkpoint: {len(dit_lora)} keys, alpha={alpha}")
        else:
            print("WARNING: No DIT LoRA keys found in dual checkpoint")

        if vace_lora:
            self.pipe.load_lora(self.pipe.vace, state_dict=vace_lora, alpha=alpha)
            print(f"Loaded VACE LoRA from dual checkpoint: {len(vace_lora)} keys, alpha={alpha}")
        else:
            print("WARNING: No VACE LoRA keys found in dual checkpoint")

        # (2) 非 LoRA 全量参数 (plücker: mask_cam_embedding + vace_patch_embedding)
        if dit_other:
            miss, unexp = self.pipe.dit.load_state_dict(dit_other, strict=False)
            print(
                f"[plücker] DIT non-LoRA load: {len(dit_other)} keys; "
                f"missing={len(miss)} unexpected={len(unexp)}"
            )
        if vace_other:
            miss, unexp = self.pipe.vace.load_state_dict(vace_other, strict=False)
            print(
                f"[plücker] VACE non-LoRA load: {len(vace_other)} keys; "
                f"missing={len(miss)} unexpected={len(unexp)}"
            )
            mask_cam_keys = [k for k in vace_other if k.startswith("mask_cam_embedding")]
            if mask_cam_keys and not self.use_plucker:
                print(
                    f"WARNING: checkpoint contains {len(mask_cam_keys)} mask_cam_embedding keys "
                    f"but --use_plucker is OFF; they were silently dropped. Output will NOT use "
                    f"plücker conditioning."
                )
            if not mask_cam_keys and self.use_plucker:
                print(
                    f"WARNING: --use_plucker is ON but checkpoint has NO mask_cam_embedding keys. "
                    f"mask_cam_embedding remains random-initialized (output ≈ noise)."
                )

        print(f"Dual LoRA loaded: {checkpoint_path}")

    def _load_single_lora(self, model_name: str, lora_path: str, alpha: float):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"{model_name} LoRA path not found: {lora_path}")
        model = getattr(self.pipe, model_name, None)
        if model is None:
            raise RuntimeError(f"Pipeline has no '{model_name}' model")
        self.pipe.load_lora(model, lora_path, alpha=alpha)
        print(f"Loaded {model_name} LoRA: {lora_path}, alpha={alpha}")

    @torch.no_grad()
    def generate(
        self,
        prompt,
        vace_reference_image=None,
        vace_video=None,
        vace_rgb_video=None,
        vace_depth_video=None,
        vace_mask_video=None,
        vace_intrinsic=None,
        vace_extrinsic=None,
        negative_prompt="",
        height=336,
        width=448,
        num_frames=61,
        seed=0,
        cfg_scale=1.0,
        num_inference_steps=50,
        vace_scale=0.0,
        tiled=False,
        rand_device=None,
    ):
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=None,  # V2V_VACE 不使用 I2V input_image
            vace_reference_image=vace_reference_image,
            vace_video=vace_video,
            vace_rgb_video=vace_rgb_video,
            vace_depth_video=vace_depth_video,
            vace_mask_video=vace_mask_video,
            vace_intrinsic=vace_intrinsic,
            vace_extrinsic=vace_extrinsic,
            vace_scale=vace_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            rand_device=rand_device or self.pipe.device,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            tiled=tiled,
        )


# ─── CLI 参数 ─────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="V2V_VACE Demo 推理 (VACE-14B + 双 LoRA + vace_reference_image)"
    )

    # 模型路径
    parser.add_argument("--model_paths", type=str, default=None)
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_processor_path", type=str, default=None)

    # 双 LoRA 检查点
    parser.add_argument("--dual_lora_checkpoint", type=str, default=None,
                        help="双 LoRA 检查点 (pipe.dit.* + pipe.vace.*)")
    parser.add_argument("--dual_lora_alpha", type=float, default=1.0)

    # 单独 LoRA (备选)
    parser.add_argument("--dit_lora_path", type=str, default=None)
    parser.add_argument("--dit_lora_alpha", type=float, default=1.0)
    parser.add_argument("--vace_lora_path", type=str, default=None)
    parser.add_argument("--vace_lora_alpha", type=float, default=1.0)

    # 数据集
    parser.add_argument("--dataset_base_path", type=str, default=None)
    parser.add_argument("--cache_files", type=str, nargs="+", default=None,
                        help="预提取的 cache.pkl 路径")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="dataset_config JSON。提供后走 mixed/source dataset_config 推理分支, "
             "不再要求 --dataset_base_path/--cache_files。",
    )
    parser.add_argument(
        "--case_manifest",
        type=str,
        default=None,
        help="显式 case 清单；按 global_dataset_index 选样并使用每条记录的 seed。",
    )
    parser.add_argument(
        "--samples_per_source",
        type=int,
        default=None,
        help="dataset_config 分支下每个 source 抽样条数。<=0 表示跑完整 source。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="dataset_config 分支下总抽样条数。<=0 表示不限制总数。",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="配合 --max_samples 在所有候选样本中随机抽样；否则按 source 顺序取前 N 条。",
    )
    parser.add_argument(
        "--source_names",
        type=str,
        nargs="+",
        default=None,
        help="dataset_config 分支下只评估指定 source ref；默认按 train_mix 顺序评估全部 source。",
    )
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=101)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--camera_key", type=str, nargs="+", default=["head"])
    parser.add_argument("--min_interval", type=int, default=1)
    parser.add_argument("--max_interval", type=int, default=1)
    parser.add_argument("--load_workers", type=int, default=32)
    parser.add_argument("--proprio_stats_path", type=str, default="proprio_stats_untar")
    parser.add_argument("--proprio_debug", action="store_true")
    parser.add_argument("--proprio_event", type=str, default=None)
    parser.add_argument("--proprio_event_offset", type=int, default=None)
    parser.add_argument("--proprio_all_events", action="store_true")
    parser.add_argument("--proprio_event_ratio", type=float, default=1.0)
    parser.add_argument(
        "--detail_prompt",
        action="store_true",
        help="使用包含 init_scene_text/task_name/action_text 的详细 prompt。",
    )
    parser.add_argument(
        "--vace_condition_mode",
        type=parse_vace_condition_mode,
        default="condition_h5",
    )
    parser.add_argument(
        "--threeviews_concat",
        action="store_true",
        help="三视角 concat 推理: 数据集按 head/hand_left/hand_right 三组独立加载 RGB/depth/mask, "
             "推理 pipeline 内每个视角各自 VAE encode, 然后在 latent 宽维 (dim=-1) 拼接, "
             "输出 3 段视频(分别为各视角推理结果)。",
    )
    parser.add_argument(
        "--use_plucker",
        action="store_true",
        help="启用 plücker map VACE 条件 (mirrors training --use_plucker). 必须与训练 ckpt 一致: "
             "训练时打开则推理也必须打开, 否则 ckpt 中 mask_cam_embedding 键被丢弃导致输出异常。",
    )

    # 首帧来源
    parser.add_argument(
        "--reference_image_source",
        type=str,
        default="input_image",
        choices=["input_image", "gt", "condition_rgb"],
        help="vace_reference_image 来源: input_image=frame.png, gt=GT[0], condition_rgb=条件RGB[0]",
    )

    # 推理参数
    parser.add_argument("--output_path", type=str, default="./outputs/inference/v2v_vace_challenge")
    parser.add_argument("--seed", type=int, default=-1,
                        help="随机种子。设为 -1 或负数表示随机生成（每次运行不同），>=0 表示固定种子")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="每个 sample 重复推理次数，每次使用不同 seed (base_seed + idx*num_repeats + repeat_idx)")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--vace_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_condition", action="store_true")
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--save_pred_frames", action="store_true",
                        help="额外保存 pred 图片序列到子文件夹")
    parser.add_argument("--pad_mode", type=str, default="front",
                        choices=["front", "back"],
                        help="Padding 模式: front=首帧前 padding (默认), back=尾帧后 padding")
    parser.add_argument(
        "--pad_short_actions",
        action="store_true",
        help="dataset_config 分支备用开关；source config 中的 pad_short_actions 会优先覆盖。",
    )
    parser.add_argument(
        "--random_sample_start",
        dest="random_sample_start",
        action="store_true",
        default=False,
        help="开启随机 clip 采样 (随机 n_real + 随机 start)。默认关闭 (确定性: 最大 n_real + start=0)。",
    )
    parser.add_argument(
        "--no_random_sample_start",
        dest="random_sample_start",
        action="store_false",
        help="确定性采样: 最大 n_real + start=0 (默认行为)。",
    )
    parser.add_argument("--serial_model_load", action="store_true")
    parser.add_argument("--model_load_parallel_ranks", type=int, default=1)
    parser.add_argument("--model_load_stagger_seconds", type=float, default=0.0)
    parser.add_argument("--vram_limit", type=float, default=None)

    return parser


# ─── 推理主逻辑 ───────────────────────────────────────────────────────────

def run_inference(args, accelerator: Accelerator):
    # ── seed 解析：负数 → 随机生成，并广播到所有进程保持一致 ──
    if args.seed < 0:
        if accelerator.is_main_process:
            import random as _random
            args.seed = _random.randint(0, 2**31 - 1)
            print(f"[Seed] 随机生成 seed = {args.seed}")
        seed_payload = [args.seed]
        broadcast_object_list(seed_payload, from_process=0)
        args.seed = seed_payload[0]
    else:
        if accelerator.is_main_process:
            print(f"[Seed] 使用固定 seed = {args.seed}")

    condition_mode = normalize_vace_condition_mode(args.vace_condition_mode)
    args.vace_condition_mode = condition_mode
    os.environ["AGIBOT_VACE_CONDITION_MODE"] = condition_mode

    source_summaries = None
    if args.case_manifest:
        eval_items, source_summaries = _build_case_manifest_items(args)
        if accelerator.is_main_process:
            print(
                f"[V2V_VACE Inference] Total manifest samples: {len(eval_items)}, "
                f"manifest: {args.case_manifest}"
            )
    elif args.dataset_config:
        eval_items, source_summaries = _build_dataset_config_items(args)
        if accelerator.is_main_process:
            print(
                f"[V2V_VACE Inference] Total mixed samples: {len(eval_items)}, "
                f"dataset_config: {args.dataset_config}"
            )
    else:
        if not args.dataset_base_path:
            raise ValueError("--dataset_base_path is required unless --dataset_config is provided.")
        if not args.cache_files:
            raise ValueError("--cache_files is required unless --dataset_config is provided.")
        dataset = AgibotWorldDataset(
            ROOT=args.dataset_base_path,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            repeat=args.dataset_repeat,
            camera_key=args.camera_key,
            cache_files=args.cache_files,
            load_workers=args.load_workers,
            min_interval=args.min_interval,
            max_interval=args.max_interval,
            threeviews_concat=args.threeviews_concat,
            use_plucker=args.use_plucker,
            proprio_stats_path=args.proprio_stats_path,
            proprio_debug=args.proprio_debug,
            proprio_event=args.proprio_event,
            proprio_event_offset=args.proprio_event_offset,
            proprio_all_events=args.proprio_all_events,
            proprio_event_ratio=args.proprio_event_ratio,
            detail_prompt=args.detail_prompt,
            random_sample_start=args.random_sample_start,
        )
        eval_items = _build_cache_items(dataset)
        if accelerator.is_main_process:
            print(
                f"[V2V_VACE Inference] Total samples: {len(eval_items)}, "
                f"cache_files: {args.cache_files}"
            )

    num_repeats = max(1, int(args.num_repeats))
    if num_repeats > 1:
        expanded_items = []
        for item in eval_items:
            for repeat_idx in range(num_repeats):
                rep_item = dict(item)
                rep_item["repeat_idx"] = repeat_idx
                expanded_items.append(rep_item)
        eval_items = expanded_items
        if accelerator.is_main_process:
            print(
                f"[Repeat] num_repeats={num_repeats}, "
                f"expanded total samples: {len(eval_items)}"
            )

    total_samples = len(eval_items)
    local_items = eval_items[accelerator.process_index :: accelerator.num_processes]

    # 输出目录
    output_root = None
    if accelerator.is_main_process:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join(args.output_path, f"demo_{timestamp}")
    output_payload = [output_root]
    broadcast_object_list(output_payload, from_process=0)
    args.output_path = output_payload[0]
    os.makedirs(args.output_path, exist_ok=True)
    setup_run_log_streams(args.output_path)

    if accelerator.is_main_process:
        config_path = os.path.join(args.output_path, "inference_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            config = vars(args).copy()
            config["num_processes"] = accelerator.num_processes
            config["hostname"] = socket.gethostname()
            config["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            config["hip_visible_devices"] = os.environ.get("HIP_VISIBLE_DEVICES", "")
            config["inference_type"] = "v2v_vace"
            if source_summaries is not None:
                config["dataset_config_sources"] = source_summaries
                config["eval_samples"] = [
                    {
                        "idx": item["global_index"],
                        "pool_index": item.get("pool_index", item["global_index"]),
                        "source_name": item["source_name"],
                        "data_index": item["data_index"],
                        "condition_id": item.get("condition_id"),
                        "seed": item.get("seed"),
                        "repeat_idx": item.get("repeat_idx", 0),
                    }
                    for item in eval_items
                ]
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[Config] Saved to {config_path}")

    # 加载模型
    model_kwargs = dict(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        dual_lora_checkpoint=args.dual_lora_checkpoint,
        dual_lora_alpha=args.dual_lora_alpha,
        dit_lora_path=args.dit_lora_path,
        dit_lora_alpha=args.dit_lora_alpha,
        vace_lora_path=args.vace_lora_path,
        vace_lora_alpha=args.vace_lora_alpha,
        vram_limit=args.vram_limit,
        device=accelerator.device,
        torch_dtype=torch.bfloat16,
        use_plucker=args.use_plucker,
    )

    model = None
    if args.serial_model_load and accelerator.num_processes > 1:
        batch_size = max(1, int(args.model_load_parallel_ranks))
        if accelerator.is_main_process:
            print(
                f"[Model Load] staged load: {accelerator.num_processes} rank(s), batch_size={batch_size}"
            )
        for batch_start in range(0, accelerator.num_processes, batch_size):
            batch_end = min(batch_start + batch_size, accelerator.num_processes)
            # 使用文件屏障替代 NCCL barrier，避免模型加载耗时过长导致超时
            _file_based_barrier(
                output_dir=args.output_path,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
                timeout=3600,
                tag=f"model_load_pre_batch{batch_start}",
            )
            if batch_start <= accelerator.process_index < batch_end:
                batch_local_rank = accelerator.process_index - batch_start
                if args.model_load_stagger_seconds > 0:
                    stagger = batch_local_rank * args.model_load_stagger_seconds
                    print(f"[Model Load] rank={accelerator.process_index} sleeping {stagger:.1f}s...")
                    time.sleep(stagger)
                print(f"[Model Load] batch=[{batch_start},{batch_end}), rank={accelerator.process_index} loading...")
                model = WanV2VVaceInferenceModule(**model_kwargs)
                print(f"[Model Load] rank={accelerator.process_index} done.")
            # 使用文件屏障替代 NCCL barrier，等待当前 batch 加载完成
            _file_based_barrier(
                output_dir=args.output_path,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
                timeout=3600,
                tag=f"model_load_post_batch{batch_start}",
            )
    else:
        if args.model_load_stagger_seconds > 0 and accelerator.num_processes > 1:
            stagger = accelerator.process_index * args.model_load_stagger_seconds
            print(f"[Model Load] rank={accelerator.process_index} sleeping {stagger:.1f}s...")
            time.sleep(stagger)
        model = WanV2VVaceInferenceModule(**model_kwargs)

    # 逐样本推理
    results = []
    for step, item in enumerate(
        tqdm(
            local_items,
            desc=f"[GPU {accelerator.process_index}] V2V_VACE Inference",
            disable=not accelerator.is_local_main_process,
        ),
        start=1,
    ):
        try:
            idx = int(item["global_index"])
            data_index = int(item["data_index"])
            item_dataset = item["dataset"]
            source_name = item.get("source_name")
            repeat_idx = int(item.get("repeat_idx", 0))
            default_seed = args.seed + idx * num_repeats + repeat_idx
            sample_seed = int(item["seed"]) + repeat_idx if "seed" in item else default_seed
            sample = item_dataset[data_index]
            prompt = sample.get("prompt", "")

            clip_info = _clip_info_for_dataset_index(item_dataset, data_index)
            task_id = clip_info.get("task_id", sample.get("task_id", source_name or idx))
            episode_id = clip_info.get("episode_id", sample.get("episode_id", data_index))
            expected_condition_id = item.get("condition_id")
            if expected_condition_id:
                condition_parts = expected_condition_id.split("_", 2)
                expected_episode = "/".join(condition_parts[:2])
                if str(episode_id) != expected_episode:
                    raise ValueError(
                        f"Manifest case {idx} expected episode {expected_episode}, got {episode_id}"
                    )
                expected_n_real = item.get("n_real_frames")
                actual_n_real = sample.get("n_real_frames")
                if expected_n_real is not None and int(actual_n_real) != int(expected_n_real):
                    raise ValueError(
                        f"Manifest case {idx} expected n_real_frames={expected_n_real}, "
                        f"got {actual_n_real}"
                    )
            start_frame = clip_info.get("start_frame", "na")
            end_frame = clip_info.get("end_frame", "na")
            clip_frame_info = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames_available": clip_info.get("num_frames_available", sample.get("num_frames_available")),
            }
            if source_name:
                base_name = sanitize_filename(
                    f"{source_name}-{task_id}-{episode_id}-action_{start_frame}"
                )
            else:
                base_name = sanitize_filename(f"{task_id}-{episode_id}-action_{start_frame}")
            shared_base_name = base_name
            if num_repeats > 1:
                base_name = f"{base_name}_seed{sample_seed}"

            if args.threeviews_concat:
                # 三视角 concat: 视角顺序由 vace_contract.VIEW_NAMES 统一约定（与训练端同一常量）,
                # pipeline 内每个视角各自 VAE encode 后在 dim=-1 上 cat,顺序错位会直接污染输出。
                _VIEW_NAMES = vace_contract.VIEW_NAMES

                if condition_mode == "normal_png":
                    raise ValueError(
                        "threeviews_concat 推理目前只支持 h5 condition 模式 (condition_h5 / condition*_h5)。"
                    )
                if not is_h5_condition_mode(condition_mode):
                    raise ValueError(
                        f"threeviews_concat 需要 h5 condition 模式; 当前: {condition_mode}"
                    )

                missing_video_keys = [
                    f"{v}_video" for v in _VIEW_NAMES if f"{v}_video" not in sample
                ]
                if missing_video_keys:
                    raise KeyError(
                        f"Dataset sample missing per-view video fields: {missing_video_keys}. "
                        "Did the cache include threeviews_concat=True?"
                    )

                per_view_videos = [sample[f"{v}_video"] for v in _VIEW_NAMES]

                required_per_view_keys = []
                for v in _VIEW_NAMES:
                    for suffix in ("vace_rgb_video", "vace_depth_video"):
                        required_per_view_keys.append(f"{v}_{suffix}")
                missing_cond_keys = [k for k in required_per_view_keys if k not in sample]
                if missing_cond_keys:
                    raise KeyError(
                        f"Dataset sample missing per-view condition fields: {missing_cond_keys}."
                    )

                per_view_rgb = [sample[f"{v}_vace_rgb_video"] for v in _VIEW_NAMES]
                per_view_depth = [sample[f"{v}_vace_depth_video"] for v in _VIEW_NAMES]

                # plücker: dataset 已在 use_plucker=True 时 emit head_/hand_left_/hand_right_vace_intrinsic
                # 与 ..._vace_extrinsic 字段。聚合成 per-view list 透传给 pipeline。
                per_view_intrinsic = None
                per_view_extrinsic = None
                if args.use_plucker:
                    intr_keys = [f"{v}_vace_intrinsic" for v in _VIEW_NAMES]
                    extr_keys = [f"{v}_vace_extrinsic" for v in _VIEW_NAMES]
                    missing_cam = [k for k in intr_keys + extr_keys if k not in sample]
                    if missing_cam:
                        raise KeyError(
                            f"--use_plucker is ON but dataset sample missing camera fields: {missing_cam}. "
                            f"Ensure the dataset is built with use_plucker=True and the condition h5 contains "
                            f"cameras_intrinsic attr + global_pose datasets."
                        )
                    per_view_intrinsic = [sample[k] for k in intr_keys]
                    per_view_extrinsic = [sample[k] for k in extr_keys]

                # 各视角 input_image / first-frame 当作参考帧。pipeline 期望 per-view
                # 结构 [[head_ref], [hand_left_ref], [hand_right_ref]] (内层 list 表示该视角
                # 的 reference 帧数, 这里固定 1 帧)。
                per_view_refs = [[per_view_videos[i][0]] for i in range(len(_VIEW_NAMES))]

                # 各视角的 mask 形状一致, 共用一份 auto mask 即可;
                # _build_auto_mask 训练时也是把同一份 mask broadcast 到所有视角。
                n_real_frames = sample.get("n_real_frames")
                sample_pad_mode = sample.get("pad_mode", args.pad_mode)
                single_view_mask = _build_auto_mask_frames(
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    n_real_frames=n_real_frames,
                    pad_mode=sample_pad_mode,
                )
                per_view_mask = [list(single_view_mask) for _ in _VIEW_NAMES]

                generate_kwargs = {
                    "prompt": prompt,
                    "vace_reference_image": per_view_refs,
                    "vace_rgb_video": per_view_rgb,
                    "vace_depth_video": per_view_depth,
                    "vace_mask_video": per_view_mask,
                    "vace_intrinsic": per_view_intrinsic,
                    "vace_extrinsic": per_view_extrinsic,
                    "negative_prompt": args.negative_prompt,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "seed": sample_seed,
                    "cfg_scale": args.cfg_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "vace_scale": args.vace_scale,
                    "tiled": args.tiled,
                    "rand_device": "cpu" if args.case_manifest else None,
                }

                generated_per_view = model.generate(**generate_kwargs)
                if not isinstance(generated_per_view, (list, tuple)) or len(generated_per_view) != len(_VIEW_NAMES):
                    raise RuntimeError(
                        "Expected pipeline to return per-view video list of length "
                        f"{len(_VIEW_NAMES)}, got {type(generated_per_view).__name__} "
                        f"len={len(generated_per_view) if hasattr(generated_per_view, '__len__') else 'n/a'}"
                    )

                trim_info = _resolve_padding_trim_info(
                    sample,
                    clip_info,
                    total_frames=len(generated_per_view[0]),
                    default_pad_mode=args.pad_mode,
                )

                record = {
                    "idx": idx,
                    "pool_index": item.get("pool_index", idx),
                    "status": "success",
                    "seed": sample_seed,
                    "repeat_idx": repeat_idx,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "source_name": source_name,
                    "data_index": data_index,
                    "condition_id": expected_condition_id,
                    **clip_frame_info,
                    "prompt": prompt,
                    "reference_image_source": args.reference_image_source,
                    "threeviews_concat": True,
                    "views": list(_VIEW_NAMES),
                    **trim_info,
                }

                # 各视角先各自 trim padding, 再沿 width 方向 concat 成 1 路视频 (顺序 head→hand_left→hand_right,
                # 与训练侧 WanVideoUnit_VACE 的 torch.cat(view_contexts, dim=-1) 同向)。
                per_view_gen = [
                    _trim_padding_frames(generated_per_view[i], trim_info)
                    for i in range(len(_VIEW_NAMES))
                ]
                per_view_cond = [
                    _trim_padding_frames(per_view_rgb[i], trim_info)
                    for i in range(len(_VIEW_NAMES))
                ]
                per_view_gt = [
                    _trim_padding_frames(per_view_videos[i], trim_info)
                    for i in range(len(_VIEW_NAMES))
                ]

                pred_concat = _concat_views_horizontally(per_view_gen)
                cond_concat = _concat_views_horizontally(per_view_cond)
                gt_concat = _concat_views_horizontally(per_view_gt)

                if pred_concat is None:
                    raise RuntimeError(
                        "Failed to concat per-view predictions; one or more views are empty."
                    )

                pred_path = os.path.join(args.output_path, f"{base_name}_pred.mp4")
                save_video(pred_concat, pred_path, fps=args.fps)
                record["pred_path"] = pred_path

                if args.save_pred_frames:
                    pred_frames_dir = os.path.join(args.output_path, f"{base_name}_pred_frames")
                    try:
                        os.makedirs(pred_frames_dir, exist_ok=True)
                        for fi, frame in enumerate(pred_concat):
                            frame_img = _to_rgb_pil(frame)
                            frame_img.save(os.path.join(pred_frames_dir, f"{fi:04d}.png"))
                        record["pred_frames_status"] = "saved"
                        record["pred_frames_dir"] = pred_frames_dir
                    except Exception as frames_error:
                        record["pred_frames_status"] = "failed"
                        record["pred_frames_error"] = str(frames_error)
                        tqdm.write(
                            f"[Process {accelerator.process_index}] "
                            f"Failed to save pred frames for idx={idx}: {frames_error}"
                        )

                if args.save_condition and cond_concat and repeat_idx == 0:
                    cond_path = os.path.join(args.output_path, f"{shared_base_name}_condition.mp4")
                    try:
                        save_video(cond_concat, cond_path, fps=args.fps)
                        record["condition_status"] = "saved"
                    except Exception as cond_err:
                        record["condition_status"] = "failed"
                        record["condition_error"] = str(cond_err)

                if args.save_gt and gt_concat and repeat_idx == 0:
                    gt_path = os.path.join(args.output_path, f"{shared_base_name}_gt.mp4")
                    try:
                        save_video(gt_concat, gt_path, fps=args.fps)
                        record["gt_status"] = "saved"
                    except Exception as gt_err:
                        record["gt_status"] = "failed"
                        record["gt_error"] = str(gt_err)

                # vis: 三行竖直堆叠, 每行宽度 = W * num_views (与上面 concat 一致)
                vis_rows = []
                if cond_concat:
                    vis_rows.append(("condition", cond_concat))
                vis_rows.append(("pred", pred_concat))
                if gt_concat:
                    vis_rows.append(("gt", gt_concat))

                try:
                    vis_frames = _stack_video_rows(vis_rows)
                    vis_path = os.path.join(args.output_path, f"{base_name}_pred_v2v_vace_vis.mp4")
                    save_video(vis_frames, vis_path, fps=args.fps)
                    record["pred_vis_status"] = "saved"
                except Exception as vis_err:
                    record["pred_vis_status"] = "failed"
                    record["pred_vis_error"] = str(vis_err)

                results.append(record)

            else:
                # 单视角 (兼容历史行为)
                # 确定首帧 (vace_reference_image) 来源
                if args.reference_image_source == "input_image":
                    # 优先 input_image (frame.png), 回退 video[0]
                    vace_ref = sample.get("input_image", None)
                    if vace_ref is None and "video" in sample:
                        vace_ref = sample["video"][0]
                elif args.reference_image_source == "gt":
                    if "video" not in sample:
                        raise KeyError("Dataset sample missing 'video' for reference_image_source=gt")
                    vace_ref = sample["video"][0]
                elif args.reference_image_source == "condition_rgb":
                    if "vace_rgb_video" in sample and len(sample["vace_rgb_video"]) > 0:
                        vace_ref = sample["vace_rgb_video"][0]
                    else:
                        vace_ref = sample.get("input_image", None)
                        if vace_ref is None and "video" in sample:
                            vace_ref = sample["video"][0]
                else:
                    vace_ref = sample.get("input_image", sample.get("video", [None])[0])

                if vace_ref is None:
                    raise KeyError("Cannot determine vace_reference_image from sample")

                generate_kwargs = {
                    "prompt": prompt,
                    "vace_reference_image": vace_ref,
                    "negative_prompt": args.negative_prompt,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "seed": sample_seed,  # base_seed + idx*num_repeats + repeat_idx
                    "cfg_scale": args.cfg_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "vace_scale": args.vace_scale,
                    "tiled": args.tiled,
                    "rand_device": "cpu" if args.case_manifest else None,
                }

                # VACE 条件
                condition_frames = None
                if condition_mode == "normal_png":
                    if "vace_video" not in sample:
                        raise KeyError("Dataset sample missing 'vace_video'.")
                    generate_kwargs["vace_video"] = sample["vace_video"]
                    condition_frames = sample.get("vace_video")
                elif is_h5_condition_mode(condition_mode):
                    required_keys = ("vace_rgb_video", "vace_depth_video", "vace_mask_video")
                    missing_keys = [k for k in required_keys if k not in sample]
                    if missing_keys:
                        raise KeyError(f"Dataset sample missing {missing_keys}.")
                    generate_kwargs["vace_rgb_video"] = sample["vace_rgb_video"]
                    generate_kwargs["vace_depth_video"] = sample["vace_depth_video"]
                    # V2V_VACE: 自动生成 mask (支持前/后 padding 模式)
                    n_real_frames = sample.get("n_real_frames")
                    # 优先使用样本中的 pad_mode（来自 cache），其次使用 CLI 参数
                    sample_pad_mode = sample.get("pad_mode", args.pad_mode)
                    generate_kwargs["vace_mask_video"] = _build_auto_mask_frames(
                        num_frames=args.num_frames,
                        height=args.height,
                        width=args.width,
                        n_real_frames=n_real_frames,
                        pad_mode=sample_pad_mode,
                    )
                    # plücker (single-view): dataset emit vace_intrinsic/vace_extrinsic 时透传
                    if args.use_plucker:
                        missing_cam = [k for k in ("vace_intrinsic", "vace_extrinsic") if k not in sample]
                        if missing_cam:
                            raise KeyError(
                                f"--use_plucker is ON but dataset sample missing {missing_cam} "
                                f"(condition mode={condition_mode})."
                            )
                        generate_kwargs["vace_intrinsic"] = sample["vace_intrinsic"]
                        generate_kwargs["vace_extrinsic"] = sample["vace_extrinsic"]
                    condition_frames = sample.get("vace_rgb_video")
                elif condition_mode == "disabled":
                    condition_frames = None
                else:
                    raise ValueError(f"Unsupported vace_condition_mode: {condition_mode}")

                generated_frames = model.generate(**generate_kwargs)

                trim_info = _resolve_padding_trim_info(
                    sample,
                    clip_info,
                    total_frames=len(generated_frames),
                    default_pad_mode=args.pad_mode,
                )
                saved_generated_frames = _trim_padding_frames(generated_frames, trim_info)
                saved_condition_frames = _trim_padding_frames(condition_frames, trim_info)
                saved_gt_frames = _trim_padding_frames(sample.get("video"), trim_info)

                pred_path = os.path.join(args.output_path, f"{base_name}_pred.mp4")
                save_video(saved_generated_frames, pred_path, fps=args.fps)

                record = {
                    "idx": idx,
                    "pool_index": item.get("pool_index", idx),
                    "status": "success",
                    "seed": sample_seed,  # 记录实际使用的 seed，便于复现
                    "repeat_idx": repeat_idx,
                    "pred_path": pred_path,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "source_name": source_name,
                    "data_index": data_index,
                    "condition_id": expected_condition_id,
                    **clip_frame_info,
                    "prompt": prompt,
                    "reference_image_source": args.reference_image_source,
                    **trim_info,
                }

                # 保存 pred 图片序列
                if args.save_pred_frames:
                    pred_frames_dir = os.path.join(args.output_path, f"{base_name}_pred_frames")
                    try:
                        os.makedirs(pred_frames_dir, exist_ok=True)
                        for fi, frame in enumerate(saved_generated_frames):
                            frame_img = _to_rgb_pil(frame)
                            frame_img.save(os.path.join(pred_frames_dir, f"{fi:04d}.png"))
                        record["pred_frames_status"] = "saved"
                        record["pred_frames_dir"] = pred_frames_dir
                    except Exception as frames_error:
                        record["pred_frames_status"] = "failed"
                        record["pred_frames_error"] = str(frames_error)
                        tqdm.write(
                            f"[Process {accelerator.process_index}] "
                            f"Failed to save pred frames for idx={idx}: {frames_error}"
                        )

                if args.save_condition and saved_condition_frames and repeat_idx == 0:
                    condition_path = os.path.join(args.output_path, f"{shared_base_name}_condition.mp4")
                    try:
                        save_video(saved_condition_frames, condition_path, fps=args.fps)
                        record["condition_status"] = "saved"
                    except Exception as cond_err:
                        record["condition_status"] = "failed"
                        record["condition_error"] = str(cond_err)

                if args.save_gt and saved_gt_frames and repeat_idx == 0:
                    gt_path = os.path.join(args.output_path, f"{shared_base_name}_gt.mp4")
                    gt_frames_dir = os.path.join(args.output_path, f"{shared_base_name}_gt_frames")
                    try:
                        save_video(saved_gt_frames, gt_path, fps=args.fps)
                        os.makedirs(gt_frames_dir, exist_ok=True)
                        for fi, frame in enumerate(saved_gt_frames):
                            frame_img = _to_rgb_pil(frame)
                            frame_img.save(os.path.join(gt_frames_dir, f"{fi:04d}.png"))
                        record["gt_status"] = "saved"
                    except Exception as gt_err:
                        record["gt_status"] = "failed"
                        record["gt_error"] = str(gt_err)

                if saved_condition_frames:
                    vis_rows = [("condition", saved_condition_frames), ("pred", saved_generated_frames)]
                    if saved_gt_frames:
                        vis_rows.append(("gt", saved_gt_frames))
                    vis_path = os.path.join(args.output_path, f"{base_name}_pred_v2v_vace_vis.mp4")
                    try:
                        vis_frames = _stack_video_rows(vis_rows)
                        save_video(vis_frames, vis_path, fps=args.fps)
                        record["pred_vis_status"] = "saved"
                    except Exception as vis_err:
                        record["pred_vis_status"] = "failed"
                        record["pred_vis_error"] = str(vis_err)
                else:
                    record["pred_vis_status"] = "skipped"

                results.append(record)

        except Exception as error:
            idx = int(item.get("global_index", -1))
            data_index = int(item.get("data_index", -1))
            source_name = item.get("source_name")
            failure_record = {
                "idx": idx,
                "pool_index": item.get("pool_index", idx),
                "status": "failed",
                "error": str(error),
                "source_name": source_name,
                "data_index": data_index,
                "condition_id": item.get("condition_id"),
                "seed": item.get("seed"),
                "repeat_idx": int(item.get("repeat_idx", 0)),
            }
            try:
                failure_clip_info = _clip_info_for_dataset_index(item["dataset"], data_index)
                if failure_clip_info:
                    failure_record.update(
                        {
                            "task_id": failure_clip_info.get("task_id"),
                            "episode_id": failure_clip_info.get("episode_id"),
                            "start_frame": failure_clip_info.get("start_frame"),
                            "end_frame": failure_clip_info.get("end_frame"),
                            "num_frames_available": failure_clip_info.get("num_frames_available"),
                        }
                    )
            except Exception:
                pass
            results.append(failure_record)
            import traceback
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.log_interval and step % args.log_interval == 0:
            print(
                f"[STEP] process={accelerator.process_index} "
                f"{step}/{len(local_items)} samples done"
            )

    # 保存进程结果
    proc_file = os.path.join(
        args.output_path, f"results_process_{accelerator.process_index}.json"
    )
    with open(proc_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 使用基于文件的屏障替代 NCCL barrier (accelerator.wait_for_everyone()),
    # 避免各进程推理样本数不均时先完成的进程在 NCCL barrier 等待超时 (600s)。
    _file_based_barrier(
        output_dir=args.output_path,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        timeout=7200,  # 2 小时，足够推理完成
    )

    # 主进程合并结果
    if accelerator.is_main_process:
        merged = []
        for pi in range(accelerator.num_processes):
            fp = os.path.join(args.output_path, f"results_process_{pi}.json")
            if not os.path.exists(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))
            os.remove(fp)

        merged.sort(key=lambda x: x.get("idx", -1))

        summary = {
            "total_samples": total_samples,
            "num_processes": accelerator.num_processes,
            "success": sum(1 for r in merged if r.get("status") == "success"),
            "failed": sum(1 for r in merged if r.get("status") == "failed"),
        }
        summary["success_rate"] = (
            summary["success"] / total_samples if total_samples > 0 else 0.0
        )

        results_path = os.path.join(args.output_path, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": merged}, f, indent=2, ensure_ascii=False)

        summary_path = os.path.join(args.output_path, "inference_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("=" * 80)
        print("V2V_VACE inference completed")
        print(f"  Output      : {args.output_path}")
        print(f"  Total       : {summary['total_samples']}")
        print(f"  Success     : {summary['success']}")
        print(f"  Failed      : {summary['failed']}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Results     : {results_path}")
        print("=" * 80)
        if summary["failed"]:
            raise RuntimeError(f"{summary['failed']} of {total_samples} inference cases failed")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    # 设置 NCCL 超时为 30 分钟（默认 600s），避免模型加载或推理期间的 broadcast 超时
    import datetime as _dt
    _nccl_timeout = _dt.timedelta(seconds=1800)
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=_nccl_timeout)]
    )
    run_inference(args, accelerator)
