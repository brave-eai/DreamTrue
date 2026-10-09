#!/usr/bin/env python3
"""Add gripper retreat modifications before each gripper close/open event detected from qpos."""
import argparse
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from dataset import build_modified_datasets, get_dataset_class
from scripts.tasks._taskio import add_io_args, dump_tasks, load_tasks
from sim import SapienEnv


def random_retreat_vector(retreat: float) -> list[float]:
    while True:
        x = random.choice([retreat, -retreat, 0, 0])
        y = random.choice([retreat, -retreat, 0, 0])
        z = random.choice([-retreat, 0])
        if x != 0 or y != 0 or z != 0:
            return [x, y, z]


def retreat_vector(retreat: float, mode: str) -> list[float]:
    if mode == 'z_only':
        return [0, 0, -retreat]
    return random_retreat_vector(retreat)


def find_events(gripper_vals: np.ndarray, gripper_open_qpos: float) -> list[tuple[int, str]]:
    """Detect all closing and opening events with hysteresis, sorted by frame.

    A close event requires crossing below 0.9*gop AND subsequently reaching 0.8*gop.
    An open event requires crossing above 0.1*gop AND subsequently reaching 0.2*gop.
    Consecutive events of the same type are suppressed.
    """
    close_thresh = 0.9 * gripper_open_qpos
    open_thresh = 0.1 * gripper_open_qpos
    close_confirm = 0.8 * gripper_open_qpos
    open_confirm = 0.2 * gripper_open_qpos

    raw: list[tuple[int, str]] = []
    above_close = gripper_vals >= close_thresh
    for c in np.where(above_close[:-1] & ~above_close[1:])[0]:
        raw.append((int(c) + 1, 'close'))
    below_open = gripper_vals <= open_thresh
    for o in np.where(below_open[:-1] & ~below_open[1:])[0]:
        raw.append((int(o) + 1, 'open'))
    raw.sort(key=lambda x: x[0])

    events: list[tuple[int, str]] = []
    last_type = None
    for frame, etype in raw:
        if etype == last_type:
            continue
        if etype == 'close' and np.any(gripper_vals[frame:] <= close_confirm):
            events.append((frame, etype))
            last_type = etype
        elif etype == 'open' and np.any(gripper_vals[frame:] >= open_confirm):
            events.append((frame, etype))
            last_type = etype
    return events


def _make_segment(start: int, end: int, delta: list[float], delta_end: list[float] | None = None) -> dict:
    seg = {
        "start": start,
        "end": end,
        "delta_pose": {
            "p": list(delta),
            "q": [1, 0, 0, 0]
        },
        "frame": "local",
    }
    if delta_end is not None:
        seg["delta_pose_end"] = {"p": list(delta_end), "q": [1, 0, 0, 0]}
    return seg


def build_segments(events: list[tuple[int, str]], clip_start: int, clip_end: int, transition: int, retreat: float, mode: str) -> list[dict]:
    """Build chained segments within a clip. Delta starts at [0,0,0] per clip.
    Each event picks a new retreat vector; ramp transitions from previous end state."""
    segments: list[dict] = []
    current = [0.0, 0.0, 0.0]
    prev_end = clip_start

    for event_frame, _event_type in events:
        latest_start = event_frame - transition
        if latest_start <= prev_end:
            continue
        start = random.randint(prev_end, latest_start)
        new_target = retreat_vector(retreat, mode)

        if any(v != 0 for v in current) and prev_end < start:
            segments.append(_make_segment(prev_end, start, current))

        segments.append(_make_segment(start, start + transition, current, new_target))

        current = new_target
        prev_end = start + transition

    if any(v != 0 for v in current) and prev_end < clip_end:
        segments.append(_make_segment(prev_end, clip_end, current))

    return segments


def build_fallback_segments(clip_start: int, clip_end: int, transition: int, retreat: float, mode: str) -> list[dict]:
    """Fallback when no events in a clip: random start in first 25% of clip, hold to clip end."""
    clip_len = clip_end - clip_start
    quarter = clip_len // 4
    latest_start = max(0, quarter - transition)
    if latest_start < 0:
        return []
    start = clip_start + (random.randint(0, latest_start) if latest_start > 0 else 0)
    vec = retreat_vector(retreat, mode)
    segments = [_make_segment(start, start + transition, [0.0, 0.0, 0.0], vec)]
    if start + transition < clip_end:
        segments.append(_make_segment(start + transition, clip_end, vec))
    return segments


def _arm_moved(qpos: np.ndarray, arm_cols: np.ndarray, start: int, end: int, threshold: float) -> bool:
    """该臂任一关节列在 [start, end) 内的 peak-to-peak 达到阈值即视为移动（所有维度都小于阈值才算没动）。"""
    if end - start < 2:
        return False
    return bool(np.any(np.ptp(qpos[start:end][:, arm_cols], axis=0) >= threshold))


def process_task(task: dict, data_class: str, data_root: str, robot_path: str, retreat: float, transition: int, modification_name: str, movement_thresh: float, mode: str = 'random', seed: int | None = None) -> list[dict]:
    data_key = task.get('data_key', '')
    try:
        if seed is not None:
            random.seed(seed)
        dataset = get_dataset_class(data_class)(root=data_root, key=data_key, cams=[], use_state=False)
        dataset = build_modified_datasets(dataset, task.get('modifications', []), assets_path=SapienEnv.assets_path, robot_path=robot_path)
        qpos = dataset.qpos
        dims = dataset._qpos_dims
        gripper_open_qpos = dataset.gripper_open_qpos
        clips = dataset.clip_info
        assert len(clips) > 0
        spans: list[tuple[int, int, int]] = []  # (clip index, start row, end row) 当前视图内
        for i, clip in enumerate(clips):
            rows = dataset.clip_frames(clip)
            if len(rows) == 0:
                continue
            if not np.all(np.diff(rows) == 1):
                print(f'Warning: {data_key}: clip [{clip.start}, {clip.end}) rows are not contiguous in the current view, skipped', file=sys.stderr)
                continue
            spans.append((i, int(rows[0]), int(rows[-1]) + 1))
        clip_segs: list[defaultdict[str, list]] = [defaultdict(list) for _ in clips]
        for arm in dataset._endpose_arms:
            gripper_col = dims.get(f'gripper_{arm}') or dims.get('gripper')
            arm_col = dims.get(f'arm_{arm}') or dims.get('arm')
            if gripper_col is None or arm_col is None:
                continue
            gripper_vals = qpos[:, gripper_col].flatten()
            for i, start, end in spans:
                events = find_events(gripper_vals[start:end], gripper_open_qpos)
                events = [(f + start, t) for f, t in events]
                if events:
                    segs = build_segments(events, start, end, transition, retreat, mode)
                else:
                    if not _arm_moved(qpos, arm_col, start, end, movement_thresh):
                        continue
                    segs = build_fallback_segments(start, end, transition, retreat, mode)
                if segs:
                    clip_segs[i][arm].extend(segs)

        results: list[dict] = []
        for i, start, end in spans:
            new_segs = clip_segs[i]
            if not new_segs:
                continue
            out_task = dict(task)
            existing = list(task.get('modifications', []))
            for arm, segs in new_segs.items():
                existing.append({'class': 'IKDataset', 'name': modification_name, 'arm': arm, 'segments': segs})
            existing.append({'class': 'SlicedDataset', 'sli': f'{start}:{end}'})
            out_task['modifications'] = existing
            results.append(out_task)
        return results
    except Exception as e:
        print(f'Warning: failed to load dataset for {data_key}: {e}', file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description='Add gripper retreat modifications before close/open events.')
    add_io_args(parser)
    parser.add_argument('--data_class', type=str, default='AgibotDataset')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--robot_path', type=str, default='agibot/G1_120s/G1_120s.yaml')
    parser.add_argument('--retreat', type=float, default=0.08)
    parser.add_argument('--transition_steps', type=int, default=30)
    parser.add_argument('--mode', choices=['z_only', 'random'], default='random')
    parser.add_argument('--modification-name', required=True, metavar='NAME')
    parser.add_argument('--movement-thresh', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max_workers', type=int, default=32)
    args = parser.parse_args()
    if not args.modification_name.strip():
        parser.error('--modification-name cannot be empty')

    tasks = load_tasks(args.input)

    base_seed = args.seed
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_task,
                task,
                args.data_class,
                args.data_root,
                args.robot_path,
                args.retreat,
                args.transition_steps,
                args.modification_name,
                args.movement_thresh,
                args.mode,
                (base_seed + i) if base_seed is not None else None,
            ) for i, task in enumerate(tasks)
        ]
        results: list[list[dict]] = [fut.result() for fut in tqdm(futures, total=len(futures), desc='retreat', file=sys.stderr)]

    dump_tasks([task for result in results for task in result], args.output)


if __name__ == '__main__':
    main()
