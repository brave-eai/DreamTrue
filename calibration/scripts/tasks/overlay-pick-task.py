#!/usr/bin/env python3
"""Pair a reference clip with following single-hand clips of the other arm in the same episode.

Segments are selected by skill/text regexes (ref defaults to Pick|Open, copy to Pick). Every
qualifying pair becomes one task with an OverlayArmsDataset modification that replays the later
clip on the other hand, time-warped so both gripper close frames and both clip ends land on the
same output frames; candidates failing the overlay checks are skipped. The copy may start anywhere
after the ref ends, gaps are allowed. With --slices, each distinct ref interval of an episode also
yields one SlicedDataset control task. Input is a raw condition task JSON (clip ranges are original
episode frame coordinates); filter-ik.py can then drop tasks whose modification cannot be built.
"""
import argparse
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from dataset import get_dataset_class
from dataset.OverlayArmsDataset import DEFAULT_DELTA_RATE_LIMIT, DEFAULT_SPEED_LIMIT, OverlayArmsDataset, OverlayFailedException, OverlaySegment, build_overlay_plan
from scripts.tasks._taskio import add_io_args, dump_tasks, load_tasks


def find_clip(clips: list, start: int, skill_regexp: re.Pattern, regexp: re.Pattern | None) -> int | None:
    """start 之后（含）首个 skill 匹配 skill_regexp 且 action_text 匹配 regexp 的 clip 下标。"""
    for i in range(start, len(clips)):
        if not any(skill_regexp.match(s) for s in clips[i].skill):
            continue
        if regexp is not None and not any(regexp.search(text) for text in clips[i].action_text):
            continue
        return i
    return None


def iter_spans(
    clips: list,
    start: int,
    skill_regexp: re.Pattern,
    regexp: re.Pattern | None,
    event_arms,
    allowed: list[str] | None = None,
):
    """从 start 起逐个产出「匹配 skill/文案 且恰有一条 allowed 臂在段内闭合」的候选 (start, end, arm)；
    标注拆成相邻多段的同一次动作合并为一段。"""
    i = start
    while (found := find_clip(clips, i, skill_regexp, regexp)) is not None:
        span_start, span_end, j = clips[found].start, clips[found].end, found
        if regexp is not None:
            while j + 1 < len(clips) and clips[j + 1].start == span_end \
                    and any(skill_regexp.match(s) for s in clips[j + 1].skill) \
                    and any(regexp.search(text) for text in clips[j + 1].action_text):
                j += 1
                span_end = clips[j].end
        arms = event_arms(span_start, span_end)
        if len(arms) == 1 and (allowed is None or arms[0] in allowed):
            yield span_start, span_end, arms[0]
        i = j + 1


def process_task(
    task: dict,
    data_class: str,
    data_root: str,
    ref_skill_regexp: re.Pattern,
    ref_regexp: re.Pattern | None,
    copy_skill_regexp: re.Pattern,
    copy_regexp: re.Pattern | None,
    name: str,
    blend: int | None,
    delta_rate_limit: float | None,
    speed_limit: float | None,
    slices: bool = False,
) -> tuple[list[dict], str]:
    data_key = task.get('data_key', '')
    assert not task.get('modifications'), f'{data_key}: overlay-pick-task does not support tasks with existing modifications yet'
    try:
        dataset = get_dataset_class(data_class)(root=data_root, key=data_key, cams=[], use_state=False)
        qpos = dataset.qpos
        clips = dataset.clip_info
        arms = list(dataset._endpose_arms)
        dims = dataset._qpos_dims
        gripper_cols = {arm: np.arange(qpos.shape[1])[dims[f'gripper_{arm}']] for arm in arms}
        threshold = 0.02 * dataset.gripper_open_qpos
        signals = {arm: qpos[:, gripper_cols[arm]].mean(axis=1) for arm in arms}

        def event_arms(s: int, e: int) -> list[str]:
            return [arm for arm in arms if OverlayArmsDataset._first_close_frame(signals[arm][s:e], threshold) is not None]

        refs = list(iter_spans(clips, 0, ref_skill_regexp, ref_regexp, event_arms))
        if not refs:
            return [], 'no ref clip with a single-hand close event'

        rate_limit = DEFAULT_DELTA_RATE_LIMIT if delta_rate_limit is None else delta_rate_limit
        limit = DEFAULT_SPEED_LIMIT if speed_limit is None else speed_limit
        results, saw_candidate, slice_ranges = [], False, {}
        for ref in refs:
            others = [a for a in arms if a != ref[2]]
            copy_start = next((i for i, c in enumerate(clips) if c.start >= ref[1]), len(clips))
            for copy in iter_spans(clips, copy_start, copy_skill_regexp, copy_regexp, event_arms, allowed=others):
                saw_candidate = True
                candidate = [
                    {
                        'arm': ref[2],
                        'start': ref[0],
                        'end': ref[1]
                    },
                    {
                        'arm': copy[2],
                        'start': copy[0],
                        'end': copy[1]
                    },
                ]
                try:
                    build_overlay_plan(
                        qpos=qpos,
                        dims=dims,
                        segments=[OverlaySegment.from_dict(s) for s in candidate],
                        gripper_open_qpos=dataset.gripper_open_qpos,
                        blend=blend,
                        delta_rate_limit=rate_limit,
                        speed_limit=limit,
                    )
                except OverlayFailedException:
                    continue
                modification = {
                    'class': 'OverlayArmsDataset',
                    'name': f'{name}-{len(results)}',
                    'segments': candidate,
                }
                if blend is not None:
                    modification['blend'] = blend
                if delta_rate_limit is not None:
                    modification['delta_rate_limit'] = delta_rate_limit
                if speed_limit is not None:
                    modification['speed_limit'] = speed_limit
                out_task = dict(task)
                out_task['modifications'] = list(task.get('modifications', [])) + [modification]
                results.append(out_task)
                slice_ranges[(ref[0], ref[1])] = None
        if not results:
            return [], 'all copy candidates fail the overlay checks' if saw_candidate else 'no copy clip after ref with a close event'
        if slices:
            for start, end in slice_ranges:
                slice_task = dict(task)
                slice_task['modifications'] = list(task.get('modifications', [])) + [{'class': 'SlicedDataset', 'sli': f'{start}:{end}'}]
                results.append(slice_task)
        return results, ''
    except Exception as e:
        return [], f'{type(e).__name__}: {e}'


def main():
    parser = argparse.ArgumentParser(description='Pair a reference clip with following single-hand clips of the other arm into OverlayArmsDataset modifications.')
    add_io_args(parser)
    parser.add_argument('--data_class', type=str, default='AgibotDataset')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ref-skill', type=str, default='^(Pick|Open)$', help='skill regex of the reference clip (re.match on each clip skill; e.g. "^Pick$")')
    parser.add_argument('--ref-regex', type=str, default=None, help='optional action_text regex of the reference clip (e.g. "(?i)scanner")')
    parser.add_argument('--copy-skill', type=str, default='^Pick$', help='skill regex of the copied clip')
    parser.add_argument('--copy-regex', type=str, default=None, help='optional action_text regex of the copied clip')
    parser.add_argument('--modification-name', type=str, required=True, metavar='NAME', help='OverlayArmsDataset modification name')
    parser.add_argument('--blend', type=int, default=None, help='frames over which the frame-0 delta decays linearly (default: half of the pre-close reach)')
    parser.add_argument('--delta_rate_limit', type=float, default=None, help='max frame-0 delta correction rate in rad/frame (default: class default)')
    parser.add_argument('--speed_limit', type=float, default=None, help='max playback speed change (default: class default)')
    parser.add_argument('--slices', action='store_true', help='also emit one SlicedDataset control task per distinct ref interval of each episode')
    parser.add_argument('--max_workers', type=int, default=32)
    args = parser.parse_args()
    if not args.modification_name.strip():
        parser.error('--modification-name cannot be empty')

    ref_skill_regexp = re.compile(args.ref_skill)
    ref_regexp = re.compile(args.ref_regex) if args.ref_regex else None
    copy_skill_regexp = re.compile(args.copy_skill)
    copy_regexp = re.compile(args.copy_regex) if args.copy_regex else None

    tasks = load_tasks(args.input)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_task, task, args.data_class, args.data_root, ref_skill_regexp, ref_regexp, copy_skill_regexp, copy_regexp, args.modification_name, args.blend, args.delta_rate_limit, args.speed_limit, args.slices) for task in tasks
        ]
        results = [fut.result() for fut in tqdm(futures, total=len(futures), desc='overlay', file=sys.stderr)]

    kept = [out_task for out_tasks, _ in results for out_task in out_tasks]
    reasons = Counter(reason for _, reason in results if reason)
    for reason, count in reasons.most_common(5):
        print(f'  {count:6d}  {reason}', file=sys.stderr)
    print(f'Emitted {len(kept)} tasks from {len(tasks)} episodes.', file=sys.stderr)
    dump_tasks(kept, args.output)


if __name__ == '__main__':
    main()
