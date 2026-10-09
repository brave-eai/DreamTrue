#!/usr/bin/env python3
"""Keep only tasks whose clip skills match a regex.

skill lives in dataset.clip_info, not in the task JSON, so the dataset has to be
loaded. Datasets without skill annotations (droid, robomind, ...) match nothing.
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from dataset import get_dataset_class
from scripts.tasks._taskio import add_io_args, dump_tasks, load_tasks


def episode_clips(data_class: str, data_root: str, data_key: str) -> list[tuple[int, int, list[str]]]:
    try:
        dataset = get_dataset_class(data_class)(root=data_root, key=data_key, cams=[], use_state=False)
        return [(c.start, c.end, c.skill) for c in dataset.clip_info]
    except Exception as e:
        print(f'Warning: failed to load dataset for {data_key}: {e}', file=sys.stderr)
        return []


def sliced_range(task: dict) -> tuple[int, int] | None:
    """Clip range of an already-sliced task, or None when the task covers the whole episode."""
    for mod in task.get('modifications', []):
        if mod.get('class') == 'SlicedDataset' and ':' in str(mod.get('sli', '')):
            start, end = str(mod['sli']).split(':')
            return int(start), int(end)
    return None


def main():
    parser = argparse.ArgumentParser(description='Filter tasks by a regex on clip skills.')
    add_io_args(parser)
    parser.add_argument('pattern', help='regex matched from the start; use | for alternatives')
    parser.add_argument('--data_class', type=str, default='AgibotDataset')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--exclude', action='store_true', help='Drop matching tasks instead of keeping them')
    parser.add_argument('--max_workers', type=int, default=32)
    args = parser.parse_args()

    regexp = re.compile(args.pattern)

    tasks = load_tasks(args.input)
    data_keys = sorted({t['data_key'] for t in tasks})
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {key: executor.submit(episode_clips, args.data_class, args.data_root, key) for key in data_keys}
        clips = {key: fut.result() for key, fut in tqdm(futures.items(), total=len(futures), desc='skill', file=sys.stderr)}

    def matched(task: dict) -> bool:
        rng = sliced_range(task)
        for start, end, skills in clips[task['data_key']]:
            if rng is not None and (end <= rng[0] or start >= rng[1]):
                continue
            if any(regexp.match(s) is not None for s in skills):
                return True
        return False

    kept = [t for t in tasks if matched(t) != args.exclude]
    print(f'Kept {len(kept)} / {len(tasks)} tasks.', file=sys.stderr)
    dump_tasks(kept, args.output)


if __name__ == '__main__':
    main()
