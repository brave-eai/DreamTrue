#!/usr/bin/env python3
"""Drop tasks whose modifications cannot be solved by IK.

Rebuilds the same chain calibrate/condition.py builds later (build_modified_datasets
then touching qpos), so infeasible clips are dropped here.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from dataset import build_modified_datasets, get_dataset_class
from scripts.tasks._taskio import add_io_args, dump_tasks, load_tasks
from sim import SapienEnv


def solve(task: dict, data_class: str, data_root: str, robot_path: str) -> str | None:
    """Failure reason, or None when the IK solve succeeds."""
    try:
        dataset = get_dataset_class(data_class)(root=data_root, key=task['data_key'], cams=[], use_state=False)
        dataset = build_modified_datasets(dataset, task.get('modifications', []), assets_path=SapienEnv.assets_path, robot_path=robot_path)
        _ = dataset.qpos
    except Exception as e:
        return f'{type(e).__name__}: {e}'
    return None


def main():
    parser = argparse.ArgumentParser(description='Drop tasks whose IK cannot be solved.')
    add_io_args(parser)
    parser.add_argument('--data_class', type=str, default='AgibotDataset')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--robot_path', type=str, default='agibot/G1_120s/G1_120s.yaml')
    parser.add_argument('--max_workers', type=int, default=32)
    args = parser.parse_args()

    tasks = load_tasks(args.input)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(solve, t, args.data_class, args.data_root, args.robot_path) for t in tasks]
        reasons = [fut.result() for fut in tqdm(futures, total=len(futures), desc='ik', file=sys.stderr)]

    kept = [t for t, reason in zip(tasks, reasons, strict=True) if reason is None]
    for reason in list(dict.fromkeys(r for r in reasons if r is not None))[:5]:
        print(f'  e.g. {reason}', file=sys.stderr)
    print(f'Kept {len(kept)} / {len(tasks)} tasks.', file=sys.stderr)
    dump_tasks(kept, args.output)


if __name__ == '__main__':
    main()
