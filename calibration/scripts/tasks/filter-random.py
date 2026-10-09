#!/usr/bin/env python3
"""Randomly sample tasks from a tasks JSON file."""
import argparse
import random
import sys

from scripts.tasks._taskio import add_io_args, dump_tasks, load_tasks


def main():
    parser = argparse.ArgumentParser(description='Randomly sample tasks from a tasks JSON file.')
    add_io_args(parser)
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument('-n', '--count', type=int, metavar='N', help='Number of tasks to sample')
    size_group.add_argument('-r', '--ratio', type=float, metavar='R', help='Fraction of tasks to sample (0 < R <= 1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: None)')
    parser.add_argument('--no-shuffle', action='store_true', default=False, help='Keep original order in output (default: shuffled)')
    args = parser.parse_args()

    tasks = load_tasks(args.input)
    rng = random.Random(args.seed)

    total = len(tasks)
    if args.count is not None:
        k = min(args.count, total)
    else:
        if not (0 < args.ratio <= 1):
            parser.error('--ratio must be in (0, 1]')
        k = max(1, round(total * args.ratio))

    sampled = rng.sample(tasks, k)
    if args.no_shuffle:
        # Restore original order
        order = {id(t): i for i, t in enumerate(tasks)}
        sampled.sort(key=lambda t: order[id(t)])

    print(f'Sampled {len(sampled)} / {total} tasks.', file=sys.stderr)
    dump_tasks(sampled, args.output)


if __name__ == '__main__':
    main()
