#!/usr/bin/env python3
"""Remove modifications field from all tasks."""
import argparse
from copy import deepcopy

from scripts.tasks._taskio import add_io_args, dump_tasks, load_tasks


def main():
    parser = argparse.ArgumentParser(description='Remove modifications field from all tasks.')
    parser.add_argument('--append', action='store_true')
    add_io_args(parser)
    args = parser.parse_args()
    ori_tasks = load_tasks(args.input)
    tasks = deepcopy(ori_tasks)
    for task in tasks:
        task.pop('modifications', None)
    if args.append:
        tasks += [t for t in ori_tasks if t.get('modifications') is not None]
    dump_tasks(tasks, args.output)


if __name__ == '__main__':
    main()
