#!/usr/bin/env python3
"""Add or remove GripperKeepOpenDataset entry in modifications list."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _taskio import add_io_args, load_tasks, dump_tasks


def main():
    parser = argparse.ArgumentParser(description='Add or remove GripperKeepOpenDataset modification.')
    add_io_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--set', action='store_true', help='Add GripperKeepOpenDataset to modifications')
    group.add_argument('--unset', action='store_true', help='Remove GripperKeepOpenDataset from modifications')
    args = parser.parse_args()
    tasks = load_tasks(args.input)
    for task in tasks:
        mods = list(task.get('modifications', []))
        if args.set:
            if not any(m.get('class') == 'GripperKeepOpenDataset' for m in mods):
                mods.append(dict({'class': 'GripperKeepOpenDataset'}))
            task['modifications'] = mods
        else:
            mods = [m for m in mods if m.get('class') != 'GripperKeepOpenDataset']
            if mods:
                task['modifications'] = mods
            else:
                task.pop('modifications', None)
    dump_tasks(tasks, args.output)


if __name__ == '__main__':
    main()
