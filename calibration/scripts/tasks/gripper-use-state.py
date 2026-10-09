#!/usr/bin/env python3
"""Set or remove use_state flag on all tasks."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _taskio import add_io_args, dump_tasks, load_tasks


def main():
    parser = argparse.ArgumentParser(description='Set or remove use_state flag.')
    add_io_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--set', action='store_true', help='Set use_state=true')
    group.add_argument('--unset', action='store_true', help='Remove use_state field')
    args = parser.parse_args()
    tasks = load_tasks(args.input)
    for task in tasks:
        if args.set:
            task['use_state'] = True
        else:
            task['use_state'] = False
    dump_tasks(tasks, args.output)


if __name__ == '__main__':
    main()
