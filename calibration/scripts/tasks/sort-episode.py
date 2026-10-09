#!/usr/bin/env python3
"""Sort tasks by episode_id (the integer after '/' in data_key)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _taskio import add_io_args, load_tasks, dump_tasks


def _episode_id(task: dict) -> int:
    return int(task['data_key'].split('/')[-1])


def main():
    parser = argparse.ArgumentParser(description='Sort tasks by episode_id.')
    add_io_args(parser)
    args = parser.parse_args()
    tasks = load_tasks(args.input)
    tasks.sort(key=_episode_id)
    dump_tasks(tasks, args.output)


if __name__ == '__main__':
    main()
