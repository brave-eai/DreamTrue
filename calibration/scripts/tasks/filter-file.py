#!/usr/bin/env python3
"""Keep only tasks whose data_key appears in a given key list."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _taskio import add_io_args, dump_tasks, load_tasks


def _load_keys(path: str) -> set:
    if path.endswith('.txt') or path.endswith('.whitelist'):
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}
    with open(path) as f:
        data = json.load(f)
    if not data:
        return set()
    # list of strings
    if isinstance(data[0], str):
        return set(data)
    # list of task dicts — same format as input
    return {t['data_key'] for t in data}


def main():
    parser = argparse.ArgumentParser(description='Filter tasks by data_key list.')
    add_io_args(parser)
    parser.add_argument('--keys', '-k', required=True, nargs='+', metavar='FILE', help='one or more .txt (one key per line) or .json (list of strings or task dicts) files')
    parser.add_argument('--exclude', action='store_true', help='Exclude matching keys instead of keeping them')
    args = parser.parse_args()

    keys = set()
    for path in args.keys:
        keys |= _load_keys(path)
    tasks = load_tasks(args.input)
    if args.exclude:
        tasks = [t for t in tasks if t['data_key'] not in keys]
    else:
        tasks = [t for t in tasks if t['data_key'] in keys]
    dump_tasks(tasks, args.output)


if __name__ == '__main__':
    main()
