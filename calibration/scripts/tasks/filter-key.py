#!/usr/bin/env python3
"""Keep only tasks whose data_key matches a regex."""
import argparse
import re
import sys

from scripts.tasks._taskio import add_io_args, dump_tasks, load_tasks


def main():
    parser = argparse.ArgumentParser(description='Filter tasks by a regex on data_key.')
    add_io_args(parser)
    parser.add_argument('pattern', help='regex matched from the start; use | for alternatives')
    parser.add_argument('--field', '-f', default='data_key', help='task field to match (default: data_key)')
    parser.add_argument('--exclude', action='store_true', help='Drop matching tasks instead of keeping them')
    args = parser.parse_args()

    regexp = re.compile(args.pattern)

    tasks = load_tasks(args.input)
    kept = [t for t in tasks if (regexp.match(t[args.field]) is not None) != args.exclude]
    print(f'Kept {len(kept)} / {len(tasks)} tasks.', file=sys.stderr)
    dump_tasks(kept, args.output)


if __name__ == '__main__':
    main()
