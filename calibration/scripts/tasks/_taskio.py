import argparse
import json
import sys


def add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--input', '-i', type=str, default='-', help='Input tasks JSON (default: stdin)')
    parser.add_argument('--output', '-o', type=str, default='-', help='Output tasks JSON (default: stdout)')


def load_tasks(path: str) -> list:
    if path == '-':
        return json.load(sys.stdin)
    with open(path) as f:
        return json.load(f)


def dump_tasks(tasks: list, path: str) -> None:
    text = json.dumps(tasks) + '\n'
    if path == '-':
        sys.stdout.write(text)
    else:
        with open(path, 'w') as f:
            f.write(text)
