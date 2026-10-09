#!/usr/bin/env python3
import argparse

from scripts.tasks._taskio import add_io_args, dump_tasks, load_tasks


def main():
    parser = argparse.ArgumentParser(description='Sample tasks that have modifications.', )
    add_io_args(parser)
    args = parser.parse_args()
    tasks = load_tasks(args.input)
    sampled = [t for t in tasks if sum(len(v) for v in t.get('modifications', [])) != 0]
    dump_tasks(sampled, args.output)


if __name__ == '__main__':
    main()
