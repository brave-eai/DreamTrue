import argparse
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

# 这个脚本用来打印 hdf5 文件的结构，或者查看指定 dataset 的统计值或切片，主要用于调试和验证 hdf5 文件的正确性


def print_hdf5_tree(name: str, obj, prefix: str = "", is_last: bool = True):
    connector = "" if name == '/' else "└── " if is_last else "├── "
    short_name = name.split("/")[-1]
    if isinstance(obj, h5py.Dataset):
        print(f"{prefix}{connector}{short_name}  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{prefix}{connector}{short_name}/")
    else:
        print(f"{prefix}{connector}{short_name}  ({type(obj).__name__})")

    if hasattr(obj, "attrs") and len(obj.attrs) > 0:
        attr_prefix = prefix + ("" if name == '/' else "    " if is_last else "│   ")
        for attr_key, attr_val in obj.attrs.items():
            print(f"{attr_prefix}• @{attr_key} = {attr_val!r}")

    if isinstance(obj, h5py.Group):
        keys = list(obj.keys())
        child_prefix = prefix + ("" if name == '/' else "    " if is_last else "│   ")
        for i, k in enumerate(keys):
            child_is_last = (i == len(keys) - 1)
            print_hdf5_tree(k, obj[k], child_prefix, child_is_last)


def parse_index_token(token: str):
    token = token.strip()
    if token == "":
        return slice(None)
    if token == "...":
        return Ellipsis
    if ":" not in token:
        return int(token.removeprefix('\\'))
    parts = token.split(":")
    if len(parts) > 3:
        raise ValueError(f"非法切片片段: {token}")
    while len(parts) < 3:
        parts.append("")
    start, stop, step = (int(p.removeprefix('\\')) if p.strip() != "" else None for p in parts)
    return slice(start, stop, step)


def parse_slice_expr(expr: str):
    expr = expr.strip()
    if expr == "":
        return slice(None)
    items = [parse_index_token(p) for p in expr.split(",")]
    if len(items) == 1:
        return items[0]
    return tuple(items)


def print_dataset_summary(ds: h5py.Dataset):
    print(f"dataset: /{ds.name.lstrip('/')}")
    print(f"shape={ds.shape} dtype={ds.dtype}")


def print_dataset_minmax(ds: h5py.Dataset):
    try:
        arr = ds[()]
    except Exception as e:
        print(f"读取 dataset 失败，无法计算 min/max: {e}")
        return

    if isinstance(arr, np.ndarray) and arr.size == 0:
        print("min/max: 空数组")
        return
    if np.issubdtype(np.asarray(arr).dtype, np.number) or np.issubdtype(np.asarray(arr).dtype, np.bool_):
        print(f"min={np.min(arr)}")
        print(f"max={np.max(arr)}")
    else:
        print(f"min/max: 当前 dtype({np.asarray(arr).dtype}) 不支持")


def print_dataset_slice(ds: h5py.Dataset, slice_expr: str):
    try:
        index = parse_slice_expr(slice_expr)
        sliced = ds[index]
        print(f"slice={index}")
        print(np.asarray(sliced))
    except Exception as e:
        print(f"读取切片失败: {e}")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="打印 HDF5 结构，可选查看 dataset 的统计值或切片")
    parser.add_argument("path", type=str, help="HDF5 文件路径")
    parser.add_argument("--dataset", type=str, default=None, help="dataset 路径，例如 qpos 或 head/rgb")
    parser.add_argument("--minmax", action="store_true", help="打印指定 dataset 的最小值和最大值")
    parser.add_argument("--slice", dest="slice_expr", type=str, default=[], help="打印指定 dataset 的切片，例如 '0:10' 或 '0:5, :, 3'", nargs="+")
    return parser.parse_args()


def main():
    args = build_args()
    path = Path(args.path)

    if not path.is_file():
        print(f"错误: 找不到文件: {path}")
        sys.exit(1)

    try:
        with h5py.File(path, "r") as f:
            print(str(path), f'{os.path.getsize(path)/1024/1024:.2f} MB')
            if args.dataset is None:
                print_hdf5_tree("/", f, "", True)
                return
            dataset_path = args.dataset.strip().lstrip("/")
            if dataset_path not in f:
                print(f"错误: dataset 不存在: {args.dataset}")
                sys.exit(1)
            obj = f[dataset_path]
            if not isinstance(obj, h5py.Dataset):
                print(f"错误: 路径不是 dataset: {args.dataset}")
                sys.exit(1)
            print_dataset_summary(obj)
            if args.minmax:
                print_dataset_minmax(obj)
            for sli in args.slice_expr:
                print_dataset_slice(obj, sli)
    except OSError as e:
        print(f"无法打开 HDF5 文件: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
