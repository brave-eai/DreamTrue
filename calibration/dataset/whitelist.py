"""Whitelist helpers.

格式：一行一个 key 的纯文本文件，文件名后缀 `.whitelist`。
"""
import glob
import os


def __load_one(path: str) -> set[str]:
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_whitelist(path: str | None) -> set[str] | None:
    """加载 whitelist：path 是文件 → 单文件解析；path 是目录 → 所有 *.whitelist 取交集。

    - 单文件：跳过空行；返回去重后的 set。
    - 目录：取目录下所有 `*.whitelist` 的交集；目录下无任何 whitelist 则返回 set()。
    """
    if path is None or not os.path.exists(path):
        return None
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.whitelist')))
        if not files:
            return None
        result: set[str] = __load_one(files[0])
        for f in files[1:]:
            result &= __load_one(f)
        return result
    else:
        return __load_one(path)
