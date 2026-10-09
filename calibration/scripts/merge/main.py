import argparse
import hashlib
import json
import os
import re
from datetime import datetime

import yaml

from scripts.merge.dispatch import dispatch_sources_parallel
from scripts.merge.scan import MergeClipInfo, SourceInfo
from utils import sha256


def _split_clips(clips: list[MergeClipInfo], splits: dict[str, int], seed: int) -> dict[str, list[MergeClipInfo]]:
    """按 data_class+data_key 哈希，将 clips 稳定分配到多个命名分组。

    splits: e.g. {'train': 95, 'val': 5}，权重无需归一化。
    """
    names = sorted(splits.keys())
    total_weight = sum(splits[n] for n in names)
    # 构建累积阈值: [(threshold, name), ...]
    thresholds: list[tuple[int, str]] = []
    cumulative = 0
    for name in names:
        cumulative += splits[name]
        thresholds.append((cumulative * 10000 // total_weight, name))

    result: dict[str, list[MergeClipInfo]] = {n: [] for n in names}
    for clip in clips:
        h = int(hashlib.sha256(f"{seed}_{clip.data_class}_{clip.data_key}".encode()).hexdigest(), 16)
        bucket = h % 10000
        for threshold, name in thresholds:
            if bucket < threshold:
                result[name].append(clip)
                break
    return result


def _format_yaml_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _resolve_meta(meta_info: dict[str, dict], group_name: str) -> dict:
    """先精确匹配 meta_info 的键，否则按插入顺序用正则 fullmatch，首个命中生效。"""
    if group_name in meta_info:
        return meta_info[group_name]
    for pattern, meta in meta_info.items():
        if re.fullmatch(pattern, group_name):
            return meta
    raise KeyError(f'no meta_info entry matches group {group_name}')


def _write_dataset_config(output_dir: str, grouped: dict[str, list[MergeClipInfo]], now: str, meta_info: dict):
    os.makedirs(output_dir, exist_ok=True)
    total = sum(len(v) for v in grouped.values())
    lines = [
        f'# Generated: {now}',
        f'# Total clips: {total}',
        '',
    ]
    for group_name, clips in sorted(grouped.items()):
        if len(clips) == 0:
            continue
        group_meta = _resolve_meta(meta_info, group_name)
        assert all(c.fps == clips[0].fps for c in clips)
        json_path = os.path.join(output_dir, f'{group_name}.json')
        print(f'  {group_name}: {len(clips)} clips -> {json_path}')
        clips = sorted(clips, key=lambda c: (c.data_class, c.data_root, c.condition_root, c.data_key, c.condition_file, c.score_path or '', c.start, c.end, c.fps))
        with open(json_path, 'w') as f:
            json.dump([c.dict() for c in clips], f)
        infos = {
            'path': f'{group_name}.json',
            'sha256': sha256(json_path),
            'total': len(clips),
            'target_size': group_meta.get('target_size', len(clips)),
            'min_interval':group_meta.get('min_interval', group_meta.get('interval', max(1, round(clips[0].fps / group_meta.get('target_fps', 5))))),
            'max_interval':group_meta.get('max_interval', group_meta.get('interval', max(1, round(clips[0].fps / group_meta.get('target_fps', 5))))),
        }
        for k, v in group_meta.items():
            if k not in infos and k not in {'comment', 'target_fps','min_interval','max_interval','interval'}:
                infos[k] = v
        if 'comment' in group_meta:
            lines.append(f'# {group_meta["comment"]}')
        for i, (key, value) in enumerate(infos.items()):
            lines.append(('- ' if i == 0 else '  ') + key + ': ' + _format_yaml_value(value))
        lines.append('')

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'config -> {os.path.join(output_dir, "config.yaml")}')


def _expand_environment_variables(value):
    """递归展开配置中的 $VAR / ${VAR} 占位符；未设置的变量保持原样，
    由后续的路径错误提示需要配置哪一项。"""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_environment_variables(v) for k, v in value.items()}
    return value


def main():
    args = argparse.ArgumentParser()
    args.add_argument('config', help='YAML config file path')
    args.add_argument('--max_workers', type=int, default=None)
    args = args.parse_args()
    with open(args.config) as f:
        config = _expand_environment_variables(yaml.safe_load(f))

    sources = [SourceInfo(**s) for s in config['sources']]
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(config.get('output_dir', '/tmp/merge_output'), now)
    print(f'output to {output_dir}')

    max_workers = args.max_workers or config.get('max_workers') or os.cpu_count() or 1
    print(f'dispatching with {max_workers} workers...')
    grouped = dispatch_sources_parallel(sources=sources, filter_config=config['filter'], max_workers=max_workers)

    split_cfg = dict(config.get('split') or {})
    meta_defaults = config.get('meta_defaults', {})
    meta_info = {k: {**meta_defaults, **v} for k, v in config.get('meta_info', {}).items()}

    # 构造 [(目录, 分组数据)] 列表，有 split 时按 split 拆，否则原样输出
    outputs: list[tuple[str, dict[str, list[MergeClipInfo]]]] = []
    if split_cfg:
        seed = split_cfg.pop('seed', 0)
        splits = {k: int(v) for k, v in split_cfg.items()}
        split_results: dict[str, dict[str, list[MergeClipInfo]]] = {}
        for group_name, clips in grouped.items():
            for split_name, split_clips in _split_clips(clips, splits, seed).items():
                split_results.setdefault(split_name, {})[group_name] = split_clips
        for split_name, split_grouped in sorted(split_results.items()):
            outputs.append((f'{output_dir}_{split_name}', split_grouped))
    else:
        outputs.append((output_dir, grouped))

    for out_dir, out_grouped in outputs:
        _write_dataset_config(out_dir, out_grouped, now, meta_info)


if __name__ == '__main__':
    main()
