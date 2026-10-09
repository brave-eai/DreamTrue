"""Snapshots the full environment into output_dir at save_args time (reproducibility)."""
import json
import os

from swift.arguments.base_args.base_args import BaseArguments
from swift.utils import get_logger, is_master

logger = get_logger()

_orig_save_args = BaseArguments.save_args

def _dump_env(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    env = dict(os.environ.items())
    fpath = os.path.join(output_dir, 'env_snapshot.json')
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(env.items())), f, ensure_ascii=False, indent=2)
    logger.info(f'[save_env] {len(env)} env vars saved to: {fpath}')

def save_args(self, output_dir=None):
    ret = _orig_save_args(self, output_dir)
    if is_master():
        _dump_env(output_dir or self.output_dir)
    return ret

BaseArguments.save_args = save_args
logger.info(f'[plugin:save_env] injected: BaseArguments.save_args ({__file__})')
