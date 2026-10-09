"""Abort training when a `stop` file exists in output_dir (or its parent) on checkpoint save.

Usage: --custom_register_path .../stop_file_callback.py and --callbacks stop_file.
"""
import os

from swift.callbacks import TrainerCallback, callbacks_map
from swift.utils import get_logger

logger = get_logger()

class StopFileCallback(TrainerCallback):

    def on_save(self, args, state, control, **kwargs):
        for d in (args.output_dir, os.path.dirname(args.output_dir)):
            stop_file = os.path.join(d, 'stop')
            if os.path.exists(stop_file):
                raise RuntimeError(f'[stop_file] found {stop_file}, aborting training at step {state.global_step}.')

callbacks_map['stop_file'] = StopFileCallback
logger.info(f'[plugin:stop_file] injected: callback stop_file ({__file__})')
