"""Train-time evaluation subsetting: names eval jobs by the test_<variant> path segment."""

from collections import OrderedDict

import transformers
from swift.pipelines.train.sft import SwiftSft
from swift.pipelines.utils import get_cached_dataset
from swift.trainers.mixin import SwiftMixin
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from swift.utils import get_logger

logger = get_logger()

def _variant_name(path: str) -> str:
    clean = path.split('#', 1)[0].rstrip('/')
    for seg in reversed(clean.split('/')):
        if seg.startswith('test_'):
            return seg[len('test_'):]
    return clean.rsplit('/', 1)[-1]

def _make_prepare_dataset(orig):

    def _prepare_dataset(self):
        datasets = orig(self)
        args = self.args
        if len(getattr(args, 'cached_val_dataset', []) or []) <= 1:
            return datasets
        _, val_list = get_cached_dataset(args)
        if len(val_list) != len(args.cached_val_dataset):
            logger.warning(f'[embodied_eval_subset] val 数({len(val_list)}) 与 cached_val_dataset '
                           f'({len(args.cached_val_dataset)}) 不一致,跳过分组。')
            return datasets
        named = OrderedDict()
        for path, ds in zip(args.cached_val_dataset, val_list):
            named[_variant_name(path)] = self._post_process_datasets([None, ds])[1]
        datasets[1] = named
        logger.info(f'[embodied_eval_subset] 按变体分组评测:{list(named)}')
        ta = getattr(args, 'training_args', None)
        m = getattr(ta, 'metric_for_best_model', None) if ta is not None else None
        if ta is not None and m is not None and not m.startswith(tuple(f'eval_{n}_' for n in named)):
            logger.info(f'[embodied_eval_subset] 禁用 metric_for_best_model(原 {m!r});如需选最优请'
                        f'显式指定某子集,如 eval_{next(iter(named))}_loss。')
            ta.metric_for_best_model = None
            ta.greater_is_better = None
            ta.load_best_model_at_end = False
        return datasets

    return _prepare_dataset

def _make_evaluate(orig):

    def evaluate(self, *args, **kwargs):
        prev = getattr(self, '_subset_metric_prefix', None)
        self._subset_metric_prefix = kwargs.get('metric_key_prefix', 'eval')
        try:
            return orig(self, *args, **kwargs)
        finally:
            self._subset_metric_prefix = prev

    return evaluate

def _make_log(orig):

    def log(self, logs, *args, **kwargs):
        mode = 'train' if self.model.training else 'eval'
        prefix = getattr(self, '_subset_metric_prefix', None)
        if mode == 'eval' and prefix and prefix != 'eval':
            logs.update(self.compute_custom_metrics(self.custom_metrics['eval'], prefix + '_'))
            return transformers.Trainer.log(self, logs, *args, **kwargs)
        return orig(self, logs, *args, **kwargs)

    return log

if not getattr(SwiftSft._prepare_dataset, '_embodied_subset_patched', False):
    SwiftSft._prepare_dataset = _make_prepare_dataset(SwiftSft._prepare_dataset)
    SwiftSft._prepare_dataset._embodied_subset_patched = True
    logger.info(f'[plugin:embodied_eval_subset] injected: ({__file__})')
else:
    logger.info(f'[plugin:embodied_eval_subset] already injected: ({__file__})')

if not getattr(Seq2SeqTrainer.evaluate, '_embodied_subset_patched', False):
    Seq2SeqTrainer.evaluate = _make_evaluate(Seq2SeqTrainer.evaluate)
    Seq2SeqTrainer.evaluate._embodied_subset_patched = True

if not getattr(SwiftMixin.log, '_embodied_subset_patched', False):
    SwiftMixin.log = _make_log(SwiftMixin.log)
    SwiftMixin.log._embodied_subset_patched = True
