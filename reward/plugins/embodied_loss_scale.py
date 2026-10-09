"""Per-token loss weighting plugin (register with --loss_scale embodied).

Use --use_liger_kernel false: the template drops real-valued weights under liger.
"""
import contextvars
import re

from swift.loss_scale.base import ALL_BASE_STRATEGY, LossScale
from swift.loss_scale.mapping import get_loss_scale as _orig_get_loss_scale
from swift.utils import get_logger

logger = get_logger()

class EmbodiedLossScale(LossScale):

    is_binary = False
    _PREFIX = 'embodied'
    _PAIR_RE = re.compile(r'"(\w+)"\s*:\s*"([^"]+)"')
    _sample_weights: contextvars.ContextVar[dict | None] = contextvars.ContextVar('_sample_weights', default=None)

    def __init__(self, base_strategy: str = 'default'):
        super().__init__(base_strategy)
        assert self.is_binary is False, 'is_binary 为 True 会让实值权重被静默丢弃'

    def _inner_call(self, context, context_type, query, loss, loss_scale, is_last_round):
        if isinstance(loss_scale, dict):
            token = self._sample_weights.set(loss_scale)
            try:
                return super()._inner_call(context, context_type, query, loss, None, is_last_round)
            finally:
                self._sample_weights.reset(token)

        mult = 1.0 if loss_scale is None else float(loss_scale)
        ctx, weights = super()._inner_call(context, context_type, query, loss, None, is_last_round)
        if mult == 1.0:
            return ctx, weights
        return ctx, [w * mult for w in weights]

    def get_loss_scale(self, context, **kwargs):
        if not isinstance(context, str):
            return super().get_loss_scale(context, **kwargs)

        sample_weights = self._sample_weights.get()
        if not sample_weights:
            return [context], [1.0]

        te = context.find('</think>')
        region_start = te + len('</think>') if te != -1 else 0

        segs, weights, cur = [], [], 0
        for m in self._PAIR_RE.finditer(context, region_start):
            dim = m.group(1)
            w = float(sample_weights.get(dim, 1.0))
            vs, ve = m.span(2)
            if vs > cur:
                segs.append(context[cur:vs])
                weights.append(1.0)
            segs.append(context[vs:ve])
            weights.append(w)
            cur = ve
        if cur < len(context):
            segs.append(context[cur:])
            weights.append(1.0)
        if not segs:
            return [context], [1.0]
        return segs, weights

def _patched_get_loss_scale(loss_scale: str) -> LossScale:
    base_strategy = 'default'
    spec = loss_scale
    if '+' in loss_scale:
        maybe_base, rest = loss_scale.split('+', 1)
        if maybe_base in ALL_BASE_STRATEGY:
            base_strategy, spec = maybe_base, rest
    if spec == EmbodiedLossScale._PREFIX:
        return EmbodiedLossScale(base_strategy)
    return _orig_get_loss_scale(loss_scale)

import swift.loss_scale as _ls_pkg  # noqa: E402

_ls_pkg.get_loss_scale = _patched_get_loss_scale
logger.info(f'[plugin:embodied_loss_scale] injected: swift.loss_scale.get_loss_scale ({__file__})')
