"""Workaround for a ROCm flash-attn 2.8.3 backward stride bug (contiguous qkv views)."""

import functools
import os

from swift.utils import get_logger

logger = get_logger()

def _install():
    if os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") != "TRUE":
        return
    try:
        import flash_attn
        import flash_attn.flash_attn_interface as fai
    except ImportError:
        return

    orig_func = fai.flash_attn_func
    orig_varlen = fai.flash_attn_varlen_func

    @functools.wraps(orig_func)
    def func_wrap(q, k, v, *a, **kw):
        return orig_func(q.contiguous(), k.contiguous(), v.contiguous(), *a, **kw)

    @functools.wraps(orig_varlen)
    def varlen_wrap(q, k, v, *a, **kw):
        return orig_varlen(q.contiguous(), k.contiguous(), v.contiguous(), *a, **kw)

    fai.flash_attn_func = func_wrap
    fai.flash_attn_varlen_func = varlen_wrap
    flash_attn.flash_attn_func = func_wrap
    flash_attn.flash_attn_varlen_func = varlen_wrap
    logger.info(f"[plugin:fa_rocm_stride_fix] wrapped flash_attn funcs with contiguous() ({__file__})")

_install()
