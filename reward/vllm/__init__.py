"""Placeholder vllm package (stub) -- NOT the real vLLM.

ms-swift 4.2.3's GRPO import chain (swift/rlhf_trainers/rollout_mixin.py
-> swift.rollout.multi_turn -> swift.infer_engine.grpo_vllm_engine -> vllm_engine)
imports vllm unconditionally at top level, which crashes when the package is absent
even with --use_vllm false. This repository's training uses HF rollout only
(use_vllm=false) and never instantiates these classes, so empty placeholders are
enough to satisfy the imports.

Note: running python from the repository root shadows any real vllm installation.
If you need a real vLLM, remove this directory or run from another working directory.
"""

__version__ = '0.10.2'  # pretends to satisfy trl's lower bound to silence version warnings


class _Stub:

    def __init__(self, *args, **kwargs):
        raise RuntimeError('vllm stub: no real vLLM installed; use_vllm=true is unavailable.')


class AsyncEngineArgs(_Stub):
    pass


class AsyncLLMEngine(_Stub):
    pass


class EngineArgs(_Stub):
    pass


class LLMEngine(_Stub):
    pass


class SamplingParams(_Stub):
    pass


class LLM(_Stub):
    pass


class RequestOutput(_Stub):
    pass
