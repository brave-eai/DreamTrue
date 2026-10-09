"""vllm stub submodule; see vllm/__init__.py.

swift/rlhf_trainers/utils.py patches StatelessProcessGroup.create for IPv6 at
import time -- it only reflects the signature (inspect.signature), never calls it.
"""


class StatelessProcessGroup:

    @staticmethod
    def create(host, port, rank, world_size, data_expiration_seconds=3600, store_timeout=300, **kwargs):
        raise RuntimeError('vllm stub: no real vLLM installed; this path is unavailable.')
