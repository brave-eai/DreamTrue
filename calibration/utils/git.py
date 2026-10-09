import os
import subprocess
from functools import lru_cache

try:
    from cachetools import TTLCache, cached
    _cache = lambda maxsize, ttl: cached(cache=TTLCache(maxsize=maxsize, ttl=ttl))
except ImportError:
    _cache = lambda maxsize, ttl: lru_cache(maxsize=maxsize)


def git_status(root: str, ignore_error: bool = False) -> str:
    try:
        if os.path.isfile(root):
            root = os.path.dirname(root)
        root = os.path.abspath(root)
        return _git_status(root)
    except Exception:
        if ignore_error:
            return ''
        raise


@_cache(maxsize=128, ttl=60)
def _git_status(root: str) -> str:
    log = subprocess.check_output(
        ['git', '--no-optional-locks', '-C', root, 'log', '-1', '--pretty=format:%H (%D) | %ai | %s'],
        stderr=subprocess.STDOUT,
        text=True,
    )
    status = subprocess.check_output(
        ['git', '--no-optional-locks', '-C', root, 'status', '-sb', '--untracked-files=no'],
        stderr=subprocess.STDOUT,
        text=True,
    )
    return root + '\n' + log + '\n' + status
