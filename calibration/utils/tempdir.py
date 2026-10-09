import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory as _TemporaryDirectory


@contextmanager
def TemporaryDirectory(dir: str | None = None):
    if dir is not None:
        with _TemporaryDirectory(dir=dir) as t:
            yield t
    elif 'TEMP_DIR' in os.environ:
        os.makedirs(os.environ['TEMP_DIR'], exist_ok=True)
        with _TemporaryDirectory(dir=os.environ['TEMP_DIR']) as t:
            yield t
    elif os.path.exists('/dev/shm'):
        with _TemporaryDirectory(dir='/dev/shm') as t:
            yield t
    else:
        with _TemporaryDirectory(None) as t:
            yield t
