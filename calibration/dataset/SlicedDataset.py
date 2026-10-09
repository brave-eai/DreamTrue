import hashlib
import json
from collections import OrderedDict
from functools import cached_property

import numpy as np

from .BaseDataset import BaseDataset, ModifiedDataset


class SlicedDataset(ModifiedDataset):

    def __init__(self, dataset: BaseDataset, sli: int | str | slice | np.ndarray | list[int] | None = None, *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        if isinstance(sli, int):
            sli = slice(sli, sli + 1, 1)
        elif isinstance(sli, str):
            sli = slice(*[int(p) if p else None for p in sli.split(':')])
        elif sli is None:
            sli = slice(None)
        elif isinstance(sli, list):
            sli = np.array(sli, dtype=np.int64)
        self._sli = sli

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict((k, v[self._sli]) for k, v in self._d.rgbs.items())

    @cached_property
    def position(self) -> np.ndarray:
        return self._d.position[self._sli]

    @cached_property
    def _len(self) -> int:
        return len(self.indices)

    @cached_property
    def qpos(self) -> np.ndarray:
        return self._d.qpos[self._sli]

    @cached_property
    def indices(self) -> np.ndarray:
        return self._d.indices[self._sli]

    @cached_property
    def dataset_name(self) -> list[str]:
        if isinstance(self._sli, slice) and self._sli == slice(None):
            return self._d.dataset_name
        if isinstance(self._sli, slice):
            payload = ['slice', self._sli.start, self._sli.stop, self._sli.step]
        else:
            payload = ['indices', np.asarray(self._sli, dtype=np.int64).tolist()]
        h = hashlib.sha256()
        h.update(json.dumps(payload, separators=(',', ':')).encode())
        return self._d.dataset_name + [f'sliced_{h.hexdigest()[:16]}']


class ShuffledDataset(SlicedDataset):

    def __init__(self, dataset: BaseDataset, *args, **kwargs):
        super().__init__(dataset=dataset, sli=np.random.permutation(len(dataset)), *args, **kwargs)

    @cached_property
    def dataset_name(self) -> list[str]:
        return self._d.dataset_name + ['shuffled']
