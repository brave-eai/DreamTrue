import os
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache

import h5py
import hdf5plugin
import numpy as np

from dataset import ClipInfo as DatasetClipInfo
from dataset import SourceDataset, get_dataset_class


@dataclass(frozen=True, kw_only=True)
class SourceInfo():
    data_class: str
    data_root: str

    condition_root: str
    condition_file: str

    score_root: str | None = None
    score_file: str | None = None

    @property
    def cls(self) -> type[SourceDataset]:
        return get_dataset_class(self.data_class)

    @cached_property
    def condition_file_regexp(self) -> re.Pattern:
        return re.compile(self.condition_file)

    def __post_init__(self):
        assert isinstance(self.data_class, str) and isinstance(self.data_root, str)
        assert isinstance(self.condition_root, str) and isinstance(self.condition_file, str)
        assert (self.score_file is None and self.score_root is None) or (isinstance(self.score_file, str) and isinstance(self.score_root, str))


@dataclass(frozen=True, kw_only=True)
class MergeClipInfo():
    data_class: str
    data_root: str

    condition_root: str
    condition_file: str

    data_key: str
    start: int
    end: int
    fps: int

    score_root: str | None = None
    score_file: str | None = None

    @property
    def condition_path(self) -> str:
        return os.path.join(self.condition_root, self.data_key, self.condition_file)

    @property
    def score_path(self) -> str | None:
        return None if self.score_root is None or self.score_file is None else os.path.join(self.score_root, self.data_key, self.score_file)

    def dict(self) -> dict:
        return {
            **self.__dict__,
            'condition_path': self.condition_path,
            'score_path': self.score_path,
        }

    def __post_init__(self):
        assert isinstance(self.data_class, str) and isinstance(self.data_root, str)
        assert isinstance(self.condition_root, str) and isinstance(self.condition_file, str)
        assert isinstance(self.data_key, str) and isinstance(self.start, int) and isinstance(self.end, int) and isinstance(self.fps, int)
        assert (self.score_file is None and self.score_root is None) or (isinstance(self.score_file, str) and isinstance(self.score_root, str))


@dataclass(frozen=True, kw_only=True)
class ClipContext(DatasetClipInfo, MergeClipInfo):

    @property
    def cls(self) -> type[SourceDataset]:
        return get_dataset_class(self.data_class)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _load_indices(condition_path: str) -> np.ndarray | None:
        with h5py.File(condition_path, 'r', locking=False) as f:
            return np.asarray(f['indices'], dtype=np.int64) if 'indices' in f else None

    @cached_property
    def condition_indices(self) -> np.ndarray | None:
        return ClipContext._load_indices(self.condition_path)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _load_position(data_class: str, data_root: str, data_key: str) -> np.ndarray:
        cls = get_dataset_class(data_class)
        return cls(root=data_root, key=data_key, cams=cls.cams, use_state=False).position

    @cached_property
    def position(self) -> np.ndarray:
        return ClipContext._load_position(self.data_class, self.data_root, self.data_key)

    @property
    def clip(self) -> MergeClipInfo:
        return MergeClipInfo(
            data_class=self.data_class,
            data_root=self.data_root,
            condition_root=self.condition_root,
            condition_file=self.condition_file,
            data_key=self.data_key,
            start=self.start,
            end=self.end,
            fps=self.fps,
            score_root=self.score_root,
            score_file=self.score_file,
        )


def list_condition_files(source: SourceInfo, data_key: str) -> list[str]:
    data_dir = os.path.join(source.condition_root, data_key)
    if not os.path.isdir(data_dir):
        return []
    return [name for name in sorted(os.listdir(data_dir)) if source.condition_file_regexp.fullmatch(name)]
