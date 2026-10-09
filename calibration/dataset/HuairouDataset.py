import os
from collections import OrderedDict
from functools import cached_property
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq

from utils import video_read

from .BaseDataset import ClipInfo, SourceDataset


class HuairouPiperDataset(SourceDataset):

    fps: int = 30
    cams = ('cam_high', 'cam_left_wrist', 'cam_right_wrist', 'cam_third')

    _VIDEO_KEYS = {
        'cam_high': 'observation.images.cam_high',
        'cam_left_wrist': 'observation.images.cam_left_wrist',
        'cam_right_wrist': 'observation.images.cam_right_wrist',
        'cam_third': 'observation.images.cam_third',
    }
    _SOURCE_QPOS_ORDER = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13], dtype=np.int64)
    _RAW_GRIPPER_OPEN_QPOS = 0.11

    def __init__(self, root: str, key: str, cams: Sequence[str | tuple[str, tuple[int, int]]], use_state: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert key.count('/') == 1, f'expect 2-segment key "019/000010", got {key!r}'
        chunk, episode = key.split('/')
        assert chunk.isdigit() and episode.isdigit()
        self._root = root
        self._key = key
        self._cams: list[tuple[str, None | tuple[int, int]]] = [cam if isinstance(cam, tuple) else (cam, None) for cam in cams]
        self._use_state = use_state
        for cam, _ in self._cams:
            assert cam in self._VIDEO_KEYS, f'unknown camera {cam!r}; available={self.cams}'

    @property
    def root(self):
        return self._root

    @property
    def key(self):
        return self._key

    @property
    def _chunk(self) -> str:
        return 'chunk-' + self._key.split('/')[0]

    @property
    def _episode(self) -> str:
        return 'episode_' + self._key.split('/')[1]

    @property
    def _parquet_path(self) -> str:
        return os.path.join(self._root, 'data', self._chunk, self._episode + '.parquet')

    def _video_path(self, cam: str) -> str:
        return os.path.join(self._root, 'videos', self._chunk, self._VIDEO_KEYS[cam], self._episode + '.mp4')

    @classmethod
    def list_keys(cls, root: str) -> Iterator[str]:
        data_root = os.path.join(root, 'data')
        if not os.path.isdir(data_root):
            return
        with os.scandir(data_root) as chunk_it:
            for chunk in sorted((e for e in chunk_it if e.is_dir()), key=lambda e: e.name):
                chunk_id = chunk.name.removeprefix('chunk-')
                with os.scandir(chunk.path) as ep_it:
                    for ep in sorted((e for e in ep_it if e.is_file() and e.name.endswith('.parquet')), key=lambda e: e.name):
                        yield f'{chunk_id}/{ep.name.removesuffix(".parquet").removeprefix("episode_")}'

    @cached_property
    def _table(self):
        return pq.read_table(self._parquet_path)

    @cached_property
    def _len(self) -> int:
        return self._table.num_rows

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        result: OrderedDict[str, np.ndarray] = OrderedDict()
        for cam, r in self._cams:
            frames = video_read(self._video_path(cam), resolution=r)
            assert frames.shape[0] == len(self), f'{cam} decoded {frames.shape[0]} frames, expected {len(self)}'
            result[cam] = frames
        return result

    @cached_property
    def position(self) -> np.ndarray:
        out = np.zeros((len(self), 7), dtype=np.float64)
        out[:, 3] = 1.0
        return out

    @cached_property
    def qpos(self) -> np.ndarray:
        column = 'observation.state' if self._use_state else 'action'
        raw = np.asarray(self._table[column].to_pylist(), dtype=np.float64)
        assert raw.shape == (len(self), 14), raw.shape
        out = raw[:, self._SOURCE_QPOS_ORDER]
        out[:, 12:14] = np.clip(out[:, 12:14], 0.0, self._RAW_GRIPPER_OPEN_QPOS) / 2.0
        return out

    @cached_property
    def endpose(self) -> np.ndarray:
        raise NotImplementedError('HuairouPiperDataset not contain end-effector pose yet.')

    @cached_property
    def _qpos_dims(self) -> dict[str, slice | np.ndarray]:
        return {
            'arm': slice(0, 12),
            'arm_left': slice(0, 6),
            'arm_right': slice(6, 12),
            'gripper': slice(12, 14),
            'gripper_left': slice(12, 13),
            'gripper_right': slice(13, 14),
        }

    @cached_property
    def _endpose_arms(self) -> dict[str, int]:
        return {'left': 0, 'right': 1}

    @cached_property
    def gripper_open_qpos(self) -> float:
        return self._RAW_GRIPPER_OPEN_QPOS / 2.0

    @cached_property
    def dataset_name(self) -> list[str]:
        return ['lerobot_state'] if self._use_state else ['lerobot_action']

    @cached_property
    def base_files_exists(self) -> bool:
        if not os.path.exists(self._parquet_path):
            return False
        return all(os.path.exists(self._video_path(cam)) for cam, _ in self._cams)

    @cached_property
    def clip_info(self) -> list[ClipInfo]:
        return [ClipInfo(start=0, end=len(self))]
