import os
import os.path
from collections import OrderedDict
from functools import cached_property
from typing import Iterator, Sequence

import cv2
import h5py
import numpy as np

from .BaseDataset import ClipInfo, SourceDataset


class AgibotCompetitionDataset(SourceDataset):

    # 5 Hz（与 Agibot 完整版的 30 Hz 不同）。
    fps: int = 5
    # 比赛数据无视频流，rgbs 只会从 frame.png 取 'head' 通道单帧。
    cams = ('head', )

    def __init__(self, root: str, key: str, cams: Sequence[str | tuple[str, tuple[int, int]]], use_state: bool = False, *args, **kwargs):
        """key: '{task_id}/{episode_id}'；use_state 必须为 True（无 action 数据源），cams 至少含 'head'。"""
        super().__init__(*args, **kwargs)
        assert key.count('/') == 1, f'got {key}'
        self._root = root
        self._key = key
        self._cams: list[tuple[str, None | tuple[int, int]]] = [cam if isinstance(cam, tuple) else (cam, None) for cam in cams]
        self._use_state = use_state

    @property
    def root(self):
        return self._root

    @property
    def key(self):
        return self._key

    @classmethod
    def list_keys(cls, root: str) -> Iterator[str]:
        """枚举 {root}/{task_id}/{episode_id}。"""
        if not os.path.isdir(root):
            return
        with os.scandir(root) as task_it:
            for t in task_it:
                if not t.is_dir():
                    continue
                with os.scandir(t.path) as ep_it:
                    for e in ep_it:
                        if e.is_dir():
                            yield f'{t.name}/{e.name}'

    @property
    def _proprio_path(self):
        return os.path.join(self._root, self._key, 'proprio_stats.h5')

    @property
    def _videos_path(self):
        """比赛数据无视频流，访问视频路径直接报错。"""
        raise NotImplementedError('Competition dataset does not support videos')

    @property
    def _frame_path(self):
        return os.path.join(self._root, self._key, 'frame.png')

    @property
    def _sam3_path(self):
        return os.path.join(self._root, self._key, 'sam3.png')

    @cached_property
    def frame(self) -> np.ndarray:
        """(H, W, 3) uint8 单帧初始 RGB 图，按 cams 中 'head' 的目标分辨率 resize。"""
        r = dict(self._cams)['head']
        img = cv2.cvtColor(cv2.imread(self._frame_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, r) if r is not None else img
        return img

    @cached_property
    def sam3(self) -> np.ndarray:
        """(H, W) SAM3 实例分割掩膜（按原始像素读入，不做色彩空间转换）。"""
        return cv2.imread(self._sam3_path, cv2.IMREAD_UNCHANGED)

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        result = OrderedDict()
        for k, r in self._cams:
            if k == 'head':
                result[k] = np.stack([self.frame] + [np.zeros_like(self.frame)] * (len(self) - 1), axis=0)
        return result

    @cached_property
    def _len(self) -> int:
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            db = f[f'state/head/position']
            assert isinstance(db, h5py.Dataset)
            return db.shape[0]

    @cached_property
    def _qpos_dims(self) -> dict[str, slice | np.ndarray]:
        return {
            'waist': slice(0, 2),
            'head': slice(2, 4),
            'arm': slice(4, 18),
            'arm_left': slice(4, 11),
            'arm_right': slice(11, 18),
            'gripper': slice(18, 20),
            'gripper_left': slice(18, 19),
            'gripper_right': slice(19, 20),
        }

    @cached_property
    def _endpose_arms(self) -> dict[str, int]:
        return {'left': 0, 'right': 1}

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 1.0

    @classmethod
    def _gripper_state_normalize(cls, state: np.ndarray) -> np.ndarray:
        """与 AgibotDataset._gripper_state_normalize 同：raw 范围 [31.288, 124.75714] → [0,1] 后翻转为 0=closed / 1=open；
        qpos 阶段再乘 gripper_open_qpos 落到 BaseDataset 约定的 [0, gripper_open_qpos]。"""
        return 1 - np.clip((state-31.288) / (124.75714-31.288), 0, 1)

    @cached_property
    def qpos(self) -> np.ndarray:
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            assert self._use_state, 'Competition dataset only supports gripper use state'
            gripper = self._gripper_state_normalize(np.array(f['state/effector/position']))
            results = np.concatenate([
                np.array(f[f'state/waist/position'])[..., ::-1],
                np.array(f[f'state/head/position']),
                np.array(f[f'state/joint/position']),
                gripper * self.gripper_open_qpos,
            ], axis=-1)
            return results

    @cached_property
    def position(self) -> np.ndarray:
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            position = np.array(f['state/robot/position'])
            orientation = np.array(f['state/robot/orientation'])
            if orientation.shape[0] == 0:
                orientation = np.zeros((len(self), 4))
                orientation[:, 3] = 1.0  # unit quaternion
            if position.shape[0] == 0:
                position = np.zeros((len(self), 3))
            return np.concatenate([
                position,
                orientation[..., [3, 0, 1, 2]],
            ], axis=-1)

    @cached_property
    def endpose(self) -> np.ndarray:
        with h5py.File(self._proprio_path, 'r', locking=False) as f:
            return np.concatenate([
                np.array(f['state/end/position']),
                np.array(f['state/end/orientation'])[..., [3, 0, 1, 2]],
            ], axis=-1)

    @cached_property
    def dataset_name(self) -> list[str]:
        return ['agibot_state'] if self._use_state else ['agibot_action']

    @cached_property
    def base_files_exists(self) -> bool:
        """proprio_stats.h5 必有；frame.png 仅在配置 'head' cam 时需要。clip_info 本类不支持，不算 base。"""
        if not os.path.exists(self._proprio_path):
            return False
        if any(k == 'head' for k, _ in self._cams) and not os.path.exists(self._frame_path):
            return False
        return True

    @cached_property
    def clip_info(self) -> list[ClipInfo]:
        raise NotImplementedError('AgibotCompetitionDataset does not support clip_info')
