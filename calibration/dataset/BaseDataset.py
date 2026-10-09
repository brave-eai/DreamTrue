from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator, Sequence, TypeAlias, overload

import numpy as np


@dataclass(frozen=True)
class ClipInfo:
    """[start, end) 半开；start/end 是原 episode 帧坐标；空字段为 []。"""
    start: int
    end: int
    action_text: list[str] = field(default_factory=list)
    skill: list[str] = field(default_factory=list)
    task_name: list[str] = field(default_factory=list)
    init_scene_text: list[str] = field(default_factory=list)


class _BaseDataset(abc.ABC):

    @cached_property
    @abc.abstractmethod
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        """各路相机的 RGB 帧序列，value shape (T, H, W, 3) uint8。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def position(self) -> np.ndarray:
        """底盘位姿 (T, 7)，最后一维 [x, y, z, qw, qx, qy, qz]；静止底盘填 [0,0,0,1,0,0,0]。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def _len(self) -> int:
        raise NotImplementedError

    def __len__(self):
        return self._len

    @cached_property
    @abc.abstractmethod
    def qpos(self) -> np.ndarray:
        """关节角 + gripper (T, D)；gripper ∈ [0, gripper_open_qpos]，0=closed。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def endpose(self) -> np.ndarray:
        """末端位姿 (T, A, 7)，最后一维 [x, y, z, qw, qx, qy, qz]；A=臂的数量。
        每一列对应哪条臂由 _endpose_arms 声明。
        """
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def indices(self) -> np.ndarray:
        """(T,) int64，每帧对应原 episode 帧索引。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def _qpos_dims(self) -> dict[str, slice | np.ndarray]:
        """qpos 的列布局，name → 列切片/列索引。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def _endpose_arms(self) -> dict[str, int]:
        """endpose 第二维 (T, A, 7) 的 arm 名 → column 下标映射。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def gripper_open_qpos(self) -> float:
        """gripper 完全张开时 qpos 的取值；qpos 中 gripper 段 ∈ [0, gripper_open_qpos]，0=closed。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def dataset_name(self) -> list[str]:
        """数据来源标识。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def clip_info(self) -> list[ClipInfo]:
        """语义片段列表；start/end 是原 episode 帧坐标，要拿当前视图的帧位置用 clip_frames。"""
        raise NotImplementedError

    @cached_property
    @abc.abstractmethod
    def base_files_exists(self) -> bool:
        """构造此实例所需的全部 base 文件（满足 rgbs/qpos/endpose/position/clip_info）是否齐全。"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def source(self) -> 'SourceDataset':
        """追溯到最原始的 SourceDataset 实例；如果本身就是 SourceDataset 则返回自己。"""
        raise NotImplementedError

    def clip_frames(self, clip: ClipInfo) -> np.ndarray:
        """按 indices 反查当前视图中落在 [clip.start, clip.end) 内的帧位置。"""
        idx = self.indices
        return np.where((idx >= clip.start) & (idx < clip.end))[0]

    def find_clips(self, frame_idx: int) -> list[ClipInfo]:
        """按原 episode 帧索引查找覆盖该帧的 clip_info 列表（可能重叠，可能为空）。

        用例：取当前视图第 i 帧所属的 clip
            clips = ds.find_clips(int(ds.indices[i]))
            skill = clips[0].skill if clips else []
        """
        return [c for c in self.clip_info if c.start <= frame_idx < c.end]

    @staticmethod
    def zip_camera(cams: OrderedDict[str, np.ndarray]) -> list[OrderedDict[str, np.ndarray]]:
        """{cam: (T,...)} 横转为长度 T 的列表，每帧一个 {cam: 单帧} 字典。"""
        return [OrderedDict(zip(cams.keys(), a)) for a in zip(*cams.values())]

    @staticmethod
    def smooth_action(current: np.ndarray, target: np.ndarray, max_vel: float = 5e-2):
        """从 current 线性插值到 target，确保任一维单步变化不超过 max_vel。"""
        diff = np.abs(target - current)
        if np.all(diff <= max_vel):
            return np.array([target])
        num_steps = max(int(np.ceil(np.max(diff) / max_vel)), 2)
        t = np.linspace(0, 1, num_steps)[:, np.newaxis]
        return current + t * (target-current)


BaseDataset: TypeAlias = "SourceDataset | ModifiedDataset"


class SourceDataset(_BaseDataset):
    """从磁盘读取原始 episode 的具体数据集的公共基类。"""

    # 数据集采集名义帧率 (Hz)，所有具体子类必须以类属性形式设定。
    fps: int
    # 该数据集可用的相机 key 集合，对 __init__ 入参 cams 的合法取值；所有具体子类必须设定。
    cams: tuple[str, ...]

    @abc.abstractmethod
    def __init__(self, root: str = ..., key: str = ..., cams: Sequence[str | tuple[str, tuple[int, int]]] = ..., use_state: bool = ..., *args, **kwargs):
        """子类构造签名约定；root/key/cams 由具体子类自行存储，本基类只把剩余 *args/**kwargs 透传到 MRO 上游。"""
        super().__init__(*args, **kwargs)

    @property
    def source(self) -> 'SourceDataset':
        return self

    @property
    @abc.abstractmethod
    def root(self) -> str:
        """数据集根目录。"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def key(self) -> str:
        """episode 在数据集内的标识键。"""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def list_keys(cls, root: str) -> Iterator[str]:
        """惰性枚举 root 下所有候选 episode key（不验证 base_files_exists，调用方自行过滤）。"""
        raise NotImplementedError

    @cached_property
    def indices(self) -> np.ndarray:
        """source 数据集默认每帧对应原 episode 自身帧索引 [0, 1, ..., T-1]。"""
        return np.arange(len(self), dtype=np.int64)


class ModifiedDataset(_BaseDataset):
    """包装另一个 BaseDataset，默认对所有字段做透传；子类按需覆盖个别属性。"""

    __extra_init_kwargs__ = []  # for usage in build_modified_datasets

    def __init__(self, dataset: BaseDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._d: BaseDataset = dataset

    @cached_property
    def fps(self) -> int:
        if isinstance(self._d, SourceDataset):
            return self._d.fps
        elif isinstance(self._d, ModifiedDataset):
            return self._d.fps
        else:
            raise NotImplementedError(f'Unsupported dataset type: {type(self._d)}')

    @property
    def source(self) -> 'SourceDataset':
        """追溯到最原始的 SourceDataset 实例。"""
        return self._d.source

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        return self._d.rgbs

    @cached_property
    def position(self) -> np.ndarray:
        return self._d.position

    @cached_property
    def _len(self) -> int:
        return len(self._d)

    @cached_property
    def qpos(self) -> np.ndarray:
        return self._d.qpos

    @cached_property
    def endpose(self) -> np.ndarray:
        return self._d.endpose

    @cached_property
    def indices(self) -> np.ndarray:
        return self._d.indices

    @cached_property
    def _qpos_dims(self) -> dict[str, slice | np.ndarray]:
        return self._d._qpos_dims

    @cached_property
    def _endpose_arms(self) -> dict[str, int]:
        return self._d._endpose_arms

    @cached_property
    def gripper_open_qpos(self) -> float:
        return self._d.gripper_open_qpos

    @cached_property
    def clip_info(self) -> list[ClipInfo]:
        return self._d.clip_info

    @cached_property
    def base_files_exists(self) -> bool:
        return self._d.base_files_exists
