import abc
import os
import traceback
from collections import OrderedDict
from functools import cached_property
from typing import Iterator, Sequence

import cv2
import h5py
import numpy as np

from .BaseDataset import ClipInfo, SourceDataset


class _RobomindDataset(SourceDataset, abc.ABC):

    _status_dir: dict[str, str] = {'success': 'success_episodes', 'failed': 'failed_episodes'}
    _embodiment: str
    _cvt_color: int
    _arm_dof: int

    def __init__(self, root: str, key: str, cams: Sequence[str | tuple[str, tuple[int, int]]], use_state: bool = False, *args, **kwargs):
        """key: '{task}/{status}/{ts}'；use_state=False 取 master (action)，True 取 puppet (state)。"""
        # task   = task name（如 'add_sauce_to_pink_cup_with_both_arms'）
        # status ∈ {'success', 'failed'}（映射到 success_episodes / failed_episodes）
        # ts     = 时间戳目录名（如 '0509_130459'）
        super().__init__(*args, **kwargs)
        assert key.count('/') == 2, f'expect 3-segment key "task/status/ts", got {key!r}'
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

    @property
    def _trajectory_path(self) -> str:
        """拼 trajectory.hdf5：{root}/data/franka/{task}/{status_dir}/{ts}/data/trajectory.hdf5。"""
        task, status, ts = self._key.split('/')
        return os.path.join(
            self._root,
            'data',
            self._embodiment,
            task,
            self._status_dir[status],
            ts,
            'data',
            'trajectory.hdf5',
        )

    @classmethod
    def list_keys(cls, root: str) -> Iterator[str]:
        """枚举 {root}/data/{embodiment}/{task}/{success|failed}_episodes/{ts}，dir 名反向映射 status。"""
        base = os.path.join(root, 'data', cls._embodiment)
        if not os.path.isdir(base):
            return
        reverse = {v: k for k, v in cls._status_dir.items()}
        with os.scandir(base) as task_it:
            for task in task_it:
                if not task.is_dir():
                    continue
                with os.scandir(task.path) as st_it:
                    for sd in st_it:
                        if not (sd.is_dir() and sd.name in reverse):
                            continue
                        status = reverse[sd.name]
                        with os.scandir(sd.path) as ts_it:
                            for ts in ts_it:
                                if ts.is_dir():
                                    yield f'{task.name}/{status}/{ts.name}'

    @cached_property
    def base_files_exists(self) -> bool:
        return os.path.exists(self._trajectory_path)

    @cached_property
    def collection_info(self) -> tuple[str, str]:
        """(collector, collection_time)，取自 h5 metadata.attrs。"""
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            attrs = f['metadata'].attrs
            return str(attrs['collector']), str(attrs['collection_time'])

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        """{cam: (T, H, W, 3) uint8}，源字段 camera_observations/color_images/{cam} 存逐帧 JPG/PNG bytes，cv2 解码后转 RGB。"""
        result: OrderedDict[str, np.ndarray] = OrderedDict()
        with h5py.File(self._trajectory_path, 'r', locking=False) as f:
            for cam, r in self._cams:
                buf = f[f'camera_observations/color_images/{cam}']
                assert isinstance(buf, h5py.Dataset)
                frames = []
                for raw in buf:
                    arr = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
                    arr = cv2.cvtColor(arr, self._cvt_color)
                    if r is not None and arr.shape[:2][::-1] != tuple(r):
                        arr = cv2.resize(arr, r, interpolation=cv2.INTER_LINEAR)
                    frames.append(arr)
                result[cam] = np.stack(frames, axis=0)
        return result

    @cached_property
    def position(self) -> np.ndarray:
        n = len(self)
        out = np.zeros((n, 7), dtype=np.float64)
        out[:, 3] = 1.0
        return out

    @cached_property
    def _len(self) -> int:
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            db = f['master/arm_left_position_align/data']
            assert isinstance(db, h5py.Dataset)
            return db.shape[0]

    @cached_property
    def _qpos_dims(self) -> dict[str, slice | np.ndarray]:
        d = self._arm_dof
        return {
            'arm': slice(0, 2 * d),
            'arm_left': slice(0, d),
            'arm_right': slice(d, 2 * d),
            'gripper': slice(2 * d, 2*d + 2),
            'gripper_left': slice(2 * d, 2*d + 1),
            'gripper_right': slice(2*d + 1, 2*d + 2),
        }

    @cached_property
    def _endpose_arms(self) -> dict[str, int]:
        return {'left': 0, 'right': 1}

    @property
    def _qpos_path(self) -> str:
        """预提取的 qpos.hdf5，与 trajectory.hdf5 同目录。"""
        task, status, ts = self._key.split('/')
        return os.path.join(
            self._root,
            'data',
            self._embodiment,
            task,
            self._status_dir[status],
            ts,
            'data',
            'qpos.hdf5',
        )

    @property
    def _auto_qpos_path(self) -> str:
        return self._qpos_path if os.path.exists(self._qpos_path) else self._trajectory_path

    @cached_property
    def dataset_name(self) -> list[str]:
        return ['robomind_state'] if self._use_state else ['robomind_action']

    @cached_property
    def clip_info(self) -> list[ClipInfo]:
        """整段 episode 一个 clip；action_text 取 metadata.language_instruction，task_name 取 key 的 task 段。"""
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            instr = str(f['metadata'].attrs.get('language_instruction', ''))
        return [ClipInfo(
            start=0,
            end=len(self),
            action_text=[s] if (s := instr) else [],
            task_name=[self._key.split('/')[0]],
        )]


class RobomindFrankaDataset(_RobomindDataset):
    # 15 Hz：实测 Part-1 共 900 episode，时间戳秒级精度下 median ≈ 13.6 Hz / max ≈ 15.5 Hz。
    fps = 15
    cams = ('camera_front', 'camera_left', 'camera_right', 'camera_top', 'camera_wrist_left', 'camera_wrist_right')
    _cvt_color = cv2.COLOR_BGR2RGB
    _embodiment = 'franka'
    _arm_dof = 7

    @cached_property
    def qpos(self) -> np.ndarray:
        """(T, 16)：[arm_left(7) | arm_right(7) | gripper_left(1) | gripper_right(1)]。"""
        # 源字段 {prefix}/arm_{left,right}_position_align/data 各 8 列（前 7 关节角 + 第 8 gripper），
        # prefix 由 use_state 决定（True=puppet 机器人侧 / False=master 遥操作侧）。
        # Gripper 翻转：raw 0=open / 1=closed → BaseDataset 约定 [0, gripper_open_qpos]，0=closed。
        # 预提取的 qpos.hdf5 不存在时回退到 主文件 trajectory.hdf5。
        prefix = 'puppet' if self._use_state else 'master'
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            arm_l = np.array(f[f'{prefix}/arm_left_position_align/data'])
            arm_r = np.array(f[f'{prefix}/arm_right_position_align/data'])
        return np.concatenate([
            arm_l[:, :7],
            arm_r[:, :7],
            (1.0 - np.clip(arm_l[:, 7:8], 0.0, 1.0)) * self.gripper_open_qpos,
            (1.0 - np.clip(arm_r[:, 7:8], 0.0, 1.0)) * self.gripper_open_qpos,
        ], axis=-1).astype(np.float64)

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 0.725

    @cached_property
    def endpose(self) -> np.ndarray:
        prefix = 'puppet' if self._use_state else 'master'
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            left = np.array(f[f'{prefix}/end_effector_left_pose_align/data'])
            right = np.array(f[f'{prefix}/end_effector_right_pose_align/data'])
        left[:, 1] += 0.47
        right[:, 1] += -0.6
        # HDF5 quaternion is [qx,qy,qz,qw]; reorder to [qw,qx,qy,qz]
        left = left[:, [0, 1, 2, 6, 3, 4, 5]]
        right = right[:, [0, 1, 2, 6, 3, 4, 5]]
        return np.stack([left, right], axis=1).astype(np.float64)


class RobomindUR5Dataset(_RobomindDataset):
    # ~7 Hz：实测多 episode 时间戳秒级精度下 fps ≈ 7–8 Hz。
    fps = 7
    cams = ('camera_front', 'camera_left', 'camera_right', 'camera_top', 'camera_wrist_left', 'camera_wrist_right')
    _cvt_color = cv2.COLOR_BGR2RGB
    _embodiment = 'ur'
    _arm_dof = 6

    @cached_property
    def qpos(self) -> np.ndarray:
        """(T, 14)：[arm_left(6) | arm_right(6) | gripper_left(1) | gripper_right(1)]。
        UR5 每臂 6 关节角，gripper 在单独字段 end_effector_{side}_position_align/data。
        Gripper 翻转：raw 0=open / 1=closed → [0, gripper_open_qpos]，0=closed。
        """
        prefix = 'puppet' if self._use_state else 'master'
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            arm_l = np.array(f[f'{prefix}/arm_left_position_align/data'])
            arm_r = np.array(f[f'{prefix}/arm_right_position_align/data'])
            grip_l = np.array(f[f'{prefix}/end_effector_left_position_align/data'])
            grip_r = np.array(f[f'{prefix}/end_effector_right_position_align/data'])
        return np.concatenate([
            arm_l[:, :6],
            arm_r[:, :6],
            (1.0 - np.clip(grip_l, 0.0, 1.0)) * self.gripper_open_qpos,
            (1.0 - np.clip(grip_r, 0.0, 1.0)) * self.gripper_open_qpos,
        ], axis=-1).astype(np.float64)

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 1.0

    @cached_property
    def endpose(self) -> np.ndarray:
        raise NotImplementedError('受到臂的相对位置不确定的影响，目前 endpose 数据质量较差，暂不提供。')


class RobomindAgilexDataset(_RobomindDataset):
    cams = ('camera_front', 'camera_left', 'camera_right')
    _cvt_color = cv2.COLOR_BGR2RGB
    _arm_dof = 6
    # ~100 Hz：实测多 episode 时间戳秒级精度下 fps ≈ 99–102 Hz。
    fps = 100
    _embodiment = 'agilex'

    @cached_property
    def qpos(self) -> np.ndarray:
        """(T, 14)：[arm_left(6) | arm_right(6) | gripper_left(1) | gripper_right(1)]。
        Gripper 无需翻转，raw 已满足 0=closed 约定，直接 clip 负值到 0。
        """
        prefix = 'puppet' if self._use_state else 'master'
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            arm_l = np.array(f[f'{prefix}/arm_left_position_align/data'])
            arm_r = np.array(f[f'{prefix}/arm_right_position_align/data'])
            grip_l = np.array(f[f'{prefix}/end_effector_left_position_align/data'])
            grip_r = np.array(f[f'{prefix}/end_effector_right_position_align/data'])
        return np.concatenate([
            arm_l[:, :6],
            arm_r[:, :6],
            np.maximum(grip_l.reshape(-1, 1), 0.0),
            np.maximum(grip_r.reshape(-1, 1), 0.0),
        ], axis=-1).astype(np.float64)

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 1.0

    @cached_property
    def endpose(self) -> np.ndarray:
        raise NotImplementedError('受到臂的相对位置不确定的影响，目前 endpose 数据质量较差，暂不提供。')


class RobomindAgilexMobileDataset(_RobomindDataset):
    # ~60 Hz：实测多 episode 时间戳秒级精度下 fps ≈ 52–68 Hz，median ≈ 63 Hz。
    fps = 60
    _embodiment = 'agilex_mobile'
    cams = ('camera_front', 'camera_left', 'camera_right')
    _cvt_color = cv2.COLOR_BGR2RGB
    _arm_dof = 6

    @cached_property
    def qpos(self) -> np.ndarray:
        """(T, 14)：[arm_left(6) | arm_right(6) | gripper_left(1) | gripper_right(1)]。
        Gripper 无需翻转，raw 已满足 0=closed 约定，直接 clip 负值到 0。
        """
        prefix = 'puppet' if self._use_state else 'master'
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            arm_l = np.array(f[f'{prefix}/arm_left_position_align/data'])
            arm_r = np.array(f[f'{prefix}/arm_right_position_align/data'])
            grip_l = np.array(f[f'{prefix}/end_effector_left_position_align/data'])
            grip_r = np.array(f[f'{prefix}/end_effector_right_position_align/data'])
        return np.concatenate([
            arm_l[:, :6],
            arm_r[:, :6],
            np.maximum(grip_l.reshape(-1, 1), 0.0),
            np.maximum(grip_r.reshape(-1, 1), 0.0),
        ], axis=-1).astype(np.float64)

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 1.0

    @cached_property
    def position(self) -> np.ndarray:
        """(T, 7) 底盘位姿 [x,y,z, qw,qx,qy,qz]，从 chassis_pose_align 读取并重排四元数。"""
        prefix = 'puppet' if self._use_state else 'master'
        with h5py.File(self._auto_qpos_path, 'r', locking=False) as f:
            raw = np.array(f[f'{prefix}/chassis_pose_align/data'])
        # raw: [x,y,z, qx,qy,qz,qw] → [x,y,z, qw,qx,qy,qz]
        return raw[:, [0, 1, 2, 6, 3, 4, 5]].astype(np.float64)

    @cached_property
    def endpose(self) -> np.ndarray:
        raise NotImplementedError('受到臂的相对位置不确定的影响，目前 endpose 数据质量较差，暂不提供。')
