import abc
import json
import os
from collections import OrderedDict
from functools import cached_property
from typing import Iterator, Sequence

import cv2
import h5py
import numpy as np
from quaternion import as_float_array, as_rotation_matrix, from_float_array, from_rotation_matrix

from .BaseDataset import ClipInfo, SourceDataset


class _RobotwinDataset(SourceDataset):
    fps: int = int(round(250 / 15))  # simulation timestep 250, save_freq 15
    _arm_dof: int

    def __init__(self, root: str, key: str, cams: Sequence[str | tuple[str, tuple[int, int]]], use_state: bool = False, *args, **kwargs):
        """key: '{task}/{robot_variant}/{episode_id}'；use_state 暂未使用（仿真数据无 state/action 区分）。"""
        # task          = task name（如 'adjust_bottle'）
        # robot_variant = 机器人 + 变体（如 'franka_clean_50'）
        # episode_id    = episode 标识（如 'episode0'）
        super().__init__(*args, **kwargs)
        assert key.count('/') == 2, f'expect 3-segment key "task/robot_variant/episode_id", got {key!r}'
        self._root = root
        self._key = key
        self._cams: list[tuple[str, None | tuple[int, int]]] = [cam if isinstance(cam, tuple) else (cam, None) for cam in cams]
        self._use_state = use_state
        assert self._is_robot_type(self._key.split('/')[1]), f'incompatible robot type in key: {self._key.split("/")[1]!r}'

    @property
    def root(self):
        return self._root

    @property
    def key(self):
        return self._key

    @property
    def _episode_dir(self) -> str:
        """拼 episode 目录：{root}/{task}/{robot_variant}。"""
        task, robot_variant, _ = self._key.split('/')
        return os.path.join(self._root, task, robot_variant)

    @property
    def _episode_id(self) -> str:
        return self._key.split('/')[2]

    @property
    def _hdf5_path(self) -> str:
        return os.path.join(self._episode_dir, 'data', f'{self._episode_id}.hdf5')

    @property
    def _instruction_path(self) -> str:
        return os.path.join(self._episode_dir, 'instructions', f'{self._episode_id}.json')

    @classmethod
    @abc.abstractmethod
    def _is_robot_type(cls, variant: str) -> bool:
        raise NotImplementedError

    @classmethod
    def list_keys(cls, root: str) -> Iterator[str]:
        """枚举 {root}/{task}/{robot_variant}/data/episode{N}.hdf5。"""
        if not os.path.isdir(root):
            return
        with os.scandir(root) as task_it:
            for task in task_it:
                if not task.is_dir():
                    continue
                with os.scandir(task.path) as variant_it:
                    for variant in variant_it:
                        if not variant.is_dir():
                            continue
                        if not cls._is_robot_type(variant.name):
                            continue
                        data_dir = os.path.join(variant.path, 'data')
                        if not os.path.isdir(data_dir):
                            continue
                        with os.scandir(data_dir) as ep_it:
                            for ep in ep_it:
                                if ep.is_file() and ep.name.endswith('.hdf5'):
                                    episode_id = ep.name[:-5]  # strip .hdf5
                                    yield f'{task.name}/{variant.name}/{episode_id}'

    @cached_property
    def base_files_exists(self) -> bool:
        return os.path.exists(self._hdf5_path)

    @cached_property
    def _len(self) -> int:
        with h5py.File(self._hdf5_path, 'r', locking=False) as f:
            d = f['joint_action/vector']
            assert isinstance(d, h5py.Dataset)
            return d.shape[0]

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        """{cam: (T, H, W, 3) uint8}，源字段 observation/{cam}/rgb 存逐帧 JPEG bytes，cv2 解码后转 RGB。"""
        result: OrderedDict[str, np.ndarray] = OrderedDict()
        with h5py.File(self._hdf5_path, 'r', locking=False) as f:
            for cam, r in self._cams:
                buf = f[f'observation/{cam}/rgb']
                assert isinstance(buf, h5py.Dataset)
                frames = []
                for raw in buf:
                    img = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if r is not None and img.shape[:2][::-1] != tuple(r):
                        img = cv2.resize(img, r, interpolation=cv2.INTER_LINEAR)
                    frames.append(img)
                result[cam] = np.stack(frames, axis=0)
        return result

    @cached_property
    def position(self) -> np.ndarray:
        n = len(self)
        out = np.zeros((n, 7), dtype=np.float64)
        out[:, 3] = 1.0
        return out

    @cached_property
    def _endpose_arms(self) -> dict[str, int]:
        return {'left': 0, 'right': 1}

    @cached_property
    def dataset_name(self) -> list[str]:
        return ['robotwin_action']

    @cached_property
    def clip_info(self) -> list[ClipInfo]:
        """整段 episode 一个 clip；action_text 取 instructions/{episode_id}.json 的全部，task_name 取 key 的 task 段。"""
        if not os.path.exists(self._instruction_path):
            return [ClipInfo(start=0, end=len(self))]
        with open(self._instruction_path) as f:
            data = json.load(f)
        return [ClipInfo(
            start=0,
            end=len(self),
            action_text=data.get('seen', []) + data.get('unseen', []),
            task_name=[self._key.split('/')[0]],
        )]

    @cached_property
    def endpose(self) -> np.ndarray:
        with h5py.File(self._hdf5_path, 'r', locking=False) as f:
            left = np.array(f['endpose/left_endpose'], dtype=np.float64)
            right = np.array(f['endpose/right_endpose'], dtype=np.float64)
        return np.stack([self._convert_endpose(left), self._convert_endpose(right)], axis=1)

    @cached_property
    def qpos(self) -> np.ndarray:
        """(T, 2*_arm_dof+2)：[arm_left(D) | arm_right(D) | gripper_left(1) | gripper_right(1)]。"""
        with h5py.File(self._hdf5_path, 'r', locking=False) as f:
            left_arm = np.array(f['joint_action/left_arm'])
            right_arm = np.array(f['joint_action/right_arm'])
            left_gripper = np.array(f['joint_action/left_gripper'])
            right_gripper = np.array(f['joint_action/right_gripper'])
        assert left_arm.shape[1] == self._arm_dof
        assert right_arm.shape[1] == self._arm_dof
        return np.concatenate([
            left_arm,
            right_arm,
            left_gripper[:, np.newaxis] * self.gripper_open_qpos,
            right_gripper[:, np.newaxis] * self.gripper_open_qpos,
        ], axis=-1).astype(np.float64)

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

    _T_SCENE: np.ndarray
    _T_CORR: np.ndarray

    @classmethod
    def _convert_endpose(cls, ep: np.ndarray) -> np.ndarray:
        T = np.concatenate([
            np.concatenate([
                as_rotation_matrix(from_float_array(ep[:, 3:7])),
                ep[:, 0:3, np.newaxis],
            ], axis=2),
            np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float64), (len(ep), 4))[:, np.newaxis],
        ], axis=1)

        T_out = np.einsum('ij,tjk,kl->til', cls._T_SCENE, T, cls._T_CORR)

        return np.concatenate([
            T_out[:, :3, 3],
            as_float_array(from_rotation_matrix(T_out[:, :3, :3])),
        ], axis=-1)


class RobotwinFrankaDataset(_RobotwinDataset):

    cams = ('head_camera', 'left_camera', 'right_camera')
    _arm_dof: int = 7

    # HDF5 存的是 _trans_endpose(is_endpose=False)，不是 panda_hand link pose。
    # 实测：R_stored @ DM ≈ R_fk（误差 < 0.1°），不需要额外逆 GTM。
    # _T_CORR 右乘：逆 delta_matrix 旋转并撤销 dis=-0.04 平移
    # _T_SCENE 左乘：RoboTwin 场景 y=-0.65 平移到 y=0
    _T_CORR = np.array([
        [0.0, 0.0, 1.0, 0.04],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    _T_SCENE = np.array([
        [0.0, 1.0, 0.0, 0.65],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.75],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    @classmethod
    def _is_robot_type(cls, variant: str) -> bool:
        return variant.split('_')[0] == 'franka'

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 0.04


class RobotwinAlohaAgilexDataset(_RobotwinDataset):

    cams = ('head_camera', 'front_camera', 'left_camera', 'right_camera')
    _arm_dof: int = 6

    _T_CORR = np.eye(4, dtype=np.float64)
    _T_SCENE = np.array([
        [0.0, 1.0, 0.0, 0.419],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.782],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    @classmethod
    def _is_robot_type(cls, variant: str) -> bool:
        return variant.startswith('aloha-agilex')

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 0.04765


class RobotwinPiperDataset(_RobotwinDataset):

    cams = ('head_camera', 'left_camera', 'right_camera')
    _arm_dof: int = 6

    _T_CORR = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    _T_SCENE = np.array([
        [0.0, 1.0, 0.0, 0.45],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.75],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    @classmethod
    def _is_robot_type(cls, variant: str) -> bool:
        return variant.startswith('piper')

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 0.04


class RobotwinArxX5Dataset(_RobotwinDataset):

    cams = ('head_camera', 'left_camera', 'right_camera')
    _arm_dof: int = 6

    _T_CORR = np.eye(4, dtype=np.float64)
    _T_SCENE = np.array([
        [0.0, 1.0, 0.0, 0.35],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.784],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    @classmethod
    def _is_robot_type(cls, variant: str) -> bool:
        return variant.startswith('arx-x5')

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 0.044


class RobotwinWidowX250Dataset(_RobotwinDataset):

    cams = ('head_camera', 'left_camera', 'right_camera')
    _arm_dof: int = 6

    _T_CORR = np.array([
        [1.0, 0.0, 0.0, 0.153575],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    _T_SCENE = np.array([
        [0.0, 1.0, 0.0, 0.3],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.75],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    @classmethod
    def _is_robot_type(cls, variant: str) -> bool:
        return variant.startswith('widowx-250s')

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 0.027


class RobotwinUR5Dataset(_RobotwinDataset):

    cams = ('head_camera', 'left_camera', 'right_camera')
    _arm_dof: int = 6

    _T_CORR = np.array([
        [1.0, 0.0, 0.0, 0.01],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # 旧 URDF 左臂 base_z=0.65，右臂 base_z=0.75，T_SCENE 按左臂计算
    _T_SCENE = np.array([
        [0.0, 1.0, 0.0, 0.65],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.65],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    @classmethod
    def _is_robot_type(cls, variant: str) -> bool:
        return variant.split('_')[0] == 'ur5'

    @cached_property
    def gripper_open_qpos(self) -> float:
        return 0.055
