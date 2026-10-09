import datetime
import glob
import json
import os
from collections import OrderedDict
from functools import cached_property, lru_cache
from typing import Iterator, Sequence

import h5py
import numpy as np
from quaternion import as_float_array, from_rotation_vector

from utils import video_read

from .BaseDataset import ClipInfo, SourceDataset


# DROID 单臂 Franka 数据。
# 目录约定：{root}/{version}/{lab}/{status}/{date}/{ts}/{metadata_{uuid}.json, trajectory.h5, recordings/MP4/{serial}.mp4}
class DroidDataset(SourceDataset):

    # 15 Hz：RobotEnv.control_hz = 15
    # https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/robot_env.py#L31
    # raw trajectory.h5 中 cam estimated_capture 时间戳实测平均间隔 ≈ 67-71 ms，与 15 Hz 一致。
    fps = 15
    # ext1/ext2 = 两台外部 ZED；left/right = ZED 立体的左右目；wrist = 腕部 cam。
    # 单 episode 实际可用 cam 看 _cam_mp4_paths（从 metadata.*_mp4_path 字段反解）。
    cams = ('left', 'right', 'wrist')
    _AGGREGATED_ANNOTATIONS_FILENAME = 'aggregated-annotations-030724.json'

    def __init__(self, root: str, key: str, cams: Sequence[str | tuple[str, tuple[int, int]]], use_state: bool = False, *args, **kwargs):
        """key: '{version}/{lab}/{status}/{ts}'，date 段从 ts 反向解析；use_state 决定取 action/* 还是 observation/robot_state/*。"""
        super().__init__(*args, **kwargs)
        assert key.count('/') == 3, f'expect 4-segment key "version/lab/status/ts", got {key!r}'
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

    @staticmethod
    def _date_from_ts(ts: str) -> str:
        return datetime.datetime.strptime(ts.replace('_', ' '), '%a %b %d %H:%M:%S %Y').strftime('%Y-%m-%d')

    @classmethod
    def list_keys(cls, root: str) -> Iterator[str]:
        """枚举 {root}/{version}/{lab}/{status}/{date}/{ts}；key 输出跳过 date 段。"""
        if not os.path.isdir(root):
            return
        with os.scandir(root) as ver_it:
            for v in ver_it:
                if not v.is_dir():
                    continue
                with os.scandir(v.path) as lab_it:
                    for lab in lab_it:
                        if not lab.is_dir():
                            continue
                        with os.scandir(lab.path) as st_it:
                            for st in st_it:
                                if not (st.is_dir() and st.name in ('success', 'failure')):
                                    continue
                                with os.scandir(st.path) as date_it:
                                    for d in date_it:
                                        if not d.is_dir():
                                            continue
                                        with os.scandir(d.path) as ts_it:
                                            for ts in ts_it:
                                                if ts.is_dir():
                                                    yield f'{v.name}/{lab.name}/{st.name}/{ts.name}'

    @cached_property
    def _episode_dir(self) -> str:
        version, lab, status, ts = self._key.split('/')
        return os.path.join(self._root, version, lab, status, self._date_from_ts(ts), ts)

    @cached_property
    def _metadata(self) -> dict:
        metas = sorted(glob.glob(os.path.join(self._episode_dir, 'metadata_*.json')))
        assert len(metas) >= 1, f'no metadata json under {self._episode_dir}'
        with open(metas[0]) as f:
            return json.load(f)

    @cached_property
    def _trajectory_path(self) -> str:
        return os.path.join(self._episode_dir, 'trajectory.h5')

    @cached_property
    def _cam_mp4_paths(self) -> OrderedDict[str, str]:
        """从 metadata 中扫描所有 <name>_mp4_path 字段并解析为绝对路径，返回 name → abs_path。"""
        out: OrderedDict[str, str] = OrderedDict()
        _, _, status, ts = self._key.split('/')
        prefix_in_meta = f'{status}/{self._date_from_ts(ts)}/{ts}/'
        suffix = '_mp4_path'
        for k, rel in self._metadata.items():
            if not (isinstance(k, str) and k.endswith(suffix) and isinstance(rel, str) and rel):
                continue
            assert rel.startswith(prefix_in_meta), f'unexpected mp4 path {rel!r} (expect prefix {prefix_in_meta!r})'
            abs_path = os.path.join(self._episode_dir, rel[len(prefix_in_meta):])
            if os.path.exists(abs_path):
                out[k[:-len(suffix)]] = abs_path
        return out

    @cached_property
    def _raw_qpos(self) -> np.ndarray:
        """(h5_len, 8)：[arm(7) | gripper(1)]；gripper 已翻转为 0=closed / gripper_open_qpos=open。"""
        # use_state=True  → observation/robot_state/{joint_positions, gripper_position}
        # use_state=False → action/{joint_position, gripper_position}
        # raw gripper：0=open / 1=closed，本类翻转为 BaseDataset 约定 [0, gripper_open_qpos]，0=closed。
        if self._use_state:
            arm_key, grip_key = 'observation/robot_state/joint_positions', 'observation/robot_state/gripper_position'
        else:
            arm_key, grip_key = 'action/joint_position', 'action/gripper_position'
        with h5py.File(self._trajectory_path, 'r', locking=False) as f:
            arm = np.asarray(f[arm_key], dtype=np.float64)
            grip = np.asarray(f[grip_key], dtype=np.float64).reshape(-1, 1)
        return np.concatenate([arm, (1.0 - np.clip(grip, 0.0, 1.0)) * self.gripper_open_qpos], axis=-1)

    @cached_property
    def _raw_endpose(self) -> np.ndarray:
        """(h5_len, 7)：[xyz, qw, qx, qy, qz]，来自 observation/robot_state/cartesian_position。"""
        # cartesian_position = xyz + roll/pitch/yaw（'xyz' extrinsic 欧拉），与 DROID 采集端
        # quat_to_euler 反向一致：
        #   https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/misc/transformations.py#L6-L8
        #   写入处：https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/franka/robot.py#L162
        # 对应矩阵 R = R_z @ R_y @ R_x，四元数乘法顺序 q = q_z * q_y * q_x。
        # quaternion 库的 from_euler_angles 是 ZYZ 约定不能直接用
        #   https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible
        # 改走三个单轴 from_rotation_vector 合成，已通过 LeRobot v3 数据逐帧核对。
        #
        # 与 Bridge 不同：DROID euler 是世界系下的**绝对**末端朝向，没有 "相对 home" 偏置。
        with h5py.File(self._trajectory_path, 'r', locking=False) as f:
            cp = np.asarray(f['observation/robot_state/cartesian_position'], dtype=np.float64)
        r, p, y = cp[:, 3], cp[:, 4], cp[:, 5]
        zero = np.zeros_like(r)
        qx = from_rotation_vector(np.stack([r, zero, zero], axis=-1))
        qy = from_rotation_vector(np.stack([zero, p, zero], axis=-1))
        qz = from_rotation_vector(np.stack([zero, zero, y], axis=-1))
        quat_wxyz = as_float_array(qz * qy * qx)
        return np.concatenate([cp[:, :3], quat_wxyz], axis=-1)

    @cached_property
    def rgbs(self) -> OrderedDict[str, np.ndarray]:
        # 每路帧数 == len(self) == h5_len - 1（断言）。
        result: OrderedDict[str, np.ndarray] = OrderedDict()
        for cam, r in self._cams:
            path = self._cam_mp4_paths.get(cam)
            assert path is not None, (f'cam {cam!r} not available for {self._key}')
            frames = video_read(path, resolution=r)
            assert frames.shape[0] == len(self), (f'cam {cam!r} decoded {frames.shape[0]} frames, expected {len(self)}')
            result[cam] = frames
        return result

    @cached_property
    def _len(self) -> int:
        # h5 比 mp4 多 1 帧；以 mp4 帧数（= h5_len - 1）为准，所有数组截到 len(self)。
        with h5py.File(self._trajectory_path, 'r', locking=False) as f:
            db = f['action/joint_position']
            assert isinstance(db, h5py.Dataset)
            return db.shape[0] - 1

    @cached_property
    def qpos(self) -> np.ndarray:
        return self._raw_qpos[:len(self)]

    @cached_property
    def endpose(self) -> np.ndarray:
        return self._raw_endpose[:len(self), None, :]

    @cached_property
    def position(self) -> np.ndarray:
        out = np.zeros((len(self), 7), dtype=np.float64)
        out[:, 3] = 1.0
        return out

    @cached_property
    def _qpos_dims(self) -> dict[str, slice | np.ndarray]:
        return {'arm': slice(0, 7), 'gripper': slice(7, 8)}

    @cached_property
    def _endpose_arms(self) -> dict[str, int]:
        return {'right': 0}

    @cached_property
    def gripper_open_qpos(self) -> float:
        # Robotiq 2F-85 driver 关节「全开」对应的角度，单位 rad。
        # assets/droid_franka/droid_franka.urdf 已把 finger_joint 反向写成 [0, 0.725]：0=closed / 0.725=open，
        # 这样 qpos[gripper] 可以直接喂给 sapien.action 不再做映射。
        # 0.725 是 ros-industrial robotiq_arg2f_85_model_macro.xacro 里关节合拢前的「自然闭合」角度。
        return 0.725

    @cached_property
    def dataset_name(self) -> list[str]:
        return ['droid_state'] if self._use_state else ['droid_action']

    @cached_property
    def base_files_exists(self) -> bool:
        """trajectory.h5 + 至少一个 metadata_*.json + 已配置 cam 的 mp4 都在（_cam_mp4_paths 内部按文件存在性过滤）。"""
        if not os.path.exists(self._trajectory_path):
            return False
        if not glob.glob(os.path.join(self._episode_dir, 'metadata_*.json')):
            return False
        avail = self._cam_mp4_paths
        return all(cam in avail for cam, _ in self._cams)

    @staticmethod
    @lru_cache(maxsize=None)
    def ann_load(root: str, version: str, filename: str) -> dict:
        path = os.path.join(root, version, filename)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    @cached_property
    def clip_info(self) -> list[ClipInfo]:
        """整段 episode 一个 clip；action_text 合并 metadata 的 current_task/fixed_tasks/new_tasks，再追加 aggregated annotations 文件里 language_instruction1/2/3。"""
        m = self._metadata
        texts: list[str] = (([s] if (s := m.get('current_task')) else []) + m.get('fixed_tasks', []) + m.get('new_tasks', []))
        version, *_ = self._key.split('/')
        ann = self.ann_load(self._root, version, self._AGGREGATED_ANNOTATIONS_FILENAME).get(m.get('uuid', ''), {})
        for k in ('language_instruction1', 'language_instruction2', 'language_instruction3'):
            texts += [s] if (s := ann.get(k)) else []
        return [ClipInfo(
            start=0,
            end=len(self),
            action_text=list(set(texts)),
            task_name=[s] if (s := m.get('current_task')) else [],
        )]


# LeRobot v3 ep_idx -> DroidDataset key，由 .workmd/droid_verify_alignment.py 核对。
LEROBOT_V3_RAW_KEY_SAMPLES: dict[int, str] = {
    2: '1.0.1/TRI/success/Wed_Dec_13_15:55:58_2023',
    4: '1.0.1/AUTOLab/failure/Thu_Nov_23_18:22:14_2023',
    5: '1.0.1/AUTOLab/failure/Mon_Nov__6_16:13:40_2023',
    6: '1.0.1/AUTOLab/failure/Sun_Aug_27_23:44:33_2023',
    7: '1.0.1/TRI/success/Wed_Nov__8_10:22:23_2023',
    9: '1.0.1/IPRL/success/Sun_Apr_30_19:43:58_2023',
    10: '1.0.1/PennPAL/success/Mon_Oct__9_21:38:45_2023',
    11: '1.0.1/TRI/success/Mon_Dec__4_15:40:36_2023',
}
