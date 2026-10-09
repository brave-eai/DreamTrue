import hashlib
import json
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from quaternion import from_float_array, slerp

from sim import ArmKinematics, IKFailedException
from utils import Pose

from .BaseDataset import BaseDataset, ModifiedDataset


def _wrap_pi(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class IKDatasetConfig:
    start: int
    """Start  index (inclusive)."""

    end: int
    """End frame index (exclusive)."""

    delta_pose: Pose
    """The pose delta to apply at ``start`` (or uniformly when ``delta_pose_end`` is None)."""

    delta_pose_end: Pose | None = None
    """Optional end delta; when set, the delta is interpolated toward this value at ``end - 1``."""

    frame: Literal['local', 'world'] = 'local'
    """Coordinate frame for the delta:

    - ``'local'``: expressed in the end-effector frame. ``target = current @ delta``.
    - ``'world'``: ``target.p = current.p + delta.p``,
      ``target.q = delta.q * current.q`` (world-frame pre-multiply).
    """

    @classmethod
    def from_dict(cls, d: dict) -> 'IKDatasetConfig':
        return cls(
            start=int(d['start']),
            end=int(d['end']),
            delta_pose=Pose.from_dict(d['delta_pose']),
            delta_pose_end=Pose.from_dict(d['delta_pose_end']) if ('delta_pose_end' in d and d['delta_pose_end'] is not None) else None,
            frame=d['frame'],
        )

    @property
    def dict(self) -> dict:
        return {
            'start': self.start,
            'end': self.end,
            'delta_pose': self.delta_pose.dict,
            'frame': self.frame,
            'delta_pose_end': self.delta_pose_end.dict if self.delta_pose_end is not None else None,
        }


class IKDataset(ModifiedDataset):

    __extra_init_kwargs__ = ['assets_path', 'robot_path']

    def __init__(
        self,
        dataset: BaseDataset,
        arm: Literal['left', 'right'],
        segments: list[IKDatasetConfig | dict],
        name: str,
        assets_path: str,
        robot_path: str,
        vel_eps: float = 0.5,
        ik_threshold: float = 5e-3,
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, *args, **kwargs)
        self._arm = arm
        self._segments = [s if isinstance(s, IKDatasetConfig) else IKDatasetConfig.from_dict(s) for s in segments]
        self._modifier_name = name
        self._yaml_path = os.path.join(assets_path, robot_path)
        self._vel_eps = float(vel_eps)
        self._ik_threshold = float(ik_threshold)

    @property
    def modifier_name(self) -> str:
        return self._modifier_name

    @cached_property
    def dataset_name(self) -> list[str]:
        payload = {
            'arm': self._arm,
            'segments': [[
                s.start,
                s.end,
                s.delta_pose.p.tolist(),
                s.delta_pose.q.tolist(),
                None if s.delta_pose_end is None else [s.delta_pose_end.p.tolist(), s.delta_pose_end.q.tolist()],
                s.frame,
            ] for s in self._segments],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
        return self._d.dataset_name + [f'{self._modifier_name}_{digest}']

    @cached_property
    def _arm_kin(self) -> ArmKinematics:
        return ArmKinematics(self._yaml_path, self._arm)

    @staticmethod
    def __interpolate_delta(c: IKDatasetConfig, frame_idx: int) -> Pose:
        if c.delta_pose_end is None:
            return c.delta_pose
        assert c.end > c.start, f'Invalid segment with start {c.start} and end {c.end}'
        t = float(np.clip((frame_idx - c.start) / max(c.end - c.start - 1, 1), 0.0, 1.0))
        return Pose(
            p=(1.0-t) * c.delta_pose.p + t * c.delta_pose_end.p,
            q=slerp(from_float_array(c.delta_pose.q), from_float_array(c.delta_pose_end.q), 0, 1, t),
        )

    @staticmethod
    def __apply_delta(current: Pose, delta: Pose, frame: Literal['local', 'world']) -> Pose:
        if frame == 'local':
            return current @ delta
        elif frame == 'world':
            return delta @ current
        else:
            raise NotImplementedError(f'Unknown frame type: {frame}')

    @cached_property
    def qpos(self) -> np.ndarray:
        qpos = self._d.qpos.copy()
        n = len(qpos)
        active_segments = [s for s in self._segments if s.start < n and s.end > 0]
        if not active_segments:
            return qpos
        kin = self._arm_kin
        for segment in active_segments:
            prev_mp: np.ndarray | None = None
            for frame_idx in range(max(0, segment.start), min(n, segment.end)):
                cur_pose = kin.fk(qpos[frame_idx])
                tgt_pose = self.__apply_delta(cur_pose, self.__interpolate_delta(segment, frame_idx), segment.frame)
                try:
                    qpos[frame_idx], prev_mp = kin.ik(
                        dataset_qpos_row=qpos[frame_idx],
                        target=tgt_pose,
                        ik_seed_mplib=prev_mp,
                        threshold=self._ik_threshold,
                    )
                except IKFailedException as e:
                    raise IKFailedException(f'IK failed at frame {frame_idx} ({self._arm} arm): {e}') from e
        speed = np.abs(_wrap_pi(np.diff(qpos, axis=0)))
        limit = 2.0 * np.abs(_wrap_pi(np.diff(self._d.qpos, axis=0))) + self._vel_eps
        bad = np.argwhere(speed > limit)
        if len(bad) > 0:
            f, j = (int(x) for x in bad[0])
            raise IKFailedException(f'IK result jumps at frame {f}->{f+1} joint {j} ({self._arm} arm): {speed[f, j]:.4f} > limit {limit[f, j]:.4f} rad/step')
        return qpos

    @cached_property
    def endpose(self) -> np.ndarray:
        endpose = self._d.endpose.copy()
        n = len(endpose)
        kin = self._arm_kin
        col = self._d._endpose_arms[self._arm]
        qpos = self.qpos
        for frame_idx in range(n):
            pose = kin.fk(qpos[frame_idx])
            endpose[frame_idx, col, 0:3] = pose.p
            endpose[frame_idx, col, 3:7] = pose.q
        return endpose
