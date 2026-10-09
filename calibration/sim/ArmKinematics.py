import os
import re
from functools import cached_property

import numpy as np
import yaml
from mplib import Planner
from mplib.urdf_utils import generate_srdf

from utils import Pose


class IKFailedException(Exception):
    pass


class ArmKinematics:
    """yaml 里 `parts[arm].end_effector` 指定的 move_group 对应的单臂 FK/IK 求解器。"""

    def __init__(self, yaml_path: str, arm: str):
        self._yaml_path = yaml_path
        self._arm = arm

    @cached_property
    def _config(self) -> dict:
        with open(self._yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @cached_property
    def planner(self) -> Planner:
        urdf_path = os.path.join(os.path.dirname(self._yaml_path), self._config['urdf'])
        srdf_path = urdf_path.removesuffix('.urdf') + '_mplib.srdf'
        if not os.path.exists(srdf_path):
            generate_srdf(urdf_path)
        move_group = self._config['parts'][self._arm]['end_effector']
        return Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            move_group=move_group,
        )

    @cached_property
    def _mplib_joint_names(self) -> list[str]:
        return self.planner.user_joint_names

    @cached_property
    def _dataset_joint_names(self) -> list[str]:
        return [dof['joint'] for dof in self._config.get('dof', [])]

    @cached_property
    def n_mplib(self) -> int:
        return len(self._mplib_joint_names)

    @cached_property
    def dataset_to_mplib(self) -> np.ndarray:
        """长度=len(dataset joints)；值=mplib 列序号，-1 表示该 dataset dof 不出现在 mplib。"""
        idx = {n: i for i, n in enumerate(self._mplib_joint_names)}
        return np.array([idx.get(n, -1) for n in self._dataset_joint_names], dtype=np.intp)

    @cached_property
    def _valid_dataset(self) -> np.ndarray:
        return self.dataset_to_mplib >= 0

    @cached_property
    def move_link_idx(self) -> int:
        return self.planner.link_name_2_idx[self.planner.move_group]

    @cached_property
    def ik_mask(self) -> np.ndarray:
        """mplib 关节里 *不属于* 目标臂的位置为 True；IK 时用作锁定 mask。"""
        pat = re.compile(self._config['parts'][self._arm]['move_group'])
        return np.array([not bool(pat.match(n)) for n in self._mplib_joint_names], dtype=bool)

    def to_mplib_qpos(self, dataset_qpos_row: np.ndarray) -> np.ndarray:
        assert dataset_qpos_row.shape == (len(self.dataset_to_mplib), ), f'Expected dataset_qpos_row shape ({len(self.dataset_to_mplib)},), got {dataset_qpos_row.shape}'
        out = np.zeros(self.n_mplib)
        out[self.dataset_to_mplib[self._valid_dataset]] = dataset_qpos_row[self._valid_dataset]
        return out

    def _write_back(self, dataset_qpos_row: np.ndarray, mplib_qpos: np.ndarray) -> np.ndarray:
        out = dataset_qpos_row.copy()
        out[self._valid_dataset] = mplib_qpos[self.dataset_to_mplib[self._valid_dataset]]
        return out

    def fk(self, dataset_qpos_row: np.ndarray) -> Pose:
        """一帧 dataset 列序 qpos → move_group 末端世界位姿。"""
        mp = self.to_mplib_qpos(dataset_qpos_row)
        self.planner.pinocchio_model.compute_forward_kinematics(mp)
        return Pose.from_mplib(self.planner.pinocchio_model.get_link_pose(self.move_link_idx))

    def ik(
        self,
        dataset_qpos_row: np.ndarray,
        target: Pose,
        ik_seed_mplib: np.ndarray | None = None,
        n_init_qpos: int = 200,
        threshold: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """IK 求解；返回 (新 dataset 列序 qpos 行, mplib 列序结果 —— 下一帧可作 seed)。"""
        seed = self.to_mplib_qpos(dataset_qpos_row)
        if ik_seed_mplib is not None:
            assert ik_seed_mplib.shape == (self.n_mplib, ), f'Expected ik_seed_mplib shape ({self.n_mplib},), got {ik_seed_mplib.shape}'
            seed[~self.ik_mask] = ik_seed_mplib[~self.ik_mask]
        status, result = self.planner.IK(
            goal_pose=target.mplib,
            start_qpos=seed,
            mask=self.ik_mask,
            return_closest=True,
            n_init_qpos=n_init_qpos,
            threshold=threshold,
        )
        if not (status == 'Success' and result is not None):
            raise IKFailedException(f'IK failed for arm {self._arm!r}: {status}')
        result = np.asarray(result)
        return self._write_back(dataset_qpos_row, result), result
