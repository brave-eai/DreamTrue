from functools import cached_property

import numpy as np

from .BaseDataset import ModifiedDataset


class GripperKeepOpenDataset(ModifiedDataset):

    @cached_property
    def qpos(self) -> np.ndarray:
        results = self._d.qpos.copy()
        dims = self._qpos_dims['gripper']
        results[..., dims] = np.full_like(results[..., dims], self.gripper_open_qpos)
        return results

    @cached_property
    def dataset_name(self) -> list[str]:
        return self._d.dataset_name + ['gripper_keep_open']
