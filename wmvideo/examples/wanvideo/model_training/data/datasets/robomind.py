import io
import os
import os.path as osp

import h5py
import numpy as np
from PIL import Image

from .condition_source import ConditionSourceDataset


class RoboMindDataset(ConditionSourceDataset):
    DATASET_TYPE = "robomind"
    DEFAULT_CONDITION_NAME = "condition-robomind_action.h5"
    LOGICAL_TO_PHYSICAL_CAMERA_KEYS = {
        "head": "camera_top",
        "hand_left": "camera_wrist_left",
        "hand_right": "camera_wrist_right",
        "camera_top": "camera_top",
        "camera_front": "camera_front",
        "camera_left": "camera_left",
        "camera_right": "camera_right",
        "camera_wrist_left": "camera_wrist_left",
        "camera_wrist_right": "camera_wrist_right",
    }
    CONDITION_CAMERA_GROUP_ALIASES = {
        "camera_top": ("top",),
        "camera_front": ("front",),
        "camera_left": ("left",),
        "camera_right": ("right",),
        "camera_wrist_left": ("wrist_left",),
        "camera_wrist_right": ("wrist_right",),
    }
    CAMERA_ALIASES = {
        "top": "head",
        "front": "camera_front",
        "wrist_left": "hand_left",
        "wrist_right": "hand_right",
    }
    _STATUS_DIR = {"success": "success_episodes", "failed": "failed_episodes"}
    _EMBODIMENT = "franka"

    @classmethod
    def list_source_keys(cls, root):
        base = osp.join(root, "data", cls._EMBODIMENT)
        if not osp.isdir(base):
            return
        reverse = {value: key for key, value in cls._STATUS_DIR.items()}
        with os.scandir(base) as task_it:
            for task in task_it:
                if not task.is_dir():
                    continue
                with os.scandir(task.path) as status_it:
                    for status_dir in status_it:
                        if not (status_dir.is_dir() and status_dir.name in reverse):
                            continue
                        status = reverse[status_dir.name]
                        with os.scandir(status_dir.path) as ts_it:
                            for ts in ts_it:
                                if ts.is_dir():
                                    yield f"{task.name}/{status}/{ts.name}"

    def _trajectory_path(self, key, root=None):
        root = root or self.ROOT
        task, status, ts = key.split("/")
        return osp.join(
            root,
            "data",
            self._EMBODIMENT,
            task,
            self._STATUS_DIR[status],
            ts,
            "data",
            "trajectory.hdf5",
        )

    def _color_dataset_path(self, physical_camera):
        return f"camera_observations/color_images/{physical_camera}"

    def _frame_count(self, h5_file, physical_cameras):
        lengths = []
        for cam in physical_cameras:
            ds_path = self._color_dataset_path(cam)
            if ds_path in h5_file:
                lengths.append(len(h5_file[ds_path]))
        return int(min(lengths)) if lengths else 0

    def _build_clips_for_key(self, seq_id, key, min_frames_required):
        path = self._trajectory_path(key)
        if not osp.exists(path):
            return []
        physical_cameras = [self._camera_key_to_physical(cam) for cam in self.camera_keys]
        with h5py.File(path, "r", locking=False) as f:
            if not all(self._color_dataset_path(cam) in f for cam in physical_cameras):
                return []
            frame_count = self._frame_count(f, physical_cameras)
            if frame_count < min_frames_required:
                return []
            instr = ""
            if "metadata" in f:
                instr = str(f["metadata"].attrs.get("language_instruction", ""))
        task_name = key.split("/")[0]
        return self._clips_for_episode(
            seq_id=seq_id,
            source_key=key,
            frame_count=frame_count,
            action_text=[instr] if instr else [],
            task_name=[task_name],
            skill=[key.split("/")[1]],
        )

    @staticmethod
    def _decode_color_frame(raw):
        if isinstance(raw, np.ndarray):
            if raw.ndim == 0:
                raw = raw.item()
            elif raw.dtype == np.uint8 and raw.ndim in (2, 3):
                return Image.fromarray(raw).convert("RGB")
            elif raw.dtype == np.uint8 and raw.ndim == 1:
                raw = raw.tobytes()
            else:
                try:
                    raw = raw.item()
                except Exception:
                    raw = raw.tobytes()
        if isinstance(raw, np.void):
            raw = raw.tobytes()
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, bytearray):
            raw = bytes(raw)
        if isinstance(raw, bytes):
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        if isinstance(raw, str):
            if osp.exists(raw):
                with Image.open(raw) as image:
                    return image.convert("RGB")
            with Image.open(io.BytesIO(raw.encode("latin1"))) as image:
                return image.convert("RGB")
        raise TypeError(f"Unsupported RoboMind color frame type: {type(raw).__name__}")

    def _read_rgb_frames(self, clip_info, physical_camera, actual_indices):
        path = self._trajectory_path(
            clip_info["source_key"],
            root=self._clip_data_root(clip_info),
        )
        ds_path = self._color_dataset_path(physical_camera)
        with h5py.File(path, "r", locking=False) as f:
            if ds_path not in f:
                raise KeyError(f"Missing camera dataset: {ds_path}")
            ds = f[ds_path]
            out = []
            for actual in actual_indices:
                idx = min(int(actual), len(ds) - 1)
                out.append((actual, self._decode_color_frame(ds[idx])))
            return out


__all__ = ["RoboMindDataset"]
