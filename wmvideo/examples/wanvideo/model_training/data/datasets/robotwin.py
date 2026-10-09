import io
import json
import os
import os.path as osp

import cv2
import h5py
import numpy as np
from PIL import Image

from .condition_source import ConditionSourceDataset


class RoboTwinDataset(ConditionSourceDataset):
    DATASET_TYPE = "robotwin"
    DEFAULT_CONDITION_NAME = "condition.h5"
    LOGICAL_TO_PHYSICAL_CAMERA_KEYS = {
        "head": "head_camera",
        "hand_left": "left_camera",
        "hand_right": "right_camera",
        "front": "front_camera",
        "head_camera": "head_camera",
        "left_camera": "left_camera",
        "right_camera": "right_camera",
        "front_camera": "front_camera",
    }
    CONDITION_CAMERA_GROUP_ALIASES = {
        "head_camera": ("head",),
        "left_camera": ("left",),
        "right_camera": ("right",),
        "front_camera": ("front",),
    }
    CAMERA_ALIASES = {
        "left": "hand_left",
        "right": "hand_right",
    }
    _ROBOT_VARIANT_PREFIXES = (
        "aloha-agilex",
        "arx-x5",
        "franka",
        "piper",
        "ur5",
    )

    @classmethod
    def _is_robot_type(cls, variant):
        variant = str(variant)
        return any(
            variant == prefix
            or variant.startswith(f"{prefix}_")
            or variant.startswith(f"{prefix}-")
            for prefix in cls._ROBOT_VARIANT_PREFIXES
        )

    @classmethod
    def list_source_keys(cls, root):
        if not osp.isdir(root):
            return
        with os.scandir(root) as task_it:
            for task in task_it:
                if not task.is_dir():
                    continue
                with os.scandir(task.path) as variant_it:
                    for variant in variant_it:
                        if not (variant.is_dir() and cls._is_robot_type(variant.name)):
                            continue
                        data_dir = osp.join(variant.path, "data")
                        if not osp.isdir(data_dir):
                            continue
                        with os.scandir(data_dir) as ep_it:
                            for ep in ep_it:
                                if ep.is_file() and ep.name.endswith(".hdf5"):
                                    episode_id = ep.name[:-5]
                                    yield f"{task.name}/{variant.name}/{episode_id}"

    def _hdf5_path(self, key, root=None):
        root = root or self.ROOT
        task, robot_variant, episode_id = key.split("/")
        return osp.join(root, task, robot_variant, "data", f"{episode_id}.hdf5")

    def _instruction_path(self, key, root=None):
        root = root or self.ROOT
        task, robot_variant, episode_id = key.split("/")
        return osp.join(root, task, robot_variant, "instructions", f"{episode_id}.json")

    def _rgb_dataset_path(self, physical_camera):
        return f"observation/{physical_camera}/rgb"

    def _frame_count(self, h5_file, physical_cameras):
        lengths = []
        if "joint_action/vector" in h5_file:
            lengths.append(len(h5_file["joint_action/vector"]))
        for cam in physical_cameras:
            ds_path = self._rgb_dataset_path(cam)
            if ds_path not in h5_file:
                return 0
            lengths.append(len(h5_file[ds_path]))
        return int(min(lengths)) if lengths else 0

    @staticmethod
    def _texts_from_instruction(path):
        if not osp.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, str):
            return [data]
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        texts = []
        for key in ("seen", "unseen", "instruction", "instructions"):
            value = data.get(key)
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, list):
                texts.extend(value)
        return texts

    def _build_clips_for_key(self, seq_id, key, min_frames_required):
        path = self._hdf5_path(key)
        if not osp.exists(path):
            return []
        physical_cameras = [self._camera_key_to_physical(cam) for cam in self.camera_keys]
        with h5py.File(path, "r", locking=False) as f:
            frame_count = self._frame_count(f, physical_cameras)
        if frame_count < min_frames_required:
            return []
        task_name, robot_variant, _ = key.split("/")
        return self._clips_for_episode(
            seq_id=seq_id,
            source_key=key,
            frame_count=frame_count,
            action_text=self._texts_from_instruction(self._instruction_path(key)),
            task_name=[task_name],
            skill=[robot_variant],
        )

    @staticmethod
    def _decode_rgb_frame(raw):
        if isinstance(raw, np.ndarray):
            if raw.ndim == 0:
                raw = raw.item()
            elif raw.dtype == np.uint8 and raw.ndim in (2, 3):
                return Image.fromarray(raw).convert("RGB")
            elif raw.dtype == np.uint8:
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
            # RoboTwin 的 JPEG 是"反色"存的：用 cv2 原生序(不翻通道)解出来才是真彩色。
            # 与 Condition_process d6e5114 (fix rgb->bgr bug) 对齐——不能用 PIL 的 RGB 解码，
            # 否则 GT 目标帧会红/蓝调反，且与 condition_h5 里的条件 RGB 通道序相反。
            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode failed on RoboTwin RGB frame bytes")
            return Image.fromarray(img)
        if isinstance(raw, str):
            if osp.exists(raw):
                with Image.open(raw) as image:
                    return image.convert("RGB")
            with Image.open(io.BytesIO(raw.encode("latin1"))) as image:
                return image.convert("RGB")
        raise TypeError(f"Unsupported RoboTwin RGB frame type: {type(raw).__name__}")

    def _read_rgb_frames(self, clip_info, physical_camera, actual_indices):
        path = self._hdf5_path(
            clip_info["source_key"],
            root=self._clip_data_root(clip_info),
        )
        ds_path = self._rgb_dataset_path(physical_camera)
        with h5py.File(path, "r", locking=False) as f:
            if ds_path not in f:
                raise KeyError(f"Missing camera dataset: {ds_path}")
            ds = f[ds_path]
            out = []
            for actual in actual_indices:
                idx = min(int(actual), len(ds) - 1)
                out.append((actual, self._decode_rgb_frame(ds[idx])))
            return out


RobotwinDataset = RoboTwinDataset

__all__ = ["RoboTwinDataset", "RobotwinDataset"]
