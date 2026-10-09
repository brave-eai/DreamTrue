import datetime
import glob
import json
import os
import os.path as osp

import h5py
import imageio
from PIL import Image

from .condition_source import ConditionSourceDataset


class DroidDataset(ConditionSourceDataset):
    DATASET_TYPE = "droid"
    DEFAULT_CONDITION_NAME = "condition-droid_action.h5"
    LOGICAL_TO_PHYSICAL_CAMERA_KEYS = {
        "head": "left",
        "hand_left": "right",
        "hand_right": "wrist",
        "ext1": "ext1",
        "ext2": "ext2",
        "wrist": "wrist",
        "left": "left",
        "right": "right",
    }
    _AGGREGATED_ANNOTATIONS_FILENAME = "aggregated-annotations-030724.json"

    @classmethod
    def list_source_keys(cls, root):
        if not osp.isdir(root):
            return
        with os.scandir(root) as ver_it:
            for version in ver_it:
                if not version.is_dir():
                    continue
                with os.scandir(version.path) as lab_it:
                    for lab in lab_it:
                        if not lab.is_dir():
                            continue
                        with os.scandir(lab.path) as status_it:
                            for status in status_it:
                                if not (status.is_dir() and status.name in ("success", "failure")):
                                    continue
                                with os.scandir(status.path) as date_it:
                                    for date_dir in date_it:
                                        if not date_dir.is_dir():
                                            continue
                                        with os.scandir(date_dir.path) as ts_it:
                                            for ts in ts_it:
                                                if ts.is_dir():
                                                    yield f"{version.name}/{lab.name}/{status.name}/{ts.name}"

    @staticmethod
    def _date_from_ts(ts):
        return datetime.datetime.strptime(
            ts.replace("_", " "), "%a %b %d %H:%M:%S %Y"
        ).strftime("%Y-%m-%d")

    def _episode_dir(self, key, root=None):
        root = root or self.ROOT
        version, lab, status, ts = key.split("/")
        return osp.join(root, version, lab, status, self._date_from_ts(ts), ts)

    def _metadata_path(self, key, root=None):
        metas = sorted(glob.glob(osp.join(self._episode_dir(key, root=root), "metadata_*.json")))
        return metas[0] if metas else None

    def _metadata(self, key, root=None):
        path = self._metadata_path(key, root=root)
        if path is None:
            raise FileNotFoundError(f"no metadata_*.json under {self._episode_dir(key, root=root)}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _trajectory_path(self, key, root=None):
        return osp.join(self._episode_dir(key, root=root), "trajectory.h5")

    def _cam_mp4_paths(self, key, metadata, root=None):
        out = {}
        _, _, status, ts = key.split("/")
        prefix = f"{status}/{self._date_from_ts(ts)}/{ts}/"
        suffix = "_mp4_path"
        for meta_key, rel in metadata.items():
            if not (
                isinstance(meta_key, str)
                and meta_key.endswith(suffix)
                and isinstance(rel, str)
                and rel
            ):
                continue
            if not rel.startswith(prefix):
                continue
            path = osp.join(self._episode_dir(key, root=root), rel[len(prefix):])
            if osp.exists(path):
                out[meta_key[: -len(suffix)]] = path
        return out

    def _frame_count(self, key, root=None):
        path = self._trajectory_path(key, root=root)
        if not osp.exists(path):
            return 0
        with h5py.File(path, "r", locking=False) as f:
            if "action/joint_position" in f:
                return max(0, int(f["action/joint_position"].shape[0]) - 1)
            if "observation/robot_state/joint_positions" in f:
                return max(0, int(f["observation/robot_state/joint_positions"].shape[0]) - 1)
        return 0

    def _texts_for_episode(self, key, metadata, root=None):
        root = root or self.ROOT
        texts = []
        if metadata.get("current_task"):
            texts.append(metadata["current_task"])
        texts.extend(metadata.get("fixed_tasks", []) or [])
        texts.extend(metadata.get("new_tasks", []) or [])
        version = key.split("/")[0]
        ann_path = osp.join(root, version, self._AGGREGATED_ANNOTATIONS_FILENAME)
        if osp.exists(ann_path):
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f).get(metadata.get("uuid", ""), {})
                for ann_key in (
                    "language_instruction1",
                    "language_instruction2",
                    "language_instruction3",
                ):
                    if ann.get(ann_key):
                        texts.append(ann[ann_key])
            except Exception:
                pass
        return texts

    def _build_clips_for_key(self, seq_id, key, min_frames_required):
        metadata = self._metadata(key)
        available = self._cam_mp4_paths(key, metadata)
        needed = [self._camera_key_to_physical(cam) for cam in self.camera_keys]
        if not all(cam in available for cam in needed):
            return []
        frame_count = self._frame_count(key)
        if frame_count < min_frames_required:
            return []
        return self._clips_for_episode(
            seq_id=seq_id,
            source_key=key,
            frame_count=frame_count,
            action_text=self._texts_for_episode(key, metadata),
            task_name=[metadata["current_task"]] if metadata.get("current_task") else [],
            skill=[key.split("/")[1]],
        )

    def _read_rgb_frames(self, clip_info, physical_camera, actual_indices):
        root = self._clip_data_root(clip_info)
        metadata = self._metadata(clip_info["source_key"], root=root)
        path = self._cam_mp4_paths(clip_info["source_key"], metadata, root=root).get(physical_camera)
        if path is None:
            raise FileNotFoundError(
                f"DROID camera {physical_camera!r} not available for {clip_info['source_key']}"
            )
        reader = imageio.get_reader(path)
        try:
            return [
                (actual, Image.fromarray(reader.get_data(int(actual))).convert("RGB"))
                for actual in actual_indices
            ]
        finally:
            reader.close()


__all__ = ["DroidDataset"]
