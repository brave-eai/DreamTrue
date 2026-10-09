import json
import os
import os.path as osp
import re

import imageio
from PIL import Image

from .condition_source import ConditionSourceDataset
from ... import vace_contract


class AgibotWorldDataset(ConditionSourceDataset):
    """AgiBot adapter aligned with the current Condition_process layout.

    Raw data layout:
      ROOT/
        task_info/task_{task_id}{.split,.new,}.json
        observations[_untar]/{task_id}/{episode_id}/videos/{cam}_color.mp4
        proprio_stats[_untar]/{task_id}/{episode_id}/proprio_stats.h5
        conditions/{task_id}/{episode_id}/{condition_name}

    The old, larger implementation is backed up as agibot_world_legacy.py.
    """

    DATASET_TYPE = "agibot"
    DEFAULT_CONDITION_NAME = "condition-agibot_action.h5"
    THREEVIEWS_CAMERA_KEYS = ("head", "hand_left", "hand_right")
    LOGICAL_TO_PHYSICAL_CAMERA_KEYS = {
        "head": "head",
        "hand_left": "hand_left",
        "hand_right": "hand_right",
        "head_color": "head",
        "hand_left_color": "hand_left",
        "hand_right_color": "hand_right",
    }
    CAMERA_ALIASES = {
        "left": "hand_left",
        "right": "hand_right",
    }
    _TASK_INFO_SUFFIXES = (".split.json", ".new.json", ".json")

    def __init__(
        self,
        ROOT,
        camera_key="head",
        test_task_id=None,
        exclude_task_ids=None,
        include_task_ids=None,
        action_workers=None,
        proprio_stats_path="proprio_stats_untar",
        proprio_event=None,
        proprio_event_offset=None,
        proprio_all_events=False,
        proprio_event_ratio=1.0,
        condition_name=None,
        condition_mode=None,
        **kwargs,
    ):
        self.test_task_id = self._normalize_optional_task_id(test_task_id)
        self.exclude_task_ids = self._normalize_task_id_set(exclude_task_ids)
        self.include_task_ids = self._normalize_task_id_set(include_task_ids)
        self.action_workers = action_workers

        self.proprio_event = self._normalize_proprio_event(proprio_event)
        self.has_proprio_stats = self.proprio_event is not None
        self.proprio_event_offset = self._normalize_proprio_event_offset(proprio_event_offset)
        self.proprio_all_events = self._normalize_bool(proprio_all_events)
        self.proprio_event_ratio = self._normalize_proprio_event_ratio(proprio_event_ratio)
        if osp.isabs(str(proprio_stats_path)):
            self.proprio_stats_path = str(proprio_stats_path)
        else:
            self.proprio_stats_path = osp.join(ROOT, str(proprio_stats_path))

        if condition_name is None and condition_mode is None:
            condition_mode = vace_contract.raw_condition_mode_env() or "disabled"
        condition_mode = self._normalize_condition_mode_for_condition_source(condition_mode)

        super().__init__(
            ROOT=ROOT,
            camera_key=camera_key,
            condition_name=condition_name,
            condition_mode=condition_mode,
            **kwargs,
        )

    @staticmethod
    def _normalize_condition_mode_for_condition_source(mode):
        mode_text = str(mode or "").strip().lower()
        if mode_text in ("normal", "normal_png", "png"):
            return "disabled"
        return mode

    @staticmethod
    def _normalize_optional_task_id(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
            return None
        return str(int(value)) if str(value).isdigit() else str(value)

    @classmethod
    def _normalize_task_id_set(cls, value):
        if value is None:
            return set()
        if isinstance(value, (str, int)):
            value = [value]
        return {
            cls._normalize_optional_task_id(item)
            for item in value
            if cls._normalize_optional_task_id(item) is not None
        }

    @staticmethod
    def _normalize_bool(value):
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in ("1", "true", "yes", "y", "on"):
                return True
            if norm in ("0", "false", "no", "n", "off", "none", "null", ""):
                return False
            raise ValueError(f"Unsupported boolean value: {value}")
        return bool(value)

    @staticmethod
    def _normalize_proprio_event(value):
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in ("none", "null", "false", "0", ""):
                return None
            value = norm
        if value is not None and value not in ("rise", "fall", "both"):
            raise ValueError(f"Unsupported proprio_event: {value}")
        return value

    @staticmethod
    def _normalize_proprio_event_offset(value):
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in ("none", "null", "false", ""):
                return None
            value = norm
        if value is None:
            return None
        value = int(value)
        if value < 0:
            raise ValueError("proprio_event_offset must be >= 0")
        return value

    @staticmethod
    def _normalize_proprio_event_ratio(value):
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in ("", "none", "null"):
                value = 1.0
        if value is None:
            value = 1.0
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("proprio_event_ratio must be within [0, 1]")
        return value

    def _uses_nondefault_proprio_event_ratio(self):
        return self.proprio_event == "both" and self.proprio_event_ratio != 1.0

    def _should_force_random_window_for_action(self, rng):
        if self.cache_path is None:
            return False
        if self.proprio_event != "both":
            return False
        if self.proprio_event_ratio >= 1.0:
            return False
        return float(rng.random()) >= self.proprio_event_ratio

    def _cache_config_extra(self):
        extra = {
            "test_task_id": self.test_task_id,
            "include_task_ids": sorted(self.include_task_ids),
            "exclude_task_ids": sorted(self.exclude_task_ids),
            "proprio_event": self.proprio_event,
            "proprio_event_offset": self.proprio_event_offset,
            "proprio_all_events": self.proprio_all_events,
            "proprio_stats_path": self.proprio_stats_path,
        }
        if self._uses_nondefault_proprio_event_ratio():
            extra["proprio_event_ratio"] = format(float(self.proprio_event_ratio), ".17g")
        return extra

    def _is_cache_data_compatible(self, cache_data):
        if cache_data.get("version") not in (1, 2, self.CACHE_VERSION):
            return False, f"version mismatch (cached={cache_data.get('version')}, expected={self.CACHE_VERSION})"
        if cache_data.get("dataset_type", self.DATASET_TYPE) != self.DATASET_TYPE:
            return False, "dataset_type mismatch"
        if cache_data.get("ROOT", self.ROOT) != self.ROOT:
            return False, "ROOT mismatch"
        if cache_data.get("num_frames", self.num_frames) != self.num_frames:
            return False, "num_frames mismatch"
        if cache_data.get("threeviews_concat", self.threeviews_concat) != self.threeviews_concat:
            return False, "threeviews_concat mismatch"
        if cache_data.get("test_task_id", self.test_task_id) != self.test_task_id:
            return False, "test_task_id mismatch"
        if self.proprio_event == "both":
            cached_ratio = self._normalize_proprio_event_ratio(
                cache_data.get("proprio_event_ratio", 1.0)
            )
            if cached_ratio != self.proprio_event_ratio:
                return False, (
                    f"proprio_event_ratio mismatch "
                    f"(cached={cached_ratio}, current={self.proprio_event_ratio})"
                )
        return True, "ok"

    def _postprocess_loaded_clips(self, clips):
        return self._enrich_clips_with_proprio(clips)

    def _enrich_clips_with_proprio(self, clips):
        for clip in clips:
            if self.has_proprio_stats:
                task_id = clip.get("task_id")
                episode_id = clip.get("episode_id")
                proprio_file = osp.join(
                    self.proprio_stats_path,
                    str(task_id),
                    str(episode_id),
                    "proprio_stats.h5",
                )
                clip["proprio_file"] = proprio_file
                clip["has_proprio_stats"] = osp.exists(proprio_file)
            else:
                clip["has_proprio_stats"] = False
        return clips

    @staticmethod
    def _resolve_layout_suffix(root):
        for dirname in ("proprio_stats_untar", "observations_untar"):
            if osp.isdir(osp.join(root, dirname)):
                return "_untar"
        return ""

    def _layout_suffix_for_root(self, root):
        if root == self.ROOT:
            return self._layout_suffix
        return self._resolve_layout_suffix(root)

    @property
    def _layout_suffix(self):
        if not hasattr(self, "_agibot_layout_suffix"):
            self._agibot_layout_suffix = self._resolve_layout_suffix(self.ROOT)
        return self._agibot_layout_suffix

    def _observations_path_for_root(self, root):
        return osp.join(root, f"observations{self._layout_suffix_for_root(root)}")

    @property
    def _observations_path(self):
        return self._observations_path_for_root(self.ROOT)

    @property
    def _proprio_stats_scan_path(self):
        return osp.join(self.ROOT, f"proprio_stats{self._layout_suffix}")

    @property
    def _task_info_path(self):
        return osp.join(self.ROOT, "task_info")

    @classmethod
    def _task_info_file(cls, task_info_dir, task_id):
        for suffix in cls._TASK_INFO_SUFFIXES:
            path = osp.join(task_info_dir, f"task_{task_id}{suffix}")
            if osp.exists(path):
                return path
        return None

    @classmethod
    def _load_task_info(cls, task_info_dir, task_id):
        path = cls._task_info_file(task_info_dir, str(task_id))
        if path is None:
            raise FileNotFoundError(osp.join(task_info_dir, f"task_{task_id}.json"))
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def list_source_keys(cls, root):
        suffix = cls._resolve_layout_suffix(root)
        proprio_base = osp.join(root, f"proprio_stats{suffix}")
        if osp.isdir(proprio_base):
            with os.scandir(proprio_base) as task_it:
                for task in task_it:
                    if not (task.is_dir() and task.name.isdigit()):
                        continue
                    with os.scandir(task.path) as ep_it:
                        for episode in ep_it:
                            if episode.is_dir():
                                yield f"{task.name}/{episode.name}"
            return

        task_info_dir = osp.join(root, "task_info")
        if not osp.isdir(task_info_dir):
            return
        seen = set()
        pattern = re.compile(r"task_(\d+)(?:\.split|\.new)?\.json$")
        with os.scandir(task_info_dir) as task_it:
            for item in task_it:
                if not item.is_file():
                    continue
                match = pattern.fullmatch(item.name)
                if not match:
                    continue
                task_id = match.group(1)
                for episode in cls._load_task_info(task_info_dir, task_id):
                    episode_id = str(episode.get("episode_id"))
                    key = f"{task_id}/{episode_id}"
                    if key not in seen:
                        seen.add(key)
                        yield key

    def _filter_source_keys(self, keys):
        out = []
        for key in super()._filter_source_keys(keys):
            task_id = key.split("/", 1)[0]
            if self.test_task_id is not None and task_id != self.test_task_id:
                continue
            if self.include_task_ids and task_id not in self.include_task_ids:
                continue
            if task_id in self.exclude_task_ids:
                continue
            out.append(key)
        return out

    def _video_dir(self, key, root=None):
        root = root or self.ROOT
        task_id, episode_id = key.split("/")
        return osp.join(self._observations_path_for_root(root), task_id, episode_id, "videos")

    def _video_path(self, key, physical_camera, root=None):
        return osp.join(self._video_dir(key, root=root), f"{physical_camera}_color.mp4")

    def _episode_info(self, key):
        task_id, episode_id = key.split("/")
        episodes = self._load_task_info(self._task_info_path, task_id)
        target = str(int(episode_id)) if str(episode_id).isdigit() else str(episode_id)
        for episode in episodes:
            current = str(episode.get("episode_id"))
            current_norm = str(int(current)) if current.isdigit() else current
            if current_norm == target:
                return episode
        raise KeyError(f"episode {episode_id} not found in task_{task_id}")

    def _build_clips_for_key(self, seq_id, key, min_frames_required):
        video_dir = self._video_dir(key)
        if not osp.isdir(video_dir):
            return []
        required_cameras = [self._camera_key_to_physical(cam) for cam in self.camera_keys]
        if not all(osp.exists(self._video_path(key, cam)) for cam in required_cameras):
            return []

        try:
            episode = self._episode_info(key)
        except Exception:
            return []

        task_id, episode_id = key.split("/")
        task_name = episode.get("task_name")
        init_scene_text = episode.get("init_scene_text")
        action_configs = episode.get("label_info", {}).get("action_config", [])
        clips = []
        for action in action_configs:
            start = action.get("start_frame")
            end = action.get("end_frame")
            if start is None or end is None:
                continue
            start = int(start)
            end = int(end)
            frame_count = max(0, end - start)
            if frame_count < min_frames_required:
                continue

            action_text = action.get("action_text")
            skill = action.get("skill")
            for clip in self._clips_for_episode(
                seq_id=seq_id,
                source_key=key,
                frame_count=frame_count,
                action_text=[action_text] if action_text else [],
                task_name=[task_name] if task_name else [],
                skill=[skill] if skill else [],
                init_scene_text=[init_scene_text] if init_scene_text else [],
            ):
                clip["start_frame"] = start
                clip["end_frame"] = end
                clip["num_frames_available"] = frame_count
                clip["task_id"] = int(task_id) if task_id.isdigit() else task_id
                clip["episode_id"] = int(episode_id) if episode_id.isdigit() else episode_id
                clips.append(clip)
        return clips

    def _read_rgb_frames(self, clip_info, physical_camera, actual_indices):
        path = self._video_path(
            clip_info["source_key"],
            physical_camera,
            root=self._clip_data_root(clip_info),
        )
        if not osp.exists(path):
            raise FileNotFoundError(f"AgiBot camera video not found: {path}")
        reader = imageio.get_reader(path)
        try:
            return [
                (actual, Image.fromarray(reader.get_data(int(actual))).convert("RGB"))
                for actual in actual_indices
            ]
        finally:
            reader.close()


__all__ = ["AgibotWorldDataset"]
