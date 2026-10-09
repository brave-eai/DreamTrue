import hashlib
import json
import os
import os.path as osp
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..base_dataset import BaseDataset

try:
    import hdf5plugin  # noqa: F401
except Exception:
    hdf5plugin = None


class ConditionSourceDataset(BaseDataset):
    """Training dataset adapter for Condition_process source datasets.

    Subclasses keep the raw-data layout knowledge close to the dataset type,
    while this base class owns the training-side contract: sampling, cache,
    VACE condition H5 loading, and optional per-frame camera parameters.
    """

    DEFAULT_PROMPT = "The robot is going to perform a manipulation task."
    THREEVIEWS_CAMERA_KEYS = ("head", "hand_left", "hand_right")
    LOGICAL_TO_PHYSICAL_CAMERA_KEYS = {}
    CONDITION_CAMERA_GROUP_ALIASES = {}
    CAMERA_ALIASES = {}
    DATASET_TYPE = "condition_source"
    DEFAULT_CONDITION_NAME = "condition.h5"
    CACHE_VERSION = 1

    # Sapien camera (+X forward, +Y left, +Z up) to OpenCV (+X right, +Y down, +Z forward).
    _SAPIEN_TO_OPENCV_M4 = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    _CONDITION_H5_FALLBACKS = {
        "condition.h5": "condition-state.h5",
    }

    def __init__(
        self,
        ROOT,
        camera_key="head",
        cache_path=None,
        cache_files=None,
        load_workers=32,
        min_interval=2,
        max_interval=2,
        threeviews_concat=False,
        use_plucker=False,
        pad_short_actions=False,
        detail_prompt=False,
        include_keys=None,
        exclude_keys=None,
        key_regex=None,
        condition_root=None,
        condition_name=None,
        condition_mode=None,
        manifest_path=None,
        manifest_entries=None,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.cache_path = cache_path
        self.cache_files = self._normalize_path_list(cache_files)
        self.manifest_path = manifest_path
        self.manifest_entries = self._load_manifest_entries(
            manifest_path=manifest_path,
            manifest_entries=manifest_entries,
        )
        self._manifest_cache_key = self._compute_manifest_cache_key()
        self.load_workers = max(1, int(load_workers))
        self.threeviews_concat = bool(threeviews_concat)
        self.use_plucker = bool(use_plucker)
        self.pad_short_actions = bool(pad_short_actions)
        self.detail_prompt = bool(detail_prompt)
        self.include_keys = self._normalize_key_list(include_keys)
        self.exclude_keys = self._normalize_key_list(exclude_keys)
        self.key_regex = str(key_regex) if key_regex else None
        self._key_pattern = re.compile(self.key_regex) if self.key_regex else None
        self.condition_root = condition_root or osp.join(ROOT, "conditions")
        self.vace_condition_h5_filename = self._resolve_condition_filename(
            condition_name=condition_name,
            condition_mode=condition_mode,
        )
        self.enable_vace_condition_h5 = self.vace_condition_h5_filename is not None
        self._intrinsic_attr_cache = {}

        if self.threeviews_concat:
            self.camera_keys = list(self.THREEVIEWS_CAMERA_KEYS)
        elif isinstance(camera_key, str):
            self.camera_keys = [self._normalize_camera_key(camera_key)]
        else:
            self.camera_keys = [self._normalize_camera_key(key) for key in camera_key]
        self.camera_key = self.camera_keys[0] if len(self.camera_keys) == 1 else list(self.camera_keys)

        if "base_path" not in kwargs:
            kwargs["base_path"] = ROOT
        super().__init__(
            min_interval=min_interval,
            max_interval=max_interval,
            **kwargs,
        )
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._load_data()

    @staticmethod
    def _normalize_path_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @staticmethod
    def _normalize_key_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            if osp.isfile(value):
                with open(value, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]
            if "," in value:
                return [part.strip() for part in value.split(",") if part.strip()]
            return [value]
        return [str(item) for item in value]

    @staticmethod
    def _load_manifest_entries(*, manifest_path=None, manifest_entries=None):
        if manifest_entries is None and manifest_path:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_entries = json.load(f)
        if manifest_entries is None:
            return []
        if not isinstance(manifest_entries, list):
            raise ValueError("manifest_entries must be a list of manifest records")
        return [
            {
                key: os.path.expandvars(value) if isinstance(value, str) else value
                for key, value in dict(entry).items()
            }
            for entry in manifest_entries
        ]

    def _compute_manifest_cache_key(self):
        if not self.manifest_entries:
            return ""
        digest = hashlib.md5()
        digest.update(str(osp.abspath(self.manifest_path or "")).encode("utf-8"))
        for entry in self.manifest_entries:
            for key in (
                "data_key",
                "source_key",
                "data_root",
                "start",
                "end",
                "condition_path",
                "condition_file",
            ):
                digest.update(str(entry.get(key, "")).encode("utf-8"))
                digest.update(b"\0")
        return digest.hexdigest()[:12]

    def _cache_config_extra(self):
        return {}

    def _cache_data_extra(self):
        return dict(self._cache_config_extra())

    def _postprocess_loaded_clips(self, clips):
        return clips

    def _clip_data_root(self, clip_info):
        return clip_info.get("data_root") or self.ROOT

    @classmethod
    def _physical_camera_keys(cls):
        return set(cls.LOGICAL_TO_PHYSICAL_CAMERA_KEYS.values())

    @classmethod
    def _normalize_camera_key(cls, camera_key):
        camera_key = str(camera_key)
        if camera_key in cls.LOGICAL_TO_PHYSICAL_CAMERA_KEYS:
            return camera_key
        for logical, physical in cls.LOGICAL_TO_PHYSICAL_CAMERA_KEYS.items():
            if camera_key == physical:
                return logical
        if camera_key in cls.CAMERA_ALIASES:
            return cls.CAMERA_ALIASES[camera_key]
        valid = (
            list(cls.LOGICAL_TO_PHYSICAL_CAMERA_KEYS)
            + sorted(cls._physical_camera_keys())
            + list(cls.CAMERA_ALIASES)
        )
        raise ValueError(f"Unsupported camera_key: {camera_key}. Supported keys: {valid}")

    @classmethod
    def _camera_key_to_physical(cls, camera_key):
        logical = cls._normalize_camera_key(camera_key)
        return cls.LOGICAL_TO_PHYSICAL_CAMERA_KEYS[logical]

    @classmethod
    def _condition_camera_candidates(cls, physical_camera):
        candidates = [str(physical_camera)]
        aliases = cls.CONDITION_CAMERA_GROUP_ALIASES.get(physical_camera, ())
        if isinstance(aliases, str):
            aliases = (aliases,)
        for alias in aliases:
            alias = str(alias)
            if alias not in candidates:
                candidates.append(alias)
        return candidates

    @classmethod
    def _resolve_condition_camera_key(cls, available_keys, physical_camera, condition_path, context):
        candidates = cls._condition_camera_candidates(physical_camera)
        available = set(available_keys)
        for candidate in candidates:
            if candidate in available:
                return candidate
        if len(candidates) == 1:
            raise KeyError(f"Missing camera group in {context}: {physical_camera} ({condition_path})")
        raise KeyError(
            f"Missing camera group in {context}: {physical_camera} "
            f"(tried {candidates}, found {list(available_keys)}) ({condition_path})"
        )

    @staticmethod
    def _mode_to_condition_h5_filename(mode):
        mode = str(mode or "").strip().lower()
        if mode in ("", "0", "false", "off", "none", "disabled"):
            return None
        if mode in ("condition_h5", "h5"):
            return "condition.h5"
        if re.fullmatch(r"condition[a-z0-9_-]*_h5", mode):
            return f"{mode[:-3]}.h5"
        match = re.fullmatch(r"(condition[a-z0-9_-]*)\.h5", mode)
        if match:
            return f"{match.group(1)}.h5"
        raise ValueError(f"Unsupported condition_mode: {mode}")

    def _resolve_condition_filename(self, *, condition_name, condition_mode):
        if condition_name is not None:
            condition_name = str(condition_name).strip()
            if condition_name.lower() in ("", "0", "false", "off", "none", "disabled"):
                return None
            return condition_name
        if condition_mode is not None:
            return self._mode_to_condition_h5_filename(condition_mode)
        return self.DEFAULT_CONDITION_NAME

    def _get_cache_filename(self):
        cfg = {
            "root": osp.abspath(self.ROOT) if self.ROOT else "",
            "dataset_type": self.DATASET_TYPE,
            "camera_keys": self.camera_keys,
            "threeviews_concat": self.threeviews_concat,
            "num_frames": self.num_frames,
            "min_interval": self.min_interval,
            "max_interval": self.max_interval,
            "pad_short_actions": self.pad_short_actions,
            "include_keys": self.include_keys,
            "exclude_keys": self.exclude_keys,
            "key_regex": self.key_regex,
            "condition_name": self.vace_condition_h5_filename,
            "manifest_path": osp.abspath(self.manifest_path) if self.manifest_path else "",
            "manifest_entries": len(self.manifest_entries),
            "manifest_cache_key": self._manifest_cache_key,
            "seed": self.seed,
        }
        cfg.update(self._cache_config_extra())
        config_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
        return (
            f"{self.DATASET_TYPE}_cache_"
            f"f{self.num_frames}_"
            f"c{len(self.camera_keys)}_"
            f"k{len(self.include_keys) or 'all'}_"
            f"{config_hash}.pkl"
        )

    def _load_single_explicit_cache(self, cache_file_path):
        if not osp.exists(cache_file_path):
            print(f"[Cache] Explicit cache file not found: {cache_file_path}")
            return None
        try:
            with open(cache_file_path, "rb") as f:
                cache_data = pickle.load(f)
        except Exception as e:
            print(f"[Cache] Failed to load explicit cache {cache_file_path}: {e}")
            return None

        version = cache_data.get("version")
        if version not in (1, 2, self.CACHE_VERSION):
            print(f"[Cache] Skip explicit cache with unsupported version {version}: {cache_file_path}")
            return None
        clips = cache_data.get("clips", [])
        if not clips:
            print(f"[Cache] Explicit cache contains no clips: {cache_file_path}")
            return None
        clips = self._postprocess_loaded_clips(clips)
        print(f"[Cache] Loaded {len(clips)} clips from explicit cache: {cache_file_path}")
        return clips

    def _load_from_explicit_cache_files(self):
        if not self.cache_files:
            return None

        if len(self.cache_files) == 1:
            clips = self._load_single_explicit_cache(self.cache_files[0])
            if clips:
                print(f"[Cache] Total {len(clips)} clips from 1 explicit cache file")
                return clips
            return None

        ordered_results = [None] * len(self.cache_files)
        num_workers = min(self.load_workers, len(self.cache_files))
        print(f"[Cache] Loading {len(self.cache_files)} explicit cache files with {num_workers} threads...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self._load_single_explicit_cache, path): idx
                for idx, path in enumerate(self.cache_files)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as e:
                    print(f"[Cache] Thread error loading {self.cache_files[idx]}: {e}")

        clips_all = []
        for clips in ordered_results:
            if clips:
                clips_all.extend(clips)
        if clips_all:
            print(f"[Cache] Total {len(clips_all)} clips from {len(self.cache_files)} explicit cache file(s)")
            return clips_all
        return None

    def _is_cache_data_compatible(self, cache_data):
        if cache_data.get("version") != self.CACHE_VERSION:
            return False, (
                f"version mismatch (cached={cache_data.get('version')}, "
                f"expected={self.CACHE_VERSION})"
            )
        if cache_data.get("dataset_type") != self.DATASET_TYPE:
            return False, "dataset_type mismatch"
        return True, "ok"

    def _load_from_cache(self):
        if self.cache_path is None:
            return None
        path = osp.join(self.cache_path, self._get_cache_filename())
        if not osp.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                cache_data = pickle.load(f)
        except Exception as e:
            print(f"[Cache] Failed to load {path}: {e}")
            return None
        compatible, reason = self._is_cache_data_compatible(cache_data)
        if not compatible:
            print(f"[Cache] Cache {reason}, regenerating...")
            return None
        clips = cache_data.get("clips", [])
        if not clips:
            return None
        clips = self._postprocess_loaded_clips(clips)
        print(f"[Cache] Loaded {len(clips)} clips from {path}")
        return clips

    def _save_to_cache(self, clips):
        if self.cache_path is None:
            return
        os.makedirs(self.cache_path, exist_ok=True)
        path = osp.join(self.cache_path, self._get_cache_filename())
        cache_data = {
            "version": self.CACHE_VERSION,
            "dataset_type": self.DATASET_TYPE,
            "ROOT": self.ROOT,
            "camera_keys": self.camera_keys,
            "threeviews_concat": self.threeviews_concat,
            "num_frames": self.num_frames,
            "min_interval": self.min_interval,
            "max_interval": self.max_interval,
            "pad_short_actions": self.pad_short_actions,
            "include_keys": self.include_keys,
            "exclude_keys": self.exclude_keys,
            "key_regex": self.key_regex,
            "condition_name": self.vace_condition_h5_filename,
            "manifest_path": self.manifest_path,
            "manifest_entries": len(self.manifest_entries),
            "manifest_cache_key": self._manifest_cache_key,
            "clips": clips,
        }
        cache_data.update(self._cache_data_extra())
        try:
            with open(path, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"[Cache] Saved {len(clips)} clips to {path}")
        except Exception as e:
            print(f"[Cache] Failed to save cache {path}: {e}")

    @classmethod
    def list_source_keys(cls, root):
        raise NotImplementedError

    def _build_clips_for_key(self, seq_id, key, min_frames_required):
        raise NotImplementedError

    def _read_rgb_frames(self, clip_info, physical_camera, actual_indices):
        raise NotImplementedError

    def _filter_source_keys(self, keys):
        include = set(self.include_keys)
        exclude = set(self.exclude_keys)
        out = []
        for key in keys:
            if include and key not in include:
                continue
            if key in exclude:
                continue
            if self._key_pattern and not self._key_pattern.search(key):
                continue
            out.append(key)
        return out

    @staticmethod
    def _manifest_texts(entry, key):
        value = entry.get(key, [])
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    def _build_clips_from_manifest(self):
        min_frames_required = 1 if self.pad_short_actions else max(
            2, (self.num_frames - 1) * int(self.min_interval) + 1
        )
        include = set(self.include_keys)
        exclude = set(self.exclude_keys)
        clips = []
        skipped = 0
        for seq_id, entry in enumerate(self.manifest_entries):
            source_key = entry.get("data_key") or entry.get("source_key")
            if not source_key:
                raise ValueError(f"manifest entry {seq_id} missing data_key/source_key")
            source_key = str(source_key)
            if include and source_key not in include:
                skipped += 1
                continue
            if source_key in exclude:
                skipped += 1
                continue
            if self._key_pattern and not self._key_pattern.search(source_key):
                skipped += 1
                continue

            start = int(entry.get("start", entry.get("start_frame", 0)))
            if "end" in entry:
                end = int(entry["end"])
                frame_count = max(0, end - start)
            elif "num_frames_available" in entry:
                frame_count = max(0, int(entry["num_frames_available"]))
                end = start + frame_count
            else:
                raise ValueError(
                    f"manifest entry {seq_id} ({source_key}) missing end/num_frames_available"
                )
            if frame_count < min_frames_required:
                skipped += 1
                continue

            entry_clips = self._clips_for_episode(
                seq_id=seq_id,
                source_key=source_key,
                frame_count=frame_count,
                action_text=self._manifest_texts(entry, "action_text"),
                task_name=self._manifest_texts(entry, "task_name"),
                skill=self._manifest_texts(entry, "skill"),
                init_scene_text=self._manifest_texts(entry, "init_scene_text"),
            )
            for clip in entry_clips:
                clip["start_frame"] = start
                clip["end_frame"] = end
                clip["num_frames_available"] = frame_count
                clip["manifest_index"] = seq_id
                condition_path = entry.get("condition_path")
                if condition_path:
                    clip["condition_path"] = str(condition_path)
                data_root = entry.get("data_root")
                if data_root:
                    clip["data_root"] = str(data_root)
                for meta_key in ("condition_root", "condition_file", "data_class", "fps"):
                    value = entry.get(meta_key)
                    if value not in (None, ""):
                        clip[meta_key] = value
                clips.append(clip)
        print(
            f"[{self.DATASET_TYPE}] Loaded {len(clips)} clip(s) from manifest"
            f" ({len(self.manifest_entries)} entries, skipped={skipped})"
        )
        return clips

    def _load_data(self):
        if self.cache_files:
            clips = self._load_from_explicit_cache_files()
            if clips is not None:
                self.scenes = clips
                return

        clips = self._load_from_cache()
        if clips is not None:
            self.scenes = clips
            return

        if self.manifest_entries:
            self.scenes = self._postprocess_loaded_clips(self._build_clips_from_manifest())
            self._save_to_cache(self.scenes)
            return

        keys = self._filter_source_keys(list(self.list_source_keys(self.ROOT)))
        if not keys:
            self.scenes = []
            print(f"[{self.DATASET_TYPE}] No source keys found under ROOT: {self.ROOT}")
            return

        min_frames_required = 1 if self.pad_short_actions else max(
            2, (self.num_frames - 1) * int(self.min_interval) + 1
        )
        print(
            f"[{self.DATASET_TYPE}] Found {len(keys)} episode(s), "
            f"min_frames_required={min_frames_required}, workers={self.load_workers}"
        )

        all_clips = []
        indexed_keys = list(enumerate(keys))
        if self.load_workers <= 1 or len(indexed_keys) <= 1:
            iterator = (
                tqdm(indexed_keys, desc=f"Loading {self.DATASET_TYPE}", unit="episode", dynamic_ncols=True)
                if self.use_tqdm
                else indexed_keys
            )
            for seq_id, key in iterator:
                all_clips.extend(self._build_clips_for_key(seq_id, key, min_frames_required))
        else:
            with ThreadPoolExecutor(max_workers=min(self.load_workers, len(indexed_keys))) as executor:
                future_to_key = {
                    executor.submit(self._build_clips_for_key, seq_id, key, min_frames_required): key
                    for seq_id, key in indexed_keys
                }
                pbar = None
                if self.use_tqdm:
                    pbar = tqdm(total=len(indexed_keys), desc=f"Loading {self.DATASET_TYPE}", unit="episode", dynamic_ncols=True)
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        all_clips.extend(future.result())
                    except Exception as e:
                        print(f"[{self.DATASET_TYPE}] Failed to process {key}: {e}")
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({"clips": len(all_clips)})
                if pbar is not None:
                    pbar.close()

        all_clips = self._postprocess_loaded_clips(all_clips)
        self.scenes = all_clips
        print(f"[{self.DATASET_TYPE}] Loaded {len(all_clips)} clip(s)")
        self._save_to_cache(all_clips)

    def _build_clip(
        self,
        *,
        seq_id,
        source_key,
        frame_count,
        camera_key,
        action_text=None,
        task_name=None,
        skill=None,
        init_scene_text=None,
    ):
        return {
            "dataset_type": self.DATASET_TYPE,
            "source_key": source_key,
            "task_id": self.DATASET_TYPE,
            "episode_id": source_key,
            "start_frame": 0,
            "end_frame": int(frame_count),
            "num_frames_available": int(frame_count),
            "camera_key": camera_key,
            "action_text": self._dedupe_texts(action_text or []),
            "task_name": self._dedupe_texts(task_name or []),
            "skill": self._dedupe_texts(skill or []),
            "init_scene_text": self._dedupe_texts(init_scene_text or []),
            "seq_id": int(seq_id),
        }

    @staticmethod
    def _dedupe_texts(texts: Iterable[str]):
        out = []
        seen = set()
        for text in texts:
            if text is None:
                continue
            text = str(text).strip()
            if not text or text in seen:
                continue
            out.append(text)
            seen.add(text)
        return out

    def _clips_for_episode(
        self,
        *,
        seq_id,
        source_key,
        frame_count,
        action_text=None,
        task_name=None,
        skill=None,
        init_scene_text=None,
    ):
        if frame_count <= 0:
            return []
        if self.threeviews_concat:
            return [
                self._build_clip(
                    seq_id=seq_id,
                    source_key=source_key,
                    frame_count=frame_count,
                    camera_key="+".join(self.camera_keys),
                    action_text=action_text,
                    task_name=task_name,
                    skill=skill,
                    init_scene_text=init_scene_text,
                )
            ]
        return [
            self._build_clip(
                seq_id=seq_id,
                source_key=source_key,
                frame_count=frame_count,
                camera_key=cam,
                action_text=action_text,
                task_name=task_name,
                skill=skill,
                init_scene_text=init_scene_text,
            )
            for cam in self.camera_keys
        ]

    def _make_clip_rng(self, clip_info):
        base_seed = self.seed if self.seed is not None else 0
        seed_str = (
            f"{base_seed}_"
            f"{clip_info.get('dataset_type')}_"
            f"{clip_info.get('source_key')}_"
            f"{clip_info.get('start_frame')}_"
            f"{clip_info.get('end_frame')}_"
            f"{clip_info.get('camera_key')}"
        )
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:16], 16)
        return np.random.default_rng(seed)

    @staticmethod
    def _normalize_sample_indices(sample_indices):
        if isinstance(sample_indices, np.ndarray):
            return sample_indices.astype(int).tolist()
        return [int(x) for x in sample_indices]

    def _build_padded_sample_indices_random(self, num_frames_available, rng):
        interval = int(self.max_interval)
        max_valid = min(
            (num_frames_available - 1) // interval + 1 if num_frames_available > 0 else 0,
            self.num_frames,
        )
        upper = min(max_valid, self.num_frames)
        if self.random_sample_start:
            # 随机采样 (训练增强 / 数据集显式开启 random_sample_start): 随机 n_real + 随机 start
            lower = max(upper - 10, 11)
            if lower > upper:
                lower = upper
            n_real = int(rng.integers(lower, upper + 1))
            clip_length = (n_real - 1) * interval + 1
            max_start = max(0, num_frames_available - clip_length)
            start = int(rng.integers(0, max_start + 1))
        else:
            # 默认确定性: 取可用的最大 n_real, 从头 (start=0) 覆盖整条片段
            n_real = max(1, upper)
            clip_length = (n_real - 1) * interval + 1
            start = 0
        real_indices = list(range(start, start + clip_length, interval))[:n_real]
        first_idx = real_indices[0] if real_indices else 0
        return [first_idx] * (self.num_frames - n_real) + real_indices, n_real

    def _select_sample_indices(self, clip_info, rng):
        if "sample_indices" in clip_info:
            return self._normalize_sample_indices(clip_info["sample_indices"]), clip_info.get("n_real_frames")

        num_frames_available = int(clip_info["num_frames_available"])
        min_required_for_max_interval = (self.num_frames - 1) * int(self.max_interval) + 1
        if self.pad_short_actions and num_frames_available < min_required_for_max_interval:
            indices, n_real = self._build_padded_sample_indices_random(num_frames_available, rng)
            return self._normalize_sample_indices(indices), n_real

        indices, _ = self.sample_from_video(
            num_frames_available,
            self.num_frames,
            self.min_interval,
            self.max_interval,
            rng,
        )
        return self._normalize_sample_indices(indices), None

    def _actual_indices(self, clip_info, sample_indices):
        start = int(clip_info["start_frame"])
        last = max(start, start + int(clip_info["num_frames_available"]) - 1)
        return [min(max(start + int(idx), 0), last) for idx in sample_indices]

    def _build_prompt(self, clip_info):
        if not self.detail_prompt:
            return self.DEFAULT_PROMPT
        init_scene = self._dedupe_texts(clip_info.get("init_scene_text", []))
        action_text = self._dedupe_texts(clip_info.get("action_text", []))
        task_name = self._dedupe_texts(clip_info.get("task_name", []))
        parts = []
        if init_scene:
            parts.append(f"The initial scene is that {'; '.join(init_scene)}.")
        target = action_text or task_name
        if target:
            parts.append(f"The robot is going to perform the following task: {'; '.join(target)}.")
        return " ".join(parts) if parts else self.DEFAULT_PROMPT

    def _condition_h5_path(self, clip_info):
        condition_path = clip_info.get("condition_path")
        if condition_path:
            return condition_path
        filename = self.vace_condition_h5_filename or "condition.h5"
        source_key = clip_info["source_key"]
        primary = osp.join(self.condition_root, source_key, filename)
        if osp.exists(primary):
            return primary
        fallback_name = self._CONDITION_H5_FALLBACKS.get(filename)
        if fallback_name:
            fallback = osp.join(self.condition_root, source_key, fallback_name)
            if osp.exists(fallback):
                return fallback
        return primary

    def _read_intrinsic_attr(self, condition_path):
        if condition_path in self._intrinsic_attr_cache:
            return self._intrinsic_attr_cache[condition_path]
        with h5py.File(condition_path, "r") as f:
            if "cameras_intrinsic" not in f.attrs:
                raise KeyError(f"condition h5 root missing 'cameras_intrinsic' attr: {condition_path}")
            intr = json.loads(f.attrs["cameras_intrinsic"])
        self._intrinsic_attr_cache[condition_path] = intr
        return intr

    @staticmethod
    def _build_opencv_K(intr_camera_dict, target_w, target_h):
        if any(k not in intr_camera_dict for k in ("rfx", "rfy", "rcx", "rcy", "w", "h")):
            fx, fy, cx, cy = (
                intr_camera_dict["fx"],
                intr_camera_dict["fy"],
                intr_camera_dict["cx"],
                intr_camera_dict["cy"],
            )
            w0, h0 = intr_camera_dict["w"], intr_camera_dict["h"]
        else:
            fx, fy, cx, cy = (
                intr_camera_dict["rfx"],
                intr_camera_dict["rfy"],
                intr_camera_dict["rcx"],
                intr_camera_dict["rcy"],
            )
            w0, h0 = intr_camera_dict["w"], intr_camera_dict["h"]
        sx = float(target_w) / float(w0)
        sy = float(target_h) / float(h0)
        return np.array(
            [
                [fx * sx, 0.0, cx * sx],
                [0.0, fy * sy, cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _load_camera_params_from_h5(self, clip_info, camera_key, sample_indices):
        if self.height is None or self.width is None:
            raise ValueError("use_plucker requires dataset height/width.")
        condition_path = self._condition_h5_path(clip_info)
        if not osp.exists(condition_path):
            raise FileNotFoundError(f"condition h5 not found for camera params: {condition_path}")
        physical_camera = self._camera_key_to_physical(camera_key)
        intr_all = self._read_intrinsic_attr(condition_path)
        condition_camera = self._resolve_condition_camera_key(
            intr_all.keys(),
            physical_camera,
            condition_path,
            "cameras_intrinsic JSON",
        )
        K_single = self._build_opencv_K(intr_all[condition_camera], int(self.width), int(self.height))
        actual_indices = self._actual_indices(clip_info, sample_indices)
        with h5py.File(condition_path, "r") as f:
            pose_name = f"{condition_camera}/global_pose"
            if pose_name not in f:
                raise KeyError(f"Missing dataset {pose_name} in {condition_path}")
            pose_ds = f[pose_name]
            sampled = []
            for actual in actual_indices:
                actual = min(int(actual), int(pose_ds.shape[0]) - 1)
                sampled.append(np.asarray(pose_ds[actual], dtype=np.float32))
        c2w_sapien = np.stack(sampled, axis=0)
        c2w_opencv = c2w_sapien @ self._SAPIEN_TO_OPENCV_M4
        K_per_frame = np.broadcast_to(K_single[None], (c2w_opencv.shape[0], 3, 3)).astype(np.float32, copy=True)
        return K_per_frame, c2w_opencv.astype(np.float32, copy=False)

    def _validate_condition_h5_dataset(
        self,
        *,
        camera_group,
        key,
        expected_ndim,
        expected_last_dim=None,
        expected_dtype=None,
        allowed_dtype_kinds=None,
        condition_path,
        camera_key,
    ):
        if key not in camera_group:
            raise KeyError(f"Missing dataset {camera_key}/{key} in {condition_path}")
        dataset = camera_group[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"Condition key is not an H5 dataset: {camera_key}/{key}")
        if dataset.ndim != expected_ndim:
            raise ValueError(f"Unexpected ndim for {camera_key}/{key}: shape={dataset.shape}")
        if expected_last_dim is not None and dataset.shape[-1] != expected_last_dim:
            raise ValueError(f"Unexpected channel dim for {camera_key}/{key}: shape={dataset.shape}")
        dtype = np.dtype(dataset.dtype)
        if expected_dtype is not None and dtype != np.dtype(expected_dtype):
            raise ValueError(f"Unexpected dtype for {camera_key}/{key}: got {dtype}, expected {np.dtype(expected_dtype)}")
        if allowed_dtype_kinds is not None and dtype.kind not in allowed_dtype_kinds:
            raise ValueError(f"Unexpected dtype for {camera_key}/{key}: got {dtype}")
        return dataset

    def _crop_resize_depth_if_necessary(self, depth_frame, resolution, rng, info=None):
        depth_array = np.asarray(depth_frame)
        if depth_array.ndim != 2:
            raise ValueError(f"Depth frame must be 2D, got shape={depth_array.shape}, info={info}")
        if depth_array.dtype != np.uint16:
            if np.issubdtype(depth_array.dtype, np.integer):
                depth_array = depth_array.astype(np.uint16)
            else:
                raise ValueError(f"Depth frame must be integer dtype, got dtype={depth_array.dtype}, info={info}")
        image = Image.fromarray(depth_array)
        target_resolution = np.array(resolution)
        rotate_to_portrait = False
        if self.landscape_check and image.size[0] < image.size[1] and resolution[0] != resolution[1]:
            target_resolution = np.array([resolution[1], resolution[0]])
            rotate_to_portrait = True
        noisy_resolution = target_resolution
        if self.aug_crop > 1:
            noisy_resolution = target_resolution + (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        image = image.resize(tuple(noisy_resolution), Image.NEAREST)
        left, top = np.int32(np.round((np.array(image.size) - target_resolution) / 2))
        out_width, out_height = target_resolution
        image = image.crop((left, top, left + out_width, top + out_height))
        if rotate_to_portrait:
            image = image.rotate(90 if rng.random() > 0.5 else -90, expand=True)
        arr = np.asarray(image)
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)
        return arr

    def _load_condition_h5_modalities(self, clip_info, camera_key, sample_indices, rng):
        condition_path = self._condition_h5_path(clip_info)
        if not osp.exists(condition_path):
            raise FileNotFoundError(
                f"Condition h5 not found: {condition_path}\n"
                f"Dataset: {self.DATASET_TYPE}, key: {clip_info.get('source_key')}, camera: {camera_key}"
            )

        physical_camera = self._camera_key_to_physical(camera_key)
        actual_indices = self._actual_indices(clip_info, sample_indices)
        rgb_frames = []
        depth_frames = []
        mask_frames = []
        with h5py.File(condition_path, "r") as f:
            condition_camera = self._resolve_condition_camera_key(
                f.keys(),
                physical_camera,
                condition_path,
                "condition h5",
            )
            camera_group = f[condition_camera]
            rgb_ds = self._validate_condition_h5_dataset(
                camera_group=camera_group,
                key="rgb",
                expected_ndim=4,
                expected_last_dim=3,
                expected_dtype=np.uint8,
                condition_path=condition_path,
                camera_key=condition_camera,
            )
            depth_ds = self._validate_condition_h5_dataset(
                camera_group=camera_group,
                key="depth",
                expected_ndim=3,
                expected_dtype=np.uint16,
                condition_path=condition_path,
                camera_key=condition_camera,
            )
            mask_ds = self._validate_condition_h5_dataset(
                camera_group=camera_group,
                key="mask",
                expected_ndim=3,
                allowed_dtype_kinds={"b", "u", "i"},
                condition_path=condition_path,
                camera_key=condition_camera,
            )
            n_total = int(rgb_ds.shape[0])
            for actual in actual_indices:
                actual = min(int(actual), n_total - 1)
                rgb_frame = np.asarray(rgb_ds[actual])
                depth_frame = np.asarray(depth_ds[actual], dtype=np.uint16)
                mask_frame = np.asarray(mask_ds[actual]).astype(bool)
                depth_frame = depth_frame.copy()
                depth_frame[~mask_frame] = 0
                rgb_image = Image.fromarray(rgb_frame).convert("RGB")
                mask_uint8 = mask_frame.astype(np.uint8)
                if self.height is not None and self.width is not None:
                    rgb_image = self._crop_resize_if_necessary(
                        rgb_image,
                        resolution=(self.width, self.height),
                        rng=rng,
                        info=f"{self.DATASET_TYPE}/{clip_info.get('source_key')}/{condition_camera}/rgb_{actual}",
                    )
                    depth_frame = self._crop_resize_depth_if_necessary(
                        depth_frame,
                        resolution=(self.width, self.height),
                        rng=rng,
                        info=f"{self.DATASET_TYPE}/{clip_info.get('source_key')}/{condition_camera}/depth_{actual}",
                    )
                    mask_uint8 = self._crop_resize_depth_if_necessary(
                        mask_uint8,
                        resolution=(self.width, self.height),
                        rng=rng,
                        info=f"{self.DATASET_TYPE}/{clip_info.get('source_key')}/{condition_camera}/mask_{actual}",
                    )
                rgb_frames.append(rgb_image)
                depth_frames.append(depth_frame)
                mask_frames.append((np.asarray(mask_uint8) > 0).astype(np.uint8))
        return {
            "vace_rgb_video": rgb_frames,
            "vace_depth_video": depth_frames,
            "vace_mask_video": mask_frames,
        }

    def _resize_rgb_frames(self, frames, camera_name, clip_info, rng):
        out = []
        for actual, frame in frames:
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame).convert("RGB")
            else:
                image = frame.convert("RGB")
            if self.height is not None and self.width is not None:
                image = self._crop_resize_if_necessary(
                    image,
                    resolution=(self.width, self.height),
                    rng=rng,
                    info=f"{self.DATASET_TYPE}/{clip_info.get('source_key')}/{camera_name}/frame_{actual}",
                )
            out.append(image)
        return out

    def _get_video_frames(self, idx, rng):
        clip_info = self.scenes[idx]
        prompt = self._build_prompt(clip_info)
        sample_indices, n_real_frames = self._select_sample_indices(clip_info, rng)
        actual_indices = self._actual_indices(clip_info, sample_indices)

        def load_video_frames(camera_name, indices=actual_indices):
            physical = self._camera_key_to_physical(camera_name)
            frames = self._read_rgb_frames(clip_info, physical, indices)
            return self._resize_rgb_frames(frames, physical, clip_info, rng)

        if self.threeviews_concat:
            result = {
                "head_video": load_video_frames("head"),
                "hand_left_video": load_video_frames("hand_left"),
                "hand_right_video": load_video_frames("hand_right"),
                "prompt": prompt,
                "task_id": clip_info.get("task_id"),
                "episode_id": clip_info.get("episode_id"),
                "source_key": clip_info.get("source_key"),
                "skill": clip_info.get("skill", []),
                "task_name": clip_info.get("task_name", []),
                "camera_key": clip_info.get("camera_key"),
            }
            if n_real_frames is not None:
                result["n_real_frames"] = n_real_frames
            if self.enable_vace_condition_h5:
                for view_cam in self.THREEVIEWS_CAMERA_KEYS:
                    mods = self._load_condition_h5_modalities(clip_info, view_cam, sample_indices, rng)
                    result[f"{view_cam}_vace_rgb_video"] = mods["vace_rgb_video"]
                    result[f"{view_cam}_vace_depth_video"] = mods["vace_depth_video"]
                    result[f"{view_cam}_vace_mask_video"] = mods["vace_mask_video"]
                    if self.use_plucker:
                        K_pf, c2w_pf = self._load_camera_params_from_h5(clip_info, view_cam, sample_indices)
                        result[f"{view_cam}_vace_intrinsic"] = K_pf
                        result[f"{view_cam}_vace_extrinsic"] = c2w_pf
            return result

        camera_key = clip_info["camera_key"]
        data = {
            "video": load_video_frames(camera_key),
            "prompt": prompt,
            "task_id": clip_info.get("task_id"),
            "episode_id": clip_info.get("episode_id"),
            "source_key": clip_info.get("source_key"),
            "skill": clip_info.get("skill", []),
            "task_name": clip_info.get("task_name", []),
            "camera_key": camera_key,
        }
        if n_real_frames is not None:
            data["n_real_frames"] = n_real_frames
        if self.enable_vace_condition_h5:
            mods = self._load_condition_h5_modalities(clip_info, camera_key, sample_indices, rng)
            data.update(mods)
            if self.use_plucker:
                K_pf, c2w_pf = self._load_camera_params_from_h5(clip_info, camera_key, sample_indices)
                data["vace_intrinsic"] = K_pf
                data["vace_extrinsic"] = c2w_pf
        return data
